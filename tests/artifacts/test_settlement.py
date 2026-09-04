from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from contextvars import ContextVar
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import cayu
import cayu.artifacts as artifacts
import cayu.artifacts.settlement as settlement_module
from cayu.artifacts._settlement import (
    _absent_artifact_write,
    _ArtifactWriteRegistry,
    _await_owned_sync_call,
    _committed_artifact_write,
    _settle_artifact_write,
    _unsettled_artifact_write,
)
from cayu.artifacts.base import ArtifactMetadata, ArtifactStoreUnavailableError
from cayu.artifacts.settlement import (
    ArtifactWriteSettlementEvidence,
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementObservation,
    ArtifactWriteSettlementObserver,
    ArtifactWriteSettlementPhase,
    ArtifactWriteSettlementRegistration,
    ArtifactWriteSettlementStatus,
    artifact_store_identity_sha256,
    artifact_write_settlements,
    copy_artifact_write_settlement,
    record_artifact_write_settlement,
    register_artifact_write_operation,
)

_ARTIFACT_ID = f"art_{'a' * 32}"


def _metadata() -> ArtifactMetadata:
    return ArtifactMetadata(
        id=_ARTIFACT_ID,
        filename="result.txt",
        size_bytes=4,
        session_id="sess_settlement",
    )


def _evidence() -> ArtifactWriteSettlementEvidence:
    now = datetime.now(UTC)
    return ArtifactWriteSettlementEvidence(
        operation_id=f"artifact_write_{'b' * 32}",
        artifact_id=_ARTIFACT_ID,
        store_identity_sha256=artifact_store_identity_sha256("store"),
        status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
        phase=ArtifactWriteSettlementPhase.COMMIT,
        observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
        started_at=now,
        observed_at=now,
        elapsed_ms=0,
        failure_codes=(ArtifactWriteSettlementFailureCode.COMMIT_FAILED,),
    )


def test_artifact_write_settlement_evidence_is_strict_bounded_and_frozen() -> None:
    evidence = _evidence()

    with pytest.raises(ValidationError):
        ArtifactWriteSettlementEvidence.model_validate(
            {**evidence.model_dump(), "elapsed_ms": True}
        )
    with pytest.raises(ValidationError):
        ArtifactWriteSettlementEvidence.model_validate(
            {**evidence.model_dump(), "artifact_id": "x" * 257}
        )
    with pytest.raises(ValidationError):
        ArtifactWriteSettlementEvidence.model_validate(
            {
                **evidence.model_dump(),
                "failure_codes": ["commit_failed", "commit_failed"],
            }
        )
    with pytest.raises(ValidationError):
        ArtifactWriteSettlementEvidence.model_validate(
            {**evidence.model_dump(), "status": "reconciliation_required"}
        )
    with pytest.raises(ValidationError):
        ArtifactWriteSettlementEvidence.model_validate(
            {
                **evidence.model_dump(),
                "status": ArtifactWriteSettlementStatus.COMMITTED,
                "phase": ArtifactWriteSettlementPhase.COMMIT,
            }
        )
    with pytest.raises(ValidationError):
        ArtifactWriteSettlementEvidence.model_validate(
            {**evidence.model_dump(), "phase": ArtifactWriteSettlementPhase.SETTLED}
        )
    with pytest.raises(ValidationError):
        evidence.status = ArtifactWriteSettlementStatus.COMMITTED  # type: ignore[misc]
    assert (
        ArtifactWriteSettlementEvidence.model_validate_json(evidence.model_dump_json()) == evidence
    )


def test_artifact_write_settlement_public_surface_is_exported() -> None:
    names = {
        "ArtifactWriteSettlementEvidence",
        "ArtifactWriteSettlementFailureCode",
        "ArtifactWriteSettlementObservation",
        "ArtifactWriteSettlementObserver",
        "ArtifactWriteSettlementPhase",
        "ArtifactWriteSettlementRegistration",
        "ArtifactWriteSettlementStatus",
        "artifact_store_identity_sha256",
        "artifact_write_settlements",
        "copy_artifact_write_settlement",
        "register_artifact_write_operation",
        "record_artifact_write_settlement",
    }

    for name in names:
        assert name in cayu.__all__
        assert name in artifacts.__all__
        assert getattr(cayu, name) is getattr(artifacts, name)


def test_owned_sync_call_preserves_contextvars() -> None:
    marker = ContextVar("artifact_settlement_test_marker", default="missing")
    observed: list[str] = []

    async def run() -> None:
        marker.set("captured")

        async def operation(reporter):
            observed.append(await _await_owned_sync_call(reporter, marker.get))
            return _committed_artifact_write(_metadata())

        await _settle_artifact_write(
            registry=_ArtifactWriteRegistry(),
            store_id="store",
            artifact_id=_ARTIFACT_ID,
            operation=operation,
        )

    asyncio.run(run())

    assert observed == ["captured"]


def test_artifact_write_settlement_observer_is_bounded_and_defensive() -> None:
    evidence = _evidence()
    error = RuntimeError("write failed")

    with ArtifactWriteSettlementObserver(max_operations=1) as observer:
        record_artifact_write_settlement(evidence, error=error)

    observed = observer.snapshot()
    attached = artifact_write_settlements(error)
    assert observed == (evidence,)
    assert attached == (evidence,)
    assert observed[0] is not evidence
    assert observer.drain() == (evidence,)
    assert observer.snapshot() == ()
    with pytest.raises(TypeError):
        record_artifact_write_settlement(evidence, error=object())  # type: ignore[arg-type]


@pytest.mark.parametrize("copy_kind", ("function", "model_copy"))
def test_artifact_write_settlement_copy_bounds_mutated_failure_code_iterables(
    copy_kind,
) -> None:
    class UnboundedFailureCodes:
        def __init__(self) -> None:
            self.visited = 0

        def __iter__(self):
            while True:
                self.visited += 1
                yield ArtifactWriteSettlementFailureCode.COMMIT_FAILED

    evidence = _evidence()
    failure_codes = UnboundedFailureCodes()
    object.__setattr__(evidence, "failure_codes", failure_codes)

    with pytest.raises(ValidationError):
        if copy_kind == "function":
            copy_artifact_write_settlement(evidence)
        else:
            evidence.model_copy()

    assert failure_codes.visited == 9


def test_artifact_write_settlement_bounds_exception_group_scheduling(monkeypatch) -> None:
    children = tuple(RuntimeError(f"failure-{index}") for index in range(1024))
    evidence = _evidence()
    record_artifact_write_settlement(evidence, error=children[0])
    group = BaseExceptionGroup("wide group", children)
    limits: list[int | None] = []
    original = settlement_module.exception_group_children

    def bounded_children(error, *, maximum=None):
        limits.append(maximum)
        return original(error, maximum=maximum)

    monkeypatch.setattr(settlement_module, "exception_group_children", bounded_children)

    assert artifact_write_settlements(group) == (evidence,)
    assert limits == [255]


def test_third_party_registration_exposes_active_and_late_evidence() -> None:
    error = RuntimeError("third-party write settled late")

    with ArtifactWriteSettlementObserver(max_operations=1) as observer:
        registration = register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
            operation_id=f"artifact_write_{'c' * 32}",
        )
        assert isinstance(registration, ArtifactWriteSettlementRegistration)
        registration.set_phase(ArtifactWriteSettlementPhase.COMMIT)
        active = observer.record_active_candidates()
        final = registration.record(
            status=ArtifactWriteSettlementStatus.COMMITTED,
            phase=ArtifactWriteSettlementPhase.SETTLED,
            error=error,
        )

    assert active[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert active[0].phase is ArtifactWriteSettlementPhase.COMMIT
    assert final.observation is ArtifactWriteSettlementObservation.LATE
    assert artifact_write_settlements(error) == (final,)
    assert observer.snapshot() == (active[0], final)
    assert observer.record_active_candidates() == ()


def test_third_party_registration_reports_late_without_an_observer(caplog) -> None:
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        candidates: list[ArtifactWriteSettlementEvidence] = []
        late_records: list[ArtifactWriteSettlementEvidence] = []
        retained_tasks: list[asyncio.Task[None]] = []

        async def write() -> None:
            registration = register_artifact_write_operation(
                artifact_id=_ARTIFACT_ID,
                store_id="third-party-store",
            )
            registration.set_phase(ArtifactWriteSettlementPhase.CONTENT)
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError as cancellation:
                candidates.append(
                    registration.record(
                        status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
                        error=cancellation,
                        final=False,
                    )
                )

                async def settle_late(
                    *,
                    owned_registration=registration,
                    owned_cancellation=cancellation,
                ) -> None:
                    await release.wait()
                    late_records.append(
                        owned_registration.record(
                            status=ArtifactWriteSettlementStatus.COMMITTED,
                            phase=ArtifactWriteSettlementPhase.SETTLED,
                            error=owned_cancellation,
                        )
                    )

                retained_tasks.append(asyncio.create_task(settle_late()))
                raise

        task = asyncio.create_task(write())
        await started.wait()
        task.cancel("direct SDK caller stopped")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        release.set()
        await retained_tasks[0]
        return task, raised.value, candidates[0], late_records[0]

    caplog.set_level(logging.INFO, logger="cayu.artifacts.settlement")
    task, cancellation, candidate, late = asyncio.run(scenario())

    assert task.cancelled()
    assert task.cancelling() == 1
    assert candidate.observation is ArtifactWriteSettlementObservation.CALLER_BOUNDARY
    assert late.observation is ArtifactWriteSettlementObservation.LATE
    assert artifact_write_settlements(cancellation) == (candidate, late)
    assert "status=committed" in caplog.text


def test_third_party_registration_rejects_a_retained_operation_id() -> None:
    operation_id = f"artifact_write_{'d' * 32}"

    with ArtifactWriteSettlementObserver(max_operations=1) as observer:
        first = register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
            operation_id=operation_id,
        )
        first_record = first.record(
            status=ArtifactWriteSettlementStatus.COMMITTED,
            phase=ArtifactWriteSettlementPhase.SETTLED,
        )

        with pytest.raises(ValueError, match="already active or retained"):
            register_artifact_write_operation(
                artifact_id=_ARTIFACT_ID,
                store_id="third-party-store",
                operation_id=operation_id,
            )

        assert observer.drain() == (first_record,)
        reused_after_drain = register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
            operation_id=operation_id,
        )
        reused_after_drain.close()


def test_third_party_registration_finalization_is_atomic_with_timeout_sampling(
    monkeypatch,
) -> None:
    def reject_split_release(_registration) -> None:
        raise AssertionError("terminal recording must not release through close()")

    monkeypatch.setattr(
        ArtifactWriteSettlementRegistration,
        "close",
        reject_split_release,
    )
    for index in range(50):
        observer = ArtifactWriteSettlementObserver(max_operations=1)
        with observer:
            registration = register_artifact_write_operation(
                artifact_id=_ARTIFACT_ID,
                store_id="third-party-store",
                operation_id=f"artifact_write_{index:032x}",
            )
            barrier = threading.Barrier(2)
            finals: list[ArtifactWriteSettlementEvidence] = []
            failures: list[BaseException] = []

            def finalize(
                owned_registration: ArtifactWriteSettlementRegistration,
                owned_barrier: threading.Barrier,
                observed_finals: list[ArtifactWriteSettlementEvidence],
                observed_failures: list[BaseException],
            ) -> None:
                try:
                    owned_barrier.wait(timeout=10)
                    observed_finals.append(
                        owned_registration.record(
                            status=ArtifactWriteSettlementStatus.COMMITTED,
                            phase=ArtifactWriteSettlementPhase.SETTLED,
                        )
                    )
                except BaseException as exc:
                    observed_failures.append(exc)

            thread = threading.Thread(
                target=finalize,
                args=(registration, barrier, finals, failures),
            )
            thread.start()
            barrier.wait(timeout=10)
            candidates = observer.record_active_candidates()
            thread.join(timeout=10)

        assert not thread.is_alive()
        assert failures == []
        assert len(finals) == 1
        assert all(
            candidate.status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
            for candidate in candidates
        )
        assert observer.record_active_candidates() == ()
        if candidates:
            assert finals[0].observation is ArtifactWriteSettlementObservation.LATE
        else:
            assert finals[0].observation is ArtifactWriteSettlementObservation.CALLER_BOUNDARY


def test_registered_write_rejects_lower_level_settlement_bypass() -> None:
    with ArtifactWriteSettlementObserver(max_operations=1) as observer:
        registration = register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
            operation_id=_evidence().operation_id,
        )
        with pytest.raises(RuntimeError, match="finalized through registration.record"):
            record_artifact_write_settlement(_evidence())
        active = observer.record_active_candidates()
        final = registration.record(
            status=ArtifactWriteSettlementStatus.ABSENT,
            phase=ArtifactWriteSettlementPhase.CLEANUP,
        )

    assert len(active) == 1
    assert active[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert final.observation is ArtifactWriteSettlementObservation.LATE
    assert observer.record_active_candidates() == ()


def test_registration_bounds_failure_code_iteration_before_copying() -> None:
    class FailureCodes:
        def __init__(self) -> None:
            self.calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls > 9:
                raise AssertionError("failure-code iterator was consumed past its bound")
            return ArtifactWriteSettlementFailureCode.COMMIT_FAILED

    failure_codes = FailureCodes()
    with ArtifactWriteSettlementObserver(max_operations=1) as observer:
        registration = register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
        )
        with pytest.raises(ValidationError, match="at most 8 entries"):
            registration.record(
                status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
                phase=ArtifactWriteSettlementPhase.COMMIT,
                failure_codes=failure_codes,
            )
        assert failure_codes.calls == 9
        assert len(observer.record_active_candidates()) == 1
        registration.close()


def test_registration_is_not_a_scope_owned_context_manager() -> None:
    registration = register_artifact_write_operation(
        artifact_id=_ARTIFACT_ID,
        store_id="third-party-store",
    )

    assert not hasattr(registration, "__enter__")
    assert not hasattr(registration, "__exit__")
    registration.close()


def test_third_party_registration_validates_before_observer_mutation() -> None:
    with ArtifactWriteSettlementObserver(max_operations=1) as observer:
        registration = register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
        )
        with pytest.raises(TypeError, match="error must be a BaseException"):
            registration.record(
                status=ArtifactWriteSettlementStatus.ABSENT,
                error=object(),  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="non-final settlement"):
            registration.record(
                status=ArtifactWriteSettlementStatus.COMMITTED,
                phase=ArtifactWriteSettlementPhase.SETTLED,
                final=False,
            )
        assert observer.snapshot() == ()
        assert len(observer.record_active_candidates()) == 1
        registration.close()

    with pytest.raises((TypeError, ValueError)):
        register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
            operation_id="",
        )
    with pytest.raises((TypeError, ValueError)):
        register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
            operation_id=False,  # type: ignore[arg-type]
        )


def test_third_party_late_logging_failure_cannot_retain_operation(monkeypatch) -> None:
    def fail_log(*_args, **_kwargs) -> None:
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr("cayu.artifacts.settlement._LOGGER.info", fail_log)
    with ArtifactWriteSettlementObserver(max_operations=1) as observer:
        registration = register_artifact_write_operation(
            artifact_id=_ARTIFACT_ID,
            store_id="third-party-store",
        )
        observer.record_active_candidates()
        evidence = registration.record(
            status=ArtifactWriteSettlementStatus.COMMITTED,
            phase=ArtifactWriteSettlementPhase.SETTLED,
        )

    assert evidence.observation is ArtifactWriteSettlementObservation.LATE
    assert observer.record_active_candidates() == ()


def test_rejected_mutated_settlement_does_not_leak_invalid_or_sibling_values(
    caplog,
    capsys,
) -> None:
    secret = "rejected-settlement-secret-canary"

    class SecretValue:
        def __str__(self) -> str:
            return secret

        def __repr__(self) -> str:
            return secret

    evidence = _evidence().model_copy(update={"backend_locator": secret})
    object.__setattr__(evidence, "artifact_id", SecretValue())

    with (
        warnings.catch_warnings(record=True) as captured_warnings,
        pytest.raises((TypeError, ValidationError)) as caught,
    ):
        record_artifact_write_settlement(evidence)

    output = capsys.readouterr()
    diagnostics = (
        str(caught.value),
        repr(caught.value),
        output.out,
        output.err,
        caplog.text,
        *(str(item.message) for item in captured_warnings),
    )
    assert all(secret not in diagnostic for diagnostic in diagnostics)


def test_settlement_accessor_avoids_extension_exception_accessors() -> None:
    class HostileError(RuntimeError):
        def __getattribute__(self, name):
            raise AssertionError(f"unexpected exception attribute access: {name}")

        def __str__(self) -> str:
            raise AssertionError("unexpected exception rendering")

        def __repr__(self) -> str:
            raise AssertionError("unexpected exception rendering")

    error = HostileError()
    evidence = _evidence()

    record_artifact_write_settlement(evidence, error=error)

    assert artifact_write_settlements(error) == (evidence,)


def test_settlement_owner_does_not_dispatch_through_hostile_backend_error() -> None:
    class HostileError(RuntimeError):
        def __getattribute__(self, name):
            raise AssertionError(f"unexpected exception attribute access: {name}")

        def __str__(self) -> str:
            raise AssertionError("unexpected exception rendering")

        def __repr__(self) -> str:
            raise AssertionError("unexpected exception rendering")

    failure = HostileError()

    async def run() -> BaseException:
        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.RECONCILIATION)
            return _unsettled_artifact_write(
                failure,
                phase=ArtifactWriteSettlementPhase.RECONCILIATION,
                failure_codes=(ArtifactWriteSettlementFailureCode.RECONCILIATION_FAILED,),
            )

        with pytest.raises(HostileError) as caught:
            await _settle_artifact_write(
                registry=_ArtifactWriteRegistry(),
                store_id="store",
                artifact_id=_ARTIFACT_ID,
                operation=operation,
            )
        return caught.value

    error = asyncio.run(run())

    assert error is failure
    assert artifact_write_settlements(error)[0].status is (
        ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    )


def test_exception_attached_artifact_write_settlements_are_bounded() -> None:
    error = RuntimeError("write failed")

    for index in range(20):
        record_artifact_write_settlement(
            _evidence().model_copy(update={"operation_id": f"artifact_write_{index:032x}"}),
            error=error,
        )

    records = artifact_write_settlements(error)
    assert len(records) == 16
    assert records[0].operation_id == f"artifact_write_{4:032x}"
    assert records[-1].operation_id == f"artifact_write_{19:032x}"


def test_artifact_write_settlement_preserves_real_cancellation_and_commit() -> None:
    async def run() -> tuple[asyncio.Task[ArtifactMetadata], BaseException]:
        started = asyncio.Event()
        release = asyncio.Event()
        registry = _ArtifactWriteRegistry()

        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CONTENT)
            started.set()
            await release.wait()
            return _committed_artifact_write(_metadata())

        task = asyncio.create_task(
            _settle_artifact_write(
                registry=registry,
                store_id="store",
                artifact_id=_ARTIFACT_ID,
                operation=operation,
                settlement_timeout_s=1.0,
            )
        )
        await started.wait()
        task.cancel("caller stopped")
        release.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return task, exc_info.value

    task, cancellation = asyncio.run(run())

    assert task.cancelled()
    assert task.cancelling() == 1
    assert cancellation.args == ("caller stopped",)
    records = artifact_write_settlements(cancellation)
    assert len(records) == 1
    assert records[0].status is ArtifactWriteSettlementStatus.COMMITTED
    assert records[0].observation is ArtifactWriteSettlementObservation.CALLER_BOUNDARY


def test_artifact_write_settlement_cancellation_before_dispatch_is_absent() -> None:
    async def run() -> tuple[bool, asyncio.CancelledError]:
        dispatched = False

        async def operation(_reporter):
            nonlocal dispatched
            dispatched = True
            return _committed_artifact_write(_metadata())

        task = asyncio.create_task(
            _settle_artifact_write(
                registry=_ArtifactWriteRegistry(),
                store_id="store",
                artifact_id=_ARTIFACT_ID,
                operation=operation,
            )
        )
        await asyncio.sleep(0)
        task.cancel("before dispatch")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return dispatched, exc_info.value

    dispatched, cancellation = asyncio.run(run())

    assert dispatched is False
    records = artifact_write_settlements(cancellation)
    assert len(records) == 1
    assert records[0].status is ArtifactWriteSettlementStatus.ABSENT
    assert records[0].phase is ArtifactWriteSettlementPhase.PRE_DISPATCH


def test_artifact_write_settlement_capacity_fails_before_dispatch() -> None:
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        second_dispatched = False
        registry = _ArtifactWriteRegistry(max_operations=1)

        async def first_operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CONTENT)
            started.set()
            await release.wait()
            return _committed_artifact_write(_metadata())

        async def second_operation(_reporter):
            nonlocal second_dispatched
            second_dispatched = True
            return _committed_artifact_write(_metadata())

        first = asyncio.create_task(
            _settle_artifact_write(
                registry=registry,
                store_id="store",
                artifact_id=_ARTIFACT_ID,
                operation=first_operation,
            )
        )
        await started.wait()
        with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
            await _settle_artifact_write(
                registry=registry,
                store_id="store",
                artifact_id=_ARTIFACT_ID,
                operation=second_operation,
            )
        release.set()
        await first
        return second_dispatched, exc_info.value, len(registry)

    dispatched, error, retained = asyncio.run(run())

    assert dispatched is False
    assert retained == 0
    records = artifact_write_settlements(error)
    assert len(records) == 1
    assert records[0].status is ArtifactWriteSettlementStatus.ABSENT
    assert records[0].failure_codes == (ArtifactWriteSettlementFailureCode.CAPACITY_EXHAUSTED,)


def test_artifact_write_settlement_retains_late_work_and_reports_final_state() -> None:
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        registry = _ArtifactWriteRegistry()
        observer = ArtifactWriteSettlementObserver()

        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CONTENT)
            started.set()
            await release.wait()
            return _committed_artifact_write(_metadata())

        with observer:
            task = asyncio.create_task(
                _settle_artifact_write(
                    registry=registry,
                    store_id="store",
                    artifact_id=_ARTIFACT_ID,
                    operation=operation,
                    settlement_timeout_s=0.0,
                )
            )
            await started.wait()
            task.cancel("deadline")
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
        assert len(registry) == 1
        release.set()
        for _ in range(20):
            if len(registry) == 0:
                break
            await asyncio.sleep(0)
        return exc_info.value, observer.snapshot(), len(registry)

    cancellation, records, retained = asyncio.run(run())

    assert retained == 0
    assert [record.observation for record in records] == [
        ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
        ArtifactWriteSettlementObservation.LATE,
    ]
    assert records[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert records[1].status is ArtifactWriteSettlementStatus.COMMITTED
    assert artifact_write_settlements(cancellation) == (records[0],)


def test_artifact_write_settlement_reports_late_positive_absence() -> None:
    failure = RuntimeError("mutation was cleaned")

    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        registry = _ArtifactWriteRegistry()
        observer = ArtifactWriteSettlementObserver()

        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CLEANUP)
            started.set()
            await release.wait()
            return _absent_artifact_write(
                failure,
                phase=ArtifactWriteSettlementPhase.CLEANUP,
                failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
            )

        with observer:
            task = asyncio.create_task(
                _settle_artifact_write(
                    registry=registry,
                    store_id="store",
                    artifact_id=_ARTIFACT_ID,
                    operation=operation,
                    settlement_timeout_s=0.0,
                )
            )
            await started.wait()
            task.cancel("deadline")
            with pytest.raises(asyncio.CancelledError):
                await task
        release.set()
        for _ in range(20):
            records = observer.snapshot()
            if len(records) == 2:
                return records, len(registry)
            await asyncio.sleep(0)
        return observer.snapshot(), len(registry)

    records, retained = asyncio.run(run())

    assert retained == 0
    assert [record.status for record in records] == [
        ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
        ArtifactWriteSettlementStatus.ABSENT,
    ]
    assert records[-1].observation is ArtifactWriteSettlementObservation.LATE
    assert records[-1].phase is ArtifactWriteSettlementPhase.CLEANUP


def test_artifact_write_settlement_caller_timeout_retains_identifiable_work() -> None:
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        observer = ArtifactWriteSettlementObserver()

        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CONTENT)
            started.set()
            await release.wait()
            return _committed_artifact_write(_metadata())

        with observer:
            try:
                async with asyncio.timeout(0.01):
                    await _settle_artifact_write(
                        registry=_ArtifactWriteRegistry(),
                        store_id="store",
                        artifact_id=_ARTIFACT_ID,
                        operation=operation,
                        settlement_timeout_s=0.0,
                    )
            except TimeoutError as error:
                timeout = error
            else:  # pragma: no cover - deadline must expire
                raise AssertionError("caller timeout did not expire")
        assert started.is_set()
        release.set()
        for _ in range(20):
            if len(observer.snapshot()) == 2:
                break
            await asyncio.sleep(0)
        return timeout, observer.snapshot()

    timeout, records = asyncio.run(run())

    attached = artifact_write_settlements(timeout)
    assert len(attached) == 1
    assert attached[0].artifact_id == _ARTIFACT_ID
    assert attached[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert [record.status for record in records] == [
        ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
        ArtifactWriteSettlementStatus.COMMITTED,
    ]


def test_artifact_write_settlement_preserves_repeated_cancellation_count() -> None:
    async def run() -> asyncio.Task[ArtifactMetadata]:
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CONTENT)
            started.set()
            await release.wait()
            return _committed_artifact_write(_metadata())

        task = asyncio.create_task(
            _settle_artifact_write(
                registry=_ArtifactWriteRegistry(),
                store_id="store",
                artifact_id=_ARTIFACT_ID,
                operation=operation,
                settlement_timeout_s=1.0,
            )
        )
        await started.wait()
        task.cancel("first")
        task.cancel("second")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task

    task = asyncio.run(run())

    assert task.cancelled()
    assert task.cancelling() == 2


def test_artifact_write_settlement_classifies_child_cancellation_as_failure() -> None:
    child_cancellation = asyncio.CancelledError("child")

    async def run() -> BaseException:
        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CONTENT)
            raise child_cancellation

        with pytest.raises(RuntimeError) as exc_info:
            await _settle_artifact_write(
                registry=_ArtifactWriteRegistry(),
                store_id="store",
                artifact_id=_ARTIFACT_ID,
                operation=operation,
                settlement_timeout_s=0.0,
            )
        return exc_info.value

    error = asyncio.run(run())
    records = artifact_write_settlements(error)
    assert len(records) == 1
    assert records[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert records[0].failure_codes == (ArtifactWriteSettlementFailureCode.CHILD_CANCELLED,)


def test_artifact_write_settlement_normalizes_late_child_cancellation() -> None:
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        observer = ArtifactWriteSettlementObserver()

        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CONTENT)
            started.set()
            await release.wait()
            return _unsettled_artifact_write(
                asyncio.CancelledError("child"),
                phase=ArtifactWriteSettlementPhase.CONTENT,
                failure_codes=(),
            )

        with observer:
            task = asyncio.create_task(
                _settle_artifact_write(
                    registry=_ArtifactWriteRegistry(),
                    store_id="store",
                    artifact_id=_ARTIFACT_ID,
                    operation=operation,
                    operation_name="Test artifact publication",
                    settlement_timeout_s=0.0,
                )
            )
            await started.wait()
            task.cancel("caller")
            with pytest.raises(asyncio.CancelledError):
                await task
        release.set()
        for _ in range(20):
            records = observer.snapshot()
            if len(records) == 2:
                return records
            await asyncio.sleep(0)
        return observer.snapshot()

    records = asyncio.run(run())

    assert len(records) == 2
    assert records[-1].observation is ArtifactWriteSettlementObservation.LATE
    assert records[-1].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert records[-1].failure_codes == (ArtifactWriteSettlementFailureCode.CHILD_CANCELLED,)


def test_artifact_write_late_telemetry_is_bounded_and_content_free(
    caplog,
    capsys,
) -> None:
    secret = "provider-raw-response-secret"

    async def run() -> tuple[ArtifactWriteSettlementEvidence, ...]:
        started = asyncio.Event()
        release = asyncio.Event()
        observer = ArtifactWriteSettlementObserver()

        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.RECONCILIATION)
            started.set()
            await release.wait()
            return _unsettled_artifact_write(
                RuntimeError(secret),
                phase=ArtifactWriteSettlementPhase.RECONCILIATION,
                failure_codes=(ArtifactWriteSettlementFailureCode.RECONCILIATION_FAILED,),
            )

        with observer:
            task = asyncio.create_task(
                _settle_artifact_write(
                    registry=_ArtifactWriteRegistry(),
                    store_id="safe-store",
                    artifact_id=_ARTIFACT_ID,
                    operation=operation,
                    settlement_timeout_s=0.0,
                )
            )
            await started.wait()
            task.cancel("caller")
            with pytest.raises(asyncio.CancelledError):
                await task
        release.set()
        for _ in range(20):
            records = observer.snapshot()
            if len(records) == 2:
                return records
            await asyncio.sleep(0)
        return observer.snapshot()

    caplog.set_level(logging.INFO, logger="cayu.artifacts.settlement")
    with warnings.catch_warnings(record=True) as captured_warnings:
        records = asyncio.run(run())

    output = capsys.readouterr()
    assert len(records) == 2
    assert "status=reconciliation_required" in caplog.text
    assert "phase=reconciliation" in caplog.text
    assert f"artifact_id={_ARTIFACT_ID}" in caplog.text
    assert artifact_store_identity_sha256("safe-store") in caplog.text
    assert secret not in caplog.text
    assert secret not in output.out
    assert secret not in output.err
    assert all(secret not in str(item.message) for item in captured_warnings)


def test_artifact_write_settlement_records_positive_absence() -> None:
    failure = RuntimeError("mutation failed")

    async def run() -> BaseException:
        async def operation(reporter):
            reporter.set(ArtifactWriteSettlementPhase.CLEANUP)
            return _absent_artifact_write(
                failure,
                phase=ArtifactWriteSettlementPhase.CLEANUP,
                failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
            )

        with pytest.raises(RuntimeError) as exc_info:
            await _settle_artifact_write(
                registry=_ArtifactWriteRegistry(),
                store_id="store",
                artifact_id=_ARTIFACT_ID,
                operation=operation,
            )
        return exc_info.value

    error = asyncio.run(run())

    assert error is failure
    records = artifact_write_settlements(error)
    assert len(records) == 1
    assert records[0].status is ArtifactWriteSettlementStatus.ABSENT
