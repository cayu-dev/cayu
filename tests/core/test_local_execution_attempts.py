from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    LocalExecutionAttemptConflict,
    LocalExecutionAttemptCoordinator,
    LocalExecutionAttemptEffectOutcome,
    LocalExecutionAttemptLifetime,
    LocalExecutionAttemptLimits,
    LocalExecutionAttemptQuiescence,
    LocalExecutionAttemptReceipt,
    LocalExecutionAttemptRecoveryClaim,
    LocalExecutionAttemptRequest,
    LocalExecutionAttemptSettlement,
    LocalExecutionAttemptStart,
    LocalExecutionAttemptUnavailable,
    LocalExecutionAttemptUnsettled,
    LocalExecutionEffectPolicy,
    LocalExecutionProcessIdentity,
    SQLiteTaskStore,
    TaskCreate,
    TaskRetryPolicy,
    TaskStatus,
    build_local_execution_attempt_authority,
    local_execution_attempt_capability_evidence,
    local_execution_parent_death_containment_platform_candidate,
)
from cayu.runtime.local_execution_attempts import (
    _authenticate_local_execution_attempt_settlement,
    local_execution_attempt_receipt_sha256,
    local_execution_boot_id,
    local_execution_host_identity,
)
from cayu.storage.migrations import LATEST_REVISION, SchemaMode
from cayu.vaults import SecretRedactor


def _store(kind: str, path: Path):
    return InMemoryTaskStore() if kind == "memory" else SQLiteTaskStore(path)


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _set_store_wall_clock(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    value: list[datetime],
) -> None:
    class StoreDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = value[0]
            return current if tz is not None else current.replace(tzinfo=None)

    monkeypatch.setattr(
        "cayu.runtime.tasks.datetime" if kind == "memory" else "cayu.storage.sqlite.datetime",
        StoreDatetime,
    )


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def _assert_cayu_tracebacks_exclude(error: BaseException, canary: str) -> None:
    pending: list[BaseException] = [error]
    observed: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in observed:
            continue
        observed.add(id(current))
        traceback = current.__traceback__
        while traceback is not None:
            if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
                assert canary not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


async def _claimed_task(store, *, task_id: str = "local-attempt"):
    await store.create_task(TaskCreate(task_id=task_id, type="local-execution"))
    task = await store.claim_task("worker-a", lease_seconds=300)
    assert task is not None
    return task


def _request(
    *,
    effect_policy: LocalExecutionEffectPolicy = LocalExecutionEffectPolicy.LOCAL_ONLY,
) -> LocalExecutionAttemptRequest:
    return LocalExecutionAttemptRequest(
        effect_lineage_id="effect-a",
        argv=("/usr/bin/true",),
        effect_policy=effect_policy,
        idempotency_key=(
            "external-operation-a"
            if effect_policy is LocalExecutionEffectPolicy.IDEMPOTENT_EXTERNAL
            else None
        ),
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "deadline_seconds",
        "startup_timeout_seconds",
        "term_grace_seconds",
        "kill_grace_seconds",
    ],
)
def test_local_attempt_limits_reject_boolean_seconds(field_name: str) -> None:
    with pytest.raises(ValueError):
        LocalExecutionAttemptLimits(**{field_name: True})


def test_local_attempt_drain_rejects_boolean_timeout(tmp_path: Path) -> None:
    coordinator = LocalExecutionAttemptCoordinator(
        InMemoryTaskStore(),
        state_dir=tmp_path / "attempt-state",
    )

    async def scenario() -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            await coordinator.drain(timeout_seconds=True)

    asyncio.run(scenario())


def _start(authority, *, root: bool) -> LocalExecutionAttemptStart:
    supervisor = LocalExecutionProcessIdentity(
        pid=100,
        process_group=100,
        start_tick=1000,
        proc_inode=2000,
    )
    return LocalExecutionAttemptStart(
        attempt_id=authority.attempt_id,
        request_sha256=authority.request_sha256,
        host_identity=local_execution_host_identity(),
        boot_id=local_execution_boot_id(),
        supervisor_nonce="a" * 64,
        rendezvous_identity="b" * 64,
        supervisor=supervisor,
        root=(
            LocalExecutionProcessIdentity(
                pid=101,
                process_group=100,
                start_tick=1001,
                proc_inode=2001,
            )
            if root
            else None
        ),
        started_at=datetime.now(UTC),
    )


def _receipt(
    start: LocalExecutionAttemptStart,
    *,
    quiescence: LocalExecutionAttemptQuiescence = LocalExecutionAttemptQuiescence.QUIESCENT,
    outcome: LocalExecutionAttemptEffectOutcome = LocalExecutionAttemptEffectOutcome.SUCCEEDED,
) -> LocalExecutionAttemptReceipt:
    payload = {
        "attempt_id": start.attempt_id,
        "boot_id": start.boot_id,
        "descendants_observed": 2,
        "effect_outcome": outcome.value,
        "exit_code": 0 if outcome is LocalExecutionAttemptEffectOutcome.SUCCEEDED else None,
        "host_identity": start.host_identity,
        "kill_sent": False,
        "quiescence": quiescence.value,
        "request_sha256": start.request_sha256,
        "root": None if start.root is None else start.root.model_dump(mode="json"),
        "settled_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "supervisor_nonce": start.supervisor_nonce,
        "term_sent": False,
        "terminal_reason": "test_settlement",
    }
    payload["receipt_sha256"] = local_execution_attempt_receipt_sha256(payload)
    return LocalExecutionAttemptReceipt.model_validate(payload)


def _settlement(authority, receipt: LocalExecutionAttemptReceipt):
    return _authenticate_local_execution_attempt_settlement(
        LocalExecutionAttemptSettlement(
            attempt_id=authority.attempt_id,
            request_sha256=authority.request_sha256,
            receipt=receipt,
        )
    )


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_local_attempt_store_replays_exact_authority_and_fences_claims(
    kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = _store(kind, tmp_path / "local-attempt.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store)
            request = _request()
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )
            assert authority == build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )

            prepared = await store.prepare_local_execution_attempt(authority)
            replayed = await store.prepare_local_execution_attempt(authority)
            assert replayed == prepared
            assert prepared.retry_admissible is False

            started = _start(authority, root=True)
            await store.start_local_execution_attempt(started.model_copy(update={"root": None}))
            running = await store.start_local_execution_attempt(started)
            assert running.quiescence is LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT

            await store.release_task(task.id, "worker-a")
            assert await store.claim_task("worker-b", lease_seconds=300) is None

            settled = await store.settle_local_execution_attempt(
                _settlement(authority, _receipt(started))
            )
            assert settled.retry_admissible is True
            replacement = await store.claim_task("worker-b", lease_seconds=300)
            assert replacement is not None
            assert replacement.id == task.id
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_unsettled_local_attempt_fences_retry_deadline_terminalization(
    kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        started_at = datetime(2026, 8, 26, 12, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = (
            InMemoryTaskStore(clock=clock)
            if kind == "memory"
            else SQLiteTaskStore(tmp_path / "retry-deadline-fence.sqlite", clock=clock)
        )
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            await store.create_task(
                TaskCreate(
                    task_id="retry-deadline-fence",
                    type="local-execution",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_elapsed_seconds=1,
                    ),
                )
            )
            task = await store.claim_task("worker-a", lease_seconds=300)
            assert task is not None
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(authority)
            start = _start(authority, root=True)
            await store.start_local_execution_attempt(start)
            await store.release_task(task.id, "worker-a")

            clock.value = started_at + timedelta(seconds=2)
            assert await store.claim_task("worker-b", lease_seconds=300) is None
            fenced = await store.load_task(task.id)
            assert fenced is not None
            assert fenced.status is TaskStatus.PENDING

            await store.settle_local_execution_attempt(_settlement(authority, _receipt(start)))
            assert await store.claim_task("worker-b", lease_seconds=300) is None
            terminal = await store.load_task(task.id)
            assert terminal is not None
            assert terminal.status is TaskStatus.FAILED
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
@pytest.mark.parametrize("replacement_worker", ["worker-a", "worker-b"])
def test_retry_admissible_attempt_mints_a_new_exact_claim_identity(
    kind: str,
    replacement_worker: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = _store(kind, tmp_path / f"claim-generation-{replacement_worker}.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            first_task = await _claimed_task(store, task_id="claim-generation")
            request = _request()
            first = build_local_execution_attempt_authority(
                app=app,
                task=first_task,
                worker_id="worker-a",
                request=request,
            )
            await store.prepare_local_execution_attempt(first)
            started = _start(first, root=True)
            await store.start_local_execution_attempt(started)
            await store.settle_local_execution_attempt(_settlement(first, _receipt(started)))
            await store.release_task(first_task.id, "worker-a")
            await asyncio.sleep(0.001)
            replacement_task = await store.claim_task(
                replacement_worker,
                lease_seconds=300,
            )
            assert replacement_task is not None

            replacement = build_local_execution_attempt_authority(
                app=app,
                task=replacement_task,
                worker_id=replacement_worker,
                request=request,
            )
            assert replacement.attempt_id != first.attempt_id
            assert replacement.task_claim_updated_at != first.task_claim_updated_at
            prepared = await store.prepare_local_execution_attempt(replacement)
            assert prepared.authority == replacement
            assert prepared.phase.value == "prepared"
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_authenticated_receipt_settles_across_a_stale_recovery_claim(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        store = _store(kind, tmp_path / "stale-recovery-claim.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store, task_id="stale-recovery-claim")
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(authority)
            started = _start(authority, root=True)
            await store.start_local_execution_attempt(started)
            await store.release_task(task.id, "worker-a")
            claimed = await store.claim_local_execution_attempt_recovery(
                LocalExecutionAttemptRecoveryClaim(
                    attempt_id=authority.attempt_id,
                    request_sha256=authority.request_sha256,
                    recovery_owner_id="lost-recovery-owner",
                    expected_recovery_generation=0,
                    lease_seconds=300,
                )
            )
            assert claimed.recovery_owner_id == "lost-recovery-owner"

            state_dir = tmp_path / f"{kind}-attempt-state"
            state_dir.mkdir(mode=0o700)
            receipt_path = state_dir / f"{authority.request_sha256}.receipt.json"
            receipt_path.write_text(
                _receipt(started).model_dump_json(),
                encoding="utf-8",
            )
            recovered = await LocalExecutionAttemptCoordinator(
                store,
                state_dir=state_dir,
            ).recover(worker_id="replacement-recovery-owner")
            assert len(recovered) == 1
            assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
            assert recovered[0].recovery_owner_id is None
            assert not receipt_path.exists()
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_local_attempt_store_rejects_competing_preparation_and_forged_receipt(
    kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = _store(kind, tmp_path / "local-conflict.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store, task_id="local-conflict")
            request = _request()
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
                attempt_id="lex_authority_a",
            )
            await store.prepare_local_execution_attempt(authority)
            competing = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
                attempt_id="lex_authority_b",
            )
            with pytest.raises(LocalExecutionAttemptUnsettled):
                await store.prepare_local_execution_attempt(competing)

            started = _start(authority, root=True)
            await store.start_local_execution_attempt(started)
            valid_receipt = _receipt(started)
            with pytest.raises(LocalExecutionAttemptConflict, match="provenance"):
                await store.settle_local_execution_attempt(
                    LocalExecutionAttemptSettlement(
                        attempt_id=authority.attempt_id,
                        request_sha256=authority.request_sha256,
                        receipt=valid_receipt,
                    )
                )
            forged_payload = valid_receipt.model_dump(mode="json")
            forged_payload["supervisor_nonce"] = "c" * 64
            forged_payload["receipt_sha256"] = local_execution_attempt_receipt_sha256(
                forged_payload
            )
            forged = LocalExecutionAttemptReceipt.model_validate(forged_payload)
            with pytest.raises(LocalExecutionAttemptConflict):
                await store.settle_local_execution_attempt(_settlement(authority, forged))
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_unknown_external_outcome_is_never_retryable_from_a_caller_declaration(
    kind: str,
    tmp_path: Path,
) -> None:
    async def settle(policy: LocalExecutionEffectPolicy, suffix: str) -> bool:
        store = _store(kind, tmp_path / f"unknown-{suffix}.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store, task_id=f"unknown-{suffix}")
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(effect_policy=policy),
            )
            await store.prepare_local_execution_attempt(authority)
            started = _start(authority, root=True)
            await store.start_local_execution_attempt(started)
            record = await store.settle_local_execution_attempt(
                _settlement(
                    authority,
                    _receipt(
                        started,
                        outcome=LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN,
                    ),
                )
            )
            return record.retry_admissible
        finally:
            await _close(store)

    assert (
        asyncio.run(settle(LocalExecutionEffectPolicy.NON_IDEMPOTENT_EXTERNAL, "unsafe")) is False
    )
    assert asyncio.run(settle(LocalExecutionEffectPolicy.IDEMPOTENT_EXTERNAL, "declared")) is False


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_claimable_snapshot_excludes_pending_tasks_with_an_unsettled_local_attempt(
    kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = _store(kind, tmp_path / "claimable-local-attempt.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store, task_id="claimable-local-attempt")
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(authority)
            started = _start(authority, root=True)
            await store.start_local_execution_attempt(started)
            await store.release_task(task.id, "worker-a")

            fenced = await store.aggregate_operational_snapshot()
            assert fenced.counts_by_status.pending == 1
            assert fenced.claimable_pending_count == 0
            assert await store.claim_task("worker-b", lease_seconds=300) is None

            await store.settle_local_execution_attempt(_settlement(authority, _receipt(started)))
            released = await store.aggregate_operational_snapshot()
            assert released.counts_by_status.pending == 1
            assert released.claimable_pending_count == 1
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_local_attempt_authority_content_binds_secret_environment_without_persisting_it() -> None:
    async def scenario() -> None:
        secret = "local-environment-secret-canary"
        store = InMemoryTaskStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        task = await _claimed_task(store, task_id="environment-authority")
        first = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=LocalExecutionAttemptRequest(
                effect_lineage_id="environment-effect",
                argv=("/usr/bin/true",),
                env={"PRIVATE_VALUE": secret},
                effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
            ),
        )
        second = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=LocalExecutionAttemptRequest(
                effect_lineage_id="environment-effect",
                argv=("/usr/bin/true",),
                env={"PRIVATE_VALUE": "different-value"},
                effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
            ),
        )
        assert first.command_sha256 != second.command_sha256
        assert first.request_sha256 != second.request_sha256
        assert secret not in first.model_dump_json()

        stored = await store.prepare_local_execution_attempt(first)
        assert secret not in stored.model_dump_json()

    asyncio.run(scenario())


def test_registered_secret_idempotency_key_is_bound_only_by_digest(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> None:
        secret = "local-idempotency-secret-canary"
        store = InMemoryTaskStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        task = await _claimed_task(store, task_id="idempotency-authority")
        request = LocalExecutionAttemptRequest(
            effect_lineage_id="idempotency-effect",
            argv=("/usr/bin/true",),
            effect_policy=LocalExecutionEffectPolicy.IDEMPOTENT_EXTERNAL,
            idempotency_key=secret,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with caplog.at_level(logging.WARNING):
                authority = build_local_execution_attempt_authority(
                    app=app,
                    task=task,
                    worker_id="worker-a",
                    request=request,
                )
                stored = await store.prepare_local_execution_attempt(authority)

        expected_digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        assert authority.idempotency_key_sha256 == expected_digest
        assert request.durable_configuration()["idempotency_key_sha256"] == expected_digest
        assert "idempotency_key" not in request.durable_configuration()
        assert secret not in authority.model_dump_json()
        assert secret not in stored.model_dump_json()

        diagnostics = "\n".join(str(item.message) for item in captured)
        diagnostics += caplog.text
        streams = capsys.readouterr()
        diagnostics += streams.out + streams.err
        assert secret not in diagnostics

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_local_attempt_state_transitions_use_the_store_clock(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        now = [datetime.now(UTC)]
        store = (
            InMemoryTaskStore(clock=lambda: now[0])
            if kind == "memory"
            else SQLiteTaskStore(
                tmp_path / "local-attempt-clock.sqlite",
                clock=lambda: now[0],
            )
        )
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store, task_id=f"{kind}-clock")
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(),
            )
            prepared = await store.prepare_local_execution_attempt(authority)
            assert prepared.created_at == now[0]
            assert prepared.updated_at == now[0]

            now[0] += timedelta(seconds=1)
            started = await store.start_local_execution_attempt(_start(authority, root=False))
            assert started.updated_at == now[0]

            await store.release_task(task.id, "worker-a")
            now[0] += timedelta(seconds=1)
            lease_before = datetime.now(UTC)
            claimed = await store.claim_local_execution_attempt_recovery(
                LocalExecutionAttemptRecoveryClaim(
                    attempt_id=authority.attempt_id,
                    request_sha256=authority.request_sha256,
                    recovery_owner_id="clock-recovery-owner",
                    expected_recovery_generation=0,
                    lease_seconds=30,
                )
            )
            lease_after = datetime.now(UTC)
            assert claimed.updated_at == now[0]
            assert claimed.recovery_owner_expires_at is not None
            assert lease_before + timedelta(seconds=30) <= claimed.recovery_owner_expires_at
            assert claimed.recovery_owner_expires_at <= lease_after + timedelta(seconds=30)
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_future_evidence_clock_cannot_expire_live_local_attempt_authority(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wall_now = [datetime(2026, 8, 27, 12, tzinfo=UTC)]
    evidence_now = datetime(2100, 1, 1, tzinfo=UTC)
    _set_store_wall_clock(kind, monkeypatch, wall_now)

    async def scenario() -> None:
        store = (
            InMemoryTaskStore(clock=lambda: evidence_now)
            if kind == "memory"
            else SQLiteTaskStore(
                tmp_path / "future-local-attempt-clock.sqlite",
                clock=lambda: evidence_now,
            )
        )
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store, task_id=f"{kind}-future-local-clock")
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(),
            )

            prepared = await store.prepare_local_execution_attempt(authority)
            assert prepared.created_at == evidence_now
            started_evidence = _start(authority, root=True)
            started = await store.start_local_execution_attempt(started_evidence)
            assert started.updated_at == evidence_now

            await store.release_task(task.id, "worker-a")
            claimed = await store.claim_local_execution_attempt_recovery(
                LocalExecutionAttemptRecoveryClaim(
                    attempt_id=authority.attempt_id,
                    request_sha256=authority.request_sha256,
                    recovery_owner_id="future-clock-owner",
                    expected_recovery_generation=0,
                    lease_seconds=30,
                )
            )
            assert claimed.updated_at == evidence_now
            assert claimed.recovery_owner_expires_at == wall_now[0] + timedelta(seconds=30)

            with pytest.raises(LocalExecutionAttemptConflict):
                await store.claim_local_execution_attempt_recovery(
                    LocalExecutionAttemptRecoveryClaim(
                        attempt_id=authority.attempt_id,
                        request_sha256=authority.request_sha256,
                        recovery_owner_id="competing-future-clock-owner",
                        expected_recovery_generation=1,
                        lease_seconds=30,
                    )
                )
            with pytest.raises(LocalExecutionAttemptConflict, match="lost ownership"):
                await store.start_local_execution_attempt(started_evidence)

            settlement = _authenticate_local_execution_attempt_settlement(
                LocalExecutionAttemptSettlement(
                    attempt_id=authority.attempt_id,
                    request_sha256=authority.request_sha256,
                    receipt=_receipt(started_evidence),
                    recovery_owner_id="future-clock-owner",
                    expected_recovery_generation=1,
                )
            )
            settled = await store.settle_local_execution_attempt(settlement)
            assert settled.updated_at == evidence_now
            assert settled.retry_admissible is True
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_past_evidence_clock_cannot_extend_expired_local_attempt_authority(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wall_now = [datetime(2026, 8, 27, 12, tzinfo=UTC)]
    evidence_now = datetime(2000, 1, 1, tzinfo=UTC)
    _set_store_wall_clock(kind, monkeypatch, wall_now)

    async def scenario() -> None:
        store = (
            InMemoryTaskStore(clock=lambda: evidence_now)
            if kind == "memory"
            else SQLiteTaskStore(
                tmp_path / "past-local-attempt-clock.sqlite",
                clock=lambda: evidence_now,
            )
        )
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store, task_id=f"{kind}-past-local-clock")
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(authority)
            await store.start_local_execution_attempt(_start(authority, root=True))

            wall_now[0] += timedelta(seconds=301)
            first = await store.claim_local_execution_attempt_recovery(
                LocalExecutionAttemptRecoveryClaim(
                    attempt_id=authority.attempt_id,
                    request_sha256=authority.request_sha256,
                    recovery_owner_id="past-clock-owner-a",
                    expected_recovery_generation=0,
                    lease_seconds=30,
                )
            )
            assert first.updated_at == evidence_now
            assert first.recovery_owner_expires_at == wall_now[0] + timedelta(seconds=30)

            wall_now[0] += timedelta(seconds=31)
            replacement = await store.claim_local_execution_attempt_recovery(
                LocalExecutionAttemptRecoveryClaim(
                    attempt_id=authority.attempt_id,
                    request_sha256=authority.request_sha256,
                    recovery_owner_id="past-clock-owner-b",
                    expected_recovery_generation=1,
                    lease_seconds=30,
                )
            )
            assert replacement.recovery_owner_id == "past-clock-owner-b"
            assert replacement.recovery_owner_expires_at == wall_now[0] + timedelta(seconds=30)
            assert replacement.updated_at == evidence_now
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_mutated_request_is_rejected_without_serializer_side_channels(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SecretObject:
        def __repr__(self) -> str:
            return "mutated-request-secret-canary"

        __str__ = __repr__

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="mutated-request")
        request = _request()
        request.env["BROKEN"] = SecretObject()  # type: ignore[assignment]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
                build_local_execution_attempt_authority(
                    app=app,
                    task=task,
                    worker_id="worker-a",
                    request=request,
                )
        diagnostics = "\n".join(str(item.message) for item in captured)
        diagnostics += caplog.text
        streams = capsys.readouterr()
        diagnostics += streams.out + streams.err
        assert "mutated-request-secret-canary" not in diagnostics

    asyncio.run(scenario())


def test_mutated_task_invocation_is_copied_before_serialization(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SecretObject:
        def __repr__(self) -> str:
            return "mutated-task-invocation-secret-canary"

        __str__ = __repr__

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="mutated-task-invocation")
        object.__setattr__(task.invocation, "origin", SecretObject())
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with caplog.at_level(logging.WARNING), pytest.raises(TypeError):
                build_local_execution_attempt_authority(
                    app=app,
                    task=task,
                    worker_id="worker-a",
                    request=_request(),
                )
        diagnostics = "\n".join(str(item.message) for item in captured)
        diagnostics += caplog.text
        streams = capsys.readouterr()
        diagnostics += streams.out + streams.err
        assert "mutated-task-invocation-secret-canary" not in diagnostics

    asyncio.run(scenario())


def test_preparation_refreshes_a_same_owner_heartbeat_race() -> None:
    class HeartbeatBeforeFirstPrepareStore(InMemoryTaskStore):
        prepare_calls = 0

        async def prepare_local_execution_attempt(self, authority):
            self.prepare_calls += 1
            if self.prepare_calls == 1:
                await self.heartbeat(authority.task_id, authority.worker_id)
            return await super().prepare_local_execution_attempt(authority)

    async def scenario() -> None:
        from cayu.runtime._local_execution_attempt_owner import (
            _prepare_current_local_execution_attempt,
        )

        store = HeartbeatBeforeFirstPrepareStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="heartbeat-preparation-race")
        request = _request()
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )

        refreshed, prepared = await _prepare_current_local_execution_attempt(
            app=app,
            task_store=store,
            authority=authority,
            request=request,
        )

        current = await store.load_task(task.id)
        assert current is not None
        assert store.prepare_calls == 2
        assert refreshed.attempt_id != authority.attempt_id
        assert refreshed.task_claim_updated_at == current.updated_at
        assert refreshed.task_claim_lease_expires_at == current.lease_expires_at
        assert prepared.authority == refreshed

    asyncio.run(scenario())


def test_preparation_prefers_exact_replay_across_a_later_heartbeat() -> None:
    async def scenario() -> None:
        from cayu.runtime._local_execution_attempt_owner import (
            _prepare_current_local_execution_attempt,
        )

        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="heartbeat-exact-replay")
        request = _request()
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        expected = await store.prepare_local_execution_attempt(authority)
        await store.heartbeat(task.id, "worker-a")

        replay_authority, replayed = await _prepare_current_local_execution_attempt(
            app=app,
            task_store=store,
            authority=authority,
            request=request,
        )

        assert replay_authority == authority
        assert replayed == expected

    asyncio.run(scenario())


def test_preparation_refresh_does_not_cross_released_task_ownership() -> None:
    class ReleaseBeforeFirstPrepareStore(InMemoryTaskStore):
        prepare_calls = 0

        async def prepare_local_execution_attempt(self, authority):
            self.prepare_calls += 1
            if self.prepare_calls == 1:
                await self.release_task(authority.task_id, authority.worker_id)
            return await super().prepare_local_execution_attempt(authority)

    async def scenario() -> None:
        from cayu.runtime._local_execution_attempt_owner import (
            _prepare_current_local_execution_attempt,
        )

        store = ReleaseBeforeFirstPrepareStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="released-preparation-race")
        request = _request()
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )

        with pytest.raises(
            LocalExecutionAttemptConflict,
            match="exact task claim generation",
        ):
            await _prepare_current_local_execution_attempt(
                app=app,
                task_store=store,
                authority=authority,
                request=request,
            )
        assert store.prepare_calls == 1
        assert await store.load_local_execution_attempt(authority.attempt_id) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [RuntimeError, TimeoutError])
def test_process_preflight_failure_precedes_durable_attempt_fence(
    failure_type: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        from cayu.mcp import _stdio_process
        from cayu.runtime._local_execution_attempt_owner import (
            run_owned_local_execution_attempt,
        )

        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="unavailable-preflight")
        request = _request()
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )

        async def unavailable(_timeout: float) -> None:
            raise failure_type("raw shared-preflight detail")

        monkeypatch.setattr(
            _stdio_process,
            "preflight_stdio_mcp_parent_death_containment",
            unavailable,
        )
        with pytest.raises(
            LocalExecutionAttemptUnavailable,
            match="containment preflight failed",
        ):
            await run_owned_local_execution_attempt(
                app=app,
                task_store=store,
                state_dir=tmp_path / "attempt-state",
                authority=authority,
                request=request,
            )
        assert await store.load_local_execution_attempt(authority.attempt_id) is None

    asyncio.run(scenario())


def test_public_run_store_failure_traceback_does_not_retain_request_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingPrepareStore(InMemoryTaskStore):
        async def prepare_local_execution_attempt(self, authority):
            raise RuntimeError("injected durable preparation failure")

    async def scenario() -> None:
        from cayu.mcp import _stdio_process

        async def available(_timeout: float) -> None:
            return None

        monkeypatch.setattr(
            "cayu.runtime.local_execution_attempts."
            "local_execution_parent_death_containment_platform_candidate",
            lambda: True,
        )
        monkeypatch.setattr(
            _stdio_process,
            "preflight_stdio_mcp_parent_death_containment",
            available,
        )
        canary = "local-attempt-store-traceback-secret-canary"
        store = FailingPrepareStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(canary),
            enable_logging=False,
        )
        task = await _claimed_task(store, task_id="traceback-safe-store")
        request = LocalExecutionAttemptRequest(
            effect_lineage_id="traceback-safe-store-effect",
            argv=("/usr/bin/true",),
            env={"PRIVATE_VALUE": canary},
            effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
        )
        with pytest.raises(RuntimeError, match="durable preparation failure") as captured:
            await LocalExecutionAttemptCoordinator(
                store,
                state_dir=tmp_path / "attempt-state",
            ).run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )
        _assert_cayu_tracebacks_exclude(captured.value, canary)

    asyncio.run(scenario())


def test_public_run_failure_traceback_does_not_retain_request_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        from cayu.mcp import _stdio_process

        secret = "local-attempt-traceback-secret-canary"
        store = InMemoryTaskStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        task = await _claimed_task(store, task_id="traceback-safe-preflight")

        async def broken_preflight(_timeout: float) -> None:
            raise RuntimeError("ordinary preflight failure")

        monkeypatch.setattr(
            _stdio_process,
            "preflight_stdio_mcp_parent_death_containment",
            broken_preflight,
        )
        request = LocalExecutionAttemptRequest(
            effect_lineage_id="traceback-safe-effect",
            argv=("/usr/bin/true",),
            env={"PRIVATE_VALUE": secret},
            effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
        )
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        with pytest.raises(LocalExecutionAttemptUnavailable) as captured:
            await coordinator.run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )
        _assert_cayu_tracebacks_exclude(captured.value, secret)

    asyncio.run(scenario())


def test_public_coordinator_rejects_an_unsupported_platform_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: False,
    )

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="unsupported-platform")
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        with pytest.raises(
            LocalExecutionAttemptUnavailable,
            match="supported Linux process primitives",
        ):
            await coordinator.run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(),
            )
        assert await store.list_unsettled_local_execution_attempts() == ()

    asyncio.run(scenario())


def test_exact_terminal_replay_bypasses_unavailable_process_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        from cayu.mcp import _stdio_process

        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="terminal-replay-without-preflight")
        request = _request()
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        await store.prepare_local_execution_attempt(authority)
        started = _start(authority, root=True)
        expected = await store.start_local_execution_attempt(started)
        expected = await store.settle_local_execution_attempt(
            _settlement(authority, _receipt(started))
        )

        monkeypatch.setattr(
            "cayu.runtime.local_execution_attempts."
            "local_execution_parent_death_containment_platform_candidate",
            lambda: False,
        )

        async def must_not_preflight(_timeout: float) -> None:
            raise AssertionError("terminal replay reached process preflight")

        monkeypatch.setattr(
            _stdio_process,
            "preflight_stdio_mcp_parent_death_containment",
            must_not_preflight,
        )
        state_dir = tmp_path / "terminal-replay-state"
        replayed = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=state_dir,
        ).run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )

        assert replayed.attempt == expected
        assert replayed.stdout == ""
        assert replayed.stderr == ""
        assert replayed.stdout_truncated is True
        assert replayed.stderr_truncated is True
        assert not state_dir.exists()

    asyncio.run(scenario())


def test_persistent_detached_terminal_replay_marks_output_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="persistent-detached-replay")
        request = _request()
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        await store.prepare_local_execution_attempt(authority)
        started = _start(authority, root=True)
        await store.start_local_execution_attempt(started)
        expected = await store.settle_local_execution_attempt(
            _settlement(
                authority,
                _receipt(
                    started,
                    quiescence=LocalExecutionAttemptQuiescence.PERSISTENT_DETACHED,
                    outcome=LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN,
                ),
            )
        )
        monkeypatch.setattr(
            "cayu.runtime.local_execution_attempts."
            "local_execution_parent_death_containment_platform_candidate",
            lambda: False,
        )

        replayed = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "persistent-detached-state",
        ).run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )

        assert replayed.attempt == expected
        assert replayed.stdout == ""
        assert replayed.stderr == ""
        assert replayed.stdout_truncated is True
        assert replayed.stderr_truncated is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("recovery_node_id", "expected_quiescence"),
    [
        ("machine-a", LocalExecutionAttemptQuiescence.QUIESCENT),
        ("machine-b", LocalExecutionAttemptQuiescence.UNAVAILABLE),
    ],
)
def test_reboot_inference_requires_exact_machine_authority(
    recovery_node_id: str,
    expected_quiescence: LocalExecutionAttemptQuiescence,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        from cayu.runtime import _local_execution_attempt_owner as owner_module

        monkeypatch.setenv("CAYU_LOCAL_EXECUTION_NODE_ID", "machine-a")
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id=f"machine-{recovery_node_id}")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        start = _start(authority, root=True).model_copy(
            update={"host_identity": local_execution_host_identity(), "boot_id": "boot-a"}
        )
        await store.start_local_execution_attempt(start)
        await store.release_task(task.id, "worker-a")

        monkeypatch.setenv("CAYU_LOCAL_EXECUTION_NODE_ID", recovery_node_id)
        monkeypatch.setattr(owner_module, "local_execution_boot_id", lambda: "boot-b")
        recovered = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        ).recover(worker_id="recovery-worker")
        assert len(recovered) == 1
        assert recovered[0].quiescence is expected_quiescence
        if recovery_node_id == "machine-a":
            assert recovered[0].receipt is not None
            assert recovered[0].receipt.terminal_reason == "host_reboot"
        else:
            assert recovered[0].retry_admissible is False

    asyncio.run(scenario())


@pytest.mark.parametrize("commit_then_raise", [False, True])
def test_recovery_promotes_an_authenticated_staged_supervisor_receipt(
    commit_then_raise: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setenv("CAYU_LOCAL_EXECUTION_NODE_ID", "staged-receipt-machine")

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="staged-supervisor-receipt")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        started = _start(authority, root=True)
        await store.start_local_execution_attempt(started)
        await store.release_task(task.id, "worker-a")
        state_dir = tmp_path / "attempt-state"
        state_dir.mkdir(mode=0o700)
        receipt_path = state_dir / f"{authority.request_sha256}.receipt.json"
        staging_path = receipt_path.with_name(f"{receipt_path.name}.staging")
        staging_path.write_text(
            _receipt(started).model_dump_json(),
            encoding="utf-8",
        )
        if commit_then_raise:
            from cayu.runtime import _local_execution_attempt_owner as owner_module

            original_replace = owner_module.os.replace

            def replace_then_lose_acknowledgement(source, target) -> None:
                original_replace(source, target)
                raise OSError("injected rename acknowledgement loss")

            monkeypatch.setattr(
                owner_module.os,
                "replace",
                replace_then_lose_acknowledgement,
            )

        recovered = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=state_dir,
        ).recover(worker_id="recovery-worker")
        assert len(recovered) == 1
        assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert recovered[0].receipt is not None
        assert recovered[0].receipt.terminal_reason == "test_settlement"
        assert not receipt_path.exists()
        assert not staging_path.exists()

    asyncio.run(scenario())


def test_recovery_does_not_promote_staging_from_a_live_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setenv("CAYU_LOCAL_EXECUTION_NODE_ID", "live-staged-receipt-machine")

    async def scenario() -> None:
        from cayu.runtime import _local_execution_attempt_owner as owner_module

        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="live-staged-supervisor-receipt")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        started = _start(authority, root=True)
        await store.start_local_execution_attempt(started)
        state_dir = tmp_path / "attempt-state"
        state_dir.mkdir(mode=0o700)
        receipt_path = state_dir / f"{authority.request_sha256}.receipt.json"
        staging_path = receipt_path.with_name(f"{receipt_path.name}.staging")
        staging_path.write_text(_receipt(started).model_dump_json(), encoding="utf-8")
        supervisor_live = True

        async def absent_rendezvous(_record, *, state_dir, deadline=None):
            assert state_dir == tmp_path / "attempt-state"
            assert deadline is None
            return None

        monkeypatch.setattr(owner_module, "_probe_rendezvous", absent_rendezvous)
        monkeypatch.setattr(
            owner_module,
            "_exact_process_is_live",
            lambda _identity: supervisor_live,
        )
        coordinator = LocalExecutionAttemptCoordinator(store, state_dir=state_dir)

        assert await coordinator.recover(worker_id="recovery-worker") == ()
        assert staging_path.is_file()
        assert not receipt_path.exists()

        supervisor_live = False
        await store.release_task(task.id, "worker-a")
        recovered = await coordinator.recover(worker_id="recovery-worker")
        assert len(recovered) == 1
        assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert not staging_path.exists()
        assert not receipt_path.exists()

    asyncio.run(scenario())


def test_recovery_reloads_a_final_receipt_renamed_as_the_supervisor_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setenv("CAYU_LOCAL_EXECUTION_NODE_ID", "rename-exit-race-machine")

    async def scenario() -> None:
        from cayu.runtime import _local_execution_attempt_owner as owner_module

        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="rename-exit-race")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        started = _start(authority, root=True)
        await store.start_local_execution_attempt(started)
        await store.release_task(task.id, "worker-a")
        state_dir = tmp_path / "attempt-state"
        state_dir.mkdir(mode=0o700)
        receipt_path = state_dir / f"{authority.request_sha256}.receipt.json"
        staging_path = receipt_path.with_name(f"{receipt_path.name}.staging")
        staging_path.write_text(_receipt(started).model_dump_json(), encoding="utf-8")

        async def closed_rendezvous(_record, *, state_dir, deadline=None):
            assert state_dir == tmp_path / "attempt-state"
            assert deadline is None
            return None

        liveness_checks = 0

        def rename_and_exit(_identity) -> bool:
            nonlocal liveness_checks
            liveness_checks += 1
            assert not receipt_path.exists()
            owner_module.os.replace(staging_path, receipt_path)
            return False

        monkeypatch.setattr(owner_module, "_probe_rendezvous", closed_rendezvous)
        monkeypatch.setattr(owner_module, "_exact_process_is_live", rename_and_exit)

        recovered = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=state_dir,
        ).recover(worker_id="recovery-worker")

        assert liveness_checks == 1
        assert len(recovered) == 1
        assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert recovered[0].retry_admissible is True
        assert recovered[0].receipt is not None
        assert recovered[0].receipt.terminal_reason == "test_settlement"
        assert not receipt_path.exists()
        assert not staging_path.exists()
        replacement = await store.claim_task("worker-b", lease_seconds=300)
        assert replacement is not None
        assert replacement.id == task.id

    asyncio.run(scenario())


def test_recovery_prefers_a_receipt_published_during_claim_without_machine_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setenv("CAYU_LOCAL_EXECUTION_NODE_ID", "")

    async def scenario() -> None:
        from cayu.runtime import _local_execution_attempt_owner as owner_module
        from cayu.runtime import _local_execution_supervisor as supervisor_module

        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="claim-receipt-without-machine")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        started = _start(authority, root=True)
        assert started.host_identity == "unavailable"
        await store.start_local_execution_attempt(started)
        await store.release_task(task.id, "worker-a")
        state_dir = tmp_path / "attempt-state"
        state_dir.mkdir(mode=0o700)
        receipt_path = state_dir / f"{authority.request_sha256}.receipt.json"

        async def closed_rendezvous(_record, *, state_dir, deadline=None):
            assert state_dir == tmp_path / "attempt-state"
            assert deadline is None
            return None

        supervisor_live = True
        liveness_checks = 0

        def process_is_live(_identity) -> bool:
            nonlocal liveness_checks
            liveness_checks += 1
            return supervisor_live

        monkeypatch.setattr(owner_module, "_probe_rendezvous", closed_rendezvous)
        monkeypatch.setattr(owner_module, "_exact_process_is_live", process_is_live)
        coordinator = LocalExecutionAttemptCoordinator(store, state_dir=state_dir)

        assert await coordinator.recover(worker_id="recovery-worker") == ()
        assert liveness_checks == 1

        original_claim = store.claim_local_execution_attempt_recovery
        claim_calls = 0

        async def claim_and_publish(claim):
            nonlocal claim_calls
            claim_calls += 1
            claimed = await original_claim(claim)
            supervisor_module._atomic_receipt(
                receipt_path,
                _receipt(started).model_dump(mode="json"),
            )
            return claimed

        monkeypatch.setattr(
            store,
            "claim_local_execution_attempt_recovery",
            claim_and_publish,
        )
        supervisor_live = False
        recovered = await coordinator.recover(worker_id="recovery-worker")

        assert claim_calls == 1
        assert len(recovered) == 1
        assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert recovered[0].retry_admissible is True
        assert recovered[0].receipt is not None
        assert recovered[0].receipt.terminal_reason == "test_settlement"
        assert not receipt_path.exists()
        replacement = await store.claim_task("worker-b", lease_seconds=300)
        assert replacement is not None
        assert replacement.id == task.id

    asyncio.run(scenario())


def test_recovery_fences_a_corrupt_staged_supervisor_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setenv("CAYU_LOCAL_EXECUTION_NODE_ID", "corrupt-staged-receipt-machine")

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="corrupt-staged-supervisor-receipt")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        await store.start_local_execution_attempt(_start(authority, root=True))
        await store.release_task(task.id, "worker-a")
        state_dir = tmp_path / "attempt-state"
        state_dir.mkdir(mode=0o700)
        receipt_path = state_dir / f"{authority.request_sha256}.receipt.json"
        staging_path = receipt_path.with_name(f"{receipt_path.name}.staging")
        staging_path.write_text('{"incomplete":', encoding="utf-8")

        recovered = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=state_dir,
        ).recover(worker_id="recovery-worker")
        assert len(recovered) == 1
        assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.UNAVAILABLE
        assert recovered[0].effect_outcome is LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN
        assert recovered[0].retry_admissible is False

    asyncio.run(scenario())


def test_sqlite_rejects_redigested_inconsistent_terminal_attempt(
    tmp_path: Path,
) -> None:
    database = tmp_path / "malformed-attempt.sqlite"

    async def seed() -> str:
        store = SQLiteTaskStore(database)
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            task = await _claimed_task(store, task_id="malformed-attempt")
            authority = build_local_execution_attempt_authority(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(authority)
            return authority.attempt_id
        finally:
            await store.close()

    attempt_id = asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT record_json FROM cayu_local_execution_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        assert row is not None
        document = json.loads(row[0])
        document["phase"] = "terminal"
        connection.execute(
            "UPDATE cayu_local_execution_attempts SET phase = ?, record_json = ? "
            "WHERE attempt_id = ?",
            ("terminal", json.dumps(document), attempt_id),
        )
        connection.commit()
    finally:
        connection.close()

    async def load() -> None:
        store = SQLiteTaskStore(database)
        try:
            with pytest.raises(
                LocalExecutionAttemptConflict,
                match="content is malformed",
            ):
                await store.load_local_execution_attempt(attempt_id)
        finally:
            await store.close()

    asyncio.run(load())


def test_capability_evidence_distinguishes_declared_verified_and_detached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setattr(
        "cayu.mcp._stdio_process.stdio_mcp_parent_death_containment_supported",
        lambda: False,
    )
    configured = local_execution_attempt_capability_evidence(
        LocalExecutionAttemptLifetime.PARENT_DEATH_CONTAINMENT
    )
    detached = local_execution_attempt_capability_evidence(
        LocalExecutionAttemptLifetime.PERSISTENT_DETACHED
    )
    assert configured.state_for("parent_death_containment") == "declared"
    assert detached.state_for("persistent_detached") == "declared"
    assert detached.state_for("graceful_cleanup") == "unsupported"

    monkeypatch.setattr(
        "cayu.mcp._stdio_process.stdio_mcp_parent_death_containment_supported",
        lambda: True,
    )
    verified = local_execution_attempt_capability_evidence(
        LocalExecutionAttemptLifetime.PARENT_DEATH_CONTAINMENT
    )
    assert verified.state_for("graceful_cleanup") == "available"
    assert verified.state_for("hard_deadline") == "available"
    assert verified.state_for("parent_death_containment") == "available"

    with pytest.raises(TypeError):
        local_execution_attempt_capability_evidence(  # type: ignore[call-arg]
            LocalExecutionAttemptLifetime.PARENT_DEATH_CONTAINMENT,
            process_preflight_proved=True,
        )


def test_local_attempt_platform_candidate_uses_complete_shared_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.mcp._stdio_process.stdio_mcp_parent_death_containment_platform_candidate",
        lambda: False,
    )

    assert local_execution_parent_death_containment_platform_candidate() is False


def test_sqlite_revision_sixty_six_installs_attempt_fencing_without_task_loss(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-65.sqlite"

    async def seed() -> None:
        store = SQLiteTaskStore(database)
        try:
            await store.create_task(TaskCreate(task_id="preserved-task", type="work"))
        finally:
            await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE cayu_local_execution_attempts")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 66")
        connection.execute("PRAGMA user_version = 65")
        connection.commit()
    finally:
        connection.close()

    async def migrate() -> None:
        store = SQLiteTaskStore(database, schema_mode=SchemaMode.MIGRATE)
        try:
            assert await store.load_task("preserved-task") is not None
        finally:
            await store.close()

    asyncio.run(migrate())
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (LATEST_REVISION,)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_local_execution_attempts'"
        ).fetchone() == ("cayu_local_execution_attempts",)
    finally:
        connection.close()


def test_local_attempt_drain_never_reports_a_live_owner_as_settled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="active-drain")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        with pytest.raises(LocalExecutionAttemptUnsettled, match="drain elapsed"):
            await coordinator.drain(timeout_seconds=0.01)

        await store.release_task(task.id, "worker-a")
        settled = await coordinator.drain(timeout_seconds=1)
        assert len(settled) == 1
        assert settled[0].quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED

    asyncio.run(scenario())


def test_local_attempt_drain_does_not_hide_terminal_unproven_quiescence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="unproven-drain")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        started = _start(authority, root=True)
        await store.start_local_execution_attempt(started)
        await store.settle_local_execution_attempt(
            _settlement(
                authority,
                _receipt(
                    started,
                    quiescence=(LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT),
                    outcome=LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN,
                ),
            )
        )
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        with pytest.raises(LocalExecutionAttemptUnsettled, match="drain elapsed"):
            await coordinator.drain(timeout_seconds=0.01)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_unproven_terminal_record_does_not_starve_recoverable_attempts(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        store = _store(kind, tmp_path / "local-recovery-order.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            old_task = await _claimed_task(store, task_id="old-unproven")
            old_authority = build_local_execution_attempt_authority(
                app=app,
                task=old_task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(old_authority)
            old_start = _start(old_authority, root=True)
            await store.start_local_execution_attempt(old_start)
            await store.settle_local_execution_attempt(
                _settlement(
                    old_authority,
                    _receipt(
                        old_start,
                        quiescence=(LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT),
                        outcome=LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN,
                    ),
                )
            )

            new_task = await _claimed_task(store, task_id="new-recoverable")
            new_authority = build_local_execution_attempt_authority(
                app=app,
                task=new_task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(new_authority)
            await store.release_task(new_task.id, "worker-a")

            recovered = await LocalExecutionAttemptCoordinator(
                store,
                state_dir=tmp_path / "attempt-state",
            ).recover(worker_id="recovery-worker", limit=1)
            assert len(recovered) == 1
            assert recovered[0].authority.attempt_id == new_authority.attempt_id
            assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED
            remaining = await store.list_unsettled_local_execution_attempts(limit=10)
            assert tuple(item.authority.attempt_id for item in remaining) == (
                old_authority.attempt_id,
            )
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_live_recovery_prefix_does_not_starve_later_recoverable_attempt(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        store = _store(kind, tmp_path / "local-live-recovery-order.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            live_task = await _claimed_task(store, task_id="old-live")
            live_authority = build_local_execution_attempt_authority(
                app=app,
                task=live_task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(live_authority)
            await store.start_local_execution_attempt(_start(live_authority, root=True))

            recoverable_task = await _claimed_task(store, task_id="later-recoverable")
            recoverable_authority = build_local_execution_attempt_authority(
                app=app,
                task=recoverable_task,
                worker_id="worker-a",
                request=_request(),
            )
            await store.prepare_local_execution_attempt(recoverable_authority)
            await store.release_task(recoverable_task.id, "worker-a")

            async def probe(record, *, state_dir):
                assert state_dir == tmp_path / "attempt-state"
                if record.authority.attempt_id == live_authority.attempt_id:
                    return {"state": "running"}
                return None

            monkeypatch.setattr(
                "cayu.runtime._local_execution_attempt_owner._probe_rendezvous",
                probe,
            )
            recovered = await LocalExecutionAttemptCoordinator(
                store,
                state_dir=tmp_path / "attempt-state",
            ).recover(worker_id="recovery-worker", limit=1)
            assert tuple(item.authority.attempt_id for item in recovered) == (
                recoverable_authority.attempt_id,
            )
            remaining = await store.list_unsettled_local_execution_attempts(limit=10)
            assert tuple(item.authority.attempt_id for item in remaining) == (
                live_authority.attempt_id,
            )
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_recovery_scan_is_bounded_and_advances_while_listing_grows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="scripted-live-prefix")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        base_record = await store.prepare_local_execution_attempt(authority)
        records = []
        next_index = 0
        list_calls = 0

        async def list_scripted_records(*, limit=100, after=None):
            nonlocal list_calls, next_index
            list_calls += 1
            for _ in range(limit):
                records.append(
                    base_record.model_copy(
                        update={
                            "authority": base_record.authority.model_copy(
                                update={"attempt_id": f"scripted-attempt-{next_index:04d}"}
                            ),
                            "created_at": base_record.created_at
                            + timedelta(microseconds=next_index),
                        }
                    )
                )
                next_index += 1
            after_key = None if after is None else (after.created_at, after.attempt_id)
            eligible = tuple(
                record
                for record in records
                if after_key is None or (record.created_at, record.authority.attempt_id) > after_key
            )
            return eligible[:limit]

        probed: list[str] = []

        async def probe(record, *, state_dir):
            assert state_dir == tmp_path / "attempt-state"
            probed.append(record.authority.attempt_id)
            return {"state": "running"}

        monkeypatch.setattr(store, "list_unsettled_local_execution_attempts", list_scripted_records)
        monkeypatch.setattr(
            "cayu.runtime._local_execution_attempt_owner._probe_rendezvous",
            probe,
        )
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )

        assert await coordinator.recover(worker_id="recovery-worker", limit=1) == ()
        assert len(probed) == 256
        assert list_calls == 8

        assert await coordinator.recover(worker_id="recovery-worker", limit=1) == ()
        assert probed[256] == "scripted-attempt-0256"
        assert len(probed) == 512
        assert list_calls == 16

    asyncio.run(scenario())


def test_local_attempt_drain_stops_recovery_at_its_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed_task(store, task_id="deadline-live-attempt")
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(),
        )
        await store.prepare_local_execution_attempt(authority)
        probe_calls = 0

        async def probe(record, *, state_dir, deadline=None):
            nonlocal probe_calls
            probe_calls += 1
            assert state_dir == tmp_path / "attempt-state"
            assert deadline is not None
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            await asyncio.sleep(remaining + 0.005)
            return {"state": "running"}

        monkeypatch.setattr(
            "cayu.runtime._local_execution_attempt_owner._probe_rendezvous",
            probe,
        )
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        started_at = asyncio.get_running_loop().time()
        with pytest.raises(LocalExecutionAttemptUnsettled, match="drain elapsed"):
            await coordinator.drain(timeout_seconds=0.02)
        elapsed = asyncio.get_running_loop().time() - started_at

        assert probe_calls == 1
        assert elapsed < 0.5

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_malformed_receipt_does_not_abort_unrelated_recovery(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        store = _store(kind, tmp_path / "local-corrupt-receipt.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            authorities = []
            for task_id in ("corrupt-receipt", "unrelated-recovery"):
                task = await _claimed_task(store, task_id=task_id)
                authority = build_local_execution_attempt_authority(
                    app=app,
                    task=task,
                    worker_id="worker-a",
                    request=_request(),
                )
                await store.prepare_local_execution_attempt(authority)
                await store.release_task(task.id, "worker-a")
                authorities.append(authority)

            state_dir = tmp_path / "attempt-state"
            state_dir.mkdir(mode=0o700)
            corrupt_path = state_dir / (f"{authorities[0].request_sha256}.receipt.json")
            corrupt_path.write_text(
                '{"rejected":"receipt-secret-canary"}',
                encoding="utf-8",
            )

            recovered = await LocalExecutionAttemptCoordinator(
                store,
                state_dir=state_dir,
            ).recover(worker_id="recovery-worker", limit=2)
            assert {item.authority.attempt_id for item in recovered} == {
                item.attempt_id for item in authorities
            }
            assert all(
                item.quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED
                for item in recovered
            )
            assert not corrupt_path.exists()
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_recovery_revalidates_owner_private_settlement_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cayu.runtime.local_execution_attempts."
        "local_execution_parent_death_containment_platform_candidate",
        lambda: True,
    )

    async def scenario() -> None:
        state_dir = tmp_path / "attempt-state"
        state_dir.mkdir(mode=0o700)
        state_dir.chmod(0o755)
        coordinator = LocalExecutionAttemptCoordinator(
            InMemoryTaskStore(),
            state_dir=state_dir,
        )
        with pytest.raises(
            LocalExecutionAttemptUnavailable,
            match="owner-private",
        ):
            await coordinator.recover(worker_id="recovery-worker")

    asyncio.run(scenario())
