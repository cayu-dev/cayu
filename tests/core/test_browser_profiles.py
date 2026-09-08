from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
import traceback
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    AESGCMBrowserProfileKeyAuthority,
    BrowserProfileAccess,
    BrowserProfileBinding,
    BrowserProfileCheckpointPolicy,
    BrowserProfileCookie,
    BrowserProfileDestinationPolicy,
    BrowserProfileKeyAuthority,
    BrowserProfileLimits,
    BrowserProfileOriginStorage,
    BrowserProfileScope,
    BrowserProfileStateV1,
    BrowserProfileStatus,
    BrowserProfileStorageEntry,
    BrowserProfileStoreConflict,
    BrowserProfileTerminalOutcome,
    BrowserProfileUnavailable,
    InMemoryBrowserProfileStore,
    SQLiteBrowserProfileStore,
)
from cayu.browser_profiles import (
    BROWSER_PROFILE_MAX_RECEIPTS,
    BrowserProfileCheckpointPlan,
    validate_browser_profile_state,
)

_KEY = b"k" * 32
_OTHER_KEY = b"x" * 32
_ORIGIN = "https://auth.browser.test"
_SECOND_ORIGIN = "https://account.browser.test"


class _PrivateRepr:
    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return self.value


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 2, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class _BoundaryClock(_Clock):
    def __init__(self) -> None:
        super().__init__()
        self.boundary_check = lambda: True
        self.require_boundary = False
        self.boundary_samples = 0

    def __call__(self) -> datetime:
        if self.require_boundary:
            assert self.boundary_check()
            self.boundary_samples += 1
        return super().__call__()


class _CommitFailureConnection:
    """Delegate SQLite while failing selected commits before they take effect."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self.failures_remaining = 0

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def execute(self, *args: Any, **kwargs: Any):
        return self._connection.execute(*args, **kwargs)

    def commit(self) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise sqlite3.OperationalError("injected browser-profile commit failure")
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def _store(
    tmp_path: Path,
    *,
    durable: bool,
    clock: _Clock,
    suffix: str = "",
):
    if durable:
        return SQLiteBrowserProfileStore(
            tmp_path / f"browser-profiles{suffix}.sqlite",
            store_id="browser-profile-test-store",
            clock=clock,
        )
    return InMemoryBrowserProfileStore(
        store_id="browser-profile-test-store",
        clock=clock,
    )


def _binding(
    store,
    *,
    key: bytes = _KEY,
    current_policy: BrowserProfileDestinationPolicy | None = None,
    expires_at: datetime | None = None,
) -> BrowserProfileBinding:
    recorded_policy = BrowserProfileDestinationPolicy.build((_ORIGIN, _SECOND_ORIGIN))
    return BrowserProfileBinding.build(
        scope=BrowserProfileScope.build(
            application_id="browser-profile-tests",
            tenant_id="test-tenant",
            sharing_scope="agent-release-v1",
        ),
        destination_policy=recorded_policy,
        current_policy=current_policy,
        browser_protocol="cayu.browser-session.v3",
        browser_worker_version="10",
        store=store,
        key_authority=AESGCMBrowserProfileKeyAuthority(
            authority_id="browser-profile-test-key",
            key=key,
        ),
        profile_id="bprof_test",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        expires_at=expires_at,
        checkpoint_policy=BrowserProfileCheckpointPolicy.ON_CLOSE,
        lease_seconds=120,
    )


def _state(*, canary: str = "profile-secret-canary") -> BrowserProfileStateV1:
    return BrowserProfileStateV1(
        cookies=(
            BrowserProfileCookie(
                name="test_session",
                value=canary,
                domain="auth.browser.test",
                path="/",
                secure=True,
                http_only=True,
                same_site="Lax",
            ),
        ),
        origins=(
            BrowserProfileOriginStorage(
                origin=_ORIGIN,
                local_storage=(
                    BrowserProfileStorageEntry(
                        name="session_state",
                        value=canary,
                    ),
                ),
            ),
        ),
    )


async def _prepared_restore(binding: BrowserProfileBinding, *, suffix: str = "1"):
    return await binding.prepare_restore(
        operation_id=f"restore_{suffix}",
        execution_profile_fingerprint=_digest("execution-profile"),
        allocation_fingerprint=_digest(f"allocation-{suffix}"),
        browser_session_id=f"browser_session_{suffix}",
    )


async def _settled_restore(binding: BrowserProfileBinding, *, suffix: str = "1"):
    material = await _prepared_restore(binding, suffix=suffix)
    receipt = await binding.complete_restore(
        material,
        outcome=BrowserProfileTerminalOutcome.SUCCEEDED,
    )
    return material, receipt


async def _checkpoint(
    binding: BrowserProfileBinding,
    material,
    *,
    operation_id: str = "checkpoint_1",
    state: BrowserProfileStateV1 | None = None,
):
    plan = await binding.reserve_checkpoint(
        material=material,
        operation_id=operation_id,
        source_revision="browser_revision_1",
        source_operation_receipt_id="browser_receipt_1",
        source_operation_fingerprint=_digest("browser-operation"),
        ambiguous_lineage=False,
    )
    receipt = await binding.publish_checkpoint(plan, state or _state())
    return plan, receipt


class _RecordingSettlementStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.restore_settlement_calls = 0
        self.checkpoint_settlement_calls = 0

    async def complete_restore(self, request, *, outcome, error_code=None):
        self.restore_settlement_calls += 1
        return await super().complete_restore(
            request,
            outcome=outcome,
            error_code=error_code,
        )

    async def fail_checkpoint(self, reservation, *, outcome, error_code):
        self.checkpoint_settlement_calls += 1
        return await super().fail_checkpoint(
            reservation,
            outcome=outcome,
            error_code=error_code,
        )


class _BypassingMutationStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.bypass_mutation = False

    async def _mutate_record(self, profile_id, operation):
        if self.bypass_mutation:
            return object()
        return await super()._mutate_record(profile_id, operation)


class _LeakingLoadPrimitiveStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.fail_load = False

    async def _load_record(self, profile_id):
        if self.fail_load:
            private_store_local = "BROWSER_PROFILE_STORE_EXCEPTION_CANARY"
            raise BrowserProfileUnavailable(private_store_local)
        return await super()._load_record(profile_id)


class _MalformedLoadPrimitiveStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.malformed_load = False

    async def _load_record(self, profile_id):
        record = await super()._load_record(profile_id)
        if self.malformed_load and record is not None:
            object.__setattr__(
                record,
                "revision",
                _PrivateRepr("BROWSER_PROFILE_STORE_RECORD_CANARY"),
            )
        return record


class _FailedRestoreSettlementStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.fail_restore_settlement = False

    async def complete_restore(self, request, *, outcome, error_code=None):
        if self.fail_restore_settlement:
            raise RuntimeError("BROWSER_PROFILE_SETTLEMENT_FAILURE_CANARY")
        return await super().complete_restore(
            request,
            outcome=outcome,
            error_code=error_code,
        )


class _RecordingPublicBoundaryStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.reconcile_calls = 0
        self.revoke_calls = 0

    async def reconcile_checkpoint(self, access, operation_id):
        self.reconcile_calls += 1
        return await super().reconcile_checkpoint(access, operation_id)

    async def revoke_profile(self, access, *, revoked_at=None):
        self.revoke_calls += 1
        return await super().revoke_profile(access, revoked_at=revoked_at)


class _LeakingPublicLookupStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.fail_lookup = False

    def __getattribute__(self, name: str) -> Any:
        if name == "reconcile_checkpoint" and object.__getattribute__(
            self,
            "fail_lookup",
        ):
            private_store_local = "BROWSER_PROFILE_STORE_LOOKUP_CANARY"
            raise RuntimeError(private_store_local)
        return super().__getattribute__(name)


def test_public_profile_mutations_validate_without_exposing_raw_inputs(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> str:
        store = _RecordingPublicBoundaryStore(clock=_Clock())
        binding = _binding(store)
        await binding.initialize()
        rendered: list[str] = []

        with pytest.raises(ValueError) as reconcile_failure:
            await binding.reconcile_checkpoint(
                _PrivateRepr("BROWSER_PROFILE_PUBLIC_INPUT_CANARY")  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError) as revoke_failure:
            await binding.revoke(
                revoked_at=_PrivateRepr(  # type: ignore[arg-type]
                    "BROWSER_PROFILE_PUBLIC_INPUT_CANARY"
                )
            )

        assert store.reconcile_calls == 0
        assert store.revoke_calls == 0
        for failure in (reconcile_failure.value, revoke_failure.value):
            assert failure.__cause__ is None
            assert failure.__context__ is None
            rendered.extend((str(failure), repr(failure)))
            rendered.extend(
                repr(frame.f_locals)
                for frame, _line_number in traceback.walk_tb(failure.__traceback__)
                if is_cayu_source_filename(frame.f_code.co_filename)
            )
        return "\n".join(rendered)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = asyncio.run(scenario())
    captured = capsys.readouterr()
    diagnostic_surface = "\n".join(
        (
            rendered,
            caplog.text,
            captured.out,
            captured.err,
            *(str(item.message) for item in caught),
        )
    )
    assert "BROWSER_PROFILE_PUBLIC_INPUT_CANARY" not in diagnostic_surface


def test_public_store_method_lookup_failure_is_diagnostically_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> str:
        store = _LeakingPublicLookupStore(clock=_Clock())
        binding = _binding(store)
        await binding.initialize()
        store.fail_lookup = True

        with pytest.raises(BrowserProfileUnavailable) as exc_info:
            await binding.reconcile_checkpoint("checkpoint_lookup_failure")

        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        return "\n".join(
            (
                str(exc_info.value),
                repr(exc_info.value),
                *(
                    repr(frame.f_locals)
                    for frame, _line_number in traceback.walk_tb(exc_info.value.__traceback__)
                    if is_cayu_source_filename(frame.f_code.co_filename)
                ),
            )
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = asyncio.run(scenario())
    captured = capsys.readouterr()
    diagnostic_surface = "\n".join(
        (
            rendered,
            caplog.text,
            captured.out,
            captured.err,
            *(str(item.message) for item in caught),
        )
    )
    assert "BROWSER_PROFILE_STORE_LOOKUP_CANARY" not in diagnostic_surface


def test_browser_profile_store_requires_exact_validated_mutation_evidence() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _BypassingMutationStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        store.bypass_mutation = True

        with pytest.raises(BrowserProfileUnavailable, match="store operation failed"):
            await store.revoke_profile(binding.access)

        store.bypass_mutation = False
        inspection = await store.inspect_profile(binding.access)
        assert inspection.status is BrowserProfileStatus.EMPTY
        assert inspection.revoked_at is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "store_type",
    [_LeakingLoadPrimitiveStore, _MalformedLoadPrimitiveStore],
    ids=["exception", "record"],
)
def test_browser_profile_store_sanitizes_extension_load_boundaries(
    store_type,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> str:
        clock = _Clock()
        store = store_type(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        if isinstance(store, _LeakingLoadPrimitiveStore):
            store.fail_load = True
        else:
            store.malformed_load = True

        with pytest.raises(BrowserProfileUnavailable) as exc_info:
            await store.current_ref(binding.access)

        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        return "\n".join(
            (
                str(exc_info.value),
                repr(exc_info.value),
                *(
                    repr(frame.f_locals)
                    for frame, _line_number in traceback.walk_tb(exc_info.value.__traceback__)
                    if is_cayu_source_filename(frame.f_code.co_filename)
                ),
            )
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = asyncio.run(scenario())
    captured = capsys.readouterr()
    diagnostic_surface = "\n".join(
        (
            rendered,
            caplog.text,
            captured.out,
            captured.err,
            *(str(item.message) for item in caught),
        )
    )
    assert "BROWSER_PROFILE_STORE_EXCEPTION_CANARY" not in diagnostic_surface
    assert "BROWSER_PROFILE_STORE_RECORD_CANARY" not in diagnostic_surface


class _ProcessControlPreparationStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock, fail_operation: str) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.fail_operation = fail_operation
        self.restore_calls = 0
        self.checkpoint_calls = 0

    async def prepare_restore(self, request, *, lease_seconds):
        self.restore_calls += 1
        if self.fail_operation != "restore":
            return await super().prepare_restore(request, lease_seconds=lease_seconds)
        raise BaseExceptionGroup(
            "profile restore process control",
            [GeneratorExit("profile restore stopped")],
        )

    async def reserve_checkpoint(self, request, *, reserved_ciphertext_bytes):
        self.checkpoint_calls += 1
        if self.fail_operation != "checkpoint":
            return await super().reserve_checkpoint(
                request,
                reserved_ciphertext_bytes=reserved_ciphertext_bytes,
            )
        raise BaseExceptionGroup(
            "profile checkpoint process control",
            [GeneratorExit("profile checkpoint stopped")],
        )


class _ConcurrentProcessControlRenewStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def renew_writer(self, access, claim, *, lease_seconds):
        self.entered.set()
        await self.release.wait()
        raise GeneratorExit("profile renewal stopped")


class _CommitThenProcessControlCheckpointStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.raise_after_publish = True

    async def publish_checkpoint(self, access, operation_id):
        receipt = await super().publish_checkpoint(access, operation_id)
        if self.raise_after_publish:
            self.raise_after_publish = False
            raise GeneratorExit("profile publication stopped")
        return receipt


def test_browser_profile_rejects_nonfixed_diagnostics_before_store_calls() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _RecordingSettlementStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        restore = await _prepared_restore(binding, suffix="invalid_restore_code")

        with pytest.raises(ValueError, match="supported browser-profile error code"):
            await binding.complete_restore(
                restore,
                outcome=BrowserProfileTerminalOutcome.FAILED,
                error_code="profile-secret-canary",
            )
        assert store.restore_settlement_calls == 0
        await binding.complete_restore(
            restore,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )

        material, _ = await _settled_restore(binding, suffix="invalid_checkpoint_code")
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_invalid_error_code",
            source_revision="revision_invalid_error_code",
            source_operation_receipt_id="receipt_invalid_error_code",
            source_operation_fingerprint=_digest("invalid-error-code"),
            ambiguous_lineage=False,
        )
        with pytest.raises(ValueError, match="supported browser-profile error code"):
            await binding.fail_checkpoint(
                plan,
                outcome=BrowserProfileTerminalOutcome.FAILED,
                error_code="profile-secret-canary",
            )
        assert store.checkpoint_settlement_calls == 0
        await binding.fail_checkpoint(
            plan,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="checkpoint_failed",
        )

    asyncio.run(scenario())


def test_browser_profile_process_control_is_never_retried_as_store_unavailability() -> None:
    async def scenario() -> None:
        clock = _Clock()
        restore_store = _ProcessControlPreparationStore(
            clock=clock,
            fail_operation="restore",
        )
        restore_binding = _binding(restore_store)
        await restore_binding.initialize()

        with pytest.raises(BaseExceptionGroup) as restore_failure:
            await _prepared_restore(restore_binding, suffix="process_control")
        assert restore_store.restore_calls == 1
        assert any(
            type(candidate) is GeneratorExit for candidate in restore_failure.value.exceptions
        )

        checkpoint_store = _ProcessControlPreparationStore(
            clock=clock,
            fail_operation="checkpoint",
        )
        checkpoint_binding = _binding(checkpoint_store)
        await checkpoint_binding.initialize()
        material, _ = await _settled_restore(
            checkpoint_binding,
            suffix="checkpoint_process_control",
        )
        with pytest.raises(BaseExceptionGroup) as checkpoint_failure:
            await checkpoint_binding.reserve_checkpoint(
                material=material,
                operation_id="checkpoint_process_control",
                source_revision="revision_process_control",
                source_operation_receipt_id="receipt_process_control",
                source_operation_fingerprint=_digest("process-control"),
                ambiguous_lineage=False,
            )
        assert checkpoint_store.checkpoint_calls == 1
        assert any(
            type(candidate) is GeneratorExit for candidate in checkpoint_failure.value.exceptions
        )

    asyncio.run(scenario())


def test_browser_profile_process_control_is_not_lost_to_concurrent_cancellation() -> None:
    def exception_tree(error: BaseException):
        yield error
        if isinstance(error, BaseExceptionGroup):
            for child in error.exceptions:
                yield from exception_tree(child)

    async def scenario() -> tuple[BaseExceptionGroup, asyncio.Task[object]]:
        clock = _Clock()
        store = _ConcurrentProcessControlRenewStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="concurrent_control")
        owner = asyncio.create_task(binding.renew_writer(material))
        await store.entered.wait()
        owner.cancel("cancel profile renewal")
        assert owner.cancelling() == 1
        store.release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await owner
        return raised.value, owner

    failure, owner = asyncio.run(scenario())
    leaves = tuple(exception_tree(failure))
    assert sum(type(item) is asyncio.CancelledError for item in leaves) == 1
    assert sum(type(item) is GeneratorExit for item in leaves) == 1
    assert owner.cancelling() == 1
    assert owner.cancelled() is False


def test_checkpoint_commit_then_process_control_remains_authoritative() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _CommitThenProcessControlCheckpointStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="publish_control")
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_commit_then_control",
            source_revision="revision_commit_then_control",
            source_operation_receipt_id="receipt_commit_then_control",
            source_operation_fingerprint=_digest("commit-then-control"),
            ambiguous_lineage=False,
        )

        with pytest.raises(GeneratorExit, match="profile publication stopped"):
            await binding.publish_checkpoint(plan, _state())

        receipt = await binding.reconcile_checkpoint("checkpoint_commit_then_control")
        assert receipt is not None
        assert receipt.outcome is BrowserProfileTerminalOutcome.SUCCEEDED
        assert receipt.published_ref.generation == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_browser_profile_store_round_trip_is_encrypted_scoped_and_restart_safe(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        initial = await binding.initialize()
        assert initial.generation == 0

        material, restore_receipt = await _settled_restore(binding)
        assert material.state == BrowserProfileStateV1()
        assert restore_receipt.profile_ref == initial
        _plan, checkpoint_receipt = await _checkpoint(binding, material)
        assert checkpoint_receipt.previous_ref == initial
        assert checkpoint_receipt.published_ref.generation == 1
        assert checkpoint_receipt.origin_count == 1
        assert checkpoint_receipt.cookie_count == 1
        assert checkpoint_receipt.storage_entry_count == 1
        await binding.release_writer(material)
        assert await binding.initialize() == checkpoint_receipt.published_ref

        inspection = await store.inspect_profile(binding.access)
        assert inspection.status is BrowserProfileStatus.AVAILABLE
        assert inspection.generation == 1
        assert inspection.origin_count == 1
        assert inspection.cookie_count == 1
        assert inspection.storage_entry_count == 1
        assert not inspection.active_writer
        inspection_dump = json.dumps(inspection.model_dump(mode="json"), sort_keys=True)
        assert "profile-secret-canary" not in inspection_dump

        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()
            raw_files = tuple(tmp_path.glob("browser-profiles.sqlite*"))
            assert raw_files
            assert all(b"profile-secret-canary" not in item.read_bytes() for item in raw_files)
            store = _store(tmp_path, durable=True, clock=clock)

        restored_binding = BrowserProfileBinding(
            authority=binding.authority,
            store=store,
            key_authority=AESGCMBrowserProfileKeyAuthority(
                authority_id="browser-profile-test-key",
                key=_KEY,
            ),
            current_policy=BrowserProfileDestinationPolicy.build((_ORIGIN,)),
            checkpoint_policy=BrowserProfileCheckpointPolicy.ON_CLOSE,
            lease_seconds=120,
        )
        restored = await _prepared_restore(restored_binding, suffix="2")
        assert restored.state == _state()
        assert restored.preparation.profile_ref == checkpoint_receipt.published_ref
        await restored_binding.complete_restore(
            restored,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_sparse_checkpoint_history_prunes_restore_authority_coherently(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock, suffix="-history")
        binding = _binding(store)
        await binding.initialize()

        checkpointed, _ = await _settled_restore(binding, suffix="checkpointed")
        _plan, published = await _checkpoint(
            binding,
            checkpointed,
            operation_id="checkpoint_sparse_history",
        )
        await binding.release_writer(checkpointed)

        for index in range(BROWSER_PROFILE_MAX_RECEIPTS):
            rejected = await _prepared_restore(binding, suffix=f"rejected_{index}")
            await binding.complete_restore(
                rejected,
                outcome=BrowserProfileTerminalOutcome.FAILED,
                error_code="restore_rejected",
            )

        inspection = await store.inspect_profile(binding.access)
        assert inspection.generation == published.published_ref.generation
        assert inspection.content_fingerprint == published.published_ref.content_fingerprint
        assert inspection.active_writer is False

        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()
            store = _store(
                tmp_path,
                durable=True,
                clock=clock,
                suffix="-history",
            )
            inspection = await store.inspect_profile(binding.access)
            assert inspection.generation == published.published_ref.generation
            assert inspection.content_fingerprint == published.published_ref.content_fingerprint
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_full_restore_history_prunes_multiple_checkpoints_for_one_restore(
    tmp_path: Path,
    durable: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Scale only the history ceiling: the public binding, record validation,
    # encryption, and both store implementations retain their production paths.
    history_limit = 4
    monkeypatch.setattr("cayu.browser_profiles.BROWSER_PROFILE_MAX_RECEIPTS", history_limit)

    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock, suffix="-full-history")
        binding = _binding(store)
        await binding.initialize()
        try:
            oldest, _ = await _settled_restore(binding, suffix="a")
            for index in range(2):
                await _checkpoint(binding, oldest, operation_id=f"checkpoint_a_{index}")
            await binding.release_writer(oldest)

            for index in range(history_limit - 2):
                material, _ = await _settled_restore(binding, suffix=f"b_{index:03d}")
                _, published = await _checkpoint(
                    binding, material, operation_id=f"checkpoint_b_{index:03d}"
                )
                await binding.release_writer(material)

            rejected = await _prepared_restore(binding, suffix="z_rejected")
            last_restore = await binding.complete_restore(
                rejected,
                outcome=BrowserProfileTerminalOutcome.FAILED,
                error_code="restore_rejected",
            )
            before = await store._load_record(binding.access.profile_id)
            assert before is not None
            assert len(before.restore_operations) == history_limit
            assert len(before.checkpoint_operations) == history_limit
            assert before.writer is None

            if durable:
                await store.close()
                store = _store(tmp_path, durable=True, clock=clock, suffix="-full-history")
                binding = _binding(store)
                await binding.initialize()

            # Pruning one of A's two checkpoints does not yet free its restore.
            # Admission must continue making safe progress in the same transaction.
            replacement = await _prepared_restore(binding, suffix="zz_replacement")
            assert replacement.state == _state()
            assert replacement.preparation.profile_ref == published.published_ref
            after = await store._load_record(binding.access.profile_id)
            assert after is not None
            assert len(after.restore_operations) == history_limit
            assert len(after.checkpoint_operations) == history_limit - 2
            assert "restore_a" not in after.restore_operations
            assert after.last_restore_receipt_id == last_restore.receipt_id
            assert after.current_envelope == before.current_envelope
            assert after.last_checkpoint_receipt_id == published.receipt_id
            assert await binding.reconcile_checkpoint(published.operation_id) == published

            replayed = await _prepared_restore(binding, suffix="zz_replacement")
            assert replayed.preparation.writer_claim == replacement.preparation.writer_claim
            await binding.complete_restore(
                replacement, outcome=BrowserProfileTerminalOutcome.SUCCEEDED
            )
            await binding.release_writer(replacement)
            if durable:
                await store.close()
                store = _store(tmp_path, durable=True, clock=clock, suffix="-full-history")
                binding = _binding(store)
                await binding.initialize()
            inspection = await store.inspect_profile(binding.access)
            assert inspection.generation == published.published_ref.generation
            assert inspection.content_fingerprint == published.published_ref.content_fingerprint
            assert inspection.active_writer is False
        finally:
            if isinstance(store, SQLiteBrowserProfileStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_browser_profile_mutation_samples_time_inside_store_write_boundary(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _BoundaryClock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        if isinstance(store, SQLiteBrowserProfileStore):
            clock.boundary_check = lambda: store._connection.in_transaction
        else:
            clock.boundary_check = store._lock.locked
        clock.require_boundary = True

        material = await _prepared_restore(binding)

        assert clock.boundary_samples == 1
        await binding.complete_restore(
            material,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


def test_sqlite_browser_profile_rolls_back_failed_commits_before_retry(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = SQLiteBrowserProfileStore(
            tmp_path / "browser-profiles-commit-failure.sqlite",
            store_id="browser-profile-test-store",
            clock=clock,
        )
        connection = _CommitFailureConnection(store._connection)
        store._connection = connection  # type: ignore[assignment]
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="commit_failure")

        # Reservation retries once for acknowledgement loss. Both attempts fail
        # before commit and must each roll back rather than exposing uncommitted
        # state to the retry/readback path.
        connection.failures_remaining = 2
        with pytest.raises(BrowserProfileUnavailable, match="store operation failed"):
            await binding.reserve_checkpoint(
                material=material,
                operation_id="checkpoint_commit_failure",
                source_revision="revision_commit_failure",
                source_operation_receipt_id="receipt_commit_failure",
                source_operation_fingerprint=_digest("commit-failure"),
                ambiguous_lineage=False,
            )
        assert connection.in_transaction is False
        assert await binding.reconcile_checkpoint("checkpoint_commit_failure") is None

        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_commit_failure",
            source_revision="revision_commit_failure",
            source_operation_receipt_id="receipt_commit_failure",
            source_operation_fingerprint=_digest("commit-failure"),
            ambiguous_lineage=False,
        )
        failed = await binding.fail_checkpoint(
            plan,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="checkpoint_failed",
        )
        assert failed.outcome is BrowserProfileTerminalOutcome.FAILED
        await binding.release_writer(material)
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_browser_profile_writer_reconstruction_renews_before_old_lease_expiry(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _receipt = await _settled_restore(binding)
        original_expiry = material.preparation.writer_claim.expires_at

        clock.advance(119)
        request = material.preparation.request
        reconstructed = await binding.resume_writer(
            execution_profile_fingerprint=request.execution_profile_fingerprint,
            allocation_fingerprint=request.allocation_fingerprint,
            browser_session_id=request.browser_session_id,
        )

        assert reconstructed.preparation.writer_claim.expires_at == (
            clock.value + timedelta(seconds=binding.lease_seconds)
        )
        assert reconstructed.preparation.writer_claim.expires_at > original_expiry
        clock.advance(2)
        with pytest.raises(BrowserProfileStoreConflict, match="active writer"):
            await _prepared_restore(binding, suffix="competing-after-old-expiry")

        await binding.release_writer(reconstructed)
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_browser_profile_rejects_conflicts_widening_wrong_scope_expiry_and_revocation(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _receipt = await _settled_restore(binding)

        with pytest.raises(BrowserProfileStoreConflict, match="active writer"):
            await _prepared_restore(binding, suffix="2")

        wrong_access = BrowserProfileAccess(
            profile_id=binding.access.profile_id,
            authority_fingerprint=binding.access.authority_fingerprint,
            owner_fingerprint=_digest("wrong-owner"),
            sharing_fingerprint=binding.access.sharing_fingerprint,
            store_id=binding.access.store_id,
        )
        with pytest.raises(BrowserProfileStoreConflict, match="access authority"):
            await store.inspect_profile(wrong_access)

        with pytest.raises(ValueError, match="cannot be widened"):
            BrowserProfileBinding(
                authority=binding.authority,
                store=store,
                key_authority=binding.key_authority,
                current_policy=BrowserProfileDestinationPolicy.build(
                    (_ORIGIN, _SECOND_ORIGIN, "https://extra.browser.test")
                ),
            )

        revoked = await binding.revoke()
        assert revoked.status is BrowserProfileStatus.REVOKED
        assert revoked.active_writer
        with pytest.raises(BrowserProfileUnavailable, match="revoked"):
            await binding.renew_writer(material)
        with pytest.raises(BrowserProfileUnavailable, match="revoked"):
            await binding.reserve_checkpoint(
                material=material,
                operation_id="checkpoint_revoked",
                source_revision="revision_1",
                source_operation_receipt_id="receipt_1",
                source_operation_fingerprint=_digest("operation"),
                ambiguous_lineage=False,
            )
        await binding.release_writer(material)

        # Retrying an implicit revocation after acknowledgement loss is idempotent.
        assert (await binding.revoke()).revoked_at == revoked.revoked_at
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

        expiring_store = _store(
            tmp_path,
            durable=durable,
            clock=clock,
            suffix="-expiry",
        )
        expiring = _binding(
            expiring_store,
            expires_at=clock.value + timedelta(seconds=5),
        )
        await expiring.initialize()
        clock.advance(5)
        with pytest.raises(BrowserProfileUnavailable, match="expired"):
            await _prepared_restore(expiring)
        assert (
            await expiring_store.inspect_profile(expiring.access)
        ).status is BrowserProfileStatus.EXPIRED
        if isinstance(expiring_store, SQLiteBrowserProfileStore):
            await expiring_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_writer_renewal_uses_durable_expiry_not_caller_supplied_expiry(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        forged_claim = material.preparation.writer_claim.model_copy(
            update={"expires_at": clock.value + timedelta(days=1)}
        )

        clock.advance(121)
        with pytest.raises(BrowserProfileStoreConflict, match="lease expired"):
            await store.renew_writer(
                binding.access,
                forged_claim,
                lease_seconds=120,
            )

        replacement = await _prepared_restore(binding, suffix="after_forged_renewal")
        assert replacement.preparation.writer_claim.fence > forged_claim.fence
        await binding.complete_restore(
            replacement,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_browser_profile_wrong_key_and_narrow_policy_fail_closed_with_safe_status(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _receipt = await _settled_restore(binding)
        await _checkpoint(binding, material)
        await binding.release_writer(material)

        wrong_key = BrowserProfileBinding(
            authority=binding.authority,
            store=store,
            key_authority=AESGCMBrowserProfileKeyAuthority(
                authority_id="browser-profile-test-key",
                key=_OTHER_KEY,
            ),
            lease_seconds=120,
        )
        with pytest.raises(BrowserProfileUnavailable, match="cannot be decrypted"):
            await _prepared_restore(wrong_key, suffix="wrong_key")
        inspection = await store.inspect_profile(binding.access)
        assert inspection.status is BrowserProfileStatus.CORRUPT
        assert inspection.safe_error_code == "profile_corrupt"

        # A fresh store proves valid ciphertext that exceeds a narrowed origin
        # policy is incompatible rather than silently dropped or broadened.
        other_store = _store(
            tmp_path,
            durable=durable,
            clock=clock,
            suffix="-policy",
        )
        full = _binding(other_store)
        await full.initialize()
        full_material, _ = await _settled_restore(full)
        state = BrowserProfileStateV1(
            origins=(
                BrowserProfileOriginStorage(
                    origin=_SECOND_ORIGIN,
                    local_storage=(BrowserProfileStorageEntry(name="state", value="value"),),
                ),
            )
        )
        await _checkpoint(full, full_material, state=state)
        await full.release_writer(full_material)
        narrow = BrowserProfileBinding(
            authority=full.authority,
            store=other_store,
            key_authority=full.key_authority,
            current_policy=BrowserProfileDestinationPolicy.build((_ORIGIN,)),
            lease_seconds=120,
        )
        with pytest.raises(BrowserProfileUnavailable, match="destination authority"):
            await _prepared_restore(narrow, suffix="narrow")
        assert (
            await other_store.inspect_profile(full.access)
        ).status is BrowserProfileStatus.INCOMPATIBLE
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()
        if isinstance(other_store, SQLiteBrowserProfileStore):
            await other_store.close()

    asyncio.run(scenario())


def test_browser_profile_state_contract_is_exact_bounded_and_diagnostic_safe() -> None:
    canary = "profile-secret-canary"
    state = _state(canary=canary)
    assert canary not in repr(state)
    assert canary not in str(state)
    assert canary not in repr(state.cookies[0])
    assert canary not in repr(state.origins[0].local_storage[0])
    assert "indexed_db" in state.omitted_categories
    assert "session_storage" in state.omitted_categories

    with pytest.raises(ValidationError):
        BrowserProfileStateV1.model_validate(
            {
                **state.model_dump(mode="python"),
                "indexed_db": [{"name": canary}],
            }
        )
    with pytest.raises(ValidationError):
        BrowserProfileCookie(
            name="session",
            value=canary,
            domain=".browser.test",
            secure=True,
        )
    with pytest.raises(ValidationError):
        BrowserProfileCookie(
            name="session",
            value=canary,
            domain="auth.browser.test",
            secure=False,
        )
    with pytest.raises(ValidationError):
        BrowserProfileStorageEntry(name="name", value="x" * (64 * 1024 + 1))


def test_browser_profile_binding_owns_immutable_authority() -> None:
    clock = _Clock()
    binding = _binding(
        InMemoryBrowserProfileStore(
            store_id="browser-profile-test-store",
            clock=clock,
        )
    )

    with pytest.raises(AttributeError, match="immutable"):
        binding.current_policy = BrowserProfileDestinationPolicy.build((_ORIGIN,))

    original_material = binding.execution_profile_material()
    exposed_authority = binding.authority
    exposed_access = binding.access
    exposed_policy = binding.current_policy
    exposed_limits = binding.limits
    object.__setattr__(exposed_authority, "profile_id", "bprof_substituted")
    object.__setattr__(exposed_access, "profile_id", "bprof_substituted")
    object.__setattr__(exposed_policy, "origins", (_ORIGIN,))
    object.__setattr__(exposed_limits, "max_plaintext_bytes", 1)

    assert binding.execution_profile_material() == original_material
    assert binding.authority.profile_id == "bprof_test"
    assert binding.access.profile_id == "bprof_test"
    assert binding.current_policy.origins == tuple(sorted((_ORIGIN, _SECOND_ORIGIN)))
    assert binding.limits.max_plaintext_bytes != 1

    changed_lease = BrowserProfileBinding(
        authority=binding.authority,
        store=binding.store,
        key_authority=binding.key_authority,
        current_policy=binding.current_policy,
        limits=binding.limits,
        checkpoint_policy=binding.checkpoint_policy,
        lease_seconds=binding.lease_seconds + 1,
    )
    assert changed_lease.execution_profile_material() != binding.execution_profile_material()


def test_explicit_blank_browser_profile_id_is_not_replaced_by_a_generated_id() -> None:
    store = InMemoryBrowserProfileStore(store_id="browser-profile-test-store")
    with pytest.raises(ValidationError):
        BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-profile-tests",
                tenant_id="test-tenant",
                sharing_scope="agent-release-v1",
            ),
            destination_policy=BrowserProfileDestinationPolicy.build((_ORIGIN,)),
            browser_protocol="cayu.browser-session.v3",
            browser_worker_version="10",
            store=store,
            key_authority=AESGCMBrowserProfileKeyAuthority(
                authority_id="browser-profile-test-key",
                key=_KEY,
            ),
            profile_id="",
        )


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_browser_profile_writer_cannot_release_before_restore_settlement(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material = await _prepared_restore(binding)

        with pytest.raises(BrowserProfileStoreConflict, match="restore settlement"):
            await binding.release_writer(material)
        with pytest.raises(BrowserProfileStoreConflict, match="active writer"):
            await _prepared_restore(binding, suffix="competing")

        await binding.complete_restore(
            material,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        replacement = await _prepared_restore(binding, suffix="replacement")
        await binding.complete_restore(
            replacement,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_expired_writer_cannot_settle_restore_or_stage_checkpoint(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()

        late_restore = await _prepared_restore(binding, suffix="late")
        clock.advance(121)
        with pytest.raises(BrowserProfileStoreConflict, match="lease expired"):
            await binding.complete_restore(
                late_restore,
                outcome=BrowserProfileTerminalOutcome.SUCCEEDED,
            )

        replacement = await _prepared_restore(binding, suffix="replacement")
        await binding.complete_restore(
            replacement,
            outcome=BrowserProfileTerminalOutcome.SUCCEEDED,
        )
        plan = await binding.reserve_checkpoint(
            material=replacement,
            operation_id="checkpoint_late",
            source_revision="browser_revision_late",
            source_operation_receipt_id="browser_receipt_late",
            source_operation_fingerprint=_digest("browser-operation-late"),
            ambiguous_lineage=False,
        )
        clock.advance(121)
        with pytest.raises(BrowserProfileStoreConflict, match="lease expired"):
            await binding.publish_checkpoint(plan, _state())
        assert (await store.current_ref(binding.access)).generation == 0

        final = await _prepared_restore(binding, suffix="final")
        abandoned = await store.load_checkpoint_receipt(
            binding.access,
            "checkpoint_late",
        )
        assert abandoned is not None
        assert abandoned.outcome is BrowserProfileTerminalOutcome.OUTCOME_UNKNOWN
        await binding.complete_restore(
            final,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_published_checkpoint_replays_after_revocation(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        _plan, receipt = await _checkpoint(binding, material)
        await binding.revoke()

        assert await binding.reconcile_checkpoint("checkpoint_1") == receipt
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


class _CommitThenRaiseStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.raise_after_publish = True

    async def publish_checkpoint(self, access, operation_id):
        result = await super().publish_checkpoint(access, operation_id)
        if self.raise_after_publish:
            self.raise_after_publish = False
            raise ConnectionError("checkpoint acknowledgement lost")
        return result


def test_browser_profile_checkpoint_adopts_commit_after_acknowledgement_loss() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _CommitThenRaiseStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        _plan, receipt = await _checkpoint(binding, material)
        assert receipt.published_ref.generation == 1
        assert await store.current_ref(binding.access) == receipt.published_ref
        assert material.preparation.writer_claim.generation == 1

    asyncio.run(scenario())


class _BlockedPublishStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def publish_checkpoint(self, access, operation_id):
        self.entered.set()
        await self.release.wait()
        return await super().publish_checkpoint(access, operation_id)


class _CancelledOncePublishStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.cancel_once = True

    async def publish_checkpoint(self, access, operation_id):
        if self.cancel_once:
            self.cancel_once = False
            raise asyncio.CancelledError("profile-private-cancellation-canary")
        return await super().publish_checkpoint(access, operation_id)


class _CancelledOnceRenewStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.cancel_once = True

    async def renew_writer(self, access, claim, *, lease_seconds):
        if self.cancel_once:
            self.cancel_once = False
            raise asyncio.CancelledError("profile-renewal-cancellation-canary")
        return await super().renew_writer(
            access,
            claim,
            lease_seconds=lease_seconds,
        )


class _CommitThenRaiseReserveStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.raise_after_reserve = True

    async def reserve_checkpoint(self, request, *, reserved_ciphertext_bytes):
        result = await super().reserve_checkpoint(
            request,
            reserved_ciphertext_bytes=reserved_ciphertext_bytes,
        )
        if self.raise_after_reserve:
            self.raise_after_reserve = False
            raise ConnectionError("checkpoint reservation acknowledgement lost")
        return result


def test_browser_profile_checkpoint_adopts_reservation_after_acknowledgement_loss() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _CommitThenRaiseReserveStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)

        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_reservation_acknowledgement_lost",
            source_revision="revision_reservation",
            source_operation_receipt_id="receipt_reservation",
            source_operation_fingerprint=_digest("operation-reservation"),
            ambiguous_lineage=False,
        )

        assert plan.reservation.request.operation_id == (
            "checkpoint_reservation_acknowledgement_lost"
        )
        failed = await binding.fail_checkpoint(
            plan,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="checkpoint_failed",
        )
        assert failed.outcome is BrowserProfileTerminalOutcome.FAILED
        await binding.release_writer(material)

    asyncio.run(scenario())


class _BlockedReserveStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def reserve_checkpoint(self, request, *, reserved_ciphertext_bytes):
        result = await super().reserve_checkpoint(
            request,
            reserved_ciphertext_bytes=reserved_ciphertext_bytes,
        )
        self.entered.set()
        await self.release.wait()
        return result


class _BlockedSQLiteReserveStore(SQLiteBrowserProfileStore):
    def __init__(self, path: Path, *, clock: _Clock) -> None:
        super().__init__(
            path,
            store_id="browser-profile-test-store",
            clock=clock,
        )
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def reserve_checkpoint(self, request, *, reserved_ciphertext_bytes):
        result = await super().reserve_checkpoint(
            request,
            reserved_ciphertext_bytes=reserved_ciphertext_bytes,
        )
        self.entered.set()
        await self.release.wait()
        return result


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_checkpoint_reservation_cancellation_settles_before_releasing_writer(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = (
            _BlockedSQLiteReserveStore(tmp_path / "blocked-reserve.sqlite", clock=clock)
            if durable
            else _BlockedReserveStore(clock=clock)
        )
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        owner = asyncio.create_task(
            binding.reserve_checkpoint(
                material=material,
                operation_id="checkpoint_cancelled_during_reservation",
                source_revision="revision_cancelled_reservation",
                source_operation_receipt_id="receipt_cancelled_reservation",
                source_operation_fingerprint=_digest("cancelled-reservation"),
                ambiguous_lineage=False,
            )
        )
        await store.entered.wait()

        owner.cancel("cancel checkpoint reservation")
        assert owner.cancelling() == 1
        assert not owner.done()
        store.release.set()
        with pytest.raises(asyncio.CancelledError, match="cancel checkpoint reservation"):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1

        receipt = await store.load_checkpoint_receipt(
            binding.access,
            "checkpoint_cancelled_during_reservation",
        )
        assert receipt is not None
        assert receipt.outcome is BrowserProfileTerminalOutcome.FAILED
        assert receipt.error_code == "checkpoint_cancelled_before_export"
        await binding.release_writer(material)
        assert (await store.inspect_profile(binding.access)).active_writer is False
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


def test_browser_profile_checkpoint_publication_survives_real_task_cancellation() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _BlockedPublishStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_cancelled",
            source_revision="revision_1",
            source_operation_receipt_id="receipt_1",
            source_operation_fingerprint=_digest("operation"),
            ambiguous_lineage=True,
        )
        owner = asyncio.create_task(binding.publish_checkpoint(plan, _state()))
        await store.entered.wait()
        owner.cancel()
        assert owner.cancelling() == 1
        assert not owner.done()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        receipt = await store.load_checkpoint_receipt(
            binding.access,
            "checkpoint_cancelled",
        )
        assert receipt is not None
        assert receipt.outcome is BrowserProfileTerminalOutcome.SUCCEEDED
        assert receipt.ambiguous_lineage
        assert (await store.current_ref(binding.access)).generation == 1

    asyncio.run(scenario())


def test_browser_profile_store_child_cancellation_is_not_caller_cancellation() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _CancelledOncePublishStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_child_cancelled",
            source_revision="revision_child_cancelled",
            source_operation_receipt_id="receipt_child_cancelled",
            source_operation_fingerprint=_digest("operation-child-cancelled"),
            ambiguous_lineage=False,
        )
        owner = asyncio.current_task()
        assert owner is not None
        requests_before = owner.cancelling()

        with pytest.raises(BrowserProfileUnavailable) as exc_info:
            await binding.publish_checkpoint(plan, _state())

        assert "profile-private-cancellation-canary" not in str(exc_info.value)
        assert "profile-private-cancellation-canary" not in repr(exc_info.value)
        assert exc_info.value.__cause__ is None
        rendered_locals = "\n".join(
            repr(frame.f_locals)
            for frame, _line_number in traceback.walk_tb(exc_info.value.__traceback__)
            if is_cayu_source_filename(frame.f_code.co_filename)
        )
        assert "profile-private-cancellation-canary" not in rendered_locals
        assert owner.cancelling() == requests_before
        published = await binding.reconcile_checkpoint("checkpoint_child_cancelled")
        assert published is not None
        assert published.outcome is BrowserProfileTerminalOutcome.SUCCEEDED

    asyncio.run(scenario())


def test_browser_profile_writer_renewal_child_cancellation_is_not_caller_cancellation() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _CancelledOnceRenewStore(clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        owner = asyncio.current_task()
        assert owner is not None
        requests_before = owner.cancelling()

        with pytest.raises(BrowserProfileUnavailable) as exc_info:
            await binding.renew_writer(material)

        assert exc_info.value.__cause__ is None
        rendered = "\n".join(
            (
                str(exc_info.value),
                repr(exc_info.value),
                *(
                    repr(frame.f_locals)
                    for frame, _line_number in traceback.walk_tb(exc_info.value.__traceback__)
                    if is_cayu_source_filename(frame.f_code.co_filename)
                ),
            )
        )
        assert "profile-renewal-cancellation-canary" not in rendered
        assert owner.cancelling() == requests_before
        renewed = await binding.renew_writer(material)
        assert renewed.writer_id == material.preparation.writer_claim.writer_id

    asyncio.run(scenario())


class _CanaryCiphertextKeyAuthority(BrowserProfileKeyAuthority):
    @property
    def authority_id(self) -> str:
        return "browser-profile-test-key"

    async def encrypt(self, *, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        del nonce, aad
        prefix = b"BROWSER_PROFILE_CIPHERTEXT_TRACE_CANARY"
        return prefix + (b"x" * (len(plaintext) + 16 - len(prefix)))

    async def decrypt(self, *, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        del nonce, ciphertext, aad
        raise AssertionError("decryption is not expected")


class _LeakingStageStore(InMemoryBrowserProfileStore):
    async def stage_checkpoint(
        self,
        reservation,
        envelope,
        *,
        origin_count,
        cookie_count,
        storage_entry_count,
    ):
        del reservation, origin_count, cookie_count, storage_entry_count
        extension_local = envelope.ciphertext().decode("ascii")
        raise RuntimeError(extension_local)


def test_checkpoint_extension_failure_cannot_publish_ciphertext_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> str:
        clock = _Clock()
        store = _LeakingStageStore(
            store_id="browser-profile-test-store",
            clock=clock,
        )
        binding = BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-profile-tests",
                tenant_id="test-tenant",
                sharing_scope="agent-release-v1",
            ),
            destination_policy=BrowserProfileDestinationPolicy.build((_ORIGIN, _SECOND_ORIGIN)),
            browser_protocol="cayu.browser-session.v3",
            browser_worker_version="10",
            store=store,
            key_authority=_CanaryCiphertextKeyAuthority(),
            profile_id="bprof_extension_diagnostic",
            lease_seconds=120,
        )
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="extension_diagnostic")
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_extension_diagnostic",
            source_revision="revision_extension_diagnostic",
            source_operation_receipt_id="receipt_extension_diagnostic",
            source_operation_fingerprint=_digest("extension-diagnostic"),
            ambiguous_lineage=False,
        )

        with pytest.raises(BrowserProfileUnavailable) as exc_info:
            await binding.publish_checkpoint(plan, _state())

        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        return "\n".join(
            (
                str(exc_info.value),
                repr(exc_info.value),
                *(
                    repr(frame.f_locals)
                    for frame, _line_number in traceback.walk_tb(exc_info.value.__traceback__)
                ),
            )
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = asyncio.run(scenario())
    captured = capsys.readouterr()
    diagnostic_surface = "\n".join(
        (
            rendered,
            caplog.text,
            captured.out,
            captured.err,
            *(str(item.message) for item in caught),
        )
    )
    assert "BROWSER_PROFILE_CIPHERTEXT_TRACE_CANARY" not in diagnostic_surface


def test_checkpoint_failure_traceback_does_not_retain_raw_ciphertext() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _FailBeforeMemoryPublish(
            store_id="browser-profile-test-store",
            clock=clock,
        )
        binding = BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-profile-tests",
                tenant_id="test-tenant",
                sharing_scope="agent-release-v1",
            ),
            destination_policy=BrowserProfileDestinationPolicy.build((_ORIGIN, _SECOND_ORIGIN)),
            browser_protocol="cayu.browser-session.v3",
            browser_worker_version="10",
            store=store,
            key_authority=_CanaryCiphertextKeyAuthority(),
            profile_id="bprof_ciphertext_trace",
            lease_seconds=120,
        )
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="ciphertext_trace")
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_ciphertext_trace",
            source_revision="revision_ciphertext_trace",
            source_operation_receipt_id="receipt_ciphertext_trace",
            source_operation_fingerprint=_digest("operation-ciphertext-trace"),
            ambiguous_lineage=False,
        )

        with pytest.raises(BrowserProfileUnavailable) as exc_info:
            await binding.publish_checkpoint(plan, _state())

        rendered_locals = "\n".join(
            repr(frame.f_locals)
            for frame, _line_number in traceback.walk_tb(exc_info.value.__traceback__)
            if is_cayu_source_filename(frame.f_code.co_filename)
        )
        assert "BROWSER_PROFILE_CIPHERTEXT_TRACE_CANARY" not in rendered_locals

    asyncio.run(scenario())


class _FailBeforeSQLitePublish(SQLiteBrowserProfileStore):
    async def publish_checkpoint(self, access, operation_id):
        del access, operation_id
        raise ConnectionError("checkpoint publication unavailable")


def test_sqlite_browser_profile_reconciles_every_checkpoint_publication_phase(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=True, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_restart",
            source_revision="revision_1",
            source_operation_receipt_id="receipt_1",
            source_operation_fingerprint=_digest("operation"),
            ambiguous_lineage=False,
        )
        await store.close()

        # Capacity reservation survives restart and can settle without having
        # captured or persisted any plaintext.
        store = _store(tmp_path, durable=True, clock=clock)
        binding = BrowserProfileBinding(
            authority=binding.authority,
            store=store,
            key_authority=binding.key_authority,
            lease_seconds=120,
        )
        failed = await binding.fail_checkpoint(
            BrowserProfileCheckpointPlan(plan.reservation, material),
            outcome=BrowserProfileTerminalOutcome.OUTCOME_UNKNOWN,
            error_code="checkpoint_outcome_unknown",
        )
        assert failed.published_ref == failed.previous_ref
        assert (await store.current_ref(binding.access)).generation == 0
        await store.close()

        staged_path = tmp_path / "browser-profiles-staged.sqlite"
        staging_store = _FailBeforeSQLitePublish(
            staged_path,
            store_id="browser-profile-test-store",
            clock=clock,
        )
        staging_binding = _binding(staging_store)
        await staging_binding.initialize()
        staging_material, _ = await _settled_restore(staging_binding, suffix="staged")
        staging_plan = await staging_binding.reserve_checkpoint(
            material=staging_material,
            operation_id="checkpoint_staged",
            source_revision="revision_staged",
            source_operation_receipt_id="receipt_staged",
            source_operation_fingerprint=_digest("operation-staged"),
            ambiguous_lineage=False,
        )
        with pytest.raises(BrowserProfileUnavailable, match="store operation failed"):
            await staging_binding.publish_checkpoint(staging_plan, _state())
        assert (await staging_store.current_ref(staging_binding.access)).generation == 0
        await staging_store.close()

        recovered_store = SQLiteBrowserProfileStore(
            staged_path,
            store_id="browser-profile-test-store",
            clock=clock,
        )
        recovered = await recovered_store.reconcile_checkpoint(
            staging_binding.access,
            "checkpoint_staged",
        )
        assert recovered is not None
        assert recovered.outcome is BrowserProfileTerminalOutcome.SUCCEEDED
        assert recovered.published_ref.generation == 1
        await recovered_store.close()

        reopened = SQLiteBrowserProfileStore(
            staged_path,
            store_id="browser-profile-test-store",
            clock=clock,
        )
        assert (
            await reopened.load_checkpoint_receipt(
                staging_binding.access,
                "checkpoint_staged",
            )
        ) == recovered
        await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_abandoned_checkpoint_reservation_is_settled_before_a_new_writer(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_abandoned_reservation",
            source_revision="revision_abandoned",
            source_operation_receipt_id="receipt_abandoned",
            source_operation_fingerprint=_digest("operation-abandoned"),
            ambiguous_lineage=True,
        )

        with pytest.raises(BrowserProfileStoreConflict, match="checkpoint settlement"):
            await binding.release_writer(material)
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()
            store = _store(tmp_path, durable=True, clock=clock)
            binding = BrowserProfileBinding(
                authority=binding.authority,
                store=store,
                key_authority=binding.key_authority,
                lease_seconds=120,
            )

        clock.advance(121)
        before_reclaim = await store.inspect_profile(binding.access)
        assert before_reclaim.status is BrowserProfileStatus.OUTCOME_UNKNOWN
        assert before_reclaim.safe_error_code == "writer_lease_expired"

        replacement = await _prepared_restore(binding, suffix="replacement")
        abandoned = await store.load_checkpoint_receipt(
            binding.access,
            "checkpoint_abandoned_reservation",
        )
        assert abandoned is not None
        assert abandoned.outcome is BrowserProfileTerminalOutcome.OUTCOME_UNKNOWN
        assert abandoned.error_code == "checkpoint_outcome_unknown"
        assert abandoned.previous_ref == abandoned.published_ref
        assert abandoned.ambiguous_lineage
        assert replacement.preparation.writer_claim.fence > material.preparation.writer_claim.fence
        assert replacement.preparation.profile_ref.generation == 0
        await binding.complete_restore(
            replacement,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


class _FailBeforeMemoryPublish(InMemoryBrowserProfileStore):
    async def publish_checkpoint(self, access, operation_id):
        del access, operation_id
        raise ConnectionError("checkpoint publication unavailable")


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_abandoned_staged_checkpoint_preserves_the_prior_generation(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        if durable:
            path = tmp_path / "browser-profiles-staged-loss.sqlite"
            store = _FailBeforeSQLitePublish(
                path,
                store_id="browser-profile-test-store",
                clock=clock,
            )
        else:
            path = None
            store = _FailBeforeMemoryPublish(
                store_id="browser-profile-test-store",
                clock=clock,
            )
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_abandoned_staged",
            source_revision="revision_staged",
            source_operation_receipt_id="receipt_staged",
            source_operation_fingerprint=_digest("operation-staged-loss"),
            ambiguous_lineage=False,
        )
        with pytest.raises(BrowserProfileUnavailable, match="store operation failed"):
            await binding.publish_checkpoint(plan, _state())
        assert (await store.current_ref(binding.access)).generation == 0

        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()
            assert path is not None
            store = SQLiteBrowserProfileStore(
                path,
                store_id="browser-profile-test-store",
                clock=clock,
            )
            binding = BrowserProfileBinding(
                authority=binding.authority,
                store=store,
                key_authority=binding.key_authority,
                lease_seconds=120,
            )

        clock.advance(121)
        replacement = await _prepared_restore(binding, suffix="after_staged_loss")
        abandoned = await store.load_checkpoint_receipt(
            binding.access,
            "checkpoint_abandoned_staged",
        )
        assert abandoned is not None
        assert abandoned.outcome is BrowserProfileTerminalOutcome.OUTCOME_UNKNOWN
        assert abandoned.previous_ref == abandoned.published_ref
        assert (await store.current_ref(binding.access)).generation == 0
        await binding.complete_restore(
            replacement,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


class _BlockingDecryptKeyAuthority(BrowserProfileKeyAuthority):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.delegate = AESGCMBrowserProfileKeyAuthority(
            authority_id="browser-profile-test-key",
            key=_KEY,
        )

    @property
    def authority_id(self) -> str:
        return self.delegate.authority_id

    async def encrypt(self, *, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        return await self.delegate.encrypt(nonce=nonce, plaintext=plaintext, aad=aad)

    async def decrypt(self, *, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        self.entered.set()
        await self.release.wait()
        return await self.delegate.decrypt(
            nonce=nonce,
            ciphertext=ciphertext,
            aad=aad,
        )


class _DiagnosticKeyAuthority(BrowserProfileKeyAuthority):
    def __init__(self, *, fail_operation: str) -> None:
        self.fail_operation = fail_operation
        self.delegate = AESGCMBrowserProfileKeyAuthority(
            authority_id="browser-profile-test-key",
            key=_KEY,
        )

    @property
    def authority_id(self) -> str:
        return self.delegate.authority_id

    async def encrypt(self, *, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        if self.fail_operation == "encrypt":
            private_extension_local = "BROWSER_PROFILE_KEY_EXCEPTION_CANARY"
            raise RuntimeError(private_extension_local)
        return await self.delegate.encrypt(nonce=nonce, plaintext=plaintext, aad=aad)

    async def decrypt(self, *, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        if self.fail_operation == "decrypt":
            private_extension_local = "BROWSER_PROFILE_KEY_EXCEPTION_CANARY"
            raise RuntimeError(private_extension_local)
        return await self.delegate.decrypt(nonce=nonce, ciphertext=ciphertext, aad=aad)


class _SubstitutingPreparationStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.substitute = False

    async def prepare_restore(self, request, *, lease_seconds):
        prepared = await super().prepare_restore(
            request,
            lease_seconds=lease_seconds,
        )
        if not self.substitute:
            return prepared
        replacement_allocation = _digest("substituted-allocation")
        replacement_request = prepared.request.model_copy(
            update={"allocation_fingerprint": replacement_allocation}
        )
        replacement_claim = prepared.writer_claim.model_copy(
            update={"allocation_fingerprint": replacement_allocation}
        )
        return prepared.model_copy(
            update={
                "request": replacement_request,
                "request_fingerprint": replacement_request.fingerprint(),
                "writer_claim": replacement_claim,
            }
        )


class _SubstitutingStageReceiptStore(InMemoryBrowserProfileStore):
    def __init__(self, *, clock: _Clock) -> None:
        super().__init__(store_id="browser-profile-test-store", clock=clock)
        self.publish_calls = 0

    async def stage_checkpoint(
        self,
        reservation,
        envelope,
        *,
        origin_count,
        cookie_count,
        storage_entry_count,
    ):
        receipt = await super().stage_checkpoint(
            reservation,
            envelope,
            origin_count=origin_count,
            cookie_count=cookie_count,
            storage_entry_count=storage_entry_count,
        )
        return receipt.model_copy(
            update={
                "published_ref": receipt.published_ref.model_copy(
                    update={"content_fingerprint": _digest("substituted-content")}
                )
            }
        )

    async def publish_checkpoint(self, access, operation_id):
        self.publish_calls += 1
        return await super().publish_checkpoint(access, operation_id)


class _CountingDecryptKeyAuthority(BrowserProfileKeyAuthority):
    def __init__(self) -> None:
        self.decrypt_calls = 0
        self.delegate = AESGCMBrowserProfileKeyAuthority(
            authority_id="browser-profile-test-key",
            key=_KEY,
        )

    @property
    def authority_id(self) -> str:
        return self.delegate.authority_id

    async def encrypt(self, *, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        return await self.delegate.encrypt(nonce=nonce, plaintext=plaintext, aad=aad)

    async def decrypt(self, *, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        self.decrypt_calls += 1
        return await self.delegate.decrypt(nonce=nonce, ciphertext=ciphertext, aad=aad)


def test_restore_rejects_valid_but_substituted_store_authority_before_decryption() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _SubstitutingPreparationStore(clock=clock)
        seed = _binding(store)
        await seed.initialize()
        material, _ = await _settled_restore(seed, suffix="substitution_seed")
        await _checkpoint(
            seed,
            material,
            operation_id="checkpoint_substitution_seed",
        )
        await seed.release_writer(material)

        key_authority = _CountingDecryptKeyAuthority()
        binding = BrowserProfileBinding(
            authority=seed.authority,
            store=store,
            key_authority=key_authority,
            lease_seconds=120,
        )
        store.substitute = True
        with pytest.raises(BrowserProfileStoreConflict, match="conflicting authority"):
            await binding.prepare_restore(
                operation_id="restore_substituted_preparation",
                execution_profile_fingerprint=_digest("substitution-execution"),
                allocation_fingerprint=_digest("substitution-allocation"),
                browser_session_id="browser_session_substitution",
            )
        assert key_authority.decrypt_calls == 0

    asyncio.run(scenario())


def test_checkpoint_rejects_substituted_store_receipt_before_publication() -> None:
    async def scenario() -> None:
        store = _SubstitutingStageReceiptStore(clock=_Clock())
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="stage_substitution")
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_stage_substitution",
            source_revision="revision_stage_substitution",
            source_operation_receipt_id="receipt_stage_substitution",
            source_operation_fingerprint=_digest("stage-substitution"),
            ambiguous_lineage=False,
        )

        with pytest.raises(BrowserProfileStoreConflict, match="envelope authority"):
            await binding.publish_checkpoint(plan, _state())

        assert store.publish_calls == 0
        assert (await store.current_ref(binding.access)).generation == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["encrypt", "decrypt"])
def test_key_authority_failures_leave_no_private_diagnostic_context(
    operation: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> str:
        clock = _Clock()
        store = _store(Path("."), durable=False, clock=clock)
        key_authority = _DiagnosticKeyAuthority(fail_operation=operation)
        binding = BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-profile-tests",
                tenant_id="test-tenant",
                sharing_scope="agent-release-v1",
            ),
            destination_policy=BrowserProfileDestinationPolicy.build((_ORIGIN, _SECOND_ORIGIN)),
            browser_protocol="cayu.browser-session.v3",
            browser_worker_version="10",
            store=store,
            key_authority=key_authority,
            profile_id=f"bprof_key_diagnostic_{operation}",
            lease_seconds=120,
        )
        await binding.initialize()
        if operation == "encrypt":
            material, _ = await _settled_restore(binding, suffix="key_encrypt")
            plan = await binding.reserve_checkpoint(
                material=material,
                operation_id="checkpoint_key_diagnostic",
                source_revision="revision_key_diagnostic",
                source_operation_receipt_id="receipt_key_diagnostic",
                source_operation_fingerprint=_digest("key-diagnostic"),
                ambiguous_lineage=False,
            )
            operation_call = binding.publish_checkpoint(plan, _state())
        else:
            seed = BrowserProfileBinding(
                authority=binding.authority,
                store=store,
                key_authority=AESGCMBrowserProfileKeyAuthority(
                    authority_id="browser-profile-test-key",
                    key=_KEY,
                ),
                lease_seconds=120,
            )
            material, _ = await _settled_restore(seed, suffix="key_decrypt_seed")
            await _checkpoint(
                seed,
                material,
                operation_id="checkpoint_key_decrypt_seed",
            )
            await seed.release_writer(material)
            operation_call = binding.prepare_restore(
                operation_id="restore_key_diagnostic",
                execution_profile_fingerprint=_digest("key-diagnostic-execution"),
                allocation_fingerprint=_digest("key-diagnostic-allocation"),
                browser_session_id="browser_session_key_diagnostic",
            )

        with pytest.raises(BrowserProfileUnavailable) as exc_info:
            await operation_call

        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        return "\n".join(
            (
                str(exc_info.value),
                repr(exc_info.value),
                *(
                    repr(frame.f_locals)
                    for frame, _line_number in traceback.walk_tb(exc_info.value.__traceback__)
                    if is_cayu_source_filename(frame.f_code.co_filename)
                ),
            )
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = asyncio.run(scenario())
    captured = capsys.readouterr()
    diagnostic_surface = "\n".join(
        (
            rendered,
            caplog.text,
            captured.out,
            captured.err,
            *(str(item.message) for item in caught),
        )
    )
    assert "BROWSER_PROFILE_KEY_EXCEPTION_CANARY" not in diagnostic_surface


class _MutableKeyAuthority(BrowserProfileKeyAuthority):
    def __init__(self) -> None:
        self.current_authority_id = "browser-profile-test-key"
        self.encrypt_calls = 0

    @property
    def authority_id(self) -> str:
        return self.current_authority_id

    async def encrypt(self, *, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        del nonce, plaintext, aad
        self.encrypt_calls += 1
        raise AssertionError("changed key authority must not receive profile plaintext")

    async def decrypt(self, *, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        del nonce, ciphertext, aad
        raise AssertionError("decryption is not expected")


class _BlockingMutableKeyAuthority(BrowserProfileKeyAuthority):
    def __init__(self) -> None:
        self.current_authority_id = "browser-profile-test-key"
        self.encrypt_entered = asyncio.Event()
        self.decrypt_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.delegate = AESGCMBrowserProfileKeyAuthority(
            authority_id="browser-profile-test-key",
            key=_KEY,
        )

    @property
    def authority_id(self) -> str:
        return self.current_authority_id

    async def encrypt(self, *, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        self.encrypt_entered.set()
        await self.release.wait()
        return await self.delegate.encrypt(nonce=nonce, plaintext=plaintext, aad=aad)

    async def decrypt(self, *, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        self.decrypt_entered.set()
        await self.release.wait()
        return await self.delegate.decrypt(nonce=nonce, ciphertext=ciphertext, aad=aad)


def test_restore_revalidates_key_authority_before_acquiring_writer() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(Path("."), durable=False, clock=clock)
        key_authority = _MutableKeyAuthority()
        binding = BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-profile-tests",
                tenant_id="test-tenant",
                sharing_scope="agent-release-v1",
            ),
            destination_policy=BrowserProfileDestinationPolicy.build((_ORIGIN,)),
            browser_protocol="cayu.browser-session.v3",
            browser_worker_version="10",
            store=store,
            key_authority=key_authority,
            profile_id="bprof_mutable_restore_key",
            lease_seconds=120,
        )
        await binding.initialize()

        key_authority.current_authority_id = "replacement-browser-profile-key"
        with pytest.raises(BrowserProfileUnavailable, match="identity changed"):
            await _prepared_restore(binding, suffix="mutable_restore_key")
        assert (await store.inspect_profile(binding.access)).active_writer is False

    asyncio.run(scenario())


def test_checkpoint_revalidates_key_authority_before_exposing_plaintext() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(Path("."), durable=False, clock=clock)
        key_authority = _MutableKeyAuthority()
        binding = BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-profile-tests",
                tenant_id="test-tenant",
                sharing_scope="agent-release-v1",
            ),
            destination_policy=BrowserProfileDestinationPolicy.build((_ORIGIN,)),
            browser_protocol="cayu.browser-session.v3",
            browser_worker_version="10",
            store=store,
            key_authority=key_authority,
            profile_id="bprof_mutable_key",
            lease_seconds=120,
        )
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="mutable_key")
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_mutable_key",
            source_revision="revision_mutable_key",
            source_operation_receipt_id="receipt_mutable_key",
            source_operation_fingerprint=_digest("operation-mutable-key"),
            ambiguous_lineage=False,
        )

        key_authority.current_authority_id = "replacement-browser-profile-key"
        with pytest.raises(BrowserProfileUnavailable, match="identity changed"):
            await binding.publish_checkpoint(plan, _state())
        assert key_authority.encrypt_calls == 0

    asyncio.run(scenario())


def test_checkpoint_revalidates_key_authority_after_encryption_await() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(Path("."), durable=False, clock=clock)
        key_authority = _BlockingMutableKeyAuthority()
        binding = BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-profile-tests",
                tenant_id="test-tenant",
                sharing_scope="agent-release-v1",
            ),
            destination_policy=BrowserProfileDestinationPolicy.build((_ORIGIN, _SECOND_ORIGIN)),
            browser_protocol="cayu.browser-session.v3",
            browser_worker_version="10",
            store=store,
            key_authority=key_authority,
            profile_id="bprof_mutable_key_during_encrypt",
            lease_seconds=120,
        )
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="mutable_encrypt")
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_mutable_key_during_encrypt",
            source_revision="revision_mutable_key_during_encrypt",
            source_operation_receipt_id="receipt_mutable_key_during_encrypt",
            source_operation_fingerprint=_digest("operation-mutable-key-during-encrypt"),
            ambiguous_lineage=False,
        )
        publication = asyncio.create_task(binding.publish_checkpoint(plan, _state()))
        await key_authority.encrypt_entered.wait()

        key_authority.current_authority_id = "replacement-browser-profile-key"
        key_authority.release.set()
        with pytest.raises(BrowserProfileUnavailable, match="identity changed"):
            await publication

        assert (await store.current_ref(binding.access)).generation == 0
        failed = await binding.fail_checkpoint(
            plan,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="checkpoint_failed",
        )
        assert failed.outcome is BrowserProfileTerminalOutcome.FAILED

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_checkpoint_encryption_remains_owned_until_cancellation_settles(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        key_authority = _BlockingMutableKeyAuthority()
        binding = BrowserProfileBinding.build(
            scope=BrowserProfileScope.build(
                application_id="browser-profile-tests",
                tenant_id="test-tenant",
                sharing_scope="agent-release-v1",
            ),
            destination_policy=BrowserProfileDestinationPolicy.build((_ORIGIN,)),
            browser_protocol="cayu.browser-session.v3",
            browser_worker_version="10",
            store=store,
            key_authority=key_authority,
            profile_id="bprof_cancelled_encryption",
            lease_seconds=120,
        )
        await binding.initialize()
        material, _ = await _settled_restore(binding, suffix="cancelled_encryption")
        plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_cancelled_during_encryption",
            source_revision="revision_cancelled_during_encryption",
            source_operation_receipt_id="receipt_cancelled_during_encryption",
            source_operation_fingerprint=_digest("cancelled-during-encryption"),
            ambiguous_lineage=False,
        )
        owner = asyncio.create_task(binding.publish_checkpoint(plan, _state()))
        await key_authority.encrypt_entered.wait()

        owner.cancel("cancel browser-profile encryption")
        assert owner.cancelling() == 1
        assert not owner.done()
        key_authority.release.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel browser-profile encryption",
        ):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1

        assert (
            await store.load_checkpoint_receipt(
                binding.access,
                "checkpoint_cancelled_during_encryption",
            )
            is None
        )
        assert (await store.current_ref(binding.access)).generation == 0
        failed = await binding.fail_checkpoint(
            plan,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="checkpoint_failed",
        )
        assert failed.outcome is BrowserProfileTerminalOutcome.FAILED
        await binding.release_writer(material)
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


def test_restore_revalidates_key_authority_after_decryption_await() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(Path("."), durable=False, clock=clock)
        original = _binding(store)
        await original.initialize()
        material, _ = await _settled_restore(original, suffix="mutable_decrypt_seed")
        await _checkpoint(
            original,
            material,
            operation_id="checkpoint_mutable_decrypt_seed",
        )
        await original.release_writer(material)

        key_authority = _BlockingMutableKeyAuthority()
        binding = BrowserProfileBinding(
            authority=original.authority,
            store=store,
            key_authority=key_authority,
            lease_seconds=120,
        )
        restoration = asyncio.create_task(
            binding.prepare_restore(
                operation_id="restore_mutable_key_during_decrypt",
                execution_profile_fingerprint=_digest("execution-mutable-decrypt"),
                allocation_fingerprint=_digest("allocation-mutable-decrypt"),
                browser_session_id="browser_session_mutable_decrypt",
            )
        )
        await key_authority.decrypt_entered.wait()

        key_authority.current_authority_id = "replacement-browser-profile-key"
        key_authority.release.set()
        with pytest.raises(BrowserProfileUnavailable, match="cannot be decrypted"):
            await restoration

        receipt = await store.load_restore_receipt(
            binding.access,
            "restore_mutable_key_during_decrypt",
        )
        assert receipt is not None
        assert receipt.outcome is BrowserProfileTerminalOutcome.FAILED
        assert receipt.error_code == "profile_unavailable"
        assert (await store.inspect_profile(binding.access)).active_writer is False

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_restore_cancellation_settles_and_releases_the_exact_writer(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        original = _binding(store)
        await original.initialize()
        material, _ = await _settled_restore(original)
        await _checkpoint(original, material)
        await original.release_writer(material)

        blocking_key = _BlockingDecryptKeyAuthority()
        binding = BrowserProfileBinding(
            authority=original.authority,
            store=store,
            key_authority=blocking_key,
            lease_seconds=120,
        )
        owner = asyncio.create_task(
            binding.prepare_restore(
                operation_id="restore_cancelled",
                execution_profile_fingerprint=_digest("execution-cancelled"),
                allocation_fingerprint=_digest("allocation-cancelled"),
                browser_session_id="browser_session_cancelled",
            )
        )
        await blocking_key.entered.wait()
        owner.cancel()
        assert owner.cancelling() == 1
        assert not owner.done()
        blocking_key.release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1

        receipt = await store.load_restore_receipt(
            binding.access,
            "restore_cancelled",
        )
        assert receipt is not None
        assert receipt.outcome is BrowserProfileTerminalOutcome.FAILED
        assert receipt.error_code == "restore_cancelled_before_import"
        inspection = await store.inspect_profile(binding.access)
        assert not inspection.active_writer

        replacement = await _prepared_restore(original, suffix="after_cancel")
        assert replacement.preparation.profile_ref.generation == 1
        await original.complete_restore(
            replacement,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


def test_restore_cancellation_remains_authoritative_when_settlement_fails() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _FailedRestoreSettlementStore(clock=clock)
        original = _binding(store)
        await original.initialize()
        material, _ = await _settled_restore(original)
        await _checkpoint(original, material)
        await original.release_writer(material)

        blocking_key = _BlockingDecryptKeyAuthority()
        binding = BrowserProfileBinding(
            authority=original.authority,
            store=store,
            key_authority=blocking_key,
            lease_seconds=120,
        )
        owner = asyncio.create_task(
            binding.prepare_restore(
                operation_id="restore_cancelled_settlement_failed",
                execution_profile_fingerprint=_digest("execution-cancelled-settlement-failed"),
                allocation_fingerprint=_digest("allocation-cancelled-settlement-failed"),
                browser_session_id="browser_session_cancelled_settlement_failed",
            )
        )
        await blocking_key.entered.wait()
        store.fail_restore_settlement = True
        owner.cancel("browser-profile-owner-cancelled")
        assert owner.cancelling() == 1
        assert not owner.done()
        blocking_key.release.set()

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await owner

        assert owner.cancelled()
        assert owner.cancelling() == 1
        pending = [exc_info.value]
        seen: set[int] = set()
        cancellations: list[asyncio.CancelledError] = []
        rendered: list[str] = []
        while pending:
            failure = pending.pop()
            if id(failure) in seen:
                continue
            seen.add(id(failure))
            rendered.extend((str(failure), repr(failure)))
            if isinstance(failure, asyncio.CancelledError):
                cancellations.append(failure)
            if isinstance(failure, BaseExceptionGroup):
                pending.extend(failure.exceptions)
            if failure.__cause__ is not None:
                pending.append(failure.__cause__)
            if failure.__context__ is not None:
                pending.append(failure.__context__)
        assert cancellations == [exc_info.value]
        assert "BROWSER_PROFILE_SETTLEMENT_FAILURE_CANARY" not in "\n".join(rendered)
        assert isinstance(exc_info.value.__cause__, BrowserProfileUnavailable)
        assert (
            await store.load_restore_receipt(
                binding.access,
                "restore_cancelled_settlement_failed",
            )
            is None
        )
        assert (await store.inspect_profile(binding.access)).active_writer

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sqlite"])
def test_repeated_checkpoints_use_fresh_nonces_and_consecutive_generations(
    tmp_path: Path,
    durable: bool,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(tmp_path, durable=durable, clock=clock)
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        _first_plan, first = await _checkpoint(
            binding,
            material,
            operation_id="checkpoint_nonce_one",
        )
        second_plan = await binding.reserve_checkpoint(
            material=material,
            operation_id="checkpoint_nonce_two",
            source_revision="browser_revision_2",
            source_operation_receipt_id="browser_receipt_2",
            source_operation_fingerprint=_digest("browser-operation-2"),
            ambiguous_lineage=False,
        )
        second = await binding.publish_checkpoint(second_plan, _state())
        assert first.published_ref.generation == 1
        assert second.published_ref.generation == 2
        assert first.published_ref.content_fingerprint != second.published_ref.content_fingerprint
        assert (
            await store.load_checkpoint_receipt(
                binding.access,
                "checkpoint_nonce_one",
            )
            == first
        )
        assert await binding.reconcile_checkpoint("checkpoint_nonce_one") == first
        await binding.release_writer(material)
        if isinstance(store, SQLiteBrowserProfileStore):
            await store.close()

    asyncio.run(scenario())


def test_browser_profile_configured_limits_are_independent_and_exact() -> None:
    state = _state(canary="value")
    serialized_bytes = len(state.canonical_bytes())
    exact = BrowserProfileLimits(
        max_origins=1,
        max_cookies=1,
        max_storage_entries=1,
        max_name_bytes=len("session_state"),
        max_value_bytes=len("value"),
        max_plaintext_bytes=serialized_bytes,
        max_ciphertext_bytes=serialized_bytes + 16,
    )
    assert (
        validate_browser_profile_state(
            state,
            limits=exact,
            current_policy=BrowserProfileDestinationPolicy.build((_ORIGIN,)),
        )
        == state
    )

    cases = (
        exact.model_copy(update={"max_origins": 1}),
        exact.model_copy(update={"max_cookies": 1}),
        exact.model_copy(update={"max_storage_entries": 1}),
        exact.model_copy(update={"max_name_bytes": len("session_state") - 1}),
        exact.model_copy(update={"max_value_bytes": len("value") - 1}),
        exact.model_copy(
            update={
                "max_plaintext_bytes": serialized_bytes - 1,
                "max_ciphertext_bytes": serialized_bytes + 15,
            }
        ),
    )
    over_limit_states = (
        BrowserProfileStateV1(
            origins=tuple(
                sorted(
                    (
                        *state.origins,
                        BrowserProfileOriginStorage(
                            origin=_SECOND_ORIGIN,
                            local_storage=(),
                        ),
                    ),
                    key=lambda item: item.origin,
                )
            ),
            cookies=state.cookies,
        ),
        BrowserProfileStateV1(
            cookies=tuple(
                sorted(
                    (
                        *state.cookies,
                        BrowserProfileCookie(
                            name="other",
                            value="value",
                            domain="auth.browser.test",
                        ),
                    ),
                    key=lambda item: (item.domain, item.path, item.name),
                ),
            ),
            origins=state.origins,
        ),
        BrowserProfileStateV1(
            cookies=state.cookies,
            origins=(
                BrowserProfileOriginStorage(
                    origin=_ORIGIN,
                    local_storage=tuple(
                        sorted(
                            (
                                *state.origins[0].local_storage,
                                BrowserProfileStorageEntry(name="other", value="value"),
                            ),
                            key=lambda item: item.name,
                        )
                    ),
                ),
            ),
        ),
        state,
        state,
        state,
    )
    full_policy = BrowserProfileDestinationPolicy.build((_ORIGIN, _SECOND_ORIGIN))
    for limits, candidate in zip(cases, over_limit_states, strict=True):
        with pytest.raises(ValueError):
            validate_browser_profile_state(
                candidate,
                limits=BrowserProfileLimits.model_validate(limits),
                current_policy=full_policy,
            )


def test_sqlite_browser_profile_detects_authenticated_ciphertext_corruption(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        path = tmp_path / "browser-profiles-corrupt.sqlite"
        store = SQLiteBrowserProfileStore(
            path,
            store_id="browser-profile-test-store",
            clock=clock,
        )
        binding = _binding(store)
        await binding.initialize()
        material, _ = await _settled_restore(binding)
        await _checkpoint(binding, material)
        await binding.release_writer(material)
        await store.close()

        connection = sqlite3.connect(path)
        try:
            row = connection.execute(
                "SELECT record_json FROM cayu_browser_profiles WHERE profile_id = ?",
                (binding.authority.profile_id,),
            ).fetchone()
            assert row is not None
            document = json.loads(row[0])
            envelope = document["current_envelope"]
            nonce = base64.b64decode(envelope["nonce_base64"], validate=True)
            ciphertext = bytearray(base64.b64decode(envelope["ciphertext_base64"], validate=True))
            ciphertext[0] ^= 1
            envelope["ciphertext_base64"] = base64.b64encode(ciphertext).decode("ascii")
            envelope["content_fingerprint"] = hashlib.sha256(nonce + ciphertext).hexdigest()
            for operation in document["checkpoint_operations"].values():
                if operation.get("published") is True:
                    operation["envelope"] = dict(envelope)
                    operation["receipt"]["published_ref"]["content_fingerprint"] = envelope[
                        "content_fingerprint"
                    ]
            connection.execute(
                "UPDATE cayu_browser_profiles SET record_json = ? WHERE profile_id = ?",
                (
                    json.dumps(document, separators=(",", ":"), sort_keys=True).encode(),
                    binding.authority.profile_id,
                ),
            )
            connection.commit()
        finally:
            connection.close()

        reopened = SQLiteBrowserProfileStore(
            path,
            store_id="browser-profile-test-store",
            clock=clock,
        )
        corrupted = BrowserProfileBinding(
            authority=binding.authority,
            store=reopened,
            key_authority=binding.key_authority,
            lease_seconds=120,
        )
        with pytest.raises(BrowserProfileUnavailable, match="cannot be decrypted"):
            await _prepared_restore(corrupted, suffix="corrupt")
        inspection = await reopened.inspect_profile(corrupted.access)
        assert inspection.status is BrowserProfileStatus.CORRUPT
        assert inspection.safe_error_code == "profile_corrupt"
        assert "profile-secret-canary" not in json.dumps(
            inspection.model_dump(mode="json"),
            sort_keys=True,
        )
        await reopened.close()

    asyncio.run(scenario())
