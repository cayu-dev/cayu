from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
import threading
import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu import (
    InMemorySessionStore,
    LocalWorkspace,
    RunRequest,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
    SessionWorkspaceBranchStore,
    SQLiteSessionStore,
    WorkspaceBranchAuthority,
    WorkspaceBranchBindingAuthority,
    WorkspaceBranchBindingAuthorityClaim,
    WorkspaceBranchBindingAuthorityProvider,
    WorkspaceBranchBindingAuthorityRegistry,
    WorkspaceBranchChange,
    WorkspaceBranchClosedError,
    WorkspaceBranchDurableState,
    WorkspaceBranchFencedError,
    WorkspaceBranchLifecycleStatus,
    WorkspaceBranchLimits,
    WorkspaceBranchOperationConflict,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchRecoveryRequest,
    WorkspaceBranchRecoveryResult,
    WorkspaceBranchRequest,
    WorkspaceBranchResourceExhaustedError,
    WorkspaceBranchRollbackRequest,
    WorkspaceRevisionObservationLimits,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.sessions import _OwnedOffThreadSessionCommitGuard
from cayu.workspaces.branches import workspace_branch_change_set_digest
from cayu.workspaces.revisions import WorkspaceIdentity, observe_deterministic_workspace


def _replace_owned_commit_guard(
    commit_guard: Callable[[], None],
    callback: Callable[[], None],
) -> Callable[[], None]:
    if type(commit_guard) is _OwnedOffThreadSessionCommitGuard:
        return _OwnedOffThreadSessionCommitGuard(callback)
    return callback


class _LoseGuardedAcknowledgementStore(InMemorySessionStore):
    supports_owned_off_thread_session_commit_guards = True

    def __init__(self) -> None:
        super().__init__()
        self.lose_next_guarded_acknowledgement = False

    async def publish_session_operation_guarded(self, session_id: str, **kwargs):
        result = await super().publish_session_operation_guarded(session_id, **kwargs)
        if self.lose_next_guarded_acknowledgement:
            self.lose_next_guarded_acknowledgement = False
            raise ConnectionError("worker lost the publication acknowledgement")
        return result


class _FailPublicationBeforeCommitStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.successes_before_failure: int | None = None

    async def publish_session_operation(self, session_id: str, **kwargs):
        remaining = self.successes_before_failure
        if remaining is not None:
            if remaining == 0:
                self.successes_before_failure = None
                raise ConnectionError("worker died before durable operation commit")
            self.successes_before_failure = remaining - 1
        return await super().publish_session_operation(session_id, **kwargs)


class _BlockAfterTerminalCommitStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.block_terminal_commit = False
        self.fail_terminal_reconciliation = False
        self.terminal_committed = asyncio.Event()
        self.release_terminal_acknowledgement = asyncio.Event()

    async def publish_session_operation(self, session_id: str, **kwargs):
        result = await super().publish_session_operation(session_id, **kwargs)
        if self.block_terminal_commit:
            record = await super().load_session_operation(
                session_id,
                kwargs["idempotency_key"],
            )
            state = None if record is None else record["payload"]["state"]
            if state in {"committed", "rolled_back", "expired", "failed"}:
                self.block_terminal_commit = False
                self.terminal_committed.set()
                await self.release_terminal_acknowledgement.wait()
        return result

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        if self.fail_terminal_reconciliation and self.terminal_committed.is_set():
            raise ConnectionError("terminal reconciliation read unavailable")
        return await super().load_session_operation(session_id, idempotency_key, **kwargs)


class _CorruptLoadedBranchRecordStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.corrupt_branch_record = False

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await super().load_session_operation(session_id, idempotency_key, **kwargs)
        if self.corrupt_branch_record and record is not None:
            record["record_digest"] = "0" * 64
        return record


class _ConcurrentInitialLoadProxy:
    """Force two creators to observe the same absent durable record."""

    supports_owned_off_thread_session_commit_guards = True

    def __init__(self, store) -> None:
        self._store = store
        self._absent_loads = 0
        self._both_absent = asyncio.Event()

    def __getattr__(self, name):
        return getattr(self._store, name)

    def _supports_owned_off_thread_session_commit_guard_protocol(self) -> bool:
        checker = getattr(
            self._store,
            "_supports_owned_off_thread_session_commit_guard_protocol",
            None,
        )
        return callable(checker) and checker() is True

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await self._store.load_session_operation(session_id, idempotency_key, **kwargs)
        if record is None and self._absent_loads < 2:
            self._absent_loads += 1
            if self._absent_loads == 2:
                self._both_absent.set()
            await self._both_absent.wait()
        return record


class _RewriteRetainedEvidenceStore(_FailPublicationBeforeCommitStore):
    def __init__(self) -> None:
        super().__init__()
        self.rewrite_retained_evidence: str | None = None

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await super().load_session_operation(session_id, idempotency_key, **kwargs)
        if self.rewrite_retained_evidence is None or record is None:
            return record
        payload = record["payload"]
        if self.rewrite_retained_evidence in {
            "publication",
            "publication_created_as_modified",
            "publication_modified_as_created",
            "publication_forged_before",
        }:
            publication = payload["publication"]
            change_set = publication["change_set"]
            prior_digest = change_set["digest"]
            change = change_set["changes"][0]
            if self.rewrite_retained_evidence == "publication":
                change["path"] = "forged.txt"
            elif self.rewrite_retained_evidence == "publication_created_as_modified":
                change["operation"] = "modified"
                change["before"] = {"sha256": "0" * 64, "bytes": 1}
            elif self.rewrite_retained_evidence == "publication_modified_as_created":
                change["operation"] = "created"
                change["before"] = None
            else:
                change["before"] = {"sha256": "0" * 64, "bytes": 1}
            change_set["digest"] = workspace_branch_change_set_digest(
                branch_id=change_set["branch_id"],
                source=WorkspaceIdentity.model_validate(change_set["source"]),
                baseline_revision=change_set["baseline_revision"],
                changes=tuple(
                    WorkspaceBranchChange.model_validate(change) for change in change_set["changes"]
                ),
            )
            for attempt in payload["publication_attempts"]:
                if attempt["change_set_digest"] == prior_digest:
                    attempt["change_set_digest"] = change_set["digest"]
            publication["staging_owner"] = hashlib.sha256(
                (
                    f"{payload['private_root_owner']}\0"
                    f"{publication['idempotency_key']}\0{change_set['digest']}"
                ).encode()
            ).hexdigest()[:32]
        elif self.rewrite_retained_evidence == "rollback":
            payload["rollback"]["paths"] = ["forged.txt"]
        elif self.rewrite_retained_evidence == "failed_baseline_overflow":
            payload["baseline_revision"] = "x" * 2048
        elif self.rewrite_retained_evidence.startswith("publication_failure_"):
            outcome = self.rewrite_retained_evidence.removeprefix("publication_failure_")
            payload["failure"]["outcome"] = outcome
            payload["failure"]["conflicts"] = (
                [{"path": "answer.txt", "actual_kind": "file"}] if outcome == "conflicted" else []
            )
        else:  # pragma: no cover - test helper invariant
            raise AssertionError("Unknown retained-evidence rewrite mode.")
        record["record_digest"] = hashlib.sha256(
            canonical_durable_json_bytes(payload, "workspace_branch.record")
        ).hexdigest()
        return record


class _RewriteDurableExpiryStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.rewrite_mode: str | None = None

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await super().load_session_operation(session_id, idempotency_key, **kwargs)
        if self.rewrite_mode is None or record is None:
            return record
        payload = record["payload"]
        if self.rewrite_mode in {"premature_intent", "premature_terminal"}:
            payload["state"] = (
                "rollback_intent" if self.rewrite_mode == "premature_intent" else "expired"
            )
            payload["rollback"] = {
                "idempotency_key": "forged-expiry",
                "reason": "expired",
                "paths": sorted([*payload["overlay"], *payload["tombstones"]]),
            }
            payload["revision"] += 1
        elif self.rewrite_mode == "malformed":
            payload["expires_at"] = "not-a-timestamp"
        elif self.rewrite_mode == "timezone_naive":
            payload["expires_at"] = payload["expires_at"].removesuffix("+00:00")
        elif self.rewrite_mode == "lifetime_mismatch":
            payload["limits"]["lifetime_ms"] += 1
        else:  # pragma: no cover - test helper invariant
            raise AssertionError("Unknown durable expiry rewrite mode.")
        record["record_digest"] = hashlib.sha256(
            canonical_durable_json_bytes(payload, "workspace_branch.record")
        ).hexdigest()
        return record


class _FailReconciliationLoadStore(InMemorySessionStore):
    supports_owned_off_thread_session_commit_guards = True

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_load = False
        self.cancellation_secondary: BaseException | None = None

    async def publish_session_operation_guarded(self, session_id: str, **kwargs):
        try:
            return await super().publish_session_operation_guarded(session_id, **kwargs)
        except asyncio.CancelledError as cancellation:
            if self.cancellation_secondary is None:
                raise
            raise cancellation from self.cancellation_secondary

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        if self.fail_next_load:
            self.fail_next_load = False
            raise ConnectionError("reconciliation read failed")
        return await super().load_session_operation(session_id, idempotency_key, **kwargs)


class _RedirectLoadedPrivateRootStore(_FailPublicationBeforeCommitStore):
    def __init__(self) -> None:
        super().__init__()
        self.redirect_private_root: Path | None = None
        self.interrupt_after_next_publish = False

    async def publish_session_operation(self, session_id: str, **kwargs):
        result = await super().publish_session_operation(session_id, **kwargs)
        if self.interrupt_after_next_publish:
            self.interrupt_after_next_publish = False
            raise KeyboardInterrupt("worker stopped after durable creation intent")
        return result

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await super().load_session_operation(session_id, idempotency_key, **kwargs)
        if self.redirect_private_root is not None and record is not None:
            payload = record["payload"]
            payload["private_root"] = str(self.redirect_private_root)
            record["record_digest"] = hashlib.sha256(
                canonical_durable_json_bytes(payload, "workspace_branch.record")
            ).hexdigest()
        return record


class _DelayedCreatingGuardStore(_RedirectLoadedPrivateRootStore):
    """Hold two stale CREATING workers on independently released guards."""

    supports_owned_off_thread_session_commit_guards = True

    def __init__(self) -> None:
        super().__init__()
        self.guarded_calls = 0
        self.first_guard_entered = asyncio.Event()
        self.second_guard_entered = asyncio.Event()
        self.release_first_guard = asyncio.Event()
        self.release_second_guard = asyncio.Event()

    async def publish_session_operation_guarded(self, session_id: str, **kwargs):
        self.guarded_calls += 1
        if self.guarded_calls == 1:
            self.first_guard_entered.set()
            await self.release_first_guard.wait()
        elif self.guarded_calls == 2:
            self.second_guard_entered.set()
            await self.release_second_guard.wait()
        return await super().publish_session_operation_guarded(session_id, **kwargs)


class _MissingTerminalEvidenceStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.terminal_state: str | None = None
        self.private_root: Path | None = None

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await super().load_session_operation(session_id, idempotency_key, **kwargs)
        if self.terminal_state is not None and record is not None:
            payload = record["payload"]
            payload["state"] = self.terminal_state
            payload["publication"] = None
            payload["rollback"] = None
            self.private_root = Path(payload["private_root"])
            record["record_digest"] = hashlib.sha256(
                canonical_durable_json_bytes(payload, "workspace_branch.record")
            ).hexdigest()
        return record


class _ContradictoryTerminalEvidenceStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.inject_rollback = False

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await super().load_session_operation(session_id, idempotency_key, **kwargs)
        if self.inject_rollback and record is not None:
            payload = record["payload"]
            payload["rollback"] = {
                "idempotency_key": "contradictory-rollback",
                "reason": "explicit",
                "paths": [],
            }
            record["record_digest"] = hashlib.sha256(
                canonical_durable_json_bytes(payload, "workspace_branch.record")
            ).hexdigest()
        return record


class _RewritePublicationAuthorityStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.rewrite_publication_authority = False

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await super().load_session_operation(session_id, idempotency_key, **kwargs)
        if self.rewrite_publication_authority and record is not None:
            payload = record["payload"]
            publication = payload.get("publication")
            if publication is not None:
                change_set = publication["change_set"]
                change_set["baseline_revision"] = "sha256:" + "f" * 64
                change_set["digest"] = workspace_branch_change_set_digest(
                    branch_id=change_set["branch_id"],
                    source=WorkspaceIdentity.model_validate(change_set["source"]),
                    baseline_revision=change_set["baseline_revision"],
                    changes=tuple(
                        WorkspaceBranchChange.model_validate(change)
                        for change in change_set["changes"]
                    ),
                )
                record["record_digest"] = hashlib.sha256(
                    canonical_durable_json_bytes(payload, "workspace_branch.record")
                ).hexdigest()
        return record


class _DriftGuardedCurrentRecordStore(InMemorySessionStore):
    supports_owned_off_thread_session_commit_guards = True

    def __init__(self) -> None:
        super().__init__()
        self.drift_next_guarded_record = False

    async def publish_session_operation_guarded(self, session_id: str, **kwargs):
        operation_transform = kwargs["operation_transform"]

        def transform_with_drift(session, checkpoint, current_record):
            if self.drift_next_guarded_record and current_record is not None:
                self.drift_next_guarded_record = False
                current_record = deepcopy(current_record)
                payload = current_record["payload"]
                payload["limits"]["max_files"] += 1
                current_record["record_digest"] = hashlib.sha256(
                    canonical_durable_json_bytes(payload, "workspace_branch.record")
                ).hexdigest()
            return operation_transform(session, checkpoint, current_record)

        return await super().publish_session_operation_guarded(
            session_id,
            **{
                **kwargs,
                "operation_transform": transform_with_drift,
            },
        )


class _CrashInsideGuardStore(InMemorySessionStore):
    supports_owned_off_thread_session_commit_guards = True

    def __init__(self) -> None:
        super().__init__()
        self.crash_inside_next_guard = False

    async def publish_session_operation_guarded(self, session_id: str, **kwargs):
        if not self.crash_inside_next_guard:
            return await super().publish_session_operation_guarded(session_id, **kwargs)
        self.crash_inside_next_guard = False
        commit_guard = kwargs["commit_guard"]

        def mutate_then_crash() -> None:
            commit_guard()
            raise ConnectionError("worker died inside the guarded commit boundary")

        return await super().publish_session_operation_guarded(
            session_id,
            **{
                **kwargs,
                "commit_guard": _replace_owned_commit_guard(
                    commit_guard,
                    mutate_then_crash,
                ),
            },
        )


class _StoreFaultProxy:
    """Inject storage-boundary failures without changing the backend under test."""

    supports_owned_off_thread_session_commit_guards = True

    def __init__(self, store) -> None:
        self._store = store
        self.fail_next_publish_before = False
        self.fail_next_guarded_before = False
        self.fail_next_guarded_after_guard = False
        self.lose_next_guarded_acknowledgement = False
        self.corrupt_branch_record = False
        self.publish_calls = 0
        self.fail_publish_before_call: int | None = None
        self.lose_publish_acknowledgement_call: int | None = None
        self.interrupt_after_publish_call: int | None = None
        self.block_next_publish = False
        self.publish_blocked = asyncio.Event()
        self.release_publish = asyncio.Event()

    def __getattr__(self, name):
        return getattr(self._store, name)

    def _supports_owned_off_thread_session_commit_guard_protocol(self) -> bool:
        checker = getattr(
            self._store,
            "_supports_owned_off_thread_session_commit_guard_protocol",
            None,
        )
        return callable(checker) and checker() is True

    async def publish_session_operation(self, session_id: str, **kwargs):
        self.publish_calls += 1
        if self.block_next_publish:
            self.block_next_publish = False
            self.publish_blocked.set()
            await self.release_publish.wait()
        if self.fail_next_publish_before or self.publish_calls == self.fail_publish_before_call:
            self.fail_next_publish_before = False
            raise ConnectionError("worker died before durable operation commit")
        result = await self._store.publish_session_operation(session_id, **kwargs)
        if self.publish_calls == self.interrupt_after_publish_call:
            raise KeyboardInterrupt("worker stopped after durable creation intent")
        if self.publish_calls == self.lose_publish_acknowledgement_call:
            raise ConnectionError("worker lost the durable operation acknowledgement")
        return result

    async def publish_session_operation_guarded(self, session_id: str, **kwargs):
        if self.fail_next_guarded_before:
            self.fail_next_guarded_before = False
            raise ConnectionError("worker died before guarded operation commit")
        if self.fail_next_guarded_after_guard:
            self.fail_next_guarded_after_guard = False
            commit_guard = kwargs["commit_guard"]

            def mutate_then_fail() -> None:
                commit_guard()
                raise ConnectionError("worker died after guarded mutation")

            return await self._store.publish_session_operation_guarded(
                session_id,
                **{
                    **kwargs,
                    "commit_guard": _replace_owned_commit_guard(
                        commit_guard,
                        mutate_then_fail,
                    ),
                },
            )
        result = await self._store.publish_session_operation_guarded(session_id, **kwargs)
        if self.lose_next_guarded_acknowledgement:
            self.lose_next_guarded_acknowledgement = False
            raise ConnectionError("worker lost the guarded operation acknowledgement")
        return result

    async def load_session_operation(self, session_id: str, idempotency_key: str, **kwargs):
        record = await self._store.load_session_operation(session_id, idempotency_key, **kwargs)
        if self.corrupt_branch_record and record is not None:
            record["record_digest"] = "0" * 64
        return record


async def _create_session(store, *, session_id: str = "branch-session") -> None:
    await store.create(
        RunRequest(agent_name="durable-branch-tests", session_id=session_id, messages=[]),
        identity=SessionIdentity(provider_name="tests", model="fake"),
    )


def _binding_authority(
    *,
    generation: str = "binding-1",
    identity: str = "workspace-alpha@binding-1",
) -> WorkspaceBranchBindingAuthority:
    return WorkspaceBranchBindingAuthority(
        environment_name="local",
        binding_generation=generation,
        binding_identity=identity,
    )


class _NonReentrantAuthorityProvider(WorkspaceBranchBindingAuthorityRegistry):
    """Test provider that rejects every callback while one claim is active."""

    def __init__(self, authority: WorkspaceBranchBindingAuthority) -> None:
        super().__init__(authority)
        self._contract_lock = threading.Lock()
        self._claim_active = False
        self.claim_count = 0

    def __call__(self) -> WorkspaceBranchBindingAuthority:
        with self._contract_lock:
            if self._claim_active:
                raise AssertionError("authority provider was re-entered")
        return super().__call__()

    def claim(
        self,
        expected: WorkspaceBranchBindingAuthority,
    ):
        with self._contract_lock:
            if self._claim_active:
                raise AssertionError("authority provider claim was nested")
            self._claim_active = True
            self.claim_count += 1
        try:
            delegate = super().claim(expected)
        except BaseException:
            with self._contract_lock:
                self._claim_active = False
            raise
        provider = self

        class _Claim:
            def release(self) -> None:
                delegate.release()
                with provider._contract_lock:
                    provider._claim_active = False

        return _Claim()


class _FailOnceReleaseAuthorityProvider(WorkspaceBranchBindingAuthorityRegistry):
    """Test provider whose claim remains active across one release failure."""

    def __init__(self, authority: WorkspaceBranchBindingAuthority) -> None:
        super().__init__(authority)
        self._failure_lock = threading.Lock()
        self._fail_next_release = False

    def arm_release_failure(self) -> None:
        with self._failure_lock:
            self._fail_next_release = True

    def claim(
        self,
        expected: WorkspaceBranchBindingAuthority,
    ) -> WorkspaceBranchBindingAuthorityClaim:
        delegate = super().claim(expected)
        provider = self

        class _Claim:
            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._released = False

            def release(self) -> None:
                with self._lock:
                    if self._released:
                        return
                    with provider._failure_lock:
                        if provider._fail_next_release:
                            provider._fail_next_release = False
                            raise ConnectionError("binding claim release failed")
                    delegate.release()
                    self._released = True

        return _Claim()


def _workspace(root: Path, store, *, resolver=None) -> LocalWorkspace:
    if resolver is None:
        authority_provider = WorkspaceBranchBindingAuthorityRegistry(_binding_authority())
    elif isinstance(resolver, WorkspaceBranchBindingAuthorityProvider):
        authority_provider = resolver
    else:
        authority_provider = WorkspaceBranchBindingAuthorityRegistry(resolver())
    return LocalWorkspace(
        root,
        workspace_id="workspace-alpha",
        branch_store=SessionWorkspaceBranchStore(store),
        branch_authority_resolver=authority_provider,
    )


async def _durable_branch(
    root: Path,
    store,
    *,
    branch_id: str = "branch-alpha",
    create_key: str = "create-alpha",
    limits: WorkspaceBranchLimits | None = None,
    session_id: str = "branch-session",
    binding_generation: str = "binding-1",
    binding_identity: str = "workspace-alpha@binding-1",
    resolver=None,
):
    source = _workspace(root, store, resolver=resolver)
    baseline = await observe_deterministic_workspace(
        source,
        observer="durable-branch-tests",
        limits=WorkspaceRevisionObservationLimits(),
    )
    request = WorkspaceBranchRequest(
        baseline=baseline,
        limits=limits or WorkspaceBranchLimits(),
        branch_id=branch_id,
        idempotency_key=create_key,
        authority=WorkspaceBranchAuthority(
            session_id=session_id,
            expected_run_epoch=0,
            environment_name="local",
            binding_generation=binding_generation,
            binding_identity=binding_identity,
            creating_authority="fork-group:alpha",
            resource_policy="local-cow-defaults-v1",
        ),
    )
    created = await source.create_branch(request)
    assert created.status is WorkspaceBranchOutcomeStatus.CREATED
    assert created.branch is not None
    return source, created.branch, request


async def _interrupted_durable_creating_branch(
    root: Path,
    store: _RedirectLoadedPrivateRootStore,
    *,
    resolver=None,
) -> tuple[LocalWorkspace, WorkspaceBranchRequest]:
    source = _workspace(root, store, resolver=resolver)
    baseline = await observe_deterministic_workspace(
        source,
        observer="durable-branch-tests",
        limits=WorkspaceRevisionObservationLimits(),
    )
    request = WorkspaceBranchRequest(
        baseline=baseline,
        branch_id="branch-alpha",
        idempotency_key="create-alpha",
        authority=WorkspaceBranchAuthority(
            session_id="branch-session",
            expected_run_epoch=0,
            environment_name="local",
            binding_generation="binding-1",
            binding_identity="workspace-alpha@binding-1",
            creating_authority="fork-group:alpha",
            resource_policy="local-cow-defaults-v1",
        ),
    )
    store.interrupt_after_next_publish = True
    with pytest.raises(KeyboardInterrupt, match="durable creation intent"):
        await source.create_branch(request)
    return source, request


def test_recovery_completes_clean_pre_capture_creating_record(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _RedirectLoadedPrivateRootStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        _source, _request = await _interrupted_durable_creating_branch(root, store)
        assert tuple(root.parent.glob(".cayu-workspace-branch-*")) == ()

        recovered = await _workspace(root, store).recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.OPEN
        assert recovered.branch is not None
        assert (await recovered.branch.read_bytes("answer.txt")).content == b"baseline"
        await recovered.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=recovered.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )

    asyncio.run(scenario())


def test_pre_capture_recovery_records_a_known_baseline_conflict(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _RedirectLoadedPrivateRootStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        _source, _request = await _interrupted_durable_creating_branch(root, store)
        (root / "answer.txt").write_bytes(b"changed before recovery")

        recovered = await _workspace(root, store).recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.FAILED
        assert recovered.evidence.outcome is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert recovered.evidence.detail_code == "workspace_branch_baseline_conflicted"
        assert (root / "answer.txt").read_bytes() == b"changed before recovery"
        assert tuple(root.parent.glob(".cayu-workspace-branch-*")) == ()

    asyncio.run(scenario())


def test_exact_durable_attachments_share_one_active_branch_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )

        repeated = await source.create_branch(request)
        recovered = await _workspace(root, store).recover_branch(_recovery_request())

        assert repeated.status is WorkspaceBranchOutcomeStatus.CREATED
        assert repeated.branch is not None
        assert recovered.state is WorkspaceBranchDurableState.OPEN
        assert recovered.branch is not None
        assert len(branch_module._ACTIVE_BRANCHES[source.resource_key]) == 1

        await branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )
        assert source.resource_key not in branch_module._ACTIVE_BRANCHES
        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


def test_exact_durable_attachments_share_terminal_lifecycle_during_failed_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        repeated = await source.create_branch(request)
        assert repeated.branch is not None
        sibling = repeated.branch
        await branch.write_bytes("answer.txt", b"candidate")
        publication_request = await _publication_request(branch)
        private_root = branch._private_root
        remove_durable_private_tree = branch_module._remove_durable_private_tree

        def fail_terminal_cleanup(*_args, **_kwargs) -> None:
            raise PermissionError("terminal private cleanup unavailable")

        monkeypatch.setattr(
            branch_module,
            "_remove_durable_private_tree",
            fail_terminal_cleanup,
        )
        committed = await branch.publish(publication_request)

        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert private_root.exists()
        assert sibling.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED
        with pytest.raises(WorkspaceBranchClosedError, match="committed"):
            await sibling.read_bytes("answer.txt")
        with pytest.raises(WorkspaceBranchClosedError, match="committed"):
            await sibling.list()
        with pytest.raises(WorkspaceBranchClosedError, match="committed"):
            await sibling.changes()

        monkeypatch.setattr(
            branch_module,
            "_remove_durable_private_tree",
            remove_durable_private_tree,
        )
        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


def test_terminal_recovery_retires_surviving_process_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        local_active_branches = branch_module._ACTIVE_BRANCHES

        branch_module._ACTIVE_BRANCHES = {}
        try:
            remote = await _workspace(root, store).create_branch(request)
            assert remote.branch is not None
            await remote.branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=remote.branch.branch_id,
                    idempotency_key="rollback-alpha",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                )
            )
        finally:
            branch_module._ACTIVE_BRANCHES = local_active_branches

        assert source.resource_key in branch_module._ACTIVE_BRANCHES
        recovered = await source.recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.ROLLED_BACK
        assert source.resource_key not in branch_module._ACTIVE_BRANCHES
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ROLLED_BACK
        with pytest.raises(WorkspaceBranchClosedError, match="rolled_back"):
            await branch.changes()
        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


def test_uncertain_private_mutation_fences_every_exact_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = _CrashInsideGuardStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        repeated = await source.create_branch(request)
        assert repeated.branch is not None
        sibling = repeated.branch

        store.crash_inside_next_guard = True
        with pytest.raises(ConnectionError, match="guarded commit boundary"):
            await branch.write_bytes("uncertain.txt", b"private mutation")

        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        assert sibling.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        with pytest.raises(WorkspaceBranchFencedError, match="fenced"):
            await sibling.write_bytes("later.txt", b"must not land")
        with pytest.raises(WorkspaceBranchFencedError, match="fenced"):
            await sibling.changes()

        recovered = await source.recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.AMBIGUOUS
        assert recovered.branch is None
        assert not (root / "uncertain.txt").exists()
        assert not (root / "later.txt").exists()

    asyncio.run(scenario())


def test_remote_ambiguity_fences_surviving_process_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = _CrashInsideGuardStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        local_active_branches = branch_module._ACTIVE_BRANCHES

        branch_module._ACTIVE_BRANCHES = {}
        try:
            remote_source = _workspace(root, store)
            remote = await remote_source.recover_branch(_recovery_request())
            assert remote.branch is not None
            store.crash_inside_next_guard = True
            with pytest.raises(ConnectionError, match="guarded commit boundary"):
                await remote.branch.write_bytes("uncertain.txt", b"private mutation")
            remote_recovery = await remote_source.recover_branch(_recovery_request())
            assert remote_recovery.state is WorkspaceBranchDurableState.AMBIGUOUS
        finally:
            branch_module._ACTIVE_BRANCHES = local_active_branches

        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
        recovered = await source.recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.AMBIGUOUS
        assert recovered.branch is None
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        with pytest.raises(WorkspaceBranchFencedError, match="fenced"):
            await branch.read_bytes("uncertain.txt")
        with pytest.raises(WorkspaceBranchFencedError, match="fenced"):
            await branch.list()
        with pytest.raises(WorkspaceBranchFencedError, match="fenced"):
            await branch.changes()

    asyncio.run(scenario())


def _recovery_request(
    *,
    recovery_id: str = "recover-alpha",
    branch_id: str = "branch-alpha",
    session_id: str = "branch-session",
    binding_generation: str = "binding-1",
    binding_identity: str = "workspace-alpha@binding-1",
) -> WorkspaceBranchRecoveryRequest:
    return WorkspaceBranchRecoveryRequest(
        branch_id=branch_id,
        session_id=session_id,
        expected_run_epoch=0,
        binding_generation=binding_generation,
        binding_identity=binding_identity,
        recovery_id=recovery_id,
    )


def _publication_request(
    branch,
    *,
    key: str = "publish-alpha",
    binding_generation: str = "binding-1",
):
    async def build():
        changes = await branch.changes()
        return WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
            idempotency_key=key,
            expected_run_epoch=0,
            binding_generation=binding_generation,
        )

    return build()


def _isolate_retained_branch_state(monkeypatch, branch_module) -> None:
    monkeypatch.setattr(
        branch_module,
        "_SOURCE_STAGING_CLEANUPS",
        type(branch_module._SOURCE_STAGING_CLEANUPS)(),
    )
    monkeypatch.setattr(
        branch_module,
        "_PRIVATE_TREE_CLEANUPS",
        type(branch_module._PRIVATE_TREE_CLEANUPS)(),
    )
    monkeypatch.setattr(
        branch_module,
        "_DURABLE_TERMINAL_SETTLEMENTS",
        type(branch_module._DURABLE_TERMINAL_SETTLEMENTS)(),
    )
    monkeypatch.setattr(
        branch_module,
        "_DURABLE_RECOVERY_CLEANUPS",
        type(branch_module._DURABLE_RECOVERY_CLEANUPS)(),
    )
    monkeypatch.setattr(
        branch_module,
        "_BINDING_CLAIM_RELEASES",
        type(branch_module._BINDING_CLAIM_RELEASES)(),
    )
    monkeypatch.setattr(branch_module, "_ACTIVE_BRANCHES", {})


async def _assert_terminal_handle_released_capacity(root: Path, store) -> None:
    _successor_source, successor, _successor_request = await _durable_branch(
        root,
        store,
        branch_id="branch-successor",
        create_key="create-successor",
        limits=WorkspaceBranchLimits(max_active_branches=1),
    )
    await successor.rollback(
        WorkspaceBranchRollbackRequest(
            branch_id=successor.branch_id,
            idempotency_key="rollback-successor",
            expected_run_epoch=0,
            binding_generation="binding-1",
        )
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_durable_branch_never_reenters_binding_authority_provider(
    tmp_path: Path,
    store_kind: str,
) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "child-cancellation.sqlite3")
        )
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        provider = _NonReentrantAuthorityProvider(_binding_authority())
        source = _workspace(root, store, resolver=provider)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )
        created = await source.create_branch(request)
        assert created.status is WorkspaceBranchOutcomeStatus.CREATED
        assert created.branch is not None
        await created.branch.write_bytes("answer.txt", b"candidate")
        await created.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=created.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )
        assert provider.claim_count == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_fresh_owner_recovers_open_branch_and_private_changes(
    tmp_path: Path,
    store_kind: str,
) -> None:
    async def scenario() -> None:
        database = tmp_path / "branches.sqlite3"
        store = InMemorySessionStore() if store_kind == "memory" else SQLiteSessionStore(database)
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "kept.txt").write_bytes(b"before")
        (root / "deleted.txt").write_bytes(b"delete")
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("kept.txt", b"after")
        await branch.write_bytes("created.txt", b"created")
        await branch.delete("deleted.txt")

        reopened_store = store if store_kind == "memory" else SQLiteSessionStore(database)
        fresh = _workspace(root, reopened_store)
        recovered = await fresh.recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.OPEN
        assert recovered.branch is not None
        assert await recovered.branch.read_bytes("kept.txt") == await branch.read_bytes("kept.txt")
        assert (await recovered.branch.read_bytes("created.txt")).content == b"created"
        with pytest.raises(FileNotFoundError):
            await recovered.branch.read_bytes("deleted.txt")
        assert (await recovered.branch.changes()).digest == (await branch.changes()).digest

    asyncio.run(scenario())


def test_publication_intent_survives_lost_store_acknowledgement_and_recovers(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = _LoseGuardedAcknowledgementStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"old")
        provider = _NonReentrantAuthorityProvider(_binding_authority())
        _source, branch, _request = await _durable_branch(
            root,
            store,
            resolver=provider,
        )
        await branch.write_bytes("answer.txt", b"new")
        publication = await _publication_request(branch)

        store.lose_next_guarded_acknowledgement = True
        with pytest.raises(ConnectionError, match="acknowledgement"):
            await branch.publish(publication)
        assert (root / "answer.txt").read_bytes() == b"new"

        fresh = _workspace(root, store, resolver=provider)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.COMMITTED
        assert recovered.publication is not None
        assert recovered.publication.status is WorkspaceBranchOutcomeStatus.COMMITTED

        repeated = await fresh.recover_branch(
            _recovery_request(recovery_id="recover-alpha-repeated")
        )
        assert repeated.publication == recovered.publication

    asyncio.run(scenario())


def test_creation_reconciles_lost_guarded_store_acknowledgement(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _LoseGuardedAcknowledgementStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        source = _workspace(root, store)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )
        store.lose_next_guarded_acknowledgement = True

        created = await source.create_branch(request)

        assert created.status is WorkspaceBranchOutcomeStatus.CREATED
        assert created.branch is not None
        assert (await created.branch.read_bytes("answer.txt")).content == b"baseline"

    asyncio.run(scenario())


def test_creation_guard_is_off_loop_and_preserves_cancellation_when_reconciliation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _FailReconciliationLoadStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        source = _workspace(root, store)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )
        capture_started = threading.Event()
        release_capture = threading.Event()
        copy_regular_tree = branch_module._copy_regular_tree

        def block_during_capture(*args, **kwargs):
            capture_started.set()
            if not release_capture.wait(timeout=5):
                raise TimeoutError("test did not release baseline capture")
            return copy_regular_tree(*args, **kwargs)

        monkeypatch.setattr(branch_module, "_copy_regular_tree", block_during_capture)
        creation = asyncio.create_task(source.create_branch(request))
        try:
            assert await asyncio.to_thread(capture_started.wait, 5)
            await asyncio.sleep(0)
            creation.cancel()
            assert creation.cancelling() == 1
            store.cancellation_secondary = ConnectionError(
                "guarded publication acknowledgement failed"
            )
            store.fail_next_load = True
        finally:
            release_capture.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await creation
        assert creation.cancelled()
        assert isinstance(raised.value.__cause__, BaseExceptionGroup)
        assert [str(error) for error in raised.value.__cause__.exceptions] == [
            "guarded publication acknowledgement failed",
            "reconciliation read failed",
        ]

        monkeypatch.setattr(branch_module, "_copy_regular_tree", copy_regular_tree)
        created = await source.create_branch(request)
        assert created.status is WorkspaceBranchOutcomeStatus.CREATED
        assert created.branch is not None
        await created.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=created.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )

    asyncio.run(scenario())


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "fork") or not hasattr(signal, "SIGKILL"),
    reason="real process-loss capture recovery requires POSIX fork and SIGKILL",
)
@pytest.mark.parametrize("crash_boundary", ["before_owner", "during_copy"])
def test_creation_retries_after_process_loss_during_staged_capture(
    tmp_path: Path,
    crash_boundary: str,
) -> None:
    database = tmp_path / "branches.sqlite3"
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "answer.txt").write_bytes(b"baseline")

    async def prepare() -> WorkspaceBranchRequest:
        store = SQLiteSessionStore(database)
        await _create_session(store)
        source = _workspace(root, store)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        return WorkspaceBranchRequest(
            baseline=baseline,
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )

    request = asyncio.run(prepare())
    capture_started = tmp_path / "capture-started"
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions run in the parent
        try:
            import cayu.workspaces._local_branch as branch_module

            def block_before_owner(_path, _owner):
                capture_started.write_text("ready")
                while True:
                    signal.pause()

            def block_after_partial_capture(
                _source_root,
                baseline_root,
                _limits,
                *,
                source_semantics,
            ):
                del source_semantics
                (baseline_root / "partial.txt").write_bytes(b"partial")
                capture_started.write_text("ready")
                while True:
                    signal.pause()

            if crash_boundary == "before_owner":
                branch_module._write_private_root_owner = block_before_owner
            else:
                branch_module._copy_regular_tree = block_after_partial_capture
            child_store = SQLiteSessionStore(database)
            asyncio.run(_workspace(root, child_store).create_branch(request))
        finally:
            os._exit(90)

    deadline = time.monotonic() + 10
    while not capture_started.exists():
        exited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if exited_pid:
            pytest.fail(
                f"capture worker exited before the process-loss boundary: "
                f"{os.waitstatus_to_exitcode(status)}"
            )
        if time.monotonic() >= deadline:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
            pytest.fail("capture worker did not reach the process-loss boundary")
        time.sleep(0.02)

    os.kill(child_pid, signal.SIGKILL)
    _reaped_pid, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL

    private_paths = tuple(root.parent.glob(".cayu-workspace-branch-*"))
    assert len(private_paths) == 1
    assert all(".capture-" in path.name for path in private_paths)

    async def retry() -> None:
        reopened = SQLiteSessionStore(database)
        created = await _workspace(root, reopened).create_branch(request)
        assert created.status is WorkspaceBranchOutcomeStatus.CREATED
        assert created.branch is not None
        assert (await created.branch.read_bytes("answer.txt")).content == b"baseline"
        assert not tuple(root.parent.glob("*.capture-*"))
        await created.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=created.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )

    asyncio.run(retry())


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "fork") or not hasattr(signal, "SIGKILL"),
    reason="real pre-capture process-loss recovery requires POSIX fork and SIGKILL",
)
def test_recovery_repeats_capture_after_process_loss_before_guard_dispatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "branches.sqlite3"
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "answer.txt").write_bytes(b"baseline")

    async def prepare() -> WorkspaceBranchRequest:
        store = SQLiteSessionStore(database)
        await _create_session(store)
        source = _workspace(root, store)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        return WorkspaceBranchRequest(
            baseline=baseline,
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )

    request = asyncio.run(prepare())
    creating_committed = tmp_path / "creating-committed"
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions run in the parent
        try:

            class StopAfterCreatingStore(SQLiteSessionStore):
                async def publish_session_operation(self, session_id: str, **kwargs):
                    await super().publish_session_operation(session_id, **kwargs)
                    creating_committed.write_text("ready")
                    while True:
                        signal.pause()

            child_store = StopAfterCreatingStore(database)
            asyncio.run(_workspace(root, child_store).create_branch(request))
        finally:
            os._exit(90)

    deadline = time.monotonic() + 10
    while not creating_committed.exists():
        exited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if exited_pid:
            pytest.fail(
                "creation worker exited before the durable CREATING boundary: "
                f"{os.waitstatus_to_exitcode(status)}"
            )
        if time.monotonic() >= deadline:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
            pytest.fail("creation worker did not reach the durable CREATING boundary")
        time.sleep(0.02)

    assert tuple(root.parent.glob(".cayu-workspace-branch-*")) == ()
    os.kill(child_pid, signal.SIGKILL)
    _reaped_pid, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL

    async def recover() -> None:
        store = SQLiteSessionStore(database)
        recovered = await _workspace(root, store).recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.OPEN
        assert recovered.branch is not None
        assert (await recovered.branch.read_bytes("answer.txt")).content == b"baseline"
        await recovered.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=recovered.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )

    asyncio.run(recover())


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "fork") or not hasattr(signal, "SIGKILL"),
    reason="real process-loss cleanup recovery requires POSIX fork and SIGKILL",
)
def test_terminal_cleanup_retries_after_process_loss_without_owner_marker(
    tmp_path: Path,
) -> None:
    database = tmp_path / "branches.sqlite3"
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "answer.txt").write_bytes(b"baseline")

    async def prepare() -> None:
        store = SQLiteSessionStore(database)
        await _create_session(store)
        await _durable_branch(root, store)

    asyncio.run(prepare())
    cleanup_started = tmp_path / "cleanup-started"
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions run in the parent
        try:
            import cayu.workspaces._local_branch as branch_module

            unlink = branch_module.os.unlink

            def block_after_owner_unlink(path, *args, **kwargs):
                unlink(path, *args, **kwargs)
                if path == branch_module._PRIVATE_ROOT_OWNER_FILE:
                    cleanup_started.write_text("ready")
                    while True:
                        signal.pause()

            branch_module.os.unlink = block_after_owner_unlink

            async def rollback() -> None:
                child_store = SQLiteSessionStore(database)
                recovered = await _workspace(root, child_store).recover_branch(_recovery_request())
                assert recovered.branch is not None
                await recovered.branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=recovered.branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )

            asyncio.run(rollback())
        finally:
            os._exit(90)

    deadline = time.monotonic() + 10
    while not cleanup_started.exists():
        exited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if exited_pid:
            pytest.fail(
                f"cleanup worker exited before the process-loss boundary: "
                f"{os.waitstatus_to_exitcode(status)}"
            )
        if time.monotonic() >= deadline:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
            pytest.fail("cleanup worker did not reach the process-loss boundary")
        time.sleep(0.02)

    os.kill(child_pid, signal.SIGKILL)
    _reaped_pid, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
    cleanup_paths = tuple(root.parent.glob("*.cleanup-*"))
    assert len(cleanup_paths) == 1
    assert list(cleanup_paths[0].iterdir()) == []

    async def retry() -> None:
        reopened = SQLiteSessionStore(database)
        recovered = await _workspace(root, reopened).recover_branch(
            _recovery_request(recovery_id="recover-after-cleanup-process-loss")
        )
        assert recovered.state is WorkspaceBranchDurableState.ROLLED_BACK
        assert not tuple(root.parent.glob("*.cleanup-*"))

    asyncio.run(retry())


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "fork") or not hasattr(signal, "SIGKILL"),
    reason="real process-loss cleanup ownership recovery requires POSIX fork and SIGKILL",
)
def test_terminal_cleanup_process_loss_restores_raced_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    _isolate_retained_branch_state(monkeypatch, branch_module)
    database = tmp_path / "branches.sqlite3"
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "answer.txt").write_bytes(b"baseline")

    async def prepare() -> Path:
        store = SQLiteSessionStore(database)
        await _create_session(store)
        _source, branch, _request = await _durable_branch(root, store)
        return branch._private_root

    private_root = asyncio.run(prepare())
    displaced = private_root.with_name(f"{private_root.name}.original")
    cleanup_started = tmp_path / "replacement-quarantined"
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions run in the parent
        try:
            import cayu.workspaces._local_branch as child_branch_module

            original_rename = child_branch_module._rename_private_root_no_replace
            replaced = False

            def replace_then_block(source, target, *, parent_fd):
                nonlocal replaced
                if source == private_root and not replaced:
                    replaced = True
                    os.rename(private_root, displaced)
                    private_root.mkdir()
                    original_rename(source, target, parent_fd=parent_fd)
                    cleanup_started.write_text("ready")
                    while True:
                        signal.pause()
                return original_rename(source, target, parent_fd=parent_fd)

            child_branch_module._rename_private_root_no_replace = replace_then_block

            async def rollback() -> None:
                store = SQLiteSessionStore(database)
                recovered = await _workspace(root, store).recover_branch(_recovery_request())
                assert recovered.branch is not None
                await recovered.branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=recovered.branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )

            asyncio.run(rollback())
        finally:
            os._exit(90)

    deadline = time.monotonic() + 10
    while not cleanup_started.exists():
        exited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if exited_pid:
            pytest.fail(
                "cleanup worker exited before replacement quarantine: "
                f"{os.waitstatus_to_exitcode(status)}"
            )
        if time.monotonic() >= deadline:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
            pytest.fail("cleanup worker did not quarantine the replacement")
        time.sleep(0.02)

    os.kill(child_pid, signal.SIGKILL)
    _reaped_pid, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
    cleanup_paths = tuple(root.parent.glob(f"{private_root.name}.cleanup-*"))
    assert len(cleanup_paths) == 1
    replacement_identity = cleanup_paths[0].stat()
    assert list(cleanup_paths[0].iterdir()) == []

    async def retry() -> None:
        reopened = SQLiteSessionStore(database)
        with pytest.raises(WorkspaceBranchFencedError, match="ownership changed"):
            await _workspace(root, reopened).recover_branch(
                _recovery_request(recovery_id="recover-raced-replacement")
            )

    asyncio.run(retry())
    restored_identity = private_root.stat()
    assert os.path.samestat(replacement_identity, restored_identity)
    assert list(private_root.iterdir()) == []
    assert displaced.is_dir()
    assert not cleanup_paths[0].exists()


@pytest.mark.parametrize("overlap_boundary", ["rename", "staging_waiter"])
def test_concurrent_terminal_recovery_converges_during_private_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlap_boundary: str,
) -> None:
    async def prepare() -> tuple[Path, Path, WorkspaceBranchRecoveryRequest]:
        import cayu.workspaces._local_branch as branch_module

        database = tmp_path / "branches.sqlite3"
        store = SQLiteSessionStore(database)
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _isolate_retained_branch_state(monkeypatch, branch_module)
        (root / "answer.txt").write_bytes(b"baseline")
        _source, branch, _request = await _durable_branch(root, store)
        private_root = branch._private_root
        cleanup_registry_type = type(branch_module._PRIVATE_TREE_CLEANUPS)

        def fail_terminal_cleanup(path, owner):
            if path == private_root:
                raise PermissionError("terminal private cleanup deferred")
            return original_remove_owned_private_tree(path, owner)

        original_remove_owned_private_tree = branch_module._remove_owned_private_tree
        with monkeypatch.context() as fault:
            fault.setattr(branch_module, "_remove_owned_private_tree", fail_terminal_cleanup)
            fault.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)
            with pytest.raises(PermissionError, match="cleanup deferred"):
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )

        assert private_root.exists()
        monkeypatch.setattr(branch_module, "_PRIVATE_TREE_CLEANUPS", cleanup_registry_type())
        monkeypatch.setattr(branch_module, "_ACTIVE_BRANCHES", {})
        return database, private_root, _recovery_request()

    database, private_root, recovery_request = asyncio.run(prepare())

    import cayu.workspaces._local_branch as branch_module

    if overlap_boundary == "rename":
        original_rename = branch_module._rename_private_root_no_replace
        rename_barrier = threading.Barrier(2)

        def overlap_final_rename(source, target, *, parent_fd):
            if source == private_root:
                rename_barrier.wait(timeout=5)
            return original_rename(source, target, parent_fd=parent_fd)

        monkeypatch.setattr(
            branch_module,
            "_rename_private_root_no_replace",
            overlap_final_rename,
        )
    else:
        original_lock = branch_module._lock_cleanup_claim_staging
        original_remove_claim = branch_module._remove_cleanup_quarantine_claim
        lock_calls = 0
        lock_calls_guard = threading.Lock()
        second_waiting = threading.Event()
        cleanup_finished = threading.Event()

        def order_staging_waiters(descriptor: int) -> None:
            nonlocal lock_calls
            with lock_calls_guard:
                lock_calls += 1
                call = lock_calls
            if call == 2:
                second_waiting.set()
            original_lock(descriptor)
            if call == 1:
                if not second_waiting.wait(timeout=5):
                    raise TimeoutError("second cleanup did not wait on staging ownership")
            elif call == 2 and not cleanup_finished.wait(timeout=5):
                raise TimeoutError("winning cleanup did not settle before the waiter resumed")

        def signal_cleanup_finished(*args, **kwargs) -> None:
            original_remove_claim(*args, **kwargs)
            cleanup_finished.set()

        monkeypatch.setattr(
            branch_module,
            "_lock_cleanup_claim_staging",
            order_staging_waiters,
        )
        monkeypatch.setattr(
            branch_module,
            "_remove_cleanup_quarantine_claim",
            signal_cleanup_finished,
        )

    def recover() -> WorkspaceBranchRecoveryResult:
        store = SQLiteSessionStore(database)
        return asyncio.run(
            _workspace(tmp_path / "workspace", store).recover_branch(recovery_request)
        )

    async def recover_concurrently() -> tuple[WorkspaceBranchRecoveryResult, ...]:
        return tuple(
            await asyncio.gather(
                asyncio.to_thread(recover),
                asyncio.to_thread(recover),
            )
        )

    first, second = asyncio.run(recover_concurrently())
    assert first.state is WorkspaceBranchDurableState.ROLLED_BACK
    assert second.state is WorkspaceBranchDurableState.ROLLED_BACK
    assert not private_root.exists()
    assert not tuple(private_root.parent.glob(f"{private_root.name}.cleanup-*"))
    if overlap_boundary == "staging_waiter":
        assert lock_calls == 2


def test_durable_rollback_is_terminal_before_private_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"old")
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"discarded")

        rolled_back = await branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )
        assert rolled_back.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
        assert (
            await branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=branch.branch_id,
                    idempotency_key="rollback-alpha",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                )
            )
        ) == rolled_back
        with pytest.raises(WorkspaceBranchOperationConflict):
            await branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=branch.branch_id,
                    idempotency_key="rollback-other",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                )
            )
        assert (root / "answer.txt").read_bytes() == b"old"

        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.ROLLED_BACK
        assert recovered.rollback == rolled_back

    asyncio.run(scenario())


def test_durable_operations_reject_stale_and_conflicting_authority(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"new")
        publication = await _publication_request(branch)

        with pytest.raises(SessionRunFenced):
            await branch.publish(publication.model_copy(update={"expected_run_epoch": 1}))
        committed = await branch.publish(publication)
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        with pytest.raises(WorkspaceBranchOperationConflict):
            await branch.publish(publication.model_copy(update={"idempotency_key": "other"}))

    asyncio.run(scenario())


def test_durable_create_requires_an_atomic_branch_store(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        with pytest.raises(TypeError, match="WorkspaceBranchStore"):
            LocalWorkspace(
                root,
                workspace_id="workspace-alpha",
                branch_store=store,  # type: ignore[arg-type]
            )
        _source, _branch, request = await _durable_branch(root, store)
        without_store = LocalWorkspace(root, workspace_id="workspace-alpha")
        with pytest.raises(RuntimeError, match="branch_store"):
            await without_store.create_branch(
                request.model_copy(update={"branch_id": "branch-beta"})
            )
        without_live_authority = LocalWorkspace(
            root,
            workspace_id="workspace-alpha",
            branch_store=SessionWorkspaceBranchStore(store),
        )
        with pytest.raises(RuntimeError, match="branch_authority_resolver"):
            await without_live_authority.create_branch(
                request.model_copy(update={"branch_id": "branch-gamma"})
            )

    asyncio.run(scenario())


def test_session_workspace_branch_store_rejects_missing_owned_guard_capability() -> None:
    class LegacySessionStore:
        def __init__(self) -> None:
            self.calls = 0

        async def load_session_operation(self, *_args, **_kwargs):
            self.calls += 1

        async def publish_session_operation(self, *_args, **_kwargs):
            self.calls += 1

        async def publish_session_operation_guarded(self, *_args, **_kwargs):
            self.calls += 1

    store = LegacySessionStore()

    with pytest.raises(TypeError, match="owned off-thread session commit guards"):
        SessionWorkspaceBranchStore(store)  # type: ignore[arg-type]

    assert store.calls == 0

    class UnsupportedInMemoryStore(InMemorySessionStore):
        supports_owned_off_thread_session_commit_guards = False

    with pytest.raises(TypeError, match="owned off-thread session commit guards"):
        SessionWorkspaceBranchStore(runtime_checkpoint_session_store(UnsupportedInMemoryStore()))

    class UnsafeGuardOverride(InMemorySessionStore):
        async def publish_session_operation_guarded(self, session_id: str, **kwargs):
            return await super().publish_session_operation_guarded(session_id, **kwargs)

    unsafe = UnsafeGuardOverride()
    with pytest.raises(TypeError, match="owned off-thread session commit guards"):
        SessionWorkspaceBranchStore(unsafe)
    with pytest.raises(TypeError, match="owned off-thread session commit guards"):
        SessionWorkspaceBranchStore(runtime_checkpoint_session_store(unsafe))

    class ConformingGuardOverride(InMemorySessionStore):
        supports_owned_off_thread_session_commit_guards = True

        async def publish_session_operation_guarded(self, session_id: str, **kwargs):
            return await super().publish_session_operation_guarded(session_id, **kwargs)

    SessionWorkspaceBranchStore(ConformingGuardOverride())
    SessionWorkspaceBranchStore(runtime_checkpoint_session_store(InMemorySessionStore()))


def test_session_workspace_branch_store_is_a_public_runtime_adapter() -> None:
    import cayu
    import cayu.runtime as runtime

    assert cayu.SessionWorkspaceBranchStore is runtime.SessionWorkspaceBranchStore
    assert "SessionWorkspaceBranchStore" in cayu.__all__
    assert "SessionWorkspaceBranchStore" in runtime.__all__


def test_creation_failure_is_durable_and_exactly_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root, store)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )

        def fail_capture(*_args, **_kwargs):
            raise OSError("durable private storage failed")

        monkeypatch.setattr(branch_module, "_capture_baseline_at_private_root", fail_capture)
        failed = await source.create_branch(request)
        assert failed.status is WorkspaceBranchOutcomeStatus.FAILED
        assert failed.evidence.detail_code == "durable_workspace_branch_creation_failed"
        assert await source.create_branch(request) == failed

        recovered = await _workspace(root, store).recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.FAILED
        assert recovered.evidence.outcome is WorkspaceBranchOutcomeStatus.FAILED
        assert recovered.evidence.detail_code == "durable_workspace_branch_creation_failed"

    asyncio.run(scenario())


def test_delayed_creating_worker_converges_after_open_branch_mutation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = _DelayedCreatingGuardStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, request = await _interrupted_durable_creating_branch(root, store)

        first = asyncio.create_task(source.create_branch(request))
        await asyncio.wait_for(store.first_guard_entered.wait(), timeout=5)
        second = asyncio.create_task(source.create_branch(request))
        await asyncio.wait_for(store.second_guard_entered.wait(), timeout=5)
        try:
            store.release_first_guard.set()
            opened = await asyncio.wait_for(first, timeout=5)
            assert opened.branch is not None
            await opened.branch.write_bytes("candidate.txt", b"candidate")
            expected_changes = await opened.branch.changes()

            store.release_second_guard.set()
            replayed = await asyncio.wait_for(second, timeout=5)

            assert replayed.status is WorkspaceBranchOutcomeStatus.CREATED
            assert replayed.branch is not None
            assert await replayed.branch.changes() == expected_changes
            assert (await replayed.branch.read_bytes("candidate.txt")).content == b"candidate"
            await opened.branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=opened.branch.branch_id,
                    idempotency_key="rollback-alpha",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                )
            )
        finally:
            store.release_first_guard.set()
            store.release_second_guard.set()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(scenario())


def test_delayed_creating_worker_replays_durable_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _DelayedCreatingGuardStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, request = await _interrupted_durable_creating_branch(root, store)

        def fail_capture(*_args, **_kwargs):
            raise OSError("durable private storage failed")

        monkeypatch.setattr(branch_module, "_capture_baseline_at_private_root", fail_capture)
        first = asyncio.create_task(source.create_branch(request))
        await asyncio.wait_for(store.first_guard_entered.wait(), timeout=5)
        second = asyncio.create_task(source.create_branch(request))
        await asyncio.wait_for(store.second_guard_entered.wait(), timeout=5)
        try:
            store.release_first_guard.set()
            failed = await asyncio.wait_for(first, timeout=5)
            assert failed.status is WorkspaceBranchOutcomeStatus.FAILED

            store.release_second_guard.set()
            replayed = await asyncio.wait_for(second, timeout=5)

            assert replayed == failed
            assert replayed.evidence.detail_code == "durable_workspace_branch_creation_failed"
            assert tuple(root.parent.glob(".cayu-workspace-branch-*")) == ()
        finally:
            store.release_first_guard.set()
            store.release_second_guard.set()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(scenario())


def test_failed_creation_recovery_settles_staging_after_cleanup_owner_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root, store)
        (root / "answer.txt").write_bytes(b"baseline")
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )
        (root / "answer.txt").write_bytes(b"changed before capture")
        cleanup_registry_type = type(branch_module._PRIVATE_TREE_CLEANUPS)
        monkeypatch.setattr(branch_module, "_PRIVATE_TREE_CLEANUPS", cleanup_registry_type())
        monkeypatch.setattr(branch_module, "_ACTIVE_BRANCHES", {})
        remove_owned_private_tree = branch_module._remove_owned_private_tree

        def fail_staging_cleanup(path, owner):
            if ".capture-" in path.name and path.exists():
                raise PermissionError("capture cleanup remained in flight")
            return remove_owned_private_tree(path, owner)

        with monkeypatch.context() as fault:
            fault.setattr(branch_module, "_remove_owned_private_tree", fail_staging_cleanup)
            fault.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)
            failed = await source.create_branch(request)

        assert failed.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert len(tuple(root.parent.glob("*.capture-*"))) == 1

        # Process replacement loses the in-memory cleanup owner and active-slot
        # registry, but the durable capture attempt remains sufficient to clean it.
        monkeypatch.setattr(branch_module, "_PRIVATE_TREE_CLEANUPS", cleanup_registry_type())
        monkeypatch.setattr(branch_module, "_ACTIVE_BRANCHES", {})
        recovered = await _workspace(root, store).recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.FAILED
        assert not tuple(root.parent.glob("*.capture-*"))

    asyncio.run(scenario())


def test_failed_recovery_validates_result_before_removing_retained_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = _RewriteRetainedEvidenceStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root, store)
        (root / "answer.txt").write_bytes(b"baseline")
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            limits=WorkspaceBranchLimits(max_evidence_bytes=1024),
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )
        (root / "answer.txt").write_bytes(b"changed before capture")
        remove_owned_private_tree = branch_module._remove_owned_private_tree

        def retain_capture_staging(path, owner):
            if ".capture-" in path.name and path.exists():
                raise PermissionError("retain failed creation evidence")
            return remove_owned_private_tree(path, owner)

        with monkeypatch.context() as fault:
            fault.setattr(branch_module, "_remove_owned_private_tree", retain_capture_staging)
            fault.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)
            failed = await source.create_branch(request)

        assert failed.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        retained = tuple(root.parent.glob("*.capture-*"))
        assert len(retained) == 1

        store.rewrite_retained_evidence = "failed_baseline_overflow"
        with pytest.raises(
            WorkspaceBranchResourceExhaustedError,
            match="result_evidence_limit_exceeded",
        ):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert retained[0].is_dir()

    asyncio.run(scenario())


def test_durable_create_retry_is_exact_and_conflicting_identity_is_rejected(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, request = await _durable_branch(root, store)

        repeated = await source.create_branch(request)
        assert repeated.status is WorkspaceBranchOutcomeStatus.CREATED
        assert repeated.branch is not None
        assert repeated.branch.branch_id == branch.branch_id
        with pytest.raises(WorkspaceBranchOperationConflict):
            await source.create_branch(
                request.model_copy(update={"idempotency_key": "different-create"})
            )

    asyncio.run(scenario())


def test_stale_owner_cannot_publish_or_discard_newer_private_work(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, stale, _request = await _durable_branch(root, store)
        recovered = await _workspace(root, store).recover_branch(_recovery_request())
        assert recovered.branch is not None
        current = recovered.branch

        await current.write_bytes("new.txt", b"new private work")
        stale_publication = await _publication_request(stale)
        stale_rollback = WorkspaceBranchRollbackRequest(
            branch_id=stale.branch_id,
            idempotency_key="stale-rollback",
            expected_run_epoch=0,
            binding_generation="binding-1",
        )

        with pytest.raises(WorkspaceBranchOperationConflict, match="changed under"):
            await stale.publish(stale_publication)
        with pytest.raises(WorkspaceBranchOperationConflict, match="changed under"):
            await stale.rollback(stale_rollback)
        assert (await current.read_bytes("new.txt")).content == b"new private work"
        assert not (root / "new.txt").exists()

        committed = await current.publish(await _publication_request(current))
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (root / "new.txt").read_bytes() == b"new private work"

    asyncio.run(scenario())


def test_private_storage_identity_includes_creating_authority(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store, session_id="branch-session-a")
        await _create_session(store, session_id="branch-session-b")
        root = tmp_path / "workspace"
        root.mkdir()
        binding_a = _binding_authority(
            generation="binding-a",
            identity="workspace-alpha@binding-a",
        )
        binding_b = _binding_authority(
            generation="binding-b",
            identity="workspace-alpha@binding-b",
        )
        _source_a, branch_a, _request_a = await _durable_branch(
            root,
            store,
            session_id="branch-session-a",
            binding_generation="binding-a",
            binding_identity="workspace-alpha@binding-a",
            resolver=lambda: binding_a,
        )
        _source_b, branch_b, _request_b = await _durable_branch(
            root,
            store,
            session_id="branch-session-b",
            binding_generation="binding-b",
            binding_identity="workspace-alpha@binding-b",
            resolver=lambda: binding_b,
        )

        await branch_a.write_bytes("a.txt", b"a")
        await branch_b.write_bytes("b.txt", b"b")
        await branch_a.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=branch_a.branch_id,
                idempotency_key="rollback-a",
                expected_run_epoch=0,
                binding_generation="binding-a",
            )
        )

        assert (await branch_b.read_bytes("b.txt")).content == b"b"
        assert not (root / "a.txt").exists()
        assert not (root / "b.txt").exists()

    asyncio.run(scenario())


def test_live_binding_authority_fences_stale_handle_before_mutation(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        live = WorkspaceBranchBindingAuthorityRegistry(_binding_authority())
        _source, branch, _request = await _durable_branch(
            root,
            store,
            resolver=live,
        )
        live.replace(
            _binding_authority(
                generation="binding-2",
                identity="workspace-alpha@binding-2",
            )
        )

        with pytest.raises(WorkspaceBranchOperationConflict, match="no longer current"):
            await branch.write_bytes("stale.txt", b"must not land")
        with pytest.raises(WorkspaceBranchOperationConflict, match="no longer current"):
            await _workspace(root, store, resolver=live).recover_branch(_recovery_request())
        assert not (root / "stale.txt").exists()

    asyncio.run(scenario())


def test_binding_replacement_is_rejected_until_publication_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        authority_provider = WorkspaceBranchBindingAuthorityRegistry(_binding_authority())
        _source, branch, _request = await _durable_branch(
            root,
            store,
            resolver=authority_provider,
        )
        await branch.write_bytes("a.txt", b"a")
        await branch.write_bytes("b.txt", b"b")
        publication = await _publication_request(branch)
        first_mutation_finished = threading.Event()
        release_publication = threading.Event()
        original_create_regular = branch_module.create_regular

        def block_after_first_source_mutation(source_root, path, content, **kwargs):
            original_create_regular(source_root, path, content, **kwargs)
            if source_root == root and path == "a.txt":
                first_mutation_finished.set()
                if not release_publication.wait(timeout=5):
                    raise TimeoutError("test did not release durable publication")

        monkeypatch.setattr(branch_module, "create_regular", block_after_first_source_mutation)
        publication_task = asyncio.create_task(branch.publish(publication))
        assert await asyncio.to_thread(first_mutation_finished.wait, 5)

        with pytest.raises(WorkspaceBranchOperationConflict, match="active generation claim"):
            authority_provider.replace(
                _binding_authority(
                    generation="binding-2",
                    identity="workspace-alpha@binding-2",
                )
            )
        assert not (root / "b.txt").exists()

        release_publication.set()
        result = await publication_task
        assert result.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (root / "a.txt").read_bytes() == b"a"
        assert (root / "b.txt").read_bytes() == b"b"

        authority_provider.replace(
            _binding_authority(
                generation="binding-2",
                identity="workspace-alpha@binding-2",
            )
        )

    asyncio.run(scenario())


def test_publication_conflict_does_not_overwrite_external_source_change(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"candidate")
        publication = await _publication_request(branch)
        (root / "answer.txt").write_bytes(b"external")

        result = await branch.publish(publication)
        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert (root / "answer.txt").read_bytes() == b"external"

        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.CONFLICTED
        assert recovered.branch is not None
        (root / "answer.txt").write_bytes(b"baseline")
        retried = await recovered.branch.publish(publication)
        assert retried.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (root / "answer.txt").read_bytes() == b"candidate"

    asyncio.run(scenario())


def test_conflicted_publication_key_stays_bound_to_its_first_change_set(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"candidate")
        first = await _publication_request(branch, key="stable-key")
        (root / "answer.txt").write_bytes(b"external")
        conflicted = await branch.publish(first)
        assert conflicted.status is WorkspaceBranchOutcomeStatus.CONFLICTED

        await branch.write_bytes("extra.txt", b"new material")
        changed = await _publication_request(branch, key="stable-key")
        with pytest.raises(WorkspaceBranchOperationConflict, match="reused"):
            await branch.publish(changed)
        assert (root / "answer.txt").read_bytes() == b"external"
        assert not (root / "extra.txt").exists()

    asyncio.run(scenario())


def test_pre_mutation_publication_failure_is_durable_and_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=2),
        )
        await branch.write_bytes("answer.txt", b"candidate")
        attached = await _workspace(root, store).recover_branch(_recovery_request())
        assert attached.branch is not None
        publication = await _publication_request(branch, key="failed-key")
        original_inspection = branch_module._inspect_source_path

        def fail_inspection(*_args, **_kwargs):
            raise OSError("source inspection unavailable")

        monkeypatch.setattr(branch_module, "_inspect_source_path", fail_inspection)
        failed = await branch.publish(publication)
        monkeypatch.setattr(branch_module, "_inspect_source_path", original_inspection)
        assert failed.status is WorkspaceBranchOutcomeStatus.FAILED
        assert failed.evidence.detail_code == "source_publication_inspection_failed"
        assert not (root / "answer.txt").exists()
        assert await branch.publish(publication) == failed
        assert await attached.branch.publish(publication) == failed
        assert attached.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED

        await _assert_terminal_handle_released_capacity(root, store)

        recovered = await _workspace(root, store).recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.FAILED
        assert recovered.evidence.outcome is WorkspaceBranchOutcomeStatus.FAILED
        assert recovered.evidence.change_set_digest == publication.change_set_digest
        assert recovered.evidence.detail_code == "source_publication_inspection_failed"

    asyncio.run(scenario())


def test_inspection_failure_after_started_publication_is_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        proxy = _StoreFaultProxy(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, proxy)
        await branch.write_bytes("a.txt", b"a")
        await branch.write_bytes("b.txt", b"b")
        publication = await _publication_request(branch, key="partial-key")
        proxy.fail_next_guarded_before = True
        with pytest.raises(ConnectionError, match="before guarded"):
            await branch.publish(publication)
        (root / "a.txt").write_bytes(b"a")

        original_inspection = branch_module._inspect_source_path

        def fail_first_path(source_root, path, **kwargs):
            if path == "a.txt":
                raise OSError("partial path cannot be inspected")
            return original_inspection(source_root, path, **kwargs)

        monkeypatch.setattr(branch_module, "_inspect_source_path", fail_first_path)
        recovered = await _workspace(root, store).recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.AMBIGUOUS
        assert (root / "a.txt").read_bytes() == b"a"
        assert not (root / "b.txt").exists()
        assert tuple(root.parent.glob(".cayu-workspace-branch-*"))

    asyncio.run(scenario())


def test_new_run_epoch_fences_private_branch_mutation_before_filesystem_change(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await store.transition_status(
            "branch-session",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )

        with pytest.raises(SessionRunFenced):
            await branch.write_bytes("stale.txt", b"must-not-land")
        assert not (root / "stale.txt").exists()
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED

    asyncio.run(scenario())


def test_concurrent_exact_publication_attempts_converge_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, first, _request = await _durable_branch(root, store)
        await first.write_bytes("answer.txt", b"one answer")
        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.branch is not None
        publication = await _publication_request(first)

        first_result, second_result = await asyncio.gather(
            first.publish(publication),
            recovered.branch.publish(publication),
        )
        assert first_result == second_result
        assert first_result.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (root / "answer.txt").read_bytes() == b"one answer"
        assert first.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED
        assert recovered.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED

        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


def test_exact_committed_replay_retires_recovered_handle(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, first, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=2),
        )
        await first.write_bytes("answer.txt", b"one answer")
        recovered = await _workspace(root, store).recover_branch(_recovery_request())
        assert recovered.branch is not None
        second = recovered.branch
        publication = await _publication_request(first)

        committed = await first.publish(publication)
        assert await second.publish(publication) == committed
        assert second.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED

        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


def test_exact_rollback_replay_retires_recovered_handle(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, first, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=2),
        )
        await first.write_bytes("discarded.txt", b"private")
        recovered = await _workspace(root, store).recover_branch(_recovery_request())
        assert recovered.branch is not None
        second = recovered.branch
        rollback = WorkspaceBranchRollbackRequest(
            branch_id=first.branch_id,
            idempotency_key="rollback-alpha",
            expected_run_epoch=0,
            binding_generation="binding-1",
        )

        rolled_back = await first.rollback(rollback)
        assert await second.rollback(rollback) == rolled_back
        assert second.lifecycle_status is WorkspaceBranchLifecycleStatus.ROLLED_BACK

        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


def test_concurrent_commit_and_rollback_choose_one_terminal_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"candidate")
        publication = await _publication_request(branch)
        rollback = WorkspaceBranchRollbackRequest(
            branch_id=branch.branch_id,
            idempotency_key="rollback-alpha",
            expected_run_epoch=0,
            binding_generation="binding-1",
        )

        outcomes = await asyncio.gather(
            branch.publish(publication),
            branch.rollback(rollback),
            return_exceptions=True,
        )
        terminals = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
        assert len(terminals) == 1

        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state in {
            WorkspaceBranchDurableState.COMMITTED,
            WorkspaceBranchDurableState.ROLLED_BACK,
        }
        if recovered.state is WorkspaceBranchDurableState.COMMITTED:
            assert (root / "answer.txt").read_bytes() == b"candidate"
        else:
            assert not (root / "answer.txt").exists()

    asyncio.run(scenario())


def test_concurrent_commit_retires_losing_rollback_handle(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        proxy = _StoreFaultProxy(store)
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, publication_owner, _request = await _durable_branch(
            root,
            proxy,
            limits=WorkspaceBranchLimits(max_active_branches=2),
        )
        await publication_owner.write_bytes("answer.txt", b"candidate")
        recovered = await _workspace(root, proxy).recover_branch(_recovery_request())
        assert recovered.branch is not None
        rollback_owner = recovered.branch
        publication = await _publication_request(publication_owner)
        rollback = WorkspaceBranchRollbackRequest(
            branch_id=rollback_owner.branch_id,
            idempotency_key="rollback-alpha",
            expected_run_epoch=0,
            binding_generation="binding-1",
        )

        proxy.block_next_publish = True
        rollback_task = asyncio.create_task(rollback_owner.rollback(rollback))
        await proxy.publish_blocked.wait()
        committed = await publication_owner.publish(publication)
        proxy.release_publish.set()
        rollback_outcome = (await asyncio.gather(rollback_task, return_exceptions=True))[0]
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert isinstance(rollback_outcome, WorkspaceBranchOperationConflict)
        settled = await _workspace(root, store).recover_branch(_recovery_request())
        assert settled.state is WorkspaceBranchDurableState.COMMITTED
        assert publication_owner.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED
        assert rollback_owner.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED
        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


def test_recovery_finishes_partially_applied_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("a.txt", b"a")
        await branch.write_bytes("b.txt", b"b")
        publication = await _publication_request(branch)
        original_create = branch_module.create_regular
        applied = 0

        def create_then_lose_process(*args, **kwargs):
            nonlocal applied
            original_create(*args, **kwargs)
            applied += 1
            if applied == 1:
                raise ConnectionError("worker died after first source mutation")

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module, "create_regular", create_then_lose_process)
            with pytest.raises(ConnectionError, match="first source mutation"):
                await branch.publish(publication)
        assert (root / "a.txt").read_bytes() == b"a"
        assert not (root / "b.txt").exists()

        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.COMMITTED
        assert (root / "a.txt").read_bytes() == b"a"
        assert (root / "b.txt").read_bytes() == b"b"

    asyncio.run(scenario())


def test_recovery_settles_persisted_source_staging_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module
    import cayu.workspaces._local_guard as guard_module
    import cayu.workspaces._mutations as mutation_module

    async def scenario() -> None:
        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"candidate")
        publication = await _publication_request(branch)
        cleanup_failure = PermissionError("source staging cleanup did not settle")

        with monkeypatch.context() as fault:
            fault.setattr(
                guard_module,
                "_unlink_staging_and_inspect",
                lambda _parent_fd, _temp_name: (False, (cleanup_failure,)),
            )
            fault.setattr(branch_module, "_schedule_source_staging_cleanup", lambda _error: None)
            with pytest.raises(guard_module._LocalGuardStagingCleanupError):
                await branch.publish(publication)

        assert (root / "answer.txt").read_bytes() == b"candidate"
        staging_paths = tuple(root.glob(".answer.txt.cayu-*"))
        assert len(staging_paths) == 1

        # Simulate process replacement: process-local cleanup ownership and the
        # permanent local source fence disappear, while the durable publication
        # record remains the sole recovery authority.
        monkeypatch.setattr(
            branch_module,
            "_SOURCE_STAGING_CLEANUPS",
            type(branch_module._SOURCE_STAGING_CLEANUPS)(),
        )
        with mutation_module._LOCAL_SOURCE_CONDITION:
            mutation_module._FENCED_LOCAL_SOURCES.discard(
                mutation_module._local_source_identity(root)
            )

        recovered = await _workspace(root, store).recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.COMMITTED
        assert (root / "answer.txt").read_bytes() == b"candidate"
        assert not tuple(root.glob(".answer.txt.cayu-*"))

    asyncio.run(scenario())


def test_publication_cancellation_retains_simultaneous_source_staging_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module
    import cayu.workspaces._local_guard as guard_module
    import cayu.workspaces._mutations as mutation_module

    async def scenario() -> None:
        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        source, branch, _request = await _durable_branch(tmp_path, store)
        await branch.write_bytes("answer.txt", b"candidate")
        publication = await _publication_request(branch)
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()

        def fail_staging_cleanup(_parent_fd, _temp_name):
            cleanup_started.set()
            if not release_cleanup.wait(timeout=5):
                raise TimeoutError("test did not release source staging cleanup")
            return False, (PermissionError("source staging cleanup did not settle"),)

        with monkeypatch.context() as fault:
            fault.setattr(
                guard_module,
                "_unlink_staging_and_inspect",
                fail_staging_cleanup,
            )
            fault.setattr(branch_module, "_schedule_source_staging_cleanup", lambda _error: None)
            publishing = asyncio.create_task(branch.publish(publication))
            try:
                assert await asyncio.to_thread(cleanup_started.wait, 5)
                publishing.cancel("publication owner cancelled")
                assert publishing.cancelling() == 1
            finally:
                release_cleanup.set()

            with pytest.raises(asyncio.CancelledError) as raised:
                await publishing

        assert publishing.cancelled()
        assert publishing.cancelling() == 1
        assert raised.value.args == ("publication owner cancelled",)
        assert isinstance(raised.value.__cause__, guard_module._LocalGuardStagingCleanupError)
        assert mutation_module.local_workspace_source_is_fenced(tmp_path)
        assert tuple(tmp_path.glob(".answer.txt.cayu-*"))
        retained = branch_module._SOURCE_STAGING_CLEANUPS.items()
        assert len(retained) == 1
        assert retained[0][1].payload.cleanup_owned

        with mutation_module._LOCAL_SOURCE_CONDITION:
            mutation_module._FENCED_LOCAL_SOURCES.discard(
                mutation_module._local_source_identity(tmp_path)
            )
        branch_module._retry_pending_source_staging_cleanups(source.resource_key)
        assert not tuple(tmp_path.glob(".answer.txt.cayu-*"))
        assert not branch_module._SOURCE_STAGING_CLEANUPS

    asyncio.run(scenario())


def test_recovery_settles_rollback_intent_before_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _FailPublicationBeforeCommitStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"old")
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"discarded")
        store.successes_before_failure = 1
        with pytest.raises(ConnectionError, match="before durable operation"):
            await branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=branch.branch_id,
                    idempotency_key="rollback-alpha",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                )
            )

        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.ROLLED_BACK
        assert recovered.rollback is not None
        assert (root / "answer.txt").read_bytes() == b"old"

    asyncio.run(scenario())


def test_recovery_expires_open_branch_durably(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(lifetime_ms=2_000),
        )
        await branch.write_bytes("discarded.txt", b"private")
        await asyncio.sleep(2.05)

        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.EXPIRED
        assert recovered.rollback is not None
        assert recovered.rollback.status is WorkspaceBranchOutcomeStatus.EXPIRED
        assert not (root / "discarded.txt").exists()

    asyncio.run(scenario())


def test_expiry_authority_cannot_be_forged_before_deadline(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)

        with pytest.raises(WorkspaceBranchOperationConflict, match="before its durable lifetime"):
            await branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=branch.branch_id,
                    idempotency_key="forged-expiry",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                    reason="expired",
                )
            )
        await branch.write_bytes("still-open.txt", b"private")
        assert (await branch.read_bytes("still-open.txt")).content == b"private"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("rewrite_mode", "message"),
    [
        ("premature_intent", "precedes its deadline"),
        ("premature_terminal", "precedes its deadline"),
        ("malformed", "schema is invalid"),
        ("timezone_naive", "schema is invalid"),
        ("lifetime_mismatch", "schema is invalid"),
    ],
)
def test_recovery_rejects_invalid_or_premature_expiry_authority(
    tmp_path: Path,
    rewrite_mode: str,
    message: str,
) -> None:
    async def scenario() -> None:
        store = _RewriteDurableExpiryStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("private.txt", b"must survive")
        private_root = branch._private_root
        store.rewrite_mode = rewrite_mode

        with pytest.raises(WorkspaceBranchFencedError, match=message):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert private_root.is_dir()
        assert (await branch.read_bytes("private.txt")).content == b"must survive"

    asyncio.run(scenario())


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "fork") or not hasattr(signal, "SIGKILL"),
    reason="cleanup-claim process-loss recovery requires a POSIX host",
)
@pytest.mark.parametrize("crash_boundary", ["partial_write", "staging_sync"])
def test_terminal_recovery_settles_cleanup_claim_staging_after_process_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    database = tmp_path / "branches.sqlite3"
    root = tmp_path / "workspace"
    root.mkdir()

    async def prepare_terminal_record() -> Path:
        store = SQLiteSessionStore(database)
        await _create_session(store)
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("private.txt", b"discarded")
        private_root = branch._private_root

        def retain_private_tree(path, owner):
            raise PermissionError("retain terminal private tree for process-loss recovery")

        with monkeypatch.context() as fault:
            fault.setattr(branch_module, "_remove_owned_private_tree", retain_private_tree)
            fault.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)
            with pytest.raises(PermissionError, match="retain terminal private tree"):
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-before-process-loss",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )
        branch._timer.cancel()
        return private_root

    private_root = asyncio.run(prepare_terminal_record())
    _isolate_retained_branch_state(monkeypatch, branch_module)
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions run in the parent
        try:
            original_open = branch_module.os.open
            original_fsync = branch_module.os.fsync
            original_write = branch_module.os.write
            cleanup_staging_fds: set[int] = set()

            def tracking_open(path, flags, mode=0o777, *, dir_fd=None):
                if dir_fd is None:
                    descriptor = original_open(path, flags, mode)
                else:
                    descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
                if os.fsdecode(path).endswith(".claim.staging") and (flags & os.O_ACCMODE) in {
                    os.O_WRONLY,
                    os.O_RDWR,
                }:
                    cleanup_staging_fds.add(descriptor)
                return descriptor

            def terminate_after_staging_sync(descriptor: int) -> None:
                original_fsync(descriptor)
                if crash_boundary == "staging_sync" and descriptor in cleanup_staging_fds:
                    os.kill(os.getpid(), signal.SIGKILL)

            def terminate_after_partial_write(descriptor: int, content) -> int:
                if crash_boundary == "partial_write" and descriptor in cleanup_staging_fds:
                    written = original_write(descriptor, content[:1])
                    os.kill(os.getpid(), signal.SIGKILL)
                    return written  # pragma: no cover - SIGKILL does not return
                return original_write(descriptor, content)

            branch_module.os.open = tracking_open
            branch_module.os.fsync = terminate_after_staging_sync
            branch_module.os.write = terminate_after_partial_write
            child_store = SQLiteSessionStore(database)
            asyncio.run(_workspace(root, child_store).recover_branch(_recovery_request()))
        finally:
            os._exit(90)

    deadline = time.monotonic() + 10
    while True:
        exited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if exited_pid:
            break
        if time.monotonic() >= deadline:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
            pytest.fail("cleanup worker did not reach the staging-sync boundary")
        time.sleep(0.02)
    assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
    staging_paths = tuple(tmp_path.glob(".cayu-cleanup-*.claim.staging"))
    assert len(staging_paths) == 1
    assert private_root.is_dir()

    async def recover_after_process_loss() -> None:
        reopened = SQLiteSessionStore(database)
        recovered = await _workspace(root, reopened).recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.ROLLED_BACK

    _isolate_retained_branch_state(monkeypatch, branch_module)
    asyncio.run(recover_after_process_loss())
    assert not private_root.exists()
    assert not tuple(tmp_path.glob(".cayu-cleanup-*.claim*"))
    assert not tuple(tmp_path.glob("*.cleanup-*"))


def test_private_cleanup_rejects_malformed_deterministic_claim_staging(
    tmp_path: Path,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    owner = "a" * 32
    private_root = tmp_path / "private"
    private_root.mkdir()
    sentinel = private_root / "private.txt"
    sentinel.write_bytes(b"must survive")
    branch_module._write_private_root_owner(private_root, owner)
    quarantine = private_root.with_name(f"{private_root.name}.cleanup-{owner}")
    staging_name = branch_module._cleanup_quarantine_claim_staging_name(quarantine)
    staging_path = tmp_path / staging_name
    staging_path.write_bytes(b"forged")

    with pytest.raises(WorkspaceBranchFencedError, match="staging identity is invalid"):
        branch_module._remove_owned_private_tree(private_root, owner)

    assert sentinel.read_bytes() == b"must survive"
    assert staging_path.read_bytes() == b"forged"
    assert not quarantine.exists()
    assert not branch_module._cleanup_quarantine_claim_path(quarantine).exists()


def test_recovery_records_ambiguous_mixed_source_state_without_guessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("a.txt", b"a")
        await branch.write_bytes("b.txt", b"b")
        publication = await _publication_request(branch)
        original_create = branch_module.create_regular
        applied = 0

        def create_then_lose_process(*args, **kwargs):
            nonlocal applied
            original_create(*args, **kwargs)
            applied += 1
            if applied == 1:
                raise ConnectionError("worker died after first source mutation")

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module, "create_regular", create_then_lose_process)
            with pytest.raises(ConnectionError):
                await branch.publish(publication)
        (root / "b.txt").write_bytes(b"unattributed external content")

        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.AMBIGUOUS
        assert recovered.publication is None
        assert (root / "a.txt").read_bytes() == b"a"
        assert (root / "b.txt").read_bytes() == b"unattributed external content"

    asyncio.run(scenario())


def test_recovery_fails_closed_on_corrupt_durable_record(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _CorruptLoadedBranchRecordStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        await _durable_branch(root, store)
        store.corrupt_branch_record = True

        fresh = _workspace(root, store)
        with pytest.raises(WorkspaceBranchFencedError, match="digest"):
            await fresh.recover_branch(_recovery_request())

    asyncio.run(scenario())


def test_recovery_rejects_publication_authority_that_conflicts_with_branch_record(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = _RewritePublicationAuthorityStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"candidate")
        (root / "answer.txt").write_bytes(b"external")
        publication = await branch.publish(await _publication_request(branch))
        assert publication.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        private_root = branch._private_root
        assert private_root.exists()

        store.rewrite_publication_authority = True
        with pytest.raises(WorkspaceBranchFencedError, match="schema"):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert private_root.exists()
        assert (root / "answer.txt").read_bytes() == b"external"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "record_state,rewrite_mode",
    [
        ("publication_intent", "publication"),
        ("committed", "publication"),
        ("rolled_back", "rollback"),
    ],
)
def test_recovery_rejects_redigested_evidence_that_conflicts_with_retained_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_state: str,
    rewrite_mode: str,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = _RewriteRetainedEvidenceStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"candidate")
        private_root = branch._private_root

        if record_state == "publication_intent":
            store.successes_before_failure = 1
            with pytest.raises(ConnectionError, match="before durable operation"):
                await branch.publish(await _publication_request(branch))
            assert not (root / "answer.txt").exists()
        else:
            original_remove = branch_module._remove_owned_private_tree

            def retain_private_tree(path, owner):
                if path == private_root:
                    raise PermissionError("retain private evidence for corruption test")
                return original_remove(path, owner)

            monkeypatch.setattr(branch_module, "_remove_owned_private_tree", retain_private_tree)
            monkeypatch.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_: None)
            if record_state == "committed":
                committed = await branch.publish(await _publication_request(branch))
                assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
                assert (root / "answer.txt").read_bytes() == b"candidate"
            else:
                with pytest.raises(PermissionError, match="retain private evidence"):
                    await branch.rollback(
                        WorkspaceBranchRollbackRequest(
                            branch_id=branch.branch_id,
                            idempotency_key="rollback-alpha",
                            expected_run_epoch=0,
                            binding_generation="binding-1",
                        )
                    )
                assert not (root / "answer.txt").exists()

        assert private_root.is_dir()
        store.rewrite_retained_evidence = rewrite_mode
        with pytest.raises(WorkspaceBranchFencedError, match="schema is invalid"):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert private_root.is_dir()
        if record_state == "committed":
            assert (root / "answer.txt").read_bytes() == b"candidate"
        else:
            assert not (root / "answer.txt").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("forged_outcome", ["conflicted", "unsupported", "resource_exhausted"])
def test_recovery_rejects_redigested_publication_failure_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forged_outcome: str,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = _RewriteRetainedEvidenceStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"candidate")
        private_root = branch._private_root
        original_remove = branch_module._remove_owned_private_tree

        def retain_private_tree(path, owner):
            if path == private_root:
                raise PermissionError("retain failed publication evidence")
            return original_remove(path, owner)

        def fail_inspection(*_args, **_kwargs):
            raise OSError("source inspection unavailable")

        with monkeypatch.context() as fault:
            fault.setattr(branch_module, "_inspect_source_path", fail_inspection)
            fault.setattr(branch_module, "_remove_owned_private_tree", retain_private_tree)
            fault.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_: None)
            failed = await branch.publish(await _publication_request(branch))

        assert failed.status is WorkspaceBranchOutcomeStatus.FAILED

        assert private_root.is_dir()
        store.rewrite_retained_evidence = f"publication_failure_{forged_outcome}"
        with pytest.raises(WorkspaceBranchFencedError, match="schema is invalid"):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert private_root.is_dir()
        assert not (root / "answer.txt").exists()
        assert source.resource_key in branch_module._ACTIVE_BRANCHES

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutation,rewrite_mode",
    [
        ("created", "publication_created_as_modified"),
        ("modified", "publication_modified_as_created"),
        ("modified", "publication_forged_before"),
        ("deleted", "publication_forged_before"),
    ],
)
def test_terminal_recovery_rejects_redigested_same_path_change_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    rewrite_mode: str,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = _RewriteRetainedEvidenceStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        if mutation != "created":
            (root / "answer.txt").write_bytes(b"baseline")
        _source, branch, _request = await _durable_branch(root, store)
        if mutation == "deleted":
            await branch.delete("answer.txt")
        else:
            await branch.write_bytes("answer.txt", b"candidate")
        private_root = branch._private_root

        original_remove = branch_module._remove_owned_private_tree

        def retain_private_tree(path, owner):
            if path == private_root:
                raise PermissionError("retain private evidence for corruption test")
            return original_remove(path, owner)

        monkeypatch.setattr(branch_module, "_remove_owned_private_tree", retain_private_tree)
        monkeypatch.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_: None)
        committed = await branch.publish(await _publication_request(branch))
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert private_root.is_dir()

        store.rewrite_retained_evidence = rewrite_mode
        with pytest.raises(
            WorkspaceBranchFencedError,
            match="publication evidence conflicts with retained branch state",
        ):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert private_root.is_dir()
        if mutation == "deleted":
            assert not (root / "answer.txt").exists()
        else:
            assert (root / "answer.txt").read_bytes() == b"candidate"

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["publish", "rollback"])
def test_attached_terminal_retry_authenticates_retained_publication_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrance: str,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = _RewriteRetainedEvidenceStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("answer.txt", b"candidate")
        publication_request = await _publication_request(branch)
        private_root = branch._private_root
        original_remove = branch_module._remove_owned_private_tree

        def retain_private_tree(path, owner):
            if path == private_root:
                raise PermissionError("retain private evidence for corruption test")
            return original_remove(path, owner)

        monkeypatch.setattr(branch_module, "_remove_owned_private_tree", retain_private_tree)
        monkeypatch.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_: None)
        committed = await branch.publish(publication_request)
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert private_root.is_dir()

        store.rewrite_retained_evidence = "publication_forged_before"
        with pytest.raises(
            WorkspaceBranchFencedError,
            match="publication evidence conflicts with retained branch state",
        ):
            if entrance == "publish":
                await branch.publish(publication_request)
            else:
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-after-commit",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )

        assert private_root.is_dir()
        assert (root / "answer.txt").read_bytes() == b"candidate"

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_action", ["publish", "rollback"])
def test_terminal_commit_settles_capacity_before_propagating_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_action: str,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _BlockAfterTerminalCommitStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        event_loop_thread = threading.get_ident()
        cleanup_threads: list[int] = []
        remove_durable_private_tree = branch_module._remove_durable_private_tree

        def observe_cleanup_thread(*args, **kwargs) -> None:
            cleanup_threads.append(threading.get_ident())
            remove_durable_private_tree(*args, **kwargs)

        monkeypatch.setattr(
            branch_module,
            "_remove_durable_private_tree",
            observe_cleanup_thread,
        )
        if terminal_action == "publish":
            await branch.write_bytes("answer.txt", b"candidate")
            terminal = asyncio.create_task(branch.publish(await _publication_request(branch)))
        else:
            terminal = asyncio.create_task(
                branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )
            )
        store.block_terminal_commit = True
        try:
            assert await asyncio.wait_for(store.terminal_committed.wait(), timeout=5)
            terminal.cancel()
            assert terminal.cancelling() == 1
        finally:
            store.release_terminal_acknowledgement.set()

        with pytest.raises(asyncio.CancelledError):
            await terminal
        assert terminal.cancelled()
        assert terminal.cancelling() == 1
        assert branch.lifecycle_status is (
            WorkspaceBranchLifecycleStatus.COMMITTED
            if terminal_action == "publish"
            else WorkspaceBranchLifecycleStatus.ROLLED_BACK
        )
        assert cleanup_threads
        assert event_loop_thread not in cleanup_threads
        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_action", ["publish", "rollback"])
def test_terminal_dual_failure_retains_owner_until_assisted_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_action: str,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = _BlockAfterTerminalCommitStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=1),
            resolver=(
                live_binding := WorkspaceBranchBindingAuthorityRegistry(_binding_authority())
            ),
        )
        if terminal_action == "publish":
            await branch.write_bytes("answer.txt", b"candidate")
            terminal = asyncio.create_task(branch.publish(await _publication_request(branch)))
        else:
            terminal = asyncio.create_task(
                branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )
            )
        store.block_terminal_commit = True
        try:
            assert await asyncio.wait_for(store.terminal_committed.wait(), timeout=5)
            store.fail_terminal_reconciliation = True
            terminal.cancel()
            assert terminal.cancelling() == 1
        finally:
            store.release_terminal_acknowledgement.set()

        with pytest.raises(asyncio.CancelledError) as caught:
            await terminal
        assert terminal.cancelled()
        assert terminal.cancelling() == 1
        assert isinstance(caught.value.__cause__, ConnectionError)
        assert "terminal reconciliation read unavailable" in str(caught.value.__cause__)
        assert len(branch_module._DURABLE_TERMINAL_SETTLEMENTS) == 1
        assert branch._private_root.is_dir()
        replacement = _binding_authority(
            generation="binding-2",
            identity="workspace-alpha@binding-2",
        )
        with pytest.raises(WorkspaceBranchOperationConflict, match="active generation claim"):
            live_binding.replace(replacement)

        store.fail_terminal_reconciliation = False
        if terminal_action == "publish":
            recovered = await source.recover_branch(_recovery_request())
            assert recovered.state is WorkspaceBranchDurableState.COMMITTED
        await _assert_terminal_handle_released_capacity(root, store)
        assert len(branch_module._DURABLE_TERMINAL_SETTLEMENTS) == 0
        assert not branch._private_root.exists()
        live_binding.replace(replacement)

    asyncio.run(scenario())


def test_fresh_process_terminal_cleanup_failure_retains_binding_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        provider = _FailOnceReleaseAuthorityProvider(_binding_authority())
        _source, branch, _request = await _durable_branch(
            root,
            store,
            resolver=provider,
        )
        private_root = branch._private_root
        original_remove_owned_private_tree = branch_module._remove_owned_private_tree

        def fail_terminal_cleanup(path, owner):
            if path == private_root:
                raise PermissionError("fresh-process cleanup remains active")
            return original_remove_owned_private_tree(path, owner)

        with monkeypatch.context() as fault:
            fault.setattr(branch_module, "_remove_owned_private_tree", fail_terminal_cleanup)
            fault.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)
            with pytest.raises(PermissionError, match="cleanup remains active"):
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )

        # Model process loss after the terminal record committed: process-local
        # branch owners disappear, while the owned private tree remains.
        branch_module._release_durable_terminal_settlement(branch)
        _isolate_retained_branch_state(monkeypatch, branch_module)
        fresh = _workspace(root, store, resolver=provider)

        with monkeypatch.context() as fault:
            fault.setattr(branch_module, "_remove_owned_private_tree", fail_terminal_cleanup)
            with pytest.raises(PermissionError, match="cleanup remains active"):
                await fresh.recover_branch(_recovery_request(recovery_id="recover-cleanup-fails"))

        assert len(branch_module._DURABLE_RECOVERY_CLEANUPS) == 1
        replacement = _binding_authority(
            generation="binding-2",
            identity="workspace-alpha@binding-2",
        )
        with pytest.raises(WorkspaceBranchOperationConflict, match="active generation claim"):
            provider.replace(replacement)

        provider.arm_release_failure()
        with pytest.raises(ConnectionError, match="binding claim release failed"):
            await fresh.recover_branch(
                _recovery_request(recovery_id="recover-cleanup-release-fails")
            )
        assert len(branch_module._DURABLE_RECOVERY_CLEANUPS) == 1
        with pytest.raises(WorkspaceBranchOperationConflict, match="active generation claim"):
            provider.replace(replacement)

        recovered = await fresh.recover_branch(
            _recovery_request(recovery_id="recover-cleanup-retry")
        )
        assert recovered.state is WorkspaceBranchDurableState.ROLLED_BACK
        assert len(branch_module._DURABLE_RECOVERY_CLEANUPS) == 0
        assert not private_root.exists()
        provider.replace(replacement)

    asyncio.run(scenario())


def test_binding_claim_release_failure_is_retained_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        provider = _FailOnceReleaseAuthorityProvider(_binding_authority())
        source, branch, _request = await _durable_branch(
            root,
            store,
            resolver=provider,
        )
        provider.arm_release_failure()

        committed_mutation = asyncio.create_task(branch.create_bytes("candidate.txt", b"candidate"))
        with pytest.raises(ConnectionError, match="binding claim release failed"):
            await committed_mutation
        assert (await branch.read_bytes("candidate.txt")).content == b"candidate"
        assert len(branch_module._BINDING_CLAIM_RELEASES) == 1

        replacement = _binding_authority(
            generation="binding-2",
            identity="workspace-alpha@binding-2",
        )
        with pytest.raises(WorkspaceBranchOperationConflict, match="active generation claim"):
            provider.replace(replacement)

        successful_mutation = asyncio.create_task(branch.write_bytes("second.txt", b"second"))
        await successful_mutation
        assert len(branch_module._BINDING_CLAIM_RELEASES) == 0

        provider.arm_release_failure()
        failed_mutation = asyncio.create_task(
            branch.create_bytes("candidate.txt", b"must-not-replace")
        )
        with pytest.raises(FileExistsError) as caught:
            await failed_mutation
        assert isinstance(caught.value.__cause__, ConnectionError)
        assert str(caught.value.__cause__) == "binding claim release failed"
        assert len(branch_module._BINDING_CLAIM_RELEASES) == 1

        recovery = await asyncio.create_task(
            source.recover_branch(_recovery_request(recovery_id="recover-after-release-failure"))
        )
        assert recovery.branch is not None
        assert len(branch_module._BINDING_CLAIM_RELEASES) == 0
        await recovery.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=recovery.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )
        provider.replace(replacement)

    asyncio.run(scenario())


def test_transferred_binding_claim_release_failure_remains_assistable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        provider = _FailOnceReleaseAuthorityProvider(_binding_authority())
        source, branch, _request = await _durable_branch(
            root,
            store,
            resolver=provider,
        )
        private_root = branch._private_root
        original_remove = branch_module._remove_owned_private_tree

        def fail_cleanup(path, owner):
            if path == private_root:
                raise PermissionError("terminal cleanup remains active")
            return original_remove(path, owner)

        with monkeypatch.context() as fault:
            fault.setattr(branch_module, "_remove_owned_private_tree", fail_cleanup)
            fault.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)
            terminal_task = asyncio.create_task(
                branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )
            )
            with pytest.raises(PermissionError, match="cleanup remains active"):
                await terminal_task

        assert len(branch_module._DURABLE_TERMINAL_SETTLEMENTS) == 1
        provider.arm_release_failure()
        first_assistance = asyncio.create_task(
            source.recover_branch(_recovery_request(recovery_id="recover-release-fails"))
        )
        with pytest.raises(ConnectionError, match="binding claim release failed"):
            await first_assistance
        assert len(branch_module._DURABLE_TERMINAL_SETTLEMENTS) == 1

        replacement = _binding_authority(
            generation="binding-2",
            identity="workspace-alpha@binding-2",
        )
        with pytest.raises(WorkspaceBranchOperationConflict, match="active generation claim"):
            provider.replace(replacement)

        second_assistance = asyncio.create_task(
            source.recover_branch(_recovery_request(recovery_id="recover-release-retries"))
        )
        recovered = await second_assistance
        assert recovered.state is WorkspaceBranchDurableState.ROLLED_BACK
        assert len(branch_module._DURABLE_TERMINAL_SETTLEMENTS) == 0
        assert not private_root.exists()
        provider.replace(replacement)

    asyncio.run(scenario())


def test_cancelling_terminal_assistance_does_not_claim_later_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        registry = branch_module._DURABLE_TERMINAL_SETTLEMENTS
        source_key = ("local", "workspace-alpha")
        first_key = (Path("first-private-root"), 1)
        second_key = (Path("second-private-root"), 2)

        class _RetainedOwner:
            def __init__(self, branch: str) -> None:
                self.branch = branch

        first = _RetainedOwner("first")
        second = _RetainedOwner("second")
        registry.retain(first_key, source_key=source_key, payload=first)
        registry.retain(second_key, source_key=source_key, payload=second)

        first_started = asyncio.Event()
        never_settles = asyncio.Event()
        attempted: list[str] = []

        async def block_first(retained) -> None:
            attempted.append(retained.branch)
            first_started.set()
            await never_settles.wait()

        monkeypatch.setattr(
            branch_module,
            "_retry_durable_terminal_settlement",
            block_first,
        )
        assistance = asyncio.create_task(
            branch_module._retry_pending_durable_terminal_settlements(source_key)
        )
        await asyncio.wait_for(first_started.wait(), timeout=5)
        records = dict(registry.items())
        assert records[first_key].claimed
        assert not records[second_key].claimed

        assistance.cancel()
        assert assistance.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await assistance
        assert assistance.cancelled()
        assert assistance.cancelling() == 1
        records = dict(registry.items())
        assert not records[first_key].claimed
        assert not records[second_key].claimed
        assert attempted == ["first"]

        keys_by_branch = {"first": first_key, "second": second_key}
        settled: list[str] = []

        async def settle(retained) -> None:
            settled.append(retained.branch)

        def release(branch: str) -> None:
            registry.forget(keys_by_branch[branch])

        monkeypatch.setattr(
            branch_module,
            "_retry_durable_terminal_settlement",
            settle,
        )
        monkeypatch.setattr(
            branch_module,
            "_release_durable_terminal_settlement",
            release,
        )
        await branch_module._retry_pending_durable_terminal_settlements(source_key)
        assert settled == ["first", "second"]
        assert len(registry) == 0

    asyncio.run(scenario())


def test_attached_terminal_cleanup_settles_capacity_before_propagating_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=2),
        )
        await branch.write_bytes("answer.txt", b"candidate")
        publication_request = await _publication_request(branch)
        recovered = await source.recover_branch(_recovery_request())
        assert recovered.branch is not None
        attached = recovered.branch
        committed = await branch.publish(publication_request)
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED

        cleanup_finished = threading.Event()
        release_cleanup = threading.Event()
        remove_durable_private_tree = branch_module._remove_durable_private_tree

        def block_after_cleanup(*args, **kwargs) -> None:
            remove_durable_private_tree(*args, **kwargs)
            cleanup_finished.set()
            if not release_cleanup.wait(timeout=5):
                raise TimeoutError("test did not release terminal cleanup")

        monkeypatch.setattr(
            branch_module,
            "_remove_durable_private_tree",
            block_after_cleanup,
        )
        repeated = asyncio.create_task(attached.publish(publication_request))
        try:
            assert await asyncio.to_thread(cleanup_finished.wait, 5)
            repeated.cancel()
            assert repeated.cancelling() == 1
        finally:
            release_cleanup.set()

        with pytest.raises(asyncio.CancelledError):
            await repeated
        assert repeated.cancelled()
        assert repeated.cancelling() == 1
        assert attached.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED
        await _assert_terminal_handle_released_capacity(root, store)

    asyncio.run(scenario())


def test_attached_terminal_cleanup_dual_failure_retains_owner_until_assisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(max_active_branches=2),
        )
        await branch.write_bytes("answer.txt", b"candidate")
        publication_request = await _publication_request(branch)
        recovered = await source.recover_branch(_recovery_request())
        assert recovered.branch is not None
        attached = recovered.branch
        committed = await branch.publish(publication_request)
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED

        cleanup_started = threading.Event()
        release_cleanup = threading.Event()
        remove_durable_private_tree = branch_module._remove_durable_private_tree

        def fail_terminal_cleanup(*_args, **_kwargs) -> None:
            cleanup_started.set()
            if not release_cleanup.wait(timeout=5):
                raise TimeoutError("test did not release terminal cleanup")
            raise PermissionError("terminal authenticated cleanup unavailable")

        monkeypatch.setattr(
            branch_module,
            "_remove_durable_private_tree",
            fail_terminal_cleanup,
        )
        repeated = asyncio.create_task(attached.publish(publication_request))
        try:
            assert await asyncio.to_thread(cleanup_started.wait, 5)
            repeated.cancel()
            assert repeated.cancelling() == 1
        finally:
            release_cleanup.set()

        with pytest.raises(asyncio.CancelledError) as caught:
            await repeated
        assert repeated.cancelled()
        assert repeated.cancelling() == 1
        assert isinstance(caught.value.__cause__, PermissionError)
        assert "terminal authenticated cleanup unavailable" in str(caught.value.__cause__)
        assert len(branch_module._DURABLE_TERMINAL_SETTLEMENTS) == 1

        monkeypatch.setattr(
            branch_module,
            "_remove_durable_private_tree",
            remove_durable_private_tree,
        )
        await _assert_terminal_handle_released_capacity(root, store)
        assert len(branch_module._DURABLE_TERMINAL_SETTLEMENTS) == 0

    asyncio.run(scenario())


def test_guarded_mutation_rejects_same_revision_immutable_record_drift(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _DriftGuardedCurrentRecordStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, _request = await _durable_branch(root, store)
        store.drift_next_guarded_record = True

        with pytest.raises(WorkspaceBranchOperationConflict, match="changed under another"):
            await branch.write_bytes("must-not-land.txt", b"candidate")

        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        fresh = await source.recover_branch(_recovery_request())
        assert fresh.state is WorkspaceBranchDurableState.OPEN
        assert fresh.branch is not None
        with pytest.raises(FileNotFoundError):
            await fresh.branch.read_bytes("must-not-land.txt")
        await fresh.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=fresh.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )

    asyncio.run(scenario())


def test_recovery_does_not_follow_private_file_replaced_after_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        _source, branch, _request = await _durable_branch(root, store)
        baseline_file = branch._baseline_root / "answer.txt"
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"must not be read")
        original_open = branch_module.os.open
        replaced = False

        def replace_before_descriptor_open(path, flags, *args, **kwargs):
            nonlocal replaced
            if (
                not replaced
                and path == "answer.txt"
                and kwargs.get("dir_fd") is not None
                and flags & getattr(os, "O_NONBLOCK", 0)
            ):
                replaced = True
                baseline_file.unlink()
                baseline_file.symlink_to(outside)
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(branch_module.os, "open", replace_before_descriptor_open)
        recovered = await _workspace(root, store).recover_branch(_recovery_request())

        assert replaced
        assert recovered.state is WorkspaceBranchDurableState.AMBIGUOUS
        assert recovered.branch is None
        assert outside.read_bytes() == b"must not be read"
        assert branch._private_root.exists()

    asyncio.run(scenario())


def test_recovery_bounds_private_file_growth_after_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"base")
        _source, branch, _request = await _durable_branch(
            root,
            store,
            limits=WorkspaceBranchLimits(
                max_file_bytes=8,
                max_baseline_bytes=8,
                max_overlay_bytes=8,
            ),
        )
        baseline_file = branch._baseline_root / "answer.txt"
        original_open = branch_module.os.open
        grown = False

        def grow_before_descriptor_open(path, flags, *args, **kwargs):
            nonlocal grown
            if (
                not grown
                and path == "answer.txt"
                and kwargs.get("dir_fd") is not None
                and flags & getattr(os, "O_NONBLOCK", 0)
            ):
                grown = True
                baseline_file.write_bytes(b"x" * 64)
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(branch_module.os, "open", grow_before_descriptor_open)
        with pytest.raises(
            WorkspaceBranchResourceExhaustedError,
            match="file_byte_limit_exceeded",
        ):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert grown
        assert branch._private_root.exists()

    asyncio.run(scenario())


def test_recovery_reconstructs_private_tree_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        source, _branch, _request = await _durable_branch(root, store)
        event_loop_thread = threading.get_ident()
        scan_private_tree = branch_module._scan_private_tree
        scan_threads: list[int] = []

        def record_scan_thread(*args, **kwargs):
            scan_threads.append(threading.get_ident())
            return scan_private_tree(*args, **kwargs)

        monkeypatch.setattr(branch_module, "_scan_private_tree", record_scan_thread)
        recovered = await source.recover_branch(_recovery_request())

        assert recovered.state is WorkspaceBranchDurableState.OPEN
        assert recovered.branch is not None
        assert scan_threads
        assert all(thread_id != event_loop_thread for thread_id in scan_threads)
        await recovered.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=recovered.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )

    asyncio.run(scenario())


def test_exact_creating_retry_rejects_redirected_private_root(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _RedirectLoadedPrivateRootStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, request = await _interrupted_durable_creating_branch(root, store)
        redirected = tmp_path / "redirected-private-root"
        store.redirect_private_root = redirected

        with pytest.raises(WorkspaceBranchFencedError, match="private branch location"):
            await source.create_branch(request)

        assert not redirected.exists()
        assert tuple(tmp_path.glob(".cayu-workspace-branch-*")) == ()

    asyncio.run(scenario())


def test_exact_creating_retry_preserves_unowned_capture_staging(tmp_path: Path) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _RedirectLoadedPrivateRootStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, request = await _interrupted_durable_creating_branch(root, store)
        stored = await store.load_session_operation(
            "branch-session",
            branch_module._durable_branch_storage_key("branch-alpha"),
        )
        assert stored is not None
        payload = stored["payload"]
        private_root = Path(payload["private_root"])
        staging_root = private_root.with_name(
            f"{private_root.name}.capture-{payload['capture_attempt_id']}"
        )
        staging_root.mkdir()
        sentinel = staging_root / "unowned.txt"
        sentinel.write_text("must survive exact retry")

        with pytest.raises(WorkspaceBranchFencedError, match="ownership"):
            await source.create_branch(request)

        assert sentinel.read_text() == "must survive exact retry"
        retained = await store.load_session_operation(
            "branch-session",
            branch_module._durable_branch_storage_key("branch-alpha"),
        )
        assert retained is not None
        assert retained["payload"]["state"] == "creating"

    asyncio.run(scenario())


def test_exact_creating_retry_rejects_replaced_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _RedirectLoadedPrivateRootStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, request = await _interrupted_durable_creating_branch(root, store)
        original_root = tmp_path / "original-workspace"

        def replace_source_after_initial_validation(_source_key) -> None:
            root.rename(original_root)
            root.mkdir()

        monkeypatch.setattr(
            branch_module,
            "_retry_pending_branch_cleanups",
            replace_source_after_initial_validation,
        )

        with pytest.raises(WorkspaceBranchFencedError, match="root identity"):
            await _workspace(root, store).create_branch(request)

        assert tuple(tmp_path.glob(".cayu-workspace-branch-*")) == ()
        assert list(root.iterdir()) == []
        assert list(original_root.iterdir()) == []

    asyncio.run(scenario())


def test_binding_replacement_is_rejected_while_exact_creating_retry_owns_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _RedirectLoadedPrivateRootStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        live_binding = WorkspaceBranchBindingAuthorityRegistry(_binding_authority())
        source, request = await _interrupted_durable_creating_branch(
            root,
            store,
            resolver=live_binding,
        )
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()

        def block_before_guarded_capture(_source_key) -> None:
            cleanup_started.set()
            if not release_cleanup.wait(timeout=5):
                raise TimeoutError("test did not release pending cleanup")

        monkeypatch.setattr(
            branch_module,
            "_retry_pending_branch_cleanups",
            block_before_guarded_capture,
        )
        creation = asyncio.create_task(source.create_branch(request))
        assert await asyncio.to_thread(cleanup_started.wait, 5)
        with pytest.raises(WorkspaceBranchOperationConflict, match="active generation claim"):
            live_binding.replace(
                _binding_authority(
                    generation="binding-2",
                    identity="workspace-alpha@binding-2",
                )
            )
        release_cleanup.set()

        created = await creation
        assert created.status is WorkspaceBranchOutcomeStatus.CREATED
        assert created.branch is not None
        await created.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=created.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )
        live_binding.replace(
            _binding_authority(
                generation="binding-2",
                identity="workspace-alpha@binding-2",
            )
        )
        assert tuple(tmp_path.glob(".cayu-workspace-branch-*")) == ()

    asyncio.run(scenario())


def test_binding_replacement_is_rejected_during_guarded_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        live_binding = WorkspaceBranchBindingAuthorityRegistry(_binding_authority())
        source = _workspace(root, store, resolver=live_binding)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            limits=WorkspaceBranchLimits(max_active_branches=1),
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-1",
                binding_identity="workspace-alpha@binding-1",
                creating_authority="fork-group:alpha",
                resource_policy="local-cow-defaults-v1",
            ),
        )
        capture_started = threading.Event()
        release_capture = threading.Event()
        copy_regular_tree = branch_module._copy_regular_tree

        def block_during_capture(*args, **kwargs):
            capture_started.set()
            if not release_capture.wait(timeout=5):
                raise TimeoutError("test did not release baseline capture")
            return copy_regular_tree(*args, **kwargs)

        monkeypatch.setattr(branch_module, "_copy_regular_tree", block_during_capture)
        creation = asyncio.create_task(source.create_branch(request))
        try:
            assert await asyncio.to_thread(capture_started.wait, 5)
            with pytest.raises(
                WorkspaceBranchOperationConflict,
                match="active generation claim",
            ):
                live_binding.replace(
                    _binding_authority(
                        generation="binding-2",
                        identity="workspace-alpha@binding-2",
                    )
                )
        finally:
            release_capture.set()

        created = await creation
        assert created.status is WorkspaceBranchOutcomeStatus.CREATED
        assert created.branch is not None
        await created.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=created.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )
        live_binding.replace(
            _binding_authority(
                generation="binding-2",
                identity="workspace-alpha@binding-2",
            )
        )
        monkeypatch.setattr(branch_module, "_copy_regular_tree", copy_regular_tree)
        replacement = await source.create_branch(
            request.model_copy(
                update={
                    "branch_id": "branch-beta",
                    "idempotency_key": "create-beta",
                    "authority": WorkspaceBranchAuthority(
                        session_id="branch-session",
                        expected_run_epoch=0,
                        environment_name="local",
                        binding_generation="binding-2",
                        binding_identity="workspace-alpha@binding-2",
                        creating_authority="fork-group:beta",
                        resource_policy="local-cow-defaults-v1",
                    ),
                }
            )
        )
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED

    asyncio.run(scenario())


@pytest.mark.parametrize("record_state", ["terminal", "rollback_intent", "failed"])
def test_recovery_rejects_redirected_private_cleanup_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_state: str,
) -> None:
    async def scenario() -> None:
        store = _RedirectLoadedPrivateRootStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        if record_state == "terminal":
            await branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=branch.branch_id,
                    idempotency_key="rollback-alpha",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                )
            )
        elif record_state == "rollback_intent":
            store.successes_before_failure = 1
            with pytest.raises(ConnectionError, match="before durable operation"):
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )
        else:
            import cayu.workspaces._local_branch as branch_module

            await branch.write_bytes("answer.txt", b"candidate")

            def fail_inspection(*_args, **_kwargs):
                raise OSError("source inspection unavailable")

            monkeypatch.setattr(branch_module, "_inspect_source_path", fail_inspection)
            publication = await branch.publish(
                await _publication_request(branch, key="failed-publication")
            )
            assert publication.status is WorkspaceBranchOutcomeStatus.FAILED

        victim = tmp_path / "unrelated-sibling"
        victim.mkdir()
        sentinel = victim / "keep.txt"
        sentinel.write_text("must survive")
        store.redirect_private_root = victim

        with pytest.raises(WorkspaceBranchFencedError, match="private branch location"):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert sentinel.read_text() == "must survive"

    asyncio.run(scenario())


@pytest.mark.parametrize("record_state", ["terminal", "rollback_intent", "failed"])
def test_recovery_rejects_replaced_source_root_before_private_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_state: str,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _FailPublicationBeforeCommitStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        private_root = branch._private_root
        cleanup_registry_type = type(branch_module._PRIVATE_TREE_CLEANUPS)
        monkeypatch.setattr(branch_module, "_PRIVATE_TREE_CLEANUPS", cleanup_registry_type())
        monkeypatch.setattr(branch_module, "_ACTIVE_BRANCHES", {})
        original_remove_owned_private_tree = branch_module._remove_owned_private_tree

        def retain_private_tree(path, owner):
            if path == private_root:
                raise PermissionError("retain private evidence for root replacement")
            return original_remove_owned_private_tree(path, owner)

        monkeypatch.setattr(
            branch_module,
            "_remove_owned_private_tree",
            retain_private_tree,
        )
        monkeypatch.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)

        if record_state == "terminal":
            with pytest.raises(PermissionError, match="retain private evidence"):
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )
        elif record_state == "rollback_intent":
            store.successes_before_failure = 1
            with pytest.raises(ConnectionError, match="before durable operation"):
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )
        else:
            await branch.write_bytes("answer.txt", b"candidate")

            def fail_inspection(*_args, **_kwargs):
                raise OSError("source inspection unavailable")

            with monkeypatch.context() as fault:
                fault.setattr(branch_module, "_inspect_source_path", fail_inspection)
                publication = await branch.publish(
                    await _publication_request(branch, key="failed-publication")
                )
            assert publication.status is WorkspaceBranchOutcomeStatus.FAILED

        assert private_root.is_dir()
        original_root = tmp_path / "original-workspace"
        root.rename(original_root)
        root.mkdir()

        with pytest.raises(WorkspaceBranchFencedError, match="root identity changed"):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert private_root.is_dir()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_state", ["committed", "rolled_back", "expired"])
def test_recovery_retains_private_tree_when_terminal_evidence_is_missing(
    tmp_path: Path,
    terminal_state: str,
) -> None:
    async def scenario() -> None:
        store = _MissingTerminalEvidenceStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        await branch.write_bytes("private.txt", b"retain until terminal evidence validates")
        store.terminal_state = terminal_state

        with pytest.raises(WorkspaceBranchFencedError, match="schema is invalid"):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert store.private_root is not None
        assert store.private_root.is_dir()

    asyncio.run(scenario())


def test_recovery_retains_private_tree_when_terminal_evidence_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _ContradictoryTerminalEvidenceStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        private_root = branch._private_root
        cleanup_registry_type = type(branch_module._PRIVATE_TREE_CLEANUPS)
        monkeypatch.setattr(branch_module, "_PRIVATE_TREE_CLEANUPS", cleanup_registry_type())
        monkeypatch.setattr(branch_module, "_ACTIVE_BRANCHES", {})
        original_remove_owned_private_tree = branch_module._remove_owned_private_tree

        def retain_private_tree(path, owner):
            if path == private_root:
                raise PermissionError("retain contradictory private evidence")
            return original_remove_owned_private_tree(path, owner)

        monkeypatch.setattr(
            branch_module,
            "_remove_owned_private_tree",
            retain_private_tree,
        )
        monkeypatch.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)

        await branch.write_bytes("answer.txt", b"candidate")
        publication = await branch.publish(await _publication_request(branch))
        assert publication.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert private_root.is_dir()
        store.inject_rollback = True

        with pytest.raises(WorkspaceBranchFencedError, match="schema is invalid"):
            await _workspace(root, store).recover_branch(_recovery_request())

        assert private_root.is_dir()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_state", ["committed", "rolled_back", "expired"])
def test_terminal_recovery_does_not_delete_reused_private_root(
    tmp_path: Path,
    terminal_state: str,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(
            root,
            store,
            limits=(WorkspaceBranchLimits(lifetime_ms=1) if terminal_state == "expired" else None),
        )
        private_root = next(root.parent.glob(".cayu-workspace-branch-*"))
        if terminal_state == "committed":
            await branch.write_bytes("answer.txt", b"published")
            await branch.publish(await _publication_request(branch))
        elif terminal_state == "rolled_back":
            await branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=branch.branch_id,
                    idempotency_key="rollback-alpha",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                )
            )
        else:
            await asyncio.sleep(0.01)
            recovered = await _workspace(root, store).recover_branch(_recovery_request())
            assert recovered.state is WorkspaceBranchDurableState.EXPIRED
        assert not private_root.exists()

        private_root.mkdir()
        sentinel = private_root / "unrelated.txt"
        sentinel.write_text("must survive terminal replay")

        with pytest.raises(WorkspaceBranchFencedError, match="ownership"):
            await _workspace(root, store).recover_branch(
                _recovery_request(recovery_id="recover-reused-private-root")
            )

        assert sentinel.read_text() == "must survive terminal replay"

    asyncio.run(scenario())


def test_terminal_cleanup_restores_path_replacement_raced_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        private_root = branch._private_root
        displaced = private_root.with_name(f"{private_root.name}.original")
        sentinel = private_root / "replacement.txt"
        original_rename = branch_module._rename_private_root_no_replace
        replaced = False

        def replace_after_open(source, target, *, parent_fd):
            nonlocal replaced
            if source == private_root and not replaced:
                replaced = True
                os.rename(private_root, displaced)
                private_root.mkdir()
                sentinel.write_text("must remain authoritative")
            return original_rename(source, target, parent_fd=parent_fd)

        with monkeypatch.context() as fault:
            fault.setattr(
                branch_module,
                "_rename_private_root_no_replace",
                replace_after_open,
            )
            fault.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)
            with pytest.raises(
                WorkspaceBranchFencedError,
                match="ownership changed during quarantine",
            ):
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="rollback-alpha",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )

        assert replaced
        assert private_root.is_dir()
        assert sentinel.read_text() == "must remain authoritative"
        assert displaced.is_dir()
        assert not tuple(root.parent.glob(f"{private_root.name}.cleanup-*"))
        branch_module._release_durable_terminal_settlement(branch)

    asyncio.run(scenario())


def test_capture_cleanup_does_not_reclassify_restored_replacement_as_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    capture_root = tmp_path / ".cayu-workspace-branch.capture-attempt"
    capture_root.mkdir()
    owner = "1" * 32
    branch_module._write_private_root_owner(capture_root, owner)
    displaced = capture_root.with_name(f"{capture_root.name}.original")
    original_rename = branch_module._rename_private_root_no_replace
    replaced = False

    def replace_after_open(source, target, *, parent_fd):
        nonlocal replaced
        if source == capture_root and not replaced:
            replaced = True
            os.rename(capture_root, displaced)
            capture_root.mkdir()
        return original_rename(source, target, parent_fd=parent_fd)

    monkeypatch.setattr(
        branch_module,
        "_rename_private_root_no_replace",
        replace_after_open,
    )
    with pytest.raises(
        WorkspaceBranchFencedError,
        match="ownership changed during quarantine",
    ):
        branch_module._discard_owned_capture_staging(capture_root, owner)

    assert replaced
    assert capture_root.is_dir()
    assert list(capture_root.iterdir()) == []
    assert displaced.is_dir()
    assert not tuple(tmp_path.glob(f"{capture_root.name}.cleanup-*"))


def test_durable_authority_copy_suppresses_serializer_diagnostics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _HostileAuthorityValue:
        def __repr__(self) -> str:
            return "MUTATED_BRANCH_AUTHORITY_SECRET_CANARY"

        __str__ = __repr__

    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root, store)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        authority = WorkspaceBranchAuthority(
            session_id="branch-session",
            expected_run_epoch=0,
            environment_name="local",
            binding_generation="binding-1",
            binding_identity="workspace-alpha@binding-1",
            creating_authority="fork-group:alpha",
            resource_policy="local-cow-defaults-v1",
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            branch_id="branch-alpha",
            idempotency_key="create-alpha",
            authority=authority,
        )
        object.__setattr__(authority, "creating_authority", _HostileAuthorityValue())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with (
                caplog.at_level(logging.WARNING),
                pytest.raises(ValidationError) as raised,
            ):
                await source.create_branch(request)

        emitted = "\n".join(str(item.message) for item in caught)
        captured = capsys.readouterr()
        diagnostics = "\n".join(
            (
                str(raised.value),
                repr(raised.value),
                emitted,
                caplog.text,
                captured.out,
                captured.err,
            )
        )
        assert caught == []
        assert "MUTATED_BRANCH_AUTHORITY_SECRET_CANARY" not in diagnostics

    asyncio.run(scenario())


def test_private_mutation_crash_becomes_durable_ambiguity(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _CrashInsideGuardStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        _source, branch, _request = await _durable_branch(root, store)
        store.crash_inside_next_guard = True
        with pytest.raises(ConnectionError, match="guarded commit"):
            await branch.write_bytes("uncertain.txt", b"private mutation")

        fresh = _workspace(root, store)
        recovered = await fresh.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.AMBIGUOUS
        assert recovered.branch is None
        assert not (root / "uncertain.txt").exists()

    asyncio.run(scenario())


def test_private_mutation_preserves_cancellation_when_reconciliation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        store = _FailReconciliationLoadStore()
        await _create_session(store)
        root = tmp_path / "workspace"
        root.mkdir()
        source, branch, _request = await _durable_branch(root, store)
        mutation_started = threading.Event()
        release_mutation = threading.Event()
        write_regular = branch_module.write_regular

        def block_during_write(*args, **kwargs):
            mutation_started.set()
            if not release_mutation.wait(timeout=5):
                raise TimeoutError("test did not release private mutation")
            return write_regular(*args, **kwargs)

        monkeypatch.setattr(branch_module, "write_regular", block_during_write)
        mutation = asyncio.create_task(branch.write_bytes("uncertain.txt", b"private mutation"))
        try:
            assert await asyncio.to_thread(mutation_started.wait, 5)
            mutation.cancel()
            assert mutation.cancelling() == 1
            store.cancellation_secondary = ConnectionError(
                "guarded publication acknowledgement failed"
            )
            store.fail_next_load = True
        finally:
            release_mutation.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await mutation
        assert mutation.cancelled()
        assert mutation.cancelling() == 1
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        assert isinstance(raised.value.__cause__, BaseExceptionGroup)
        assert [str(error) for error in raised.value.__cause__.exceptions] == [
            "guarded publication acknowledgement failed",
            "reconciliation read failed",
        ]

        monkeypatch.setattr(branch_module, "write_regular", write_regular)
        recovered = await source.recover_branch(_recovery_request())
        assert recovered.state is WorkspaceBranchDurableState.OPEN
        assert recovered.branch is not None
        assert (await recovered.branch.read_bytes("uncertain.txt")).content == b"private mutation"
        await recovered.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=recovered.branch.branch_id,
                idempotency_key="rollback-alpha",
                expected_run_epoch=0,
                binding_generation="binding-1",
            )
        )

    asyncio.run(scenario())


async def assert_durable_workspace_branch_store_conformance(store, root: Path) -> None:
    """Prove the complete durable branch contract on one session-store backend."""

    root.mkdir()

    async def create_case(
        name: str,
        *,
        branch_store=None,
        limits: WorkspaceBranchLimits | None = None,
    ):
        session_id = f"branch-session-{name}"
        branch_id = f"branch-{name}"
        generation = f"binding-{name}"
        identity = f"workspace-alpha@{generation}"
        binding = _binding_authority(generation=generation, identity=identity)
        await _create_session(store, session_id=session_id)
        case_root = root / name
        case_root.mkdir()
        source, branch, request = await _durable_branch(
            case_root,
            branch_store or store,
            branch_id=branch_id,
            create_key=f"create-{name}",
            limits=limits,
            session_id=session_id,
            binding_generation=generation,
            binding_identity=identity,
            resolver=lambda: binding,
        )
        recovery = _recovery_request(
            recovery_id=f"recover-{name}",
            branch_id=branch_id,
            session_id=session_id,
            binding_generation=generation,
            binding_identity=identity,
        )
        return case_root, source, branch, request, binding, recovery

    # Open-state reconstruction, exact creation/publication replay, and conflicts.
    case_root, source, branch, request, binding, recovery = await create_case("replay")
    await branch.write_bytes("created.txt", b"created")
    repeated_creation = await source.create_branch(request)
    assert repeated_creation.status is WorkspaceBranchOutcomeStatus.CREATED
    with pytest.raises(WorkspaceBranchOperationConflict):
        await source.create_branch(request.model_copy(update={"idempotency_key": "different"}))
    fresh = _workspace(case_root, store, resolver=lambda: binding)
    recovered = await fresh.recover_branch(recovery)
    assert recovered.state is WorkspaceBranchDurableState.OPEN
    assert recovered.branch is not None
    publication = await _publication_request(
        recovered.branch,
        key="publish-replay",
        binding_generation="binding-replay",
    )
    committed = await recovered.branch.publish(publication)
    assert await recovered.branch.publish(publication) == committed
    with pytest.raises(WorkspaceBranchOperationConflict):
        await recovered.branch.publish(publication.model_copy(update={"idempotency_key": "other"}))
    terminal = await fresh.recover_branch(
        recovery.model_copy(update={"recovery_id": "recover-replay-terminal"})
    )
    assert terminal.state is WorkspaceBranchDurableState.COMMITTED
    assert terminal.publication == committed

    async def concurrent_creation_case(
        name: str,
        *,
        conflicting: bool,
    ) -> None:
        session_id = f"branch-session-{name}"
        generation = f"binding-{name}"
        identity = f"workspace-alpha@{generation}"
        binding = _binding_authority(generation=generation, identity=identity)
        await _create_session(store, session_id=session_id)
        case_root = root / name
        case_root.mkdir()
        proxy = _ConcurrentInitialLoadProxy(store)
        source = _workspace(case_root, proxy, resolver=lambda: binding)
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-tests",
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            branch_id=f"branch-{name}",
            idempotency_key=f"create-{name}",
            authority=WorkspaceBranchAuthority(
                session_id=session_id,
                expected_run_epoch=0,
                environment_name="local",
                binding_generation=generation,
                binding_identity=identity,
                creating_authority="conformance",
                resource_policy="local-cow-defaults-v1",
            ),
        )
        competing = (
            request.model_copy(update={"idempotency_key": f"conflicting-{name}"})
            if conflicting
            else request
        )
        outcomes = await asyncio.gather(
            source.create_branch(request),
            source.create_branch(competing),
            return_exceptions=True,
        )
        successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        assert len(successes) == (1 if conflicting else 2)
        assert len(failures) == (1 if conflicting else 0)
        if failures:
            assert isinstance(failures[0], WorkspaceBranchOperationConflict)
        assert all(result.status is WorkspaceBranchOutcomeStatus.CREATED for result in successes)
        assert all(result.branch is not None for result in successes)
        rollback_request = WorkspaceBranchRollbackRequest(
            branch_id=request.branch_id or "",
            idempotency_key=f"rollback-{name}",
            expected_run_epoch=0,
            binding_generation=generation,
        )
        for result in successes:
            assert result.branch is not None
            await result.branch.rollback(rollback_request)

    await concurrent_creation_case("exact-create-race", conflicting=False)
    await concurrent_creation_case("conflicting-create-race", conflicting=True)

    ledger_root, _source, ledger_branch, _request, _binding, _recovery = await create_case(
        "attempt-ledger"
    )
    await ledger_branch.write_bytes("candidate.txt", b"candidate")
    ledger_request = await _publication_request(
        ledger_branch,
        key="stable-ledger-key",
        binding_generation="binding-attempt-ledger",
    )
    (ledger_root / "candidate.txt").write_bytes(b"external")
    ledger_conflict = await ledger_branch.publish(ledger_request)
    assert ledger_conflict.status is WorkspaceBranchOutcomeStatus.CONFLICTED
    await ledger_branch.write_bytes("extra.txt", b"different material")
    with pytest.raises(WorkspaceBranchOperationConflict, match="reused"):
        await ledger_branch.publish(
            await _publication_request(
                ledger_branch,
                key="stable-ledger-key",
                binding_generation="binding-attempt-ledger",
            )
        )

    (
        failure_root,
        _source,
        failure_branch,
        _request,
        failure_binding,
        failure_recovery,
    ) = await create_case("publication-failure")
    await failure_branch.write_bytes("answer.txt", b"candidate")
    failure_publication = await _publication_request(
        failure_branch,
        key="failed-publication",
        binding_generation="binding-publication-failure",
    )
    import cayu.workspaces._local_branch as branch_module

    original_inspection = branch_module._inspect_source_path

    def fail_inspection(*_args, **_kwargs):
        raise OSError("source inspection unavailable")

    branch_module._inspect_source_path = fail_inspection
    try:
        publication_failure = await failure_branch.publish(failure_publication)
    finally:
        branch_module._inspect_source_path = original_inspection
    assert publication_failure.status is WorkspaceBranchOutcomeStatus.FAILED
    recovered_failure = await _workspace(
        failure_root,
        store,
        resolver=lambda: failure_binding,
    ).recover_branch(failure_recovery)
    assert recovered_failure.state is WorkspaceBranchDurableState.FAILED
    assert recovered_failure.evidence.change_set_digest == failure_publication.change_set_digest

    # Crash immediately after CREATING, before capture dispatch, is recoverable.
    pre_capture_proxy = _StoreFaultProxy(store)
    pre_capture_proxy.interrupt_after_publish_call = 1
    with pytest.raises(KeyboardInterrupt, match="durable creation intent"):
        await create_case("pre-capture-crash", branch_store=pre_capture_proxy)
    pre_capture_binding = _binding_authority(
        generation="binding-pre-capture-crash",
        identity="workspace-alpha@binding-pre-capture-crash",
    )
    pre_capture_recovery = _recovery_request(
        branch_id="branch-pre-capture-crash",
        session_id="branch-session-pre-capture-crash",
        binding_generation="binding-pre-capture-crash",
        binding_identity="workspace-alpha@binding-pre-capture-crash",
    )
    pre_capture_result = await _workspace(
        root / "pre-capture-crash",
        store,
        resolver=lambda: pre_capture_binding,
    ).recover_branch(pre_capture_recovery)
    assert pre_capture_result.state is WorkspaceBranchDurableState.OPEN
    assert pre_capture_result.branch is not None
    await pre_capture_result.branch.rollback(
        WorkspaceBranchRollbackRequest(
            branch_id="branch-pre-capture-crash",
            idempotency_key="rollback-pre-capture-crash",
            expected_run_epoch=0,
            binding_generation="binding-pre-capture-crash",
        )
    )

    # Crash after CREATING is durable but before OPEN is acknowledged.
    creating_proxy = _StoreFaultProxy(store)
    creating_proxy.fail_next_guarded_after_guard = True
    with pytest.raises(ConnectionError, match="after guarded mutation"):
        await create_case("creating-crash", branch_store=creating_proxy)
    creating_binding = _binding_authority(
        generation="binding-creating-crash",
        identity="workspace-alpha@binding-creating-crash",
    )
    creating_recovery = _recovery_request(
        branch_id="branch-creating-crash",
        session_id="branch-session-creating-crash",
        binding_generation="binding-creating-crash",
        binding_identity="workspace-alpha@binding-creating-crash",
    )
    creating_result = await _workspace(
        root / "creating-crash",
        store,
        resolver=lambda: creating_binding,
    ).recover_branch(creating_recovery)
    assert creating_result.state is WorkspaceBranchDurableState.OPEN

    # Publication intent, progress, and terminal acknowledgement boundaries all replay.
    intent_proxy = _StoreFaultProxy(store)
    (
        intent_root,
        _source,
        intent_branch,
        _request,
        intent_binding,
        intent_recovery,
    ) = await create_case("intent-crash", branch_store=intent_proxy)
    await intent_branch.write_bytes("answer.txt", b"after")
    intent_publication = await _publication_request(
        intent_branch,
        key="publish-intent-crash",
        binding_generation="binding-intent-crash",
    )
    intent_proxy.fail_next_guarded_before = True
    with pytest.raises(ConnectionError, match="before guarded"):
        await intent_branch.publish(intent_publication)
    intent_recoveries = await asyncio.gather(
        _workspace(
            intent_root,
            store,
            resolver=lambda: intent_binding,
        ).recover_branch(intent_recovery),
        _workspace(
            intent_root,
            store,
            resolver=lambda: intent_binding,
        ).recover_branch(intent_recovery.model_copy(update={"recovery_id": "competing-recovery"})),
    )
    assert {result.state for result in intent_recoveries} == {WorkspaceBranchDurableState.COMMITTED}
    assert (intent_root / "answer.txt").read_bytes() == b"after"

    progress_proxy = _StoreFaultProxy(store)
    (
        progress_root,
        _source,
        progress_branch,
        _request,
        progress_binding,
        progress_recovery,
    ) = await create_case("progress-crash", branch_store=progress_proxy)
    await progress_branch.write_bytes("answer.txt", b"after")
    progress_proxy.lose_next_guarded_acknowledgement = True
    with pytest.raises(ConnectionError, match="acknowledgement"):
        await progress_branch.publish(
            await _publication_request(
                progress_branch,
                key="publish-progress-crash",
                binding_generation="binding-progress-crash",
            )
        )
    progress_recovered = await _workspace(
        progress_root,
        store,
        resolver=lambda: progress_binding,
    ).recover_branch(progress_recovery)
    assert progress_recovered.state is WorkspaceBranchDurableState.COMMITTED

    terminal_proxy = _StoreFaultProxy(store)
    terminal_root, _source, terminal_branch, _request, _binding, _recovery = await create_case(
        "terminal-ack",
        branch_store=terminal_proxy,
    )
    await terminal_branch.write_bytes("answer.txt", b"after")
    terminal_proxy.lose_publish_acknowledgement_call = 4
    terminal_commit = await terminal_branch.publish(
        await _publication_request(
            terminal_branch,
            key="publish-terminal-ack",
            binding_generation="binding-terminal-ack",
        )
    )
    assert terminal_commit.status is WorkspaceBranchOutcomeStatus.COMMITTED
    assert (terminal_root / "answer.txt").read_bytes() == b"after"

    # Recovery repairs a provable partial application and refuses an unknown one.
    partial_proxy = _StoreFaultProxy(store)
    (
        partial_root,
        _source,
        partial_branch,
        _request,
        partial_binding,
        partial_recovery,
    ) = await create_case("partial", branch_store=partial_proxy)
    await partial_branch.write_bytes("a.txt", b"a")
    await partial_branch.write_bytes("b.txt", b"b")
    partial_proxy.fail_next_guarded_before = True
    with pytest.raises(ConnectionError):
        await partial_branch.publish(
            await _publication_request(
                partial_branch,
                key="publish-partial",
                binding_generation="binding-partial",
            )
        )
    (partial_root / "a.txt").write_bytes(b"a")
    partial_result = await _workspace(
        partial_root,
        store,
        resolver=lambda: partial_binding,
    ).recover_branch(partial_recovery)
    assert partial_result.state is WorkspaceBranchDurableState.COMMITTED
    assert (partial_root / "b.txt").read_bytes() == b"b"

    ambiguous_proxy = _StoreFaultProxy(store)
    (
        ambiguous_root,
        _source,
        ambiguous_branch,
        _request,
        ambiguous_binding,
        ambiguous_recovery,
    ) = await create_case("ambiguous", branch_store=ambiguous_proxy)
    await ambiguous_branch.write_bytes("a.txt", b"a")
    await ambiguous_branch.write_bytes("b.txt", b"b")
    ambiguous_proxy.fail_next_guarded_before = True
    with pytest.raises(ConnectionError):
        await ambiguous_branch.publish(
            await _publication_request(
                ambiguous_branch,
                key="publish-ambiguous",
                binding_generation="binding-ambiguous",
            )
        )
    (ambiguous_root / "a.txt").write_bytes(b"a")
    (ambiguous_root / "b.txt").write_bytes(b"unknown")
    ambiguous_result = await _workspace(
        ambiguous_root,
        store,
        resolver=lambda: ambiguous_binding,
    ).recover_branch(ambiguous_recovery)
    assert ambiguous_result.state is WorkspaceBranchDurableState.AMBIGUOUS

    inspection_proxy = _StoreFaultProxy(store)
    (
        inspection_root,
        _source,
        inspection_branch,
        _request,
        inspection_binding,
        inspection_recovery,
    ) = await create_case("partial-inspection", branch_store=inspection_proxy)
    await inspection_branch.write_bytes("a.txt", b"a")
    await inspection_branch.write_bytes("b.txt", b"b")
    inspection_proxy.fail_next_guarded_before = True
    with pytest.raises(ConnectionError):
        await inspection_branch.publish(
            await _publication_request(
                inspection_branch,
                key="publish-partial-inspection",
                binding_generation="binding-partial-inspection",
            )
        )
    (inspection_root / "a.txt").write_bytes(b"a")
    original_inspection = branch_module._inspect_source_path

    def fail_partial_inspection(source_root, path, **kwargs):
        if path == "a.txt":
            raise OSError("partial path cannot be inspected")
        return original_inspection(source_root, path, **kwargs)

    branch_module._inspect_source_path = fail_partial_inspection
    try:
        inspection_result = await _workspace(
            inspection_root,
            store,
            resolver=lambda: inspection_binding,
        ).recover_branch(inspection_recovery)
    finally:
        branch_module._inspect_source_path = original_inspection
    assert inspection_result.state is WorkspaceBranchDurableState.AMBIGUOUS
    assert tuple(inspection_root.parent.glob(".cayu-workspace-branch-*"))

    # Rollback intent survives a crash; competing terminal owners choose one winner.
    rollback_proxy = _StoreFaultProxy(store)
    (
        rollback_root,
        _source,
        rollback_branch,
        _request,
        rollback_binding,
        rollback_recovery,
    ) = await create_case("rollback-crash", branch_store=rollback_proxy)
    await rollback_branch.write_bytes("discarded.txt", b"private")
    rollback_proxy.fail_publish_before_call = 3
    with pytest.raises(ConnectionError, match="before durable"):
        await rollback_branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=rollback_branch.branch_id,
                idempotency_key="rollback-crash",
                expected_run_epoch=0,
                binding_generation="binding-rollback-crash",
            )
        )
    rollback_result = await _workspace(
        rollback_root,
        store,
        resolver=lambda: rollback_binding,
    ).recover_branch(rollback_recovery)
    assert rollback_result.state is WorkspaceBranchDurableState.ROLLED_BACK

    (
        concurrent_root,
        _source,
        concurrent_branch,
        _request,
        concurrent_binding,
        concurrent_recovery,
    ) = await create_case("concurrent")
    await concurrent_branch.write_bytes("winner.txt", b"candidate")
    second = await _workspace(
        concurrent_root,
        store,
        resolver=lambda: concurrent_binding,
    ).recover_branch(concurrent_recovery)
    assert second.branch is not None
    concurrent_publication = await _publication_request(
        concurrent_branch,
        key="publish-concurrent",
        binding_generation="binding-concurrent",
    )
    outcomes = await asyncio.gather(
        concurrent_branch.publish(concurrent_publication),
        second.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=second.branch.branch_id,
                idempotency_key="rollback-concurrent",
                expected_run_epoch=0,
                binding_generation="binding-concurrent",
            )
        ),
        return_exceptions=True,
    )
    assert len([outcome for outcome in outcomes if not isinstance(outcome, BaseException)]) == 1

    # Expiry, stale epochs, and corrupt evidence fail closed on every backend.
    expiry_root, _source, _branch, _request, expiry_binding, expiry_recovery = await create_case(
        "expiry",
        limits=WorkspaceBranchLimits(lifetime_ms=200),
    )
    await asyncio.sleep(0.25)
    expiry_results = await asyncio.gather(
        _workspace(
            expiry_root,
            store,
            resolver=lambda: expiry_binding,
        ).recover_branch(expiry_recovery),
        _workspace(
            expiry_root,
            store,
            resolver=lambda: expiry_binding,
        ).recover_branch(expiry_recovery.model_copy(update={"recovery_id": "competing-expiry"})),
    )
    assert {result.state for result in expiry_results} == {WorkspaceBranchDurableState.EXPIRED}

    stale_root, _source, stale_branch, _request, _binding, _recovery = await create_case("stale")
    await store.transition_status(
        "branch-session-stale",
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )
    with pytest.raises(SessionRunFenced):
        await stale_branch.write_bytes("stale.txt", b"must not land")
    assert not (stale_root / "stale.txt").exists()

    await _create_session(store, session_id="branch-session-failed")
    failed_root = root / "failed"
    failed_root.mkdir()
    failed_binding = _binding_authority(
        generation="binding-failed",
        identity="workspace-alpha@binding-failed",
    )
    failed_source = _workspace(failed_root, store, resolver=lambda: failed_binding)
    failed_baseline = await observe_deterministic_workspace(
        failed_source,
        observer="durable-branch-tests",
        limits=WorkspaceRevisionObservationLimits(),
    )
    (failed_root / "unsafe-link").symlink_to("missing")
    failed_creation = await failed_source.create_branch(
        WorkspaceBranchRequest(
            baseline=failed_baseline,
            branch_id="branch-failed",
            idempotency_key="create-failed",
            authority=WorkspaceBranchAuthority(
                session_id="branch-session-failed",
                expected_run_epoch=0,
                environment_name="local",
                binding_generation="binding-failed",
                binding_identity="workspace-alpha@binding-failed",
                creating_authority="conformance",
                resource_policy="local-cow-defaults-v1",
            ),
        )
    )
    assert failed_creation.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
    failed_recovery = await failed_source.recover_branch(
        _recovery_request(
            branch_id="branch-failed",
            session_id="branch-session-failed",
            binding_generation="binding-failed",
            binding_identity="workspace-alpha@binding-failed",
        )
    )
    assert failed_recovery.state is WorkspaceBranchDurableState.FAILED
    assert failed_recovery.evidence.outcome is WorkspaceBranchOutcomeStatus.UNSUPPORTED

    corrupt_root, _source, _branch, _request, corrupt_binding, corrupt_recovery = await create_case(
        "corrupt"
    )
    corrupt_proxy = _StoreFaultProxy(store)
    corrupt_proxy.corrupt_branch_record = True
    with pytest.raises(WorkspaceBranchFencedError, match="digest"):
        await _workspace(
            corrupt_root,
            corrupt_proxy,
            resolver=lambda: corrupt_binding,
        ).recover_branch(corrupt_recovery)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_durable_workspace_branch_store_conformance(
    tmp_path: Path,
    store_kind: str,
) -> None:
    store = (
        InMemorySessionStore()
        if store_kind == "memory"
        else SQLiteSessionStore(tmp_path / "conformance.sqlite3")
    )
    asyncio.run(
        assert_durable_workspace_branch_store_conformance(
            store,
            tmp_path / "conformance",
        )
    )
