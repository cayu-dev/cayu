from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.core._session_operation_fault_harness import (
    CommitEvidence,
    CommitThenRaise,
    PublicationFaultActionKind,
    SessionOperationFaultHarness,
    SessionOperationFaultRule,
    SessionOperationSelector,
)

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ExecCommandTool,
    InMemorySessionStore,
    ListArtifactsTool,
    LocalArtifactStore,
    LocalRunner,
    LocalWorkspace,
    MaterializeSharedArtifactTool,
    Message,
    PublishWorkspaceArtifactTool,
    ReadFileTool,
    RunRequest,
    SessionExecutionSource,
    SessionIdentity,
    SessionStatus,
    SharedArtifactAuthorizationError,
    SharedArtifactMaterializationReceipt,
    SharedArtifactPolicy,
    SharedArtifactPublicationReceipt,
    SharedArtifactRef,
    SQLiteSessionStore,
    ToolContext,
    ToolResult,
    authorize_shared_artifact_materialization,
    revoke_shared_artifact_grant,
)
from cayu.core.events import event_payload_authority_is_runtime_generated
from cayu.core.tools import (
    DurableToolOperationConflict,
    DurableToolRecoveryAuthority,
    _bind_runtime_tool_invocation_authority,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import _shared_artifact_results as shared_artifact_results
from cayu.runtime._tool_round_executor import _project_staged_terminal_event
from cayu.runtime.sessions import (
    SessionOperationPublication,
    SessionStore,
    run_request_with_runtime_invocation,
)
from cayu.tools import shared_artifacts as shared_artifact_tools
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.vaults import SecretRedactor

_NOW = datetime(2026, 8, 29, 12, tzinfo=UTC)
_PROFILE_FINGERPRINT = "e" * 64


class _CommitThenBlockArtifactStore(LocalArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root, store_id="shared-store")
        self.committed = asyncio.Event()
        self.release = asyncio.Event()

    async def put_bytes(self, *args: Any, **kwargs: Any):
        artifact = await super().put_bytes(*args, **kwargs)
        self.committed.set()
        await self.release.wait()
        return artifact


class _CorruptingArtifactStore(LocalArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root, store_id="shared-store")
        self.corrupt_reads = False

    async def read_bytes(self, *args: Any, **kwargs: Any):
        result = await super().read_bytes(*args, **kwargs)
        if not self.corrupt_reads or not result.content:
            return result
        first = bytes([result.content[0] ^ 1])
        return replace(result, content=first + result.content[1:])


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _policy(**updates: Any) -> SharedArtifactPolicy:
    values: dict[str, Any] = {
        "publish_path_prefixes": ("handoff",),
        "materialize_path_prefixes": ("received",),
        "allowed_content_types": ("text/plain", "text/x-python"),
        "max_bytes": 1024,
        "max_publications_per_session": 4,
        "grant_ttl_seconds": 3600,
        "max_lineage_depth": 4,
    }
    values.update(updates)
    return SharedArtifactPolicy(**values)


async def _running_session(
    store: SessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
    source: SessionExecutionSource | None = None,
):
    parent = None if parent_session_id is None else await store.load(parent_session_id)
    request = RunRequest(
        agent_name="assistant",
        session_id=session_id,
        parent_session_id=parent_session_id,
        causal_budget_id=None if parent is None else parent.causal_budget_id,
        messages=[],
    )
    if source is not None:
        request = run_request_with_runtime_invocation(request, source=source)
    created = await store.create(request, identity=_identity())
    return await store.transition_status(
        created.id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )


def _workspace(root: Path, *, workspace_id: str) -> LocalWorkspace:
    root.mkdir(parents=True, exist_ok=True)
    return LocalWorkspace(root, workspace_id=workspace_id)


def _lineage(session: Any) -> dict[str, Any]:
    return {
        "session_id": session.id,
        "session_instance_id": session.instance_id,
        "parent_session_id": session.parent_session_id,
        "causal_budget_id": session.causal_budget_id,
        "invocation": session.invocation.model_dump(mode="json"),
    }


def _bound_context(
    *,
    session: Any,
    session_store: SessionStore,
    workspace: LocalWorkspace,
    artifact_store: LocalArtifactStore,
    tool_name: str,
    args: dict[str, Any],
    tool_call_id: str,
    authorize: bool = True,
    secret_redactor: SecretRedactor | None = None,
    secret_snapshot_provider: Callable[[], InvocationRedactorSnapshot] | None = None,
    seal_durable_output: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> ToolContext:
    idempotency_key = f"tool-key-{tool_call_id}"
    ctx = ToolContext(
        session_id=session.id,
        agent_name="assistant",
        environment_name="shared-artifact-test",
        causal_budget_id=session.causal_budget_id,
        workspace_id=workspace.id,
        artifact_store_id=artifact_store.id,
        workspace=workspace,
        artifact_store=cast("Any", artifact_store),
        idempotency_key=idempotency_key,
        invocation_secret_redactor=(None if secret_redactor is None else lambda: secret_redactor),
        invocation_secret_snapshot_provider=secret_snapshot_provider,
    )
    ctx._bind_runtime_resource_authorities(
        workspace=workspace,
        artifact_store=artifact_store,
    )

    async def load(key: str) -> dict[str, Any] | None:
        return await session_store.load_session_operation(session.id, key)

    async def compare_and_set(
        key: str,
        expected: dict[str, Any] | None,
        desired: dict[str, Any],
        secondary: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any]:
        desired_copy = json.loads(json.dumps(desired))
        secondary_copy = json.loads(json.dumps(dict(secondary)))

        def publish(current_session, checkpoint, current):
            if current_session.run_epoch != session.run_epoch or current != expected:
                raise DurableToolOperationConflict("shared-artifact CAS lost")
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: desired_copy, **secondary_copy},
            )

        await session_store.publish_session_operation(
            session.id,
            idempotency_key=key,
            operation_transform=publish,
            events=[],
            expected_statuses={SessionStatus.RUNNING},
            expected_run_epoch=session.run_epoch,
        )
        return desired_copy

    async def authorize_reference(
        reference: dict[str, Any],
        policy_fingerprint: str,
        observed_at: str,
    ) -> dict[str, Any]:
        return await authorize_shared_artifact_materialization(
            session_store=session_store,
            caller_session_id=session.id,
            caller_session_instance_id=session.instance_id,
            reference=reference,
            policy_fingerprint=policy_fingerprint,
            observed_at=observed_at,
        )

    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id=None,
        parent_run_epoch=session.run_epoch,
        model_step_id="model-step-1",
        model_attempt_id="model-attempt-1",
        tool_round_id="tool-round-1",
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        idempotency_key=idempotency_key,
        effective_arguments=args,
        execution_profile_fingerprint=_PROFILE_FINGERPRINT,
        environment_allocation_fingerprint="a" * 64,
        current_session_lineage=_lineage(session),
        load_durable_operation=load,
        authorize_shared_artifact=authorize_reference if authorize else None,
        compare_and_set_durable_operation=compare_and_set,
        seal_durable_output=(
            (lambda value: json.loads(json.dumps(value)))
            if seal_durable_output is None
            else seal_durable_output
        ),
        secret_publication_sealer=lambda: None,
    )
    return ctx


async def _publish(
    *,
    session: Any,
    session_store: SessionStore,
    workspace: LocalWorkspace,
    artifact_store: LocalArtifactStore,
    policy: SharedArtifactPolicy,
    path: str = "handoff/solver.py",
    tool_call_id: str = "publish-1",
    clock: datetime = _NOW,
    secret_redactor: SecretRedactor | None = None,
    secret_snapshot_provider: Callable[[], InvocationRedactorSnapshot] | None = None,
    seal_durable_output: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
):
    args = {"path": path}
    ctx = _bound_context(
        session=session,
        session_store=session_store,
        workspace=workspace,
        artifact_store=artifact_store,
        tool_name="publish_workspace_artifact",
        args=args,
        tool_call_id=tool_call_id,
        secret_redactor=secret_redactor,
        secret_snapshot_provider=secret_snapshot_provider,
        seal_durable_output=seal_durable_output,
    )
    return await PublishWorkspaceArtifactTool(policy, clock=lambda: clock).run(ctx, args)


async def _materialize(
    *,
    session: Any,
    session_store: SessionStore,
    workspace: LocalWorkspace,
    artifact_store: LocalArtifactStore,
    policy: SharedArtifactPolicy,
    opaque_ref: str,
    destination: str = "received/solver.py",
    tool_call_id: str = "materialize-1",
    clock: datetime = _NOW + timedelta(minutes=1),
    secret_redactor: SecretRedactor | None = None,
    secret_snapshot_provider: Callable[[], InvocationRedactorSnapshot] | None = None,
    seal_durable_output: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
):
    args = {"ref": opaque_ref, "destination": destination}
    ctx = _bound_context(
        session=session,
        session_store=session_store,
        workspace=workspace,
        artifact_store=artifact_store,
        tool_name="materialize_shared_artifact",
        args=args,
        tool_call_id=tool_call_id,
        secret_redactor=secret_redactor,
        secret_snapshot_provider=secret_snapshot_provider,
        seal_durable_output=seal_durable_output,
    )
    return await MaterializeSharedArtifactTool(policy, clock=lambda: clock).run(ctx, args)


def test_shared_artifact_reference_is_canonical_and_possession_is_not_authority() -> None:
    reference = SharedArtifactRef(
        artifact_store_id="artifact-store",
        artifact_id="art_" + "1" * 32,
        content_digest="sha256:" + "2" * 64,
        size_bytes=12,
        source_session_id="parent",
        access_grant_id="sag_" + "3" * 32,
    )

    opaque = reference.to_opaque_ref()

    assert SharedArtifactRef.from_opaque_ref(opaque) == reference
    assert "artifact-store" not in opaque
    assert "parent" not in opaque
    with pytest.raises(ValueError, match="malformed|canonical"):
        SharedArtifactRef.from_opaque_ref(opaque + "=")
    with pytest.raises((ValueError, ValidationError)):
        SharedArtifactRef.from_opaque_ref("cayu-shared-artifact-v1.e30")


def test_policy_is_sealed_bounded_and_fingerprinted() -> None:
    first = _policy()
    second = _policy(publish_path_prefixes=("handoff/./",))

    assert first.fingerprint == first.model_copy().fingerprint
    assert second.publish_path_prefixes == ("handoff",)
    assert first.fingerprint == second.fingerprint
    with pytest.raises(ValidationError):
        _policy(max_bytes=0)
    with pytest.raises(ValidationError):
        _policy(publish_path_prefixes=("../escape",))
    with pytest.raises(ValidationError):
        _policy(allowed_content_types=())


def test_secret_redactor_detects_exact_raw_byte_occurrences_without_treating_marker_as_secret() -> (
    None
):
    redactor = SecretRedactor(["credential-value", "[REDACTED_SECRET]"])

    assert redactor.contains_secret_bytes(b"prefix credential-value suffix") is True
    assert redactor.contains_secret_bytes(b"prefix [REDACTED_SECRET] suffix") is False
    assert SecretRedactor().contains_secret_bytes(b"credential-value") is False
    with pytest.raises(TypeError, match="expects bytes"):
        redactor.contains_secret_bytes(cast("Any", "credential-value"))


def test_handoff_refuses_live_invocation_secrets_and_non_exact_durable_sealing(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        policy = _policy()
        await parent_workspace.write_bytes(
            "handoff/secret.txt",
            b"prefix credential-value suffix",
        )

        source_secret = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/secret.txt",
            tool_call_id="source-secret",
            secret_redactor=SecretRedactor("credential-value"),
        )
        assert source_secret.structured == {"error": "source_contains_secret"}
        assert (await artifact_store.list(session_id=parent.id)).artifacts == ()

        await parent_workspace.write_bytes("handoff/late-secret.txt", b"late-secret")
        late_revision = 0

        def late_secret_snapshot() -> InvocationRedactorSnapshot:
            nonlocal late_revision
            late_revision += 1
            return InvocationRedactorSnapshot(
                revision=late_revision,
                redactor=SecretRedactor("late-secret") if late_revision >= 2 else SecretRedactor(),
            )

        late_secret = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/late-secret.txt",
            tool_call_id="late-secret",
            secret_snapshot_provider=late_secret_snapshot,
        )
        assert late_secret.structured == {"error": "source_contains_secret"}

        await parent_workspace.write_bytes("handoff/unstable.txt", b"safe")
        unstable_revision = 0

        def unstable_snapshot() -> InvocationRedactorSnapshot:
            nonlocal unstable_revision
            unstable_revision += 1
            return InvocationRedactorSnapshot(
                revision=unstable_revision,
                redactor=SecretRedactor(),
            )

        unstable = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/unstable.txt",
            tool_call_id="unstable-secret-scope",
            secret_snapshot_provider=unstable_snapshot,
        )
        assert unstable.structured == {"error": "secret_redaction_scope_unstable"}

        await parent_workspace.write_bytes("handoff/safe.txt", b"child-only-secret")
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/safe.txt",
            tool_call_id="safe-parent",
        )
        assert published.is_error is False
        child_secret = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            destination="received/secret.txt",
            tool_call_id="child-secret",
            secret_redactor=SecretRedactor("child-only-secret"),
        )
        assert child_secret.structured == {"error": "artifact_contains_secret"}
        with pytest.raises(FileNotFoundError):
            await child_workspace.read_bytes("received/secret.txt", max_bytes=1024)

        await parent_workspace.write_bytes("handoff/identity.txt", b"safe")

        def redact_parent_identity(value: dict[str, Any]) -> dict[str, Any]:
            redacted = SecretRedactor(parent.id).redact_json_values(value)
            assert type(redacted) is dict
            return redacted

        non_exact = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/identity.txt",
            tool_call_id="non-exact-seal",
            seal_durable_output=redact_parent_identity,
        )
        assert non_exact.structured == {"error": "durable_identity_contains_secret"}

    asyncio.run(exercise())


def test_shared_artifact_result_authority_survives_only_owned_reconstruction() -> None:
    secret_session_id = "parent-secret-canary"
    reference = SharedArtifactRef(
        artifact_store_id="artifact-store",
        artifact_id="art_" + "1" * 32,
        content_digest="sha256:" + "2" * 64,
        size_bytes=12,
        source_session_id=secret_session_id,
        access_grant_id="sag_" + "3" * 32,
    )
    receipt = SharedArtifactPublicationReceipt(
        operation_id="sao_" + "3" * 32,
        reference=reference,
        source_workspace_id="parent-workspace",
        source_path_sha256="4" * 64,
        content_type="text/plain",
        policy_fingerprint="5" * 64,
        retention_class="lineage_handoff",
        published_at=_NOW,
    )
    result = ToolResult(
        content=reference.to_opaque_ref(),
        structured={
            "shared_artifact_kind": "publication",
            "opaque_ref": reference.to_opaque_ref(),
            "shared_artifact_ref": reference.model_dump(mode="json"),
            "publication_receipt": receipt.model_dump(mode="json"),
            "recovered_from_durable_receipt": False,
        },
    )
    terminal = shared_artifact_results.attest_runtime_shared_artifact_result(
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id=secret_session_id,
            tool_name="publish_workspace_artifact",
            payload={"result": result.model_dump(mode="json")},
        ),
        result,
        tool=PublishWorkspaceArtifactTool(_policy()),
    )
    marker = terminal.payload[shared_artifact_results.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD]
    assert event_payload_authority_is_runtime_generated(
        terminal,
        field_name=shared_artifact_results.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD,
        value=marker,
    )

    persisted = Event.model_validate(terminal.model_dump(mode="json"))
    redactor = SecretRedactor(secret_session_id)
    trusted = _project_staged_terminal_event(
        persisted,
        redactor=redactor,
        trust_persisted_tool_result_authority=True,
    )
    untrusted = _project_staged_terminal_event(persisted, redactor=redactor)

    assert (
        trusted.payload["result"]["structured"]["shared_artifact_ref"]["source_session_id"]
        == secret_session_id
    )
    assert (
        untrusted.payload["result"]["structured"]["shared_artifact_ref"]["source_session_id"]
        == "[REDACTED_SECRET]"
    )
    assert shared_artifact_results.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD not in untrusted.payload

    tampered_payload = json.loads(json.dumps(persisted.payload))
    tampered_payload["result"]["structured"]["shared_artifact_ref"]["source_session_id"] = (
        "attacker"
    )
    with pytest.raises(ValueError, match="authority is malformed"):
        _project_staged_terminal_event(
            persisted.model_copy(update={"payload": tampered_payload}),
            redactor=redactor,
            trust_persisted_tool_result_authority=True,
        )


@pytest.mark.parametrize(
    "source",
    [SessionExecutionSource.SUBAGENT, SessionExecutionSource.FORK],
)
def test_parent_can_publish_and_explicit_fork_or_subagent_child_can_materialize(
    tmp_path: Path,
    source: SessionExecutionSource,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=source,
        )
        assert child.causal_budget_id == parent.causal_budget_id
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        script = b"print('shared child capability')\n"
        await parent_workspace.write_bytes("handoff/solver.py", script)
        policy = _policy()

        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=cast("Any", artifact_store),
            policy=policy,
        )
        materialized = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=cast("Any", artifact_store),
            policy=policy,
            opaque_ref=published.content,
        )

        assert published.is_error is False
        assert published.structured["shared_artifact_kind"] == "publication"
        assert materialized.is_error is False
        assert materialized.structured["shared_artifact_kind"] == "materialization"
        assert materialized.structured["materialization_receipt"]["bytes_written"] == len(script)
        copied = await child_workspace.read_bytes("received/solver.py", max_bytes=1024)
        assert copied.content == script

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("caller_source", "parented"),
    [
        (SessionExecutionSource.SDK_RUN, False),
        (SessionExecutionSource.SDK_RUN, True),
        (SessionExecutionSource.TASK, True),
        (SessionExecutionSource.WORKFLOW_STEP, True),
    ],
)
def test_reference_possession_does_not_authorize_unrelated_or_unsupported_child(
    tmp_path: Path,
    caller_source: SessionExecutionSource,
    parented: bool,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        if parented:
            caller = await _running_session(
                store,
                "caller",
                parent_session_id=parent.id,
                source=caller_source,
            )
        else:
            caller = await _running_session(store, "caller")
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        source_workspace = _workspace(tmp_path / "source", workspace_id="source-workspace")
        caller_workspace = _workspace(tmp_path / "caller", workspace_id="caller-workspace")
        await source_workspace.write_bytes("handoff/data.txt", b"sealed")
        policy = _policy()
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=source_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
        )

        result = await _materialize(
            session=caller,
            session_store=store,
            workspace=caller_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
        )

        assert result.is_error is True
        assert result.structured == {"error": "authorization_denied"}
        with pytest.raises(FileNotFoundError):
            await caller_workspace.read_bytes("received/solver.py", max_bytes=1024)

    asyncio.run(exercise())


def test_shared_publication_does_not_widen_child_read_or_list_artifact_scope(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"not a raw child artifact")
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=_policy(),
            path="handoff/data.txt",
        )
        reference = SharedArtifactRef.from_opaque_ref(published.content)
        child_context = ToolContext(
            session_id=child.id,
            environment_name="shared-artifact-test",
            artifact_store_id=artifact_store.id,
            artifact_store=cast("Any", artifact_store),
        )

        read = await ReadFileTool().run(
            child_context,
            {"artifact_id": reference.artifact_id},
        )
        listed = await ListArtifactsTool().run(child_context, {"scope": "session"})

        assert read.is_error is True
        assert read.content == "Artifact is not available in this session."
        assert reference.artifact_id not in listed.content
        assert listed.structured is not None
        assert listed.structured["artifacts"] == []

    asyncio.run(exercise())


def test_revocation_expiry_policy_and_store_identity_fail_closed(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"durable data")
        policy = _policy(grant_ttl_seconds=60)
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
        )

        expired_workspace = _workspace(tmp_path / "expired", workspace_id="expired-workspace")
        expired = await _materialize(
            session=child,
            session_store=store,
            workspace=expired_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            clock=_NOW + timedelta(seconds=60),
            tool_call_id="expired",
        )
        assert expired.structured == {"error": "authorization_denied"}

        wrong_policy_workspace = _workspace(
            tmp_path / "wrong-policy", workspace_id="wrong-policy-workspace"
        )
        wrong_policy = await _materialize(
            session=child,
            session_store=store,
            workspace=wrong_policy_workspace,
            artifact_store=artifact_store,
            policy=_policy(max_bytes=512),
            opaque_ref=published.content,
            tool_call_id="wrong-policy",
        )
        assert wrong_policy.structured == {"error": "authorization_denied"}

        wrong_store = LocalArtifactStore(tmp_path / "other-artifacts", store_id="other-store")
        wrong_store_workspace = _workspace(
            tmp_path / "wrong-store", workspace_id="wrong-store-workspace"
        )
        store_mismatch = await _materialize(
            session=child,
            session_store=store,
            workspace=wrong_store_workspace,
            artifact_store=wrong_store,
            policy=policy,
            opaque_ref=published.content,
            tool_call_id="wrong-store",
        )
        assert store_mismatch.structured == {"error": "artifact_store_mismatch"}

        await revoke_shared_artifact_grant(
            store,
            published.content,
            reason="parent withdrew handoff",
            revoked_at=_NOW + timedelta(seconds=10),
        )
        revoked_workspace = _workspace(tmp_path / "revoked", workspace_id="revoked-workspace")
        revoked = await _materialize(
            session=child,
            session_store=store,
            workspace=revoked_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            clock=_NOW + timedelta(seconds=20),
            tool_call_id="revoked",
        )
        assert revoked.structured == {"error": "authorization_denied"}

    asyncio.run(exercise())


def test_path_content_size_count_and_overwrite_policies_are_enforced(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        await parent_workspace.write_bytes("outside.txt", b"outside")
        await parent_workspace.write_bytes("handoff/blocked.bin", b"binary")
        await parent_workspace.write_bytes("handoff/too-big.txt", b"x" * 9)
        await parent_workspace.write_bytes("handoff/one.txt", b"one")
        await parent_workspace.write_bytes("handoff/two.txt", b"two")
        tight = _policy(max_bytes=8, max_publications_per_session=1)

        outside = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=tight,
            path="outside.txt",
            tool_call_id="outside",
        )
        blocked = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=tight,
            path="handoff/blocked.bin",
            tool_call_id="blocked-content",
        )
        too_big = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=tight,
            path="handoff/too-big.txt",
            tool_call_id="too-big",
        )
        assert outside.structured == {"error": "path_not_allowed"}
        assert blocked.structured == {"error": "content_type_not_allowed"}
        assert too_big.structured == {"error": "source_file_oversize"}

        first = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=tight,
            path="handoff/one.txt",
            tool_call_id="first",
        )
        second = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=tight,
            path="handoff/two.txt",
            tool_call_id="second",
        )
        assert first.is_error is False
        assert second.structured == {"error": "publication_limit_reached"}

        traversal = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=tight,
            opaque_ref=first.content,
            destination="../escape.txt",
            tool_call_id="traversal",
        )
        disallowed = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=tight,
            opaque_ref=first.content,
            destination="other/file.txt",
            tool_call_id="other-path",
        )
        await child_workspace.write_bytes("received/existing.txt", b"existing")
        overwrite = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=tight,
            opaque_ref=first.content,
            destination="received/existing.txt",
            tool_call_id="overwrite",
        )
        assert traversal.structured == {"error": "invalid_arguments"}
        assert disallowed.structured == {"error": "path_not_allowed"}
        assert overwrite.structured == {"error": "overwrite_denied"}
        current = await child_workspace.read_bytes("received/existing.txt", max_bytes=32)
        assert current.content == b"existing"

    asyncio.run(exercise())


def test_directories_and_symlinks_fail_closed_at_both_workspace_boundaries(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_root = tmp_path / "parent"
        child_root = tmp_path / "child"
        parent_workspace = _workspace(parent_root, workspace_id="parent-workspace")
        child_workspace = _workspace(child_root, workspace_id="child-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"safe data")
        policy = _policy()
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
        )

        (parent_root / "handoff" / "directory").mkdir()
        (parent_root / "handoff" / "link.txt").symlink_to(parent_root / "handoff/data.txt")
        source_directory = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/directory",
            tool_call_id="source-directory",
        )
        source_symlink = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/link.txt",
            tool_call_id="source-symlink",
        )
        assert source_directory.structured == {"error": "source_file_invalid"}
        assert source_symlink.structured == {"error": "source_file_invalid"}

        (child_root / "received").mkdir()
        (child_root / "received" / "directory").mkdir()
        (child_root / "real.txt").write_bytes(b"unchanged")
        (child_root / "received" / "link.txt").symlink_to(child_root / "real.txt")
        destination_directory = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            destination="received/directory",
            tool_call_id="destination-directory",
        )
        destination_symlink = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            destination="received/link.txt",
            tool_call_id="destination-symlink",
        )
        assert destination_directory.structured == {"error": "destination_invalid"}
        assert destination_symlink.structured == {"error": "destination_invalid"}
        assert (child_root / "real.txt").read_bytes() == b"unchanged"

    asyncio.run(exercise())


def test_policy_can_explicitly_allow_revision_guarded_overwrite(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.FORK,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"replacement")
        await child_workspace.write_bytes("received/data.txt", b"old value")
        policy = _policy(allow_overwrite=True)
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
        )

        materialized = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            destination="received/data.txt",
        )

        assert materialized.is_error is False
        current = await child_workspace.read_bytes("received/data.txt", max_bytes=64)
        assert current.content == b"replacement"

    asyncio.run(exercise())


def test_duplicate_calls_rejoin_exact_receipts_without_duplicate_publication(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"stable")
        policy = _policy()

        first = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
            tool_call_id="publish-first",
        )
        second = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
            tool_call_id="publish-second",
            clock=_NOW + timedelta(minutes=2),
        )
        assert second.content == first.content
        assert second.structured["publication_receipt"] == first.structured["publication_receipt"]
        assert second.structured["recovered_from_durable_receipt"] is True

        materialized_first = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=first.content,
            tool_call_id="materialize-first",
        )
        materialized_second = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=first.content,
            tool_call_id="materialize-second",
            clock=_NOW + timedelta(minutes=3),
        )
        assert (
            materialized_second.structured["materialization_receipt"]
            == (materialized_first.structured["materialization_receipt"])
        )
        assert materialized_second.structured["recovered_from_durable_receipt"] is True

    asyncio.run(exercise())


def test_durable_reconciler_reattaches_terminal_receipts_without_replay(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"recoverable")
        policy = _policy()
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
            tool_call_id="recover-publication",
        )
        materialized = await _materialize(
            session=child,
            session_store=store,
            workspace=child_workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            tool_call_id="recover-materialization",
        )

        async def load_parent(key: str) -> dict[str, Any] | None:
            return await store.load_session_operation(parent.id, key)

        async def load_child(key: str) -> dict[str, Any] | None:
            return await store.load_session_operation(child.id, key)

        recovered_publication = await PublishWorkspaceArtifactTool(
            policy
        ).reconcile_durable_tool_call(
            parent_session_id=parent.id,
            parent_run_epoch=parent.run_epoch,
            execution_profile_fingerprint=_PROFILE_FINGERPRINT,
            environment_name="shared-artifact-test",
            environment_allocation_fingerprint="a" * 64,
            model_step_id="model-step-1",
            model_attempt_id="model-attempt-1",
            tool_round_id="tool-round-1",
            tool_call_id="recover-publication",
            idempotency_key="tool-key-recover-publication",
            arguments={"path": "handoff/data.txt"},
            started=True,
            load_operation=load_parent,
        )
        recovered_materialization = await MaterializeSharedArtifactTool(
            policy
        ).reconcile_durable_tool_call(
            parent_session_id=child.id,
            parent_run_epoch=child.run_epoch,
            execution_profile_fingerprint=_PROFILE_FINGERPRINT,
            environment_name="shared-artifact-test",
            environment_allocation_fingerprint="a" * 64,
            model_step_id="model-step-1",
            model_attempt_id="model-attempt-1",
            tool_round_id="tool-round-1",
            tool_call_id="recover-materialization",
            idempotency_key="tool-key-recover-materialization",
            arguments={"ref": published.content, "destination": "received/solver.py"},
            started=True,
            load_operation=load_child,
        )

        assert recovered_publication is not None
        assert recovered_publication.content == published.content
        assert recovered_publication.structured is not None
        assert recovered_publication.structured["recovered_from_durable_receipt"] is True
        assert recovered_materialization is not None
        assert recovered_materialization.structured is not None
        assert (
            recovered_materialization.structured["materialization_receipt"]
            == (materialized.structured["materialization_receipt"])
        )
        assert recovered_materialization.structured["recovered_from_durable_receipt"] is True

    asyncio.run(exercise())


def test_durable_reconciler_finalizes_a_proven_materialization_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"recoverable")
        policy = _policy()
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
        )

        original_finalize = shared_artifact_tools._finalize_materialization

        async def lose_process_after_workspace_effect(**kwargs: Any):
            del kwargs
            raise ConnectionError("process exited before receipt publication")

        monkeypatch.setattr(
            shared_artifact_tools,
            "_finalize_materialization",
            lose_process_after_workspace_effect,
        )
        with pytest.raises(ConnectionError, match="before receipt publication"):
            await _materialize(
                session=child,
                session_store=store,
                workspace=child_workspace,
                artifact_store=artifact_store,
                policy=policy,
                opaque_ref=published.content,
                tool_call_id="recover-preparation",
            )
        monkeypatch.setattr(
            shared_artifact_tools,
            "_finalize_materialization",
            original_finalize,
        )
        observed = await child_workspace.read_bytes("received/solver.py", max_bytes=128)
        assert observed.content == b"recoverable"

        async def load(key: str) -> dict[str, Any] | None:
            return await store.load_session_operation(child.id, key)

        async def compare_and_set(
            key: str,
            expected: dict[str, Any] | None,
            desired: dict[str, Any],
            secondary: Mapping[str, dict[str, Any]],
        ) -> dict[str, Any]:
            def publish(current_session, checkpoint, current):
                if current_session.run_epoch != child.run_epoch or current != expected:
                    raise DurableToolOperationConflict("recovery CAS lost")
                return SessionOperationPublication(
                    checkpoint={} if checkpoint is None else checkpoint,
                    operation_records={key: desired, **secondary},
                )

            await store.publish_session_operation(
                child.id,
                idempotency_key=key,
                operation_transform=publish,
                events=[],
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=child.run_epoch,
            )
            return desired

        recovered = await MaterializeSharedArtifactTool(policy).reconcile_durable_tool_call(
            parent_session_id=child.id,
            parent_run_epoch=child.run_epoch,
            execution_profile_fingerprint=_PROFILE_FINGERPRINT,
            environment_name="shared-artifact-test",
            environment_allocation_fingerprint="a" * 64,
            model_step_id="model-step-1",
            model_attempt_id="model-attempt-1",
            tool_round_id="tool-round-1",
            tool_call_id="recover-preparation",
            idempotency_key="tool-key-recover-preparation",
            arguments={"ref": published.content, "destination": "received/solver.py"},
            started=True,
            load_operation=load,
            recovery_authority=DurableToolRecoveryAuthority(
                agent_name="assistant",
                environment_name="shared-artifact-test",
                workspace=child_workspace,
                artifact_reader=artifact_store,
                compare_and_set_operation=compare_and_set,
            ),
        )

        assert recovered is not None
        assert recovered.is_error is False
        assert recovered.structured is not None
        receipt = recovered.structured["materialization_receipt"]
        assert receipt["record_type"] == "cayu.shared-artifact-materialization-receipt"
        persisted = await load(f"cayu.shared-artifact.materialization.v1:{receipt['operation_id']}")
        assert persisted == receipt

    asyncio.run(exercise())


@pytest.mark.parametrize("_iteration", range(8))
def test_concurrent_publication_and_materialization_converge(
    tmp_path: Path,
    _iteration: int,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"one durable value")
        policy = _policy()

        publications = await asyncio.gather(
            *(
                _publish(
                    session=parent,
                    session_store=store,
                    workspace=parent_workspace,
                    artifact_store=artifact_store,
                    policy=policy,
                    path="handoff/data.txt",
                    tool_call_id=f"publish-{index}",
                )
                for index in range(8)
            )
        )
        assert len({result.content for result in publications}) == 1
        assert (
            len(
                {
                    SharedArtifactPublicationReceipt.model_validate(
                        result.structured["publication_receipt"]
                    ).model_dump_json()
                    for result in publications
                }
            )
            == 1
        )

        materializations = await asyncio.gather(
            *(
                _materialize(
                    session=child,
                    session_store=store,
                    workspace=child_workspace,
                    artifact_store=artifact_store,
                    policy=policy,
                    opaque_ref=publications[0].content,
                    tool_call_id=f"materialize-{index}",
                )
                for index in range(8)
            )
        )
        assert all(result.is_error is False for result in materializations), [
            (result.content, result.structured) for result in materializations
        ]
        assert (
            len(
                {
                    SharedArtifactMaterializationReceipt.model_validate(
                        result.structured["materialization_receipt"]
                    ).model_dump_json()
                    for result in materializations
                }
            )
            == 1
        )
        copied = await child_workspace.read_bytes("received/solver.py", max_bytes=1024)
        assert copied.content == b"one durable value"

    asyncio.run(exercise())


def test_commit_then_cancel_settles_receipt_and_retry_rejoins(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        artifact_store = _CommitThenBlockArtifactStore(tmp_path / "artifacts")
        workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        await workspace.write_bytes("handoff/data.txt", b"settle before cancellation")
        policy = _policy()

        publication = asyncio.create_task(
            _publish(
                session=parent,
                session_store=store,
                workspace=workspace,
                artifact_store=artifact_store,
                policy=policy,
                path="handoff/data.txt",
                tool_call_id="cancelled",
            )
        )
        await artifact_store.committed.wait()
        publication.cancel()
        artifact_store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await publication

        retry = await _publish(
            session=parent,
            session_store=store,
            workspace=workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
            tool_call_id="retry",
        )
        assert retry.is_error is False
        assert retry.structured["recovered_from_durable_receipt"] is True

    asyncio.run(exercise())


def test_lost_commit_acknowledgements_rejoin_publication_and_materialization(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        child_workspace = _workspace(tmp_path / "child", workspace_id="child-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"acknowledged durably")
        policy = _policy()
        rules = (
            SessionOperationFaultRule(
                rule_id="shared-artifact-publish",
                selector=SessionOperationSelector(
                    session_id=parent.id,
                    label="shared-artifact-publish",
                ),
                actions=(CommitThenRaise(),),
            ),
            SessionOperationFaultRule(
                rule_id="shared-artifact-materialize",
                selector=SessionOperationSelector(
                    session_id=child.id,
                    label="shared-artifact-materialize",
                ),
                actions=(CommitThenRaise(),),
            ),
        )
        async with SessionOperationFaultHarness(store, rules=rules) as harness:
            with (
                harness.label("shared-artifact-publish"),
                pytest.raises(ConnectionError, match="acknowledgement"),
            ):
                await _publish(
                    session=parent,
                    session_store=store,
                    workspace=parent_workspace,
                    artifact_store=artifact_store,
                    policy=policy,
                    path="handoff/data.txt",
                    tool_call_id="lost-publish-ack",
                )
            published = await _publish(
                session=parent,
                session_store=store,
                workspace=parent_workspace,
                artifact_store=artifact_store,
                policy=policy,
                path="handoff/data.txt",
                tool_call_id="rejoin-publish",
            )
            assert published.structured["recovered_from_durable_receipt"] is True

            with (
                harness.label("shared-artifact-materialize"),
                pytest.raises(ConnectionError, match="acknowledgement"),
            ):
                await _materialize(
                    session=child,
                    session_store=store,
                    workspace=child_workspace,
                    artifact_store=artifact_store,
                    policy=policy,
                    opaque_ref=published.content,
                    tool_call_id="lost-materialize-ack",
                )
            materialized = await _materialize(
                session=child,
                session_store=store,
                workspace=child_workspace,
                artifact_store=artifact_store,
                policy=policy,
                opaque_ref=published.content,
                tool_call_id="rejoin-materialize",
            )
        assert materialized.structured["recovered_from_durable_receipt"] is True
        copied = await child_workspace.read_bytes("received/solver.py", max_bytes=1024)
        assert copied.content == b"acknowledged durably"
        injected = [record for record in harness.trace if record.matched_rule_id is not None]
        assert [record.matched_rule_id for record in injected] == [
            "shared-artifact-publish",
            "shared-artifact-materialize",
        ]
        assert all(
            record.action is PublicationFaultActionKind.COMMIT_THEN_RAISE
            and record.committed is CommitEvidence.YES
            and record.acknowledgement_returned is False
            for record in injected
        )

    asyncio.run(exercise())


def test_missing_and_corrupt_artifact_bytes_fail_closed(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = _CorruptingArtifactStore(tmp_path / "artifacts")
        parent_workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        await parent_workspace.write_bytes("handoff/data.txt", b"original")
        policy = _policy()
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=parent_workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
        )
        ref = SharedArtifactRef.from_opaque_ref(published.content)
        artifact_store.corrupt_reads = True
        corrupt = await _materialize(
            session=child,
            session_store=store,
            workspace=_workspace(tmp_path / "corrupt", workspace_id="corrupt-workspace"),
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            tool_call_id="corrupt",
        )
        assert corrupt.structured == {"error": "artifact_mismatch"}

        artifact_store.corrupt_reads = False
        await artifact_store.delete(ref.artifact_id)

        missing = await _materialize(
            session=child,
            session_store=store,
            workspace=_workspace(tmp_path / "missing", workspace_id="missing-workspace"),
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=published.content,
            tool_call_id="missing",
        )
        assert missing.structured == {"error": "artifact_missing"}

    asyncio.run(exercise())


def test_sqlite_and_local_artifacts_survive_parent_process_reconstruction(tmp_path: Path) -> None:
    database = tmp_path / "sessions.sqlite"
    artifact_root = tmp_path / "artifacts"
    parent_root = tmp_path / "parent"
    child_root = tmp_path / "child"
    policy = _policy()

    async def parent_process() -> str:
        store = SQLiteSessionStore(database)
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        assert child.parent_session_id == parent.id
        artifact_store = LocalArtifactStore(artifact_root, store_id="shared-store")
        workspace = _workspace(parent_root, workspace_id="parent-workspace")
        await workspace.write_bytes("handoff/solver.py", b"print(6 * 7)\n")
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=workspace,
            artifact_store=artifact_store,
            policy=policy,
        )
        await store.close()
        return published.content

    async def child_process(opaque_ref: str) -> None:
        store = SQLiteSessionStore(database)
        child = await store.load("child")
        assert child is not None
        artifact_store = LocalArtifactStore(artifact_root, store_id="shared-store")
        workspace = _workspace(child_root, workspace_id="child-workspace")
        result = await _materialize(
            session=child,
            session_store=store,
            workspace=workspace,
            artifact_store=artifact_store,
            policy=policy,
            opaque_ref=opaque_ref,
        )
        assert result.is_error is False
        copied = await workspace.read_bytes("received/solver.py", max_bytes=128)
        assert copied.content == b"print(6 * 7)\n"
        executed = await ExecCommandTool().run(
            ToolContext(
                session_id=child.id,
                workspace_id=workspace.id,
                runner=cast("Any", LocalRunner(child_root)),
            ),
            {
                "kind": "process",
                "argv": [sys.executable, "received/solver.py"],
                "timeout_s": 10,
            },
        )
        assert executed.is_error is False
        assert executed.structured is not None
        assert executed.structured["exit_code"] == 0
        assert executed.structured["stdout"] == "42\n"
        await store.close()

    opaque_ref = asyncio.run(parent_process())
    asyncio.run(child_process(opaque_ref))


def test_fresh_cayu_app_process_materializes_through_runtime_tool_authority(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime-sessions.sqlite"
    artifact_root = tmp_path / "runtime-artifacts"
    parent_root = tmp_path / "runtime-parent"
    child_root = tmp_path / "runtime-child"
    parent_workspace = _workspace(parent_root, workspace_id="parent-workspace")
    child_workspace = _workspace(child_root, workspace_id="child-workspace")
    artifact_store = LocalArtifactStore(artifact_root, store_id="runtime-shared-store")
    policy = _policy()
    asyncio.run(parent_workspace.write_bytes("handoff/solver.py", b"print(40 + 2)\n"))

    async def parent_process() -> tuple[str, str]:
        store = SQLiteSessionStore(database)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="publish-call",
                        name="publish_workspace_artifact",
                        arguments={"path": "handoff/solver.py"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Published for the child."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="parent-environment"),
                workspace=parent_workspace,
                artifact_store=artifact_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=(PublishWorkspaceArtifactTool(policy),),
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="runtime-parent",
                    causal_budget_id="runtime-shared-budget",
                    messages=[Message.text("user", "publish the solver")],
                )
            )
        ]
        terminal = next(
            event
            for event in events
            if event.type is EventType.TOOL_CALL_COMPLETED
            and event.tool_name == "publish_workspace_artifact"
        )
        opaque_ref = terminal.payload["result"]["content"]
        assert type(opaque_ref) is str
        parent = await store.load("runtime-parent")
        assert parent is not None
        await store.close()
        return opaque_ref, parent.causal_budget_id

    async def child_process(opaque_ref: str, causal_budget_id: str) -> None:
        store = SQLiteSessionStore(database)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="materialize-call",
                        name="materialize_shared_artifact",
                        arguments={
                            "ref": opaque_ref,
                            "destination": "received/solver.py",
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="execute-call",
                        name="exec_command",
                        arguments={
                            "kind": "process",
                            "argv": [sys.executable, "received/solver.py"],
                            "timeout_s": 10,
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Materialized and executed from the parent."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="child-environment"),
                workspace=child_workspace,
                artifact_store=LocalArtifactStore(
                    artifact_root,
                    store_id="runtime-shared-store",
                ),
                runner=LocalRunner(child_root),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=(MaterializeSharedArtifactTool(policy), ExecCommandTool()),
        )
        request = run_request_with_runtime_invocation(
            RunRequest(
                agent_name="assistant",
                session_id="runtime-child",
                parent_session_id="runtime-parent",
                causal_budget_id=causal_budget_id,
                messages=[Message.text("user", f"materialize {opaque_ref}")],
            ),
            source=SessionExecutionSource.SUBAGENT,
        )
        events = [event async for event in app.run(request)]
        assert events[-1].type is EventType.SESSION_COMPLETED
        terminal = next(
            event
            for event in events
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            and event.tool_name == "materialize_shared_artifact"
        )
        assert terminal.type is EventType.TOOL_CALL_COMPLETED, terminal.payload.get("result")
        assert terminal.payload["result"]["is_error"] is False
        executed = next(
            event
            for event in events
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            and event.tool_name == "exec_command"
        )
        assert executed.type is EventType.TOOL_CALL_COMPLETED, executed.payload.get("result")
        assert executed.payload["result"]["structured"]["exit_code"] == 0
        assert executed.payload["result"]["structured"]["stdout"] == "42\n"
        await store.close()

    opaque_ref, causal_budget_id = asyncio.run(parent_process())
    asyncio.run(child_process(opaque_ref, causal_budget_id))
    assert (child_root / "received/solver.py").read_bytes() == b"print(40 + 2)\n"


def test_authorizer_rejects_wrong_caller_instance_even_with_valid_lineage(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        parent = await _running_session(store, "parent")
        child = await _running_session(
            store,
            "child",
            parent_session_id=parent.id,
            source=SessionExecutionSource.SUBAGENT,
        )
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-store")
        workspace = _workspace(tmp_path / "parent", workspace_id="parent-workspace")
        await workspace.write_bytes("handoff/data.txt", b"data")
        policy = _policy()
        published = await _publish(
            session=parent,
            session_store=store,
            workspace=workspace,
            artifact_store=artifact_store,
            policy=policy,
            path="handoff/data.txt",
        )
        reference = SharedArtifactRef.from_opaque_ref(published.content)

        with pytest.raises(SharedArtifactAuthorizationError, match="denied"):
            await authorize_shared_artifact_materialization(
                session_store=store,
                caller_session_id=child.id,
                caller_session_instance_id="wrong-instance",
                reference=reference.model_dump(mode="json"),
                policy_fingerprint=policy.fingerprint,
                observed_at=(_NOW + timedelta(seconds=1)).isoformat(),
            )

    asyncio.run(exercise())
