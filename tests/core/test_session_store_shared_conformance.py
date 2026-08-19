from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import io
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core._workload_secret_support import FakeProvider

import cayu.runtime._model_step_executor as model_step_executor_module
import cayu.runtime._session_engine as session_engine_module
import cayu.runtime.sessions as sessions_module
from cayu import (
    SQLiteSessionStore,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
)
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    extract_durable_value_error,
)
from cayu.artifacts import LocalArtifactStore, file_attachment
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    MessageRole,
    ToolCallPart,
    ToolResultPart,
)
from cayu.core.billing import BillingIdentity
from cayu.core.events import event_with_runtime_payload_authority
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.environments import (
    BoundWorkspace,
    DeterministicWorkspaceBinding,
    Environment,
    EnvironmentSpec,
    WorkspaceBinding,
    WorkspaceSnapshot,
)
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ProviderOperationState,
    ProviderOperationStatus,
    UsageDialect,
    bedrock_billing_identity,
    completed_bedrock_billing_identity,
)
from cayu.runtime import (
    AllowAllToolPolicy,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    CompactSessionRequest,
    ContextCompactor,
    EnqueueSessionMessageRequest,
    EventQuery,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfileDecisionKind,
    ExecutionProfileMismatchError,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    ForkExecutionProfileSelection,
    ForkSessionRequest,
    ForkSystemPromptReplacement,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemoryBudgetStore,
    InMemorySessionStore,
    InteractionTransitionSpec,
    InterruptSessionRequest,
    InvocationOriginClaim,
    InvocationOriginTrust,
    McpManifestBaseline,
    ModelCompactor,
    ModelCompletionStageRequest,
    ModelTarget,
    PendingToolApproval,
    PendingToolApprovalEventView,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectStatus,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    RequestFootprintConfig,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunLimits,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    RuntimePublicationRequest,
    Session,
    SessionExecutionSource,
    SessionIdentity,
    SessionMessageDeliveryMode,
    SessionMessageQueueStatus,
    SessionModelCompletionStageConflict,
    SessionModelCompletionStageIncomplete,
    SessionOperationPublication,
    SessionOrder,
    SessionQuery,
    SessionQueuedMessage,
    SessionQueuedMessagesPending,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolRoundIdentity,
    TranscriptQuery,
    UsageRollupQuery,
    runtime_publication_checkpoint_mutation,
    runtime_publication_checkpoint_value_digest,
    runtime_publication_event_reference,
)
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_projection import (
    PRIVATE_EVENT_AUTHORITY,
    REDACTED_CUSTOM_EVENT_TYPE,
    public_event_id,
    public_event_sequence,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.runtime._recovery_coordinator import (
    _checkpoint_with_legacy_approval_round,
    _pending_approval_for_atomic_claim,
)
from cayu.runtime.aggregates import estimate_usage_rollup_cost
from cayu.runtime.approvals import PendingToolCallApproval
from cayu.runtime.budgets import (
    MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY,
    BudgetLimit,
    BudgetPolicy,
    BudgetReconciliation,
    BudgetReservation,
    budget_reconciliation_payload,
    budget_settlement_id,
)
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
)
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.event_sinks import EventSink, InMemoryEventSink
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.provider_operations import (
    ProviderOperationInspectionStatus,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionRequest,
    commit_provider_operation_progress,
    inspect_provider_operation,
    load_pending_provider_operation_disposition,
    load_provider_operation_resolution,
    load_recoverable_provider_operation,
    provider_operation_progress_event_id,
    provider_operation_progress_payload,
    resolve_provider_operation_stage,
)
from cayu.runtime.sessions import (
    MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES,
    PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES,
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    BudgetReservationIdentityConflict,
    PersistedEventSideEffectDelivery,
    QueuedDispatchTerminalReceiptQuery,
    _checkpoint_with_session_run_operation,
    _deactivate_session_run_fence,
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
    fork_session_invocation,
)
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)
from cayu.runtime.usage import UsageMetrics
from cayu.runtime.workspace_observation_recovery import (
    WorkspaceObservationArtifact,
    WorkspaceObservationArtifactState,
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    WorkspaceObservationTerminalStatus,
    _admit_workspace_observation_intent,
    publish_workspace_observation_transition,
    workspace_observation_event_digest,
    workspace_observations_from_checkpoint,
)
from cayu.storage.jsonl_export import export_sessions, import_sessions
from cayu.storage.migrations import SchemaMode
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault
from cayu.workspaces import LocalWorkspace

pytestmark = pytest.mark.postgres


def _admit_test_workspace_observation_intent(
    lifecycle: WorkspaceObservationLifecycle,
) -> Any:
    return _admit_workspace_observation_intent(
        lifecycle,
        redactor=SecretRedactor(),
        configured_workspace_id=(
            None if lifecycle.workspace_id == "workspace-unavailable" else lifecycle.workspace_id
        ),
        configured_artifact_store_id=lifecycle.artifact_store_id,
    )


_POSTGRES_TABLES = (
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_public_authority_alias_config",
    "cayu_task_terminalization_receipts",
    "cayu_knowledge_publication_receipts",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_embeddings",
    "cayu_knowledge_chunks",
    "cayu_knowledge_revisions",
    "cayu_knowledge_entries",
    "cayu_event_watcher_dead_letters",
    "cayu_event_watcher_state",
    "cayu_deferred_interaction_inputs",
    "cayu_interaction_latest_events",
    "cayu_session_message_deliveries",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_budget_settlements",
    "cayu_budget_reservations",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_eval_results",
    "cayu_eval_runs",
    "cayu_eval_cases",
    "cayu_eval_suites",
    "cayu_eval_corpora",
    "cayu_schema_migrations",
)
_POSTGRES_DATA_TABLES = (
    "cayu_public_authority_alias_config",
    *(
        table
        for table in _POSTGRES_TABLES
        if table not in {"cayu_public_authority_alias_config", "cayu_schema_migrations"}
    ),
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _public_authority_alias_codec() -> PublicAuthorityAliasCodec:
    encoded_key = base64.urlsafe_b64encode(bytes([23]) * 32).decode("ascii").rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="conformance",
            keys={"conformance": SecretStr(encoded_key)},
        )
    )


def _assert_exception_omits_private_values(
    error: BaseException,
    *private_values: str,
) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered = f"{current!r} {current!s} {current.args!r}"
        for private_value in private_values:
            assert private_value not in rendered
        traceback = current.__traceback__
        while traceback is not None:
            if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
                rendered_locals = repr(traceback.tb_frame.f_locals)
                for private_value in private_values:
                    assert private_value not in rendered_locals
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def _publication_tool_round_identity(label: str) -> dict[str, str]:
    digest = sha256(label.encode("utf-8")).hexdigest()
    return {
        "model_step_id": f"mstep_{digest[:32]}",
        "model_attempt_id": f"matt_{digest[16:48]}",
        "tool_round_id": f"tround_{digest[32:]}",
    }


def _approval_checkpoint(label: str) -> tuple[dict[str, Any], PendingToolApproval]:
    identity = _publication_tool_round_identity(label)
    pending_call = PendingToolCallApproval(
        tool_call_id=f"call-{label}",
        tool_name="side_effect",
        arguments={"value": label},
        policy_decision="require_approval",
        reason="human review required",
        metadata={"policy": "test"},
    )
    pending_round = tool_round_recovery.PendingToolRound(
        **identity,
        agent_name="assistant",
        tool_calls=[pending_call],
        policy_state="planned",
        policy_context_version=1,
    )
    approval = PendingToolApproval(
        approval_id=f"approval-{label}",
        **identity,
        tool_call_id=pending_call.tool_call_id,
        tool_name=pending_call.tool_name,
        arguments=pending_call.arguments,
        agent_name="assistant",
        publish_arguments=True,
        reason=pending_call.reason,
        metadata=pending_call.metadata,
        tool_calls=[pending_call],
    )
    return (
        {
            tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY: (
                pending_round.model_dump(mode="json")
            ),
            approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: (
                approval.model_dump(mode="json")
            ),
        },
        approval,
    )


def _mcp_test_manifest_hash(
    *,
    source_manifest_hash: str,
    server_hash: str,
    tools: tuple[dict[str, str], ...] = (),
    exposed_tools: tuple[dict[str, str], ...] = (),
) -> str:
    return _mcp_authoritative_manifest_hash(
        source_manifest_hash=source_manifest_hash,
        server_hash=server_hash,
        tools=tools,
        exposed_tools=exposed_tools,
    )


async def _drop_postgres_schema(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _POSTGRES_TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


async def _reset_postgres_data(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT tablename FROM pg_catalog.pg_tables "
                "WHERE schemaname = current_schema() AND tablename = ANY(%s)",
                (list(_POSTGRES_DATA_TABLES),),
            )
            existing_tables = {row[0] for row in await cur.fetchall()}
            for table in _POSTGRES_DATA_TABLES:
                if table not in existing_tables:
                    continue
                await cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table)))
            for table, column in (
                ("cayu_events", "sequence"),
                ("cayu_transcript_messages", "sequence"),
                ("cayu_session_message_queue", "ordering_key"),
            ):
                await cur.execute(
                    sql.SQL("ALTER TABLE {} ALTER COLUMN {} RESTART WITH 1").format(
                        sql.Identifier(table),
                        sql.Identifier(column),
                    )
                )
        await conn.commit()


def _new_postgres_store(
    dsn: str,
    *,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
    schema_mode: SchemaMode = SchemaMode.VALIDATE,
) -> SessionStore:
    from cayu import PostgresSessionStore

    return PostgresSessionStore(
        dsn,
        min_size=1,
        max_size=4,
        schema_mode=schema_mode,
        public_authority_alias_codec=public_authority_alias_codec,
    )


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _private_event_for_public_event(store: SessionStore, event: Event) -> Event:
    sequence = public_event_sequence(event.id)
    assert sequence is not None
    records = await store.query_events(EventQuery(session_id=event.session_id))
    matches = [record.event for record in records if record.sequence == sequence]
    assert len(matches) == 1
    return matches[0]


async def _pending_approval_for_public_event(
    store: SessionStore,
    event: Event,
) -> PendingToolApproval:
    event_approval = PendingToolApprovalEventView.from_event(
        await _private_event_for_public_event(store, event)
    )
    checkpoint_approval = approval_support.pending_approval_from_checkpoint(
        await store.load_checkpoint(event.session_id)
    )
    assert checkpoint_approval is not None
    assert (
        event_approval.approval_id,
        event_approval.tool_call_id,
        event_approval.model_step_id,
        event_approval.model_attempt_id,
        event_approval.tool_round_id,
    ) == (
        checkpoint_approval.approval_id,
        checkpoint_approval.tool_call_id,
        checkpoint_approval.model_step_id,
        checkpoint_approval.model_attempt_id,
        checkpoint_approval.tool_round_id,
    )
    return checkpoint_approval


def _summary_with_existing(request: CompactionRequest, summary: str) -> str:
    if request.existing_summary is None:
        return summary
    return f"{request.existing_summary}|{summary}"


def _represented_existing_summary_sha256(request: CompactionRequest) -> str | None:
    if request.existing_summary is None:
        return None
    return hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()


class _ConformanceCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0
        self.fail_next = False
        self.requests: list[CompactionRequest] = []

    def provider_budget_identity(self, _session: Session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        self.requests.append(request.model_copy(deep=True))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("conformance compactor failed")
        return CompactionResult(
            summary=_summary_with_existing(request, f"summary-{self.calls}"),
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(_represented_existing_summary_sha256(request)),
        )


class _ConformanceOverlappingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.provider = _ConformanceOverlappingCompactionProvider()
        self.started = self.provider.started
        self.release = self.provider.release
        self.calls = 0

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        return await ModelCompactor(
            provider=self.provider,
            model="summary-model",
            max_input_chars=100_000,
        ).compact(request)


class _ConformanceOverlappingCompactionProvider(ModelProvider):
    name = "overlap-compactor"

    def __init__(self) -> None:
        self.started = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.calls = 0

    async def stream(self, request: ModelRequest):
        del request
        call = self.calls
        self.calls += 1
        self.started[call].set()
        await self.release[call].wait()
        yield ModelStreamEvent.text_delta(f"summary from attempt {call + 1}")
        yield ModelStreamEvent.completed(
            {
                "model": "summary-model",
                "usage": {"input_tokens": call + 1, "output_tokens": 1},
            }
        )


class _ConformanceBlockingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def provider_budget_identity(self, _session: Session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return CompactionResult(
            summary=_summary_with_existing(request, "heartbeat conformance summary"),
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(_represented_existing_summary_sha256(request)),
        )


class _ConformancePartialCompactor(ContextCompactor):
    def provider_budget_identity(self, _session: Session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        return CompactionResult(
            summary=_summary_with_existing(request, "partial coverage"),
            covered_message_count=min(1, len(request.messages)),
            represented_existing_summary_sha256=(_represented_existing_summary_sha256(request)),
        )


class _ConformancePartialCancellationCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    def provider_budget_identity(self, _session: Session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _ConformancePartialOverlapCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.calls = 0

    def provider_budget_identity(self, _session: Session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        call = self.calls
        self.calls += 1
        self.started[call].set()
        await self.release[call].wait()
        return CompactionResult(
            summary=_summary_with_existing(request, f"partial-{call}"),
            covered_message_count=1,
            represented_existing_summary_sha256=(_represented_existing_summary_sha256(request)),
        )


class _UnusedForkProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _UnusedNamedForkProvider(ModelProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RejectingPortableForkProvider(_UnusedNamedForkProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.preflight_calls: list[tuple[str, tuple[Message, ...], tuple[dict[str, Any], ...]]] = []

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        self.preflight_calls.append((model, tuple(messages), tuple(tools)))
        raise ValueError("target provider rejected source history")


class _AuthorizeForkProfilePolicy(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "test:store-conformance-fork-authority:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        assert request.authority_review_required is True
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="Authorize the conformance fork profile.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        )


class _TransitionReplayConformanceProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("completed")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _WorkspaceObservationRecoveryProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        yield ModelStreamEvent.tool_call(
            id="call-workspace-recovery",
            name="workspace_recovery_write",
            arguments={},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _WorkspaceObservationRecoveryTool(Tool):
    spec = ToolSpec(
        name="workspace_recovery_write",
        parallel_safe=False,
        workspace_mutation=True,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:workspace-recovery-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del args
        self.calls += 1
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("recovered.txt", b"written once")
        for index in range(40):
            await ctx.workspace.create_bytes(
                f"generated/{index:03}.txt",
                b"written once",
            )
        return ToolResult(content="written")


def _workspace_observation_recovery_environment_spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        name="local",
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:workspace-recovery-environment",
            behavior_version="1",
            implementation_version="1",
        ),
    )


class _ApprovalRecoveryProvider(ModelProvider):
    name = "fake"

    def __init__(self, *, complete_without_tools: bool = False) -> None:
        self._complete_without_tools = complete_without_tools
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if self._complete_without_tools:
            yield ModelStreamEvent.text_delta("approval completed")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(
            id="call_policy_recovery",
            name="stateful_effect",
            arguments={"value": "must remain gated"},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _ApprovalLimitProvider(_ApprovalRecoveryProvider):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.tool_call(
            id="call_policy_recovery",
            name="stateful_effect",
            arguments={"value": "must remain gated"},
        )
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "tool_calls",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "total_tokens": 11,
                },
            }
        )


class _ApprovalMixedRecoveryProvider(_ApprovalRecoveryProvider):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if self._complete_without_tools:
            yield ModelStreamEvent.text_delta("approval completed")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(
            id="call_policy_recovery",
            name="stateful_effect",
            arguments={"value": "historically completed"},
        )
        yield ModelStreamEvent.tool_call(
            id="call_policy_sibling",
            name="stateful_effect",
            arguments={"value": "must remain gated"},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _ApprovalRecoveryTool(Tool):
    spec = ToolSpec(
        name="stateful_effect",
        description="Record a policy-protected external effect.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:approval-recovery-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        super().__init__()
        self._calls = calls

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx
        self._calls.append(dict(args))
        return ToolResult(content="executed")


class _ApprovalMetadataTool(_ApprovalRecoveryTool):
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self._calls.append(
            {
                "arguments": dict(args),
                "condition": ctx.metadata.get("condition"),
            }
        )
        return ToolResult(content="executed")


class _ApprovalResolutionBinding(WorkspaceBinding):
    """Inject one post-resume binding failure without failing initial setup."""

    def __init__(self) -> None:
        self.bind_calls = 0
        self.fail_next = False

    async def bind(
        self,
        workspace,
        runner,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        del agent_name, environment_name, metadata
        self.bind_calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("approval binding failed after durable resume")
        return BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=runner,
            metadata={"session_id": session_id},
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        del bound, outcome, metadata
        return None


class _ChangingApprovalPolicy(ToolPolicy):
    """Require approval once, then allow, as a stateful replay probe."""

    def __init__(self, calls: list[ToolPolicyDecision]) -> None:
        self._calls = calls

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:changing-approval-policy",
            behavior_version="1",
            implementation_version="1",
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        del request
        decision = (
            ToolPolicyDecision.REQUIRE_APPROVAL if not self._calls else ToolPolicyDecision.ALLOW
        )
        self._calls.append(decision)
        return ToolPolicyResult(
            decision=decision,
            reason=f"stateful policy returned {decision.value}",
        )


def _approval_recovery_environment_spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        name="approval-environment",
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:approval-recovery-environment",
            behavior_version="1",
            implementation_version="1",
        ),
    )


class _SimulatedProcessLoss(BaseException):
    pass


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def session_store_case(request, tmp_path):
    if request.param == "memory":
        return request.param, tmp_path, None
    if request.param == "sqlite":
        return request.param, tmp_path, None
    return request.param, tmp_path, request.getfixturevalue("conformance_postgres_dsn")


@pytest.fixture(scope="module")
def conformance_postgres_dsn(postgres_dsn) -> Iterator[str]:
    async def initialize_schema() -> None:
        await _drop_postgres_schema(postgres_dsn)
        store = _new_postgres_store(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.ensure_schema()
        finally:
            await _close_store(store)

    asyncio.run(initialize_schema())
    try:
        yield postgres_dsn
    finally:
        asyncio.run(_reset_postgres_data(postgres_dsn))


def test_session_store_conformance_lists_queued_dispatch_terminal_receipts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            for session_id in (
                "queued-receipt-a",
                "queued-receipt-b",
                "queued-receipt-unrelated",
            ):
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[],
                    ),
                    identity=_identity(),
                )
            await store.append_event(
                "queued-receipt-a",
                Event(
                    id="terminal-a",
                    type=EventType.SESSION_COMPLETED,
                    session_id="queued-receipt-a",
                    payload={"session_run_operation_id": "operation-a"},
                ),
            )
            await store.checkpoint(
                "queued-receipt-a",
                {
                    "session_run_operation": {
                        "version": 1,
                        "operation_id": "operation-a",
                        "run_epoch": 1,
                        "terminal_event_id": "terminal-a",
                        "queue_task_id": "queue-a",
                    }
                },
            )
            await store.checkpoint(
                "queued-receipt-b",
                {
                    "queued_dispatch_terminal_receipts": {
                        "version": 1,
                        "receipts": {
                            "operation-b-1": {
                                "queue_task_id": "queue-b-1",
                                "terminal_event_id": "terminal-b-1",
                                "run_epoch": 1,
                            },
                            "operation-b-2": {
                                "queue_task_id": "queue-b-2",
                                "terminal_event_id": "terminal-b-2",
                                "run_epoch": 1,
                            },
                        },
                    }
                },
            )
            await store.checkpoint(
                "queued-receipt-unrelated",
                {"unrelated": True},
            )

            first_page = await store.list_queued_dispatch_terminal_receipts(
                QueuedDispatchTerminalReceiptQuery(limit=2)
            )
            assert [(receipt.session_id, receipt.operation_id) for receipt in first_page] == [
                ("queued-receipt-a", "operation-a"),
                ("queued-receipt-b", "operation-b-1"),
            ]

            store = await _reopen_store(session_store_case, store)
            second_page = await store.list_queued_dispatch_terminal_receipts(
                QueuedDispatchTerminalReceiptQuery(
                    after_session_id=first_page[-1].session_id,
                    after_operation_id=first_page[-1].operation_id,
                    limit=2,
                )
            )
            assert [receipt.model_dump(mode="json") for receipt in second_page] == [
                {
                    "session_id": "queued-receipt-b",
                    "queue_task_id": "queue-b-2",
                    "operation_id": "operation-b-2",
                    "terminal_event_id": "terminal-b-2",
                }
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_receipt_query_does_not_serialize_mutated_input(
    session_store_case,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "mutated-receipt-query-diagnostic-canary"

    class Canary:
        def __repr__(self) -> str:
            return canary

        def __str__(self) -> str:
            return canary

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            query = QueuedDispatchTerminalReceiptQuery()
            object.__setattr__(query, "limit", Canary())
            with pytest.raises(ValueError):
                await store.list_queued_dispatch_terminal_receipts(query)
        finally:
            await _close_store(store)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out
    assert canary not in captured.err
    assert all(canary not in str(record.message) for record in recwarn)


def test_in_memory_queued_dispatch_receipt_query_uses_live_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "queued-receipt-indexed"
        await store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=_identity(),
        )
        await store.checkpoint(
            session_id,
            {
                "session_run_operation": {
                    "version": 1,
                    "operation_id": "operation-indexed",
                    "run_epoch": 1,
                    "terminal_event_id": "terminal-indexed",
                    "queue_task_id": "queue-indexed",
                }
            },
        )
        assert await store.list_queued_dispatch_terminal_receipts() == []
        await store.append_event(
            session_id,
            Event(
                id="terminal-indexed",
                type=EventType.SESSION_COMPLETED,
                session_id=session_id,
                payload={"session_run_operation_id": "operation-indexed"},
            ),
        )
        with pytest.raises(ValueError, match="receipt checkpoint is invalid"):
            await store.checkpoint(
                session_id,
                {
                    "queued_dispatch_terminal_receipts": {
                        "version": 2,
                        "receipts": {},
                    }
                },
            )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["session_run_operation"]["operation_id"] == "operation-indexed"

        def reject_history_projection(*args, **kwargs):
            del args, kwargs
            raise AssertionError("receipt query rescanned checkpoint/event history")

        monkeypatch.setattr(
            sessions_module,
            "_queued_dispatch_terminal_receipts_for_session",
            reject_history_projection,
        )
        receipts = await store.list_queued_dispatch_terminal_receipts()
        assert [(receipt.session_id, receipt.operation_id) for receipt in receipts] == [
            (session_id, "operation-indexed")
        ]

    asyncio.run(run())


def test_in_memory_public_authority_aliases_reject_retired_keys() -> None:
    first = _public_authority_alias_codec()
    replacement_key = SecretStr(
        base64.urlsafe_b64encode(bytes([24]) * 32).decode("ascii").rstrip("=")
    )
    retained = first.rotated(
        active_key_id="replacement",
        key=replacement_key,
    )
    retired = retained.rotated(
        active_key_id="replacement",
        key=replacement_key,
        retire_key_ids=("conformance",),
    )
    private_session_id = "private-session"
    old_alias = first.encode(private_session_id, field_name="session_id")
    active_alias = retained.encode(private_session_id, field_name="session_id")
    store = InMemorySessionStore(public_authority_alias_codec=retained)

    async def run() -> None:
        await store.register_public_authority_alias(
            old_alias,
            field_name="session_id",
            private_value=private_session_id,
        )
        await store.register_public_authority_alias(
            active_alias,
            field_name="session_id",
            private_value=private_session_id,
        )
        assert (
            await store.resolve_public_authority_alias(
                old_alias,
                field_name="session_id",
            )
            == private_session_id
        )

        store._public_authority_alias_codec = retired

        assert (
            await store.resolve_public_authority_alias(
                old_alias,
                field_name="session_id",
            )
            is None
        )
        assert (
            await store.resolve_public_authority_alias(
                active_alias,
                field_name="session_id",
            )
            == private_session_id
        )

    asyncio.run(run())


def test_session_store_conformance_parent_deletion_preserves_root_invocation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            parent = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"invocation-parent-{session_store_case[0]}",
                    messages=[],
                ),
                identity=_identity(),
            )
            child = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"invocation-child-{session_store_case[0]}",
                    parent_session_id=parent.id,
                    messages=[],
                ),
                identity=_identity(),
            )

            await store.delete_session(parent.id)
            loaded = await store.load(child.id)

            assert loaded is not None
            assert loaded.parent_session_id is None
            assert loaded.invocation == child.invocation
            assert loaded.invocation.root_session_id == parent.id
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_parent_deletion_rejects_durable_subagent_child(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            parent = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"durable-delete-parent-{session_store_case[0]}",
                    messages=[],
                ),
                identity=_identity(),
            )
            child = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"durable-delete-child-{session_store_case[0]}",
                    parent_session_id=parent.id,
                    messages=[],
                    metadata={"subagent": {"mode": "durable"}},
                ),
                identity=_identity(),
            )

            with pytest.raises(
                ValueError,
                match="durable subagent child authority still exists",
            ):
                await store.delete_session(parent.id)

            retained_parent = await store.load(parent.id)
            retained_child = await store.load(child.id)
            assert retained_parent is not None
            assert retained_child is not None
            assert retained_child.parent_session_id == parent.id

            await store.delete_session(child.id)
            await store.delete_session(parent.id)
            assert await store.load(child.id) is None
            assert await store.load(parent.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reused_root_session_id_cannot_rebind_invocation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            root_session_id = f"invocation-reused-root-{session_store_case[0]}"
            original_root = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=root_session_id,
                    messages=[],
                    invocation_origin=InvocationOriginClaim(subject="original-user"),
                ),
                identity=_identity(),
            )
            child = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"invocation-reused-child-{session_store_case[0]}",
                    parent_session_id=original_root.id,
                    messages=[],
                ),
                identity=_identity(),
            )

            await store.delete_session(original_root.id)
            replacement_root = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=root_session_id,
                    messages=[],
                    invocation_origin=InvocationOriginClaim(subject="replacement-user"),
                ),
                identity=_identity(),
            )
            store = await _reopen_store(session_store_case, store)
            loaded_child = await store.load(child.id)
            loaded_replacement = await store.load(replacement_root.id)

            assert loaded_child is not None
            assert loaded_replacement is not None
            assert loaded_child.invocation.root_session_id == root_session_id
            assert loaded_replacement.invocation.root_session_id == root_session_id
            assert (
                loaded_child.invocation.root_invocation_id
                == original_root.invocation.root_invocation_id
            )
            assert (
                loaded_replacement.invocation.root_invocation_id
                == replacement_root.invocation.root_invocation_id
            )
            assert (
                loaded_child.invocation.root_invocation_id
                != loaded_replacement.invocation.root_invocation_id
            )
            assert loaded_child.invocation.origin.subject == "original-user"
            assert loaded_replacement.invocation.origin.subject == "replacement-user"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_exact_root_invocation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            root_request = RunRequest(
                agent_name="assistant",
                session_id=f"invocation-root-{session_store_case[0]}",
                messages=[],
                invocation_origin=InvocationOriginClaim(
                    subject="application-user",
                    tenant="customer-a",
                ),
            )
            root = await store.create(root_request, identity=_identity())
            child_request = RunRequest(
                agent_name="assistant",
                session_id=f"invocation-child-reload-{session_store_case[0]}",
                parent_session_id=root.id,
                messages=[],
            )
            child = await store.create(child_request, identity=_identity())

            with pytest.raises(ValueError, match="Session already exists"):
                await store.create(child_request, identity=_identity())

            store = await _reopen_store(session_store_case, store)
            loaded_root = await store.load(root.id)
            loaded_child = await store.load(child.id)

            assert loaded_root is not None
            assert loaded_child is not None
            assert loaded_root.invocation.origin.trust is InvocationOriginTrust.HOST_ASSERTED
            assert loaded_root.invocation.origin.subject == "application-user"
            assert loaded_root.invocation.origin.tenant == "customer-a"
            assert loaded_root.invocation.root_invocation_id == root.invocation.root_invocation_id
            assert loaded_root.invocation.root_session_id == root.id
            assert loaded_root.invocation.source is SessionExecutionSource.SDK_RUN
            assert loaded_child.invocation.origin == loaded_root.invocation.origin
            assert (
                loaded_child.invocation.root_invocation_id
                == loaded_root.invocation.root_invocation_id
            )
            assert loaded_child.invocation.root_session_id == loaded_root.id
            assert loaded_child.invocation.source is SessionExecutionSource.SDK_RUN
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_concurrent_child_creation_has_one_provenance(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            root = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"invocation-concurrent-root-{session_store_case[0]}",
                    messages=[],
                    invocation_origin=InvocationOriginClaim(subject="application-user"),
                ),
                identity=_identity(),
            )
            child_request = RunRequest(
                agent_name="assistant",
                session_id=f"invocation-concurrent-child-{session_store_case[0]}",
                parent_session_id=root.id,
                messages=[],
            )
            results = await asyncio.gather(
                store.create(child_request, identity=_identity()),
                store.create(child_request, identity=_identity()),
                return_exceptions=True,
            )

            created = [result for result in results if isinstance(result, Session)]
            rejected = [result for result in results if isinstance(result, ValueError)]
            assert len(created) == 1
            assert len(rejected) == 1
            assert "Session already exists" in str(rejected[0])
            loaded = await store.load(child_request.session_id or "")
            assert loaded is not None
            assert loaded.invocation == created[0].invocation
            assert loaded.invocation.origin == root.invocation.origin
            assert loaded.invocation.root_session_id == root.id
        finally:
            await _close_store(store)

    asyncio.run(run())


async def _open_store(
    case,
    *,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
) -> SessionStore:
    if public_authority_alias_codec is None:
        public_authority_alias_codec = _public_authority_alias_codec()
    store_kind, tmp_path, postgres_dsn = case
    if store_kind == "memory":
        return InMemorySessionStore(
            public_authority_alias_codec=public_authority_alias_codec,
        )
    if store_kind == "sqlite":
        return SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=public_authority_alias_codec,
        )
    await _reset_postgres_data(postgres_dsn)
    return _new_postgres_store(
        postgres_dsn,
        public_authority_alias_codec=public_authority_alias_codec,
    )


async def _reopen_store(case, store: SessionStore) -> SessionStore:
    store_kind, tmp_path, postgres_dsn = case
    if store_kind == "memory":
        return store
    public_authority_alias_codec = store.public_authority_alias_codec
    await _close_store(store)
    if store_kind == "sqlite":
        return SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=public_authority_alias_codec,
        )
    return _new_postgres_store(
        postgres_dsn,
        public_authority_alias_codec=public_authority_alias_codec,
    )


def test_session_store_conformance_repairs_terminal_evidence_durably(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            expected_types = {
                SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
                SessionStatus.FAILED: EventType.SESSION_FAILED,
                SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
            }
            original_epochs: dict[str, int] = {}
            for status in expected_types:
                session_id = f"terminal-evidence-{status.value}"
                await store.create(
                    RunRequest(
                        agent_name="removed_agent",
                        session_id=session_id,
                        messages=[Message.text("user", "finish")],
                    ),
                    identity=profiled_session_identity(
                        provider_name="fake",
                        model="fake-model",
                    ),
                )
                terminal = await store.update_status(session_id, status)
                original_epochs[session_id] = terminal.run_epoch
                if status == SessionStatus.INTERRUPTED:
                    await store.checkpoint(
                        session_id,
                        {
                            "pending_session_interrupt": {
                                "reason": "worker restart",
                                "metadata": {"worker": "old"},
                                "interruption_type": "operator_requested",
                                "interruption_request_id": "interrupt-before-crash",
                            }
                        },
                    )

                repaired = await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
                assert repaired.actions == (
                    IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                )

            store = await _reopen_store(session_store_case, store)
            for status, expected_type in expected_types.items():
                session_id = f"terminal-evidence-{status.value}"
                session = await store.load(session_id)
                records = await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_types=tuple(expected_types.values()),
                    )
                )
                assert session is not None
                assert session.status == status
                assert session.run_epoch == original_epochs[session_id] + 2
                assert [record.event.type for record in records] == [expected_type]
                assert await store.load_checkpoint(session_id) == {
                    CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
                }

            class CompleteThenBlockProvider(ModelProvider):
                name = "fake"

                def __init__(self) -> None:
                    self.calls = 0
                    self.second_started = asyncio.Event()

                async def stream(self, _request: Any):
                    self.calls += 1
                    if self.calls == 2:
                        self.second_started.set()
                        await asyncio.Event().wait()
                        raise AssertionError("unreachable")
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})

            provider = CompleteThenBlockProvider()
            resumed_app = CayuApp(session_store=store, enable_logging=False)
            resumed_app.register_provider(provider, default=True)
            resumed_app.register_agent(AgentSpec(name="removed_agent", model="fake-model"))
            resumed_events = [
                event
                async for event in resumed_app.resume(
                    ResumeRequest(
                        session_id="terminal-evidence-interrupted",
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            assert resumed_events[-1].type == EventType.SESSION_COMPLETED

            async def collect_second_resume() -> list[Event]:
                return [
                    event
                    async for event in resumed_app.resume(
                        ResumeRequest(
                            session_id="terminal-evidence-interrupted",
                            messages=[Message.text("user", "continue again")],
                        )
                    )
                ]

            second_resume = asyncio.create_task(collect_second_resume())
            await asyncio.wait_for(provider.second_started.wait(), timeout=5)
            independent_interrupt = [
                event
                async for event in resumed_app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="terminal-evidence-interrupted",
                        reason="new operator request",
                        metadata={"worker": "new"},
                    )
                )
            ]
            second_resume_events = await asyncio.wait_for(second_resume, timeout=5)
            assert independent_interrupt[-1].type == EventType.SESSION_INTERRUPTED
            assert second_resume_events[-1].id == independent_interrupt[-1].id

            durable_interruptions = await store.query_events(
                EventQuery(
                    session_id="terminal-evidence-interrupted",
                    event_type=EventType.SESSION_INTERRUPTED,
                )
            )
            assert len(durable_interruptions) == 2
            repaired_interruption, later_interruption = (
                record.event for record in durable_interruptions
            )
            assert repaired_interruption.payload["reason"] == "worker restart"
            assert repaired_interruption.payload["metadata"] == {"worker": "old"}
            assert (
                repaired_interruption.payload["interruption_request_id"] == "interrupt-before-crash"
            )
            assert later_interruption.payload["reason"] == "new operator request"
            assert later_interruption.payload["metadata"] == {"worker": "new"}
            assert (
                later_interruption.payload["interruption_request_id"]
                != repaired_interruption.payload["interruption_request_id"]
            )

            concurrent_session_id = "terminal-evidence-concurrent"
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=concurrent_session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(concurrent_session_id, SessionStatus.COMPLETED)
            request = IncompleteSessionRecoveryRequest(session_id=concurrent_session_id)
            concurrent_results = await asyncio.gather(
                CayuApp(session_store=store, enable_logging=False).recover_incomplete_session(
                    request
                ),
                CayuApp(session_store=store, enable_logging=False).recover_incomplete_session(
                    request
                ),
            )
            assert (
                sum(
                    result.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
                    for result in concurrent_results
                )
                == 1
            )
            assert all(
                result.actions
                in {
                    (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,),
                    (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                }
                for result in concurrent_results
            )
            concurrent_events = await store.query_events(
                EventQuery(
                    session_id=concurrent_session_id,
                    event_type=EventType.SESSION_COMPLETED,
                )
            )
            assert len(concurrent_events) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_quarantines_late_secret_tool_arguments(
    session_store_case,
) -> None:
    class ResolveLateSecretTool(Tool):
        spec = ToolSpec(
            name="resolve_late_secret",
            description="Resolve one late invocation secret.",
            input_schema={"type": "object", "additionalProperties": True},
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            assert args["provided"] == secret
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="api_key"))
            return ToolResult(content="done")

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            provider = FakeProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="call_late_secret",
                            name="resolve_late_secret",
                            arguments={"provided": secret, secret: {"nested": secret}},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="local"),
                    vault=StaticVault({"api_key": secret}),
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[ResolveLateSecretTool()],
            )
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="late-secret-store-conformance",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]

            durable_events = await store.load_events("late-secret-store-conformance")
            started = next(
                event for event in durable_events if event.type is EventType.TOOL_CALL_STARTED
            )
            assert started.payload["arguments_state"] == "quarantined"
            assert "arguments" not in started.payload
            assert secret not in repr(events)
            assert secret not in repr(await store.load_transcript("late-secret-store-conformance"))
            assert secret not in repr(provider.requests[1].messages)
        finally:
            await _close_store(store)

    secret = "late-store-tool-start-secret-canary"
    asyncio.run(run())


def test_session_store_conformance_deletes_reconciled_consecutive_interruptions(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "terminal-evidence-consecutive-interrupt-delete"
        first_payload = {
            "interruption_type": "tool_approval_required",
            "approval": {"approval_id": "approval-before-operator"},
        }
        second_payload = {
            "reason": "new operator request",
            "metadata": {"worker": "new"},
            "interruption_type": "operator_requested",
            "interruption_request_id": "independent-interrupt",
        }
        try:
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    payload=first_payload,
                ),
            )

            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=lambda _session, _checkpoint: {
                    "pending_session_interrupt": second_payload
                },
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            app = CayuApp(session_store=store, enable_logging=False)
            repaired = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
            assert repaired.events[0].payload == {
                **second_payload,
                "interruption_request_id": PRIVATE_EVENT_AUTHORITY,
            }

            durable_interruptions = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_INTERRUPTED,
                )
            )
            assert durable_interruptions[-1].event.payload == second_payload

            store = await _reopen_store(session_store_case, store)
            settled = await CayuApp(
                session_store=store,
                enable_logging=False,
            ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
            assert settled.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
            await store.delete_session(session_id)
            assert await store.load(session_id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_repairs_pre_boundary_resume_failure(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "terminal-evidence-pre-boundary-resume"
        operation_id = "93f3184b-976c-4fdc-8889-c2918890581b"
        try:
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            )
            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda current_session, checkpoint: (
                    _checkpoint_with_session_run_operation(
                        checkpoint=checkpoint,
                        current_session=current_session,
                        operation_id=operation_id,
                    )
                ),
            )
            await store.update_status(session_id, SessionStatus.FAILED)
            await store.release_run_fence(session_id)

            store = await _reopen_store(session_store_case, store)
            repaired = await CayuApp(
                session_store=store,
                enable_logging=False,
            ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
            records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(
                        EventType.SESSION_COMPLETED,
                        EventType.SESSION_FAILED,
                    ),
                )
            )
            assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
            assert [record.event.type for record in records] == [
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
            ]
            assert records[-1].event.payload["session_run_operation_id"] == operation_id
            assert await store.load_checkpoint(session_id) == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
            }
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "lifecycle_event_type",
    (
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
        EventType.SESSION_FORKED,
    ),
)
def test_session_store_conformance_repairs_terminal_older_than_current_lifecycle(
    session_store_case,
    lifecycle_event_type: EventType,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        lifecycle_label = lifecycle_event_type.value.replace(".", "-")
        session_id = f"terminal-evidence-pre-boundary-{lifecycle_label}"
        operation_id = "93f3184b-976c-4fdc-8889-c2918890581b"
        try:
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_events(
                session_id,
                [
                    Event(
                        id=f"terminal-evidence-{lifecycle_label}-older-other-operation",
                        type=EventType.SESSION_COMPLETED,
                        session_id=session_id,
                        payload={"session_run_operation_id": "older-operation"},
                    ),
                    Event(
                        id=f"terminal-evidence-{lifecycle_label}-older-matching-operation",
                        type=EventType.SESSION_COMPLETED,
                        session_id=session_id,
                        payload={"session_run_operation_id": operation_id},
                    ),
                ],
            )
            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda current_session, checkpoint: (
                    _checkpoint_with_session_run_operation(
                        checkpoint=checkpoint,
                        current_session=current_session,
                        operation_id=operation_id,
                    )
                ),
            )
            await store.append_event(
                session_id,
                Event(
                    id=f"terminal-evidence-newest-{lifecycle_label}-boundary",
                    type=lifecycle_event_type,
                    session_id=session_id,
                ),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)

            store = await _reopen_store(session_store_case, store)
            with pytest.raises(TerminalSessionEvidenceError) as missing:
                await store.load_terminal_session_evidence(session_id)
            assert missing.value.code is TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING
            await store.release_run_fence(session_id)

            repaired = await CayuApp(
                session_store=store,
                enable_logging=False,
            ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
            records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_COMPLETED,
                )
            )
            assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
            assert [record.event.type for record in records] == [
                EventType.SESSION_COMPLETED,
                EventType.SESSION_COMPLETED,
                EventType.SESSION_COMPLETED,
            ]
            assert [record.event.id for record in records[:2]] == [
                f"terminal-evidence-{lifecycle_label}-older-other-operation",
                f"terminal-evidence-{lifecycle_label}-older-matching-operation",
            ]
            assert records[-1].event.payload["terminal_evidence_repaired"] is True
            assert records[-1].event.payload["session_run_operation_id"] == operation_id
            assert await store.load_checkpoint(session_id) == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
            }

            store = await _reopen_store(session_store_case, store)
            settled = await CayuApp(
                session_store=store,
                enable_logging=False,
            ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
            settled_records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_COMPLETED,
                )
            )
            assert settled.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
            assert settled_records == records
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_recovers_abandoned_resumed_run(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "terminal-evidence-abandoned-resumed-run"
        operation_id = "terminal-evidence-abandoned-resumed-operation"
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            )
            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda current_session, checkpoint: (
                    _checkpoint_with_session_run_operation(
                        checkpoint=checkpoint,
                        current_session=current_session,
                        operation_id=operation_id,
                    )
                ),
            )
            _deactivate_session_run_fence(session_id)

            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

            store = await _reopen_store(session_store_case, store)
            session = await store.load(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            interrupted = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_INTERRUPTED,
                )
            )
            assert recovered.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
            assert session is not None
            assert session.status == SessionStatus.INTERRUPTED
            assert len(interrupted) == 1
            assert interrupted[0].event.payload["session_run_operation_id"] == operation_id
            assert checkpoint is not None
            assert "session_run_operation" not in checkpoint
            assert "pending_session_interrupt" not in checkpoint
            assert "incomplete_session_recovery_claim" not in checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_declares_usage_aggregate_support(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            assert store.supports_usage_aggregates is True
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_workspace_mutation_receipts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"workspace-receipt-{session_store_case[0]}"
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[],
                ),
                identity=_identity(),
            )
            observed_before = Event(
                id=f"evt-workspace-before-{session_store_case[0]}",
                type=EventType.WORKSPACE_REVISION_OBSERVED,
                session_id=session_id,
                payload={
                    "phase": "before",
                    "window_id": "wmut-conformance",
                    "session_run_epoch": 1,
                    "model_step": 2,
                    "tool_call_id": "call-workspace",
                    "workspace_id": "workspace-conformance",
                    "observer": "GitRepositoryBinding",
                    "status": "supported",
                    "revision": "sha256:before",
                    "head_revision": "before-head",
                    "branch": "main",
                    "paths": [],
                    "total_paths": 0,
                    "detail_code": None,
                    "model_step_id": "step-conformance",
                    "model_attempt_id": "attempt-conformance",
                    "tool_round_id": "round-conformance",
                },
            )
            observed_after = observed_before.model_copy(
                update={
                    "id": f"evt-workspace-after-{session_store_case[0]}",
                    "payload": {
                        **observed_before.payload,
                        "phase": "after",
                        "revision": "sha256:after",
                    },
                },
                deep=True,
            )
            receipt = Event(
                id=f"evt-workspace-receipt-{session_store_case[0]}",
                type=EventType.WORKSPACE_MUTATION_RECORDED,
                session_id=session_id,
                payload={
                    "window_id": "wmut-conformance",
                    "before_observation_id": observed_before.id,
                    "after_observation_id": observed_after.id,
                    "session_run_epoch": 1,
                    "model_step": 2,
                    "tool_call_id": "call-workspace",
                    "workspace_id": "workspace-conformance",
                    "observer": "GitRepositoryBinding",
                    "status": "changed",
                    "before_revision": "sha256:before",
                    "after_revision": "sha256:after",
                    "paths": [
                        {
                            "path": "created.txt",
                            "change": "added",
                            "renamed_from": None,
                        }
                    ],
                    "total_paths": 1,
                    "head_changed": False,
                    "branch_changed": False,
                    "detail_code": None,
                    "model_step_id": "step-conformance",
                    "model_attempt_id": "attempt-conformance",
                    "tool_round_id": "round-conformance",
                },
            )
            expected = [observed_before, observed_after, receipt]
            await store.append_events(session_id, expected)
            store = await _reopen_store(session_store_case, store)

            restored = await store.query_events(EventQuery(session_id=session_id))

            assert [record.event.model_dump(mode="json") for record in restored] == [
                event.model_dump(mode="json") for event in expected
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_workspace_observation_lifecycle_is_exact_and_replayable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"workspace-observation-lifecycle-{session_store_case[0]}"
            session = await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[],
                ),
                identity=_identity(),
            )
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            intent = WorkspaceObservationLifecycle(
                session_id=session_id,
                window_id="wmut-conformance-lifecycle",
                source_run_epoch=session.run_epoch,
                binding_generation_id="wbind_conformance_lifecycle",
                workspace_id="workspace-conformance-lifecycle",
                observer="ConformanceWorkspaceBinding",
                observer_authority="configured",
                artifact_store_id="artifact-conformance-lifecycle",
                agent_name="assistant",
                environment_name="local",
                tool_name="mutate",
                tool_call_id="call-workspace",
                model_step_id="mstep-workspace",
                model_attempt_id="matt-workspace",
                tool_round_id="tround-workspace",
                model_step=1,
            )
            skipped_stage = WorkspaceObservationLifecycle.model_validate(
                {
                    **intent.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                    "before_state": WorkspaceObservationEvidenceState.CAPTURED_PRIVATE.value,
                    "tool_outcome_event_id": "evt-illegal-stage-skip",
                    "tool_outcome_event_digest": "d" * 64,
                }
            )
            with pytest.raises(ValueError, match="invalid phase edge"):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=intent,
                    current=skipped_stage,
                    phase="tool-outcome",
                )

            premature_before_capture = WorkspaceObservationLifecycle.model_validate(
                {
                    **intent.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.BEFORE_CAPTURED.value,
                    "before_state": WorkspaceObservationEvidenceState.CAPTURED_PRIVATE.value,
                }
            )
            premature_after_artifact = WorkspaceObservationArtifact(
                evidence_kind="revision-after",
                artifact_id="artifact-premature-after",
                sha256="e" * 64,
                size_bytes=128,
                state=WorkspaceObservationArtifactState.INTENT,
            )
            with pytest.raises(ValueError, match="invalid evidence phase"):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=premature_before_capture,
                    current=premature_before_capture.model_copy(
                        update={"artifacts": (premature_after_artifact,)}
                    ),
                    phase="artifact-revision-after-intent",
                )

            staged_without_before_evidence = WorkspaceObservationLifecycle.model_validate(
                {
                    **premature_before_capture.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                    "tool_outcome_event_id": "evt-staged-without-before-evidence",
                    "tool_outcome_event_digest": "f" * 64,
                }
            )
            with pytest.raises(ValueError, match="preceded durable before evidence"):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=staged_without_before_evidence,
                    current=staged_without_before_evidence.model_copy(
                        update={"artifacts": (premature_after_artifact,)}
                    ),
                    phase="artifact-revision-after-intent",
                )

            original_load_checkpoint = store.load_checkpoint
            child_cancelled = False

            async def load_checkpoint_with_child_cancellation(session_id: str):
                nonlocal child_cancelled
                if not child_cancelled:
                    child_cancelled = True
                    raise asyncio.CancelledError("store-owned checkpoint cancellation")
                return await original_load_checkpoint(session_id)

            store.load_checkpoint = load_checkpoint_with_child_cancellation  # type: ignore[method-assign]
            with pytest.raises(
                RuntimeError,
                match="Workspace observation checkpoint read was cancelled",
            ):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=None,
                    current=intent,
                    phase="intent",
                    intent_admission=_admit_test_workspace_observation_intent(intent),
                )
            current_task = asyncio.current_task()
            assert current_task is not None
            assert current_task.cancelling() == 0
            assert current_task.cancelled() is False

            async def load_checkpoint_with_child_cancellation_group(session_id: str):
                del session_id
                raise BaseExceptionGroup(
                    "store-owned grouped failure",
                    [
                        asyncio.CancelledError("store-owned grouped cancellation"),
                        RuntimeError("store-owned grouped error"),
                    ],
                )

            store.load_checkpoint = (  # type: ignore[method-assign]
                load_checkpoint_with_child_cancellation_group
            )
            with pytest.raises(
                RuntimeError,
                match="Workspace observation checkpoint read reported multiple operational failures",
            ):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=None,
                    current=intent,
                    phase="intent",
                    intent_admission=_admit_test_workspace_observation_intent(intent),
                )
            assert current_task.cancelling() == 0
            assert current_task.cancelled() is False
            store.load_checkpoint = original_load_checkpoint  # type: ignore[method-assign]

            original_publish = store.publish_runtime_publication
            sibling = intent.model_copy(
                update={
                    "window_id": "wmut-unrelated-sibling",
                    "tool_call_id": "call-unrelated-sibling",
                }
            )

            async def publish_with_sibling_mutation(session_id: str, **kwargs):
                request = kwargs["request"]
                operations = []
                workspace_observations_value = None
                for operation in request.mutation.operations:
                    if operation.key != "workspace_observations":
                        operations.append(operation)
                        continue
                    value = dict(operation.value)
                    value[sibling.window_id] = sibling.model_dump(mode="json")
                    workspace_observations_value = value
                    operations.append(operation.model_copy(update={"value": value}))
                assert workspace_observations_value is not None
                conflicting_intent = dict(request.intent)
                conflicting_intent["target_root_digest"] = (
                    runtime_publication_checkpoint_value_digest(workspace_observations_value)
                )
                conflicting_request = RuntimePublicationRequest(
                    publication_id=request.publication_id,
                    kind=request.kind,
                    interaction_id=request.interaction_id,
                    intent=conflicting_intent,
                    mutation=RuntimePublicationMutation(operations=tuple(operations)),
                    transcript_messages=request.transcript_messages,
                    events=request.events,
                    referenced_events=request.referenced_events,
                )
                return await original_publish(
                    session_id,
                    request=conflicting_request,
                    **{key: value for key, value in kwargs.items() if key != "request"},
                )

            store.publish_runtime_publication = (  # type: ignore[method-assign]
                publish_with_sibling_mutation
            )
            with pytest.raises(ValueError, match="changed a sibling lifecycle"):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=None,
                    current=intent,
                    phase="intent",
                    intent_admission=_admit_test_workspace_observation_intent(intent),
                )
            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]

            returned_conflicting_acknowledgement = False

            async def publish_with_conflicting_acknowledgement(session_id: str, **kwargs):
                nonlocal returned_conflicting_acknowledgement
                result = await original_publish(session_id, **kwargs)
                if (
                    kwargs["request"].intent.get("phase") != "intent"
                    or returned_conflicting_acknowledgement
                ):
                    return result
                returned_conflicting_acknowledgement = True
                return result.model_copy(
                    update={
                        "receipt": result.receipt.model_copy(
                            update={
                                "kind": "tool-round",
                                "request_digest": "f" * 64,
                            }
                        )
                    }
                )

            store.publish_runtime_publication = (  # type: ignore[method-assign]
                publish_with_conflicting_acknowledgement
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=None,
                current=intent,
                phase="intent",
                intent_admission=_admit_test_workspace_observation_intent(intent),
            )
            assert returned_conflicting_acknowledgement is True
            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: intent}

            before_captured = WorkspaceObservationLifecycle.model_validate(
                {
                    **intent.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.BEFORE_CAPTURED.value,
                    "before_state": WorkspaceObservationEvidenceState.CAPTURED_PRIVATE.value,
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=intent,
                current=before_captured,
                phase="before-capture",
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: before_captured}

            staged = WorkspaceObservationLifecycle.model_validate(
                {
                    **before_captured.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                    "tool_outcome_event_id": "evt-tool-workspace",
                    "tool_outcome_event_digest": "a" * 64,
                }
            )
            await asyncio.gather(
                *(
                    publish_workspace_observation_transition(
                        session_store=store,
                        event_writer=event_writer,
                        session=session,
                        previous=before_captured,
                        current=staged,
                        phase="tool-outcome",
                    )
                    for _ in range(2)
                )
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: staged}

            artifact_intent = WorkspaceObservationArtifact(
                evidence_kind="revision-before",
                artifact_id="artifact-workspace-before",
                sha256="b" * 64,
                size_bytes=128,
                state=WorkspaceObservationArtifactState.INTENT,
            )
            artifact_intended = WorkspaceObservationLifecycle.model_validate(
                {
                    **staged.model_dump(mode="json"),
                    "artifacts": [artifact_intent.model_dump(mode="json")],
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=staged,
                current=artifact_intended,
                phase="artifact-revision-before-intent",
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: artifact_intended}

            artifact_published = WorkspaceObservationLifecycle.model_validate(
                {
                    **artifact_intended.model_dump(mode="json"),
                    "artifacts": [
                        artifact_intent.model_copy(
                            update={"state": WorkspaceObservationArtifactState.PUBLISHED}
                        ).model_dump(mode="json")
                    ],
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=artifact_intended,
                current=artifact_published,
                phase="artifact-revision-before-published",
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: artifact_published}

            def observation_event(
                *,
                event_id: str,
                event_type: EventType,
                phase: str | None = None,
                status: str | None = None,
            ) -> Event:
                payload = {
                    "window_id": intent.window_id,
                    "session_run_epoch": intent.source_run_epoch,
                    "binding_generation_id": intent.binding_generation_id,
                    "workspace_id": intent.workspace_id,
                    "observer": intent.observer,
                    "artifact_store_id": intent.artifact_store_id,
                    "tool_call_id": intent.tool_call_id,
                    "model_step_id": intent.model_step_id,
                    "model_attempt_id": intent.model_attempt_id,
                    "tool_round_id": intent.tool_round_id,
                    "model_step": intent.model_step,
                }
                if phase is not None:
                    payload["phase"] = phase
                    payload["status"] = "supported"
                    payload["path_scope"] = "complete"
                    if phase == "before":
                        payload.update(
                            {
                                "manifest_artifact_id": artifact_intent.artifact_id,
                                "manifest_artifact_sha256": artifact_intent.sha256,
                                "manifest_artifact_size_bytes": artifact_intent.size_bytes,
                            }
                        )
                if status is not None:
                    payload["status"] = status
                if event_type is EventType.WORKSPACE_MUTATION_RECORDED:
                    payload.update(
                        {
                            "before_observation_id": before_published.before_observation_id,
                            "after_observation_id": after_captured.after_observation_id,
                            "tool_outcome_event_id": staged.tool_outcome_event_id,
                            "tool_outcome_event_digest": staged.tool_outcome_event_digest,
                        }
                    )
                return Event(
                    id=event_id,
                    type=event_type,
                    session_id=session_id,
                    agent_name=intent.agent_name,
                    environment_name=intent.environment_name,
                    tool_name=intent.tool_name,
                    payload=payload,
                )

            before_event = observation_event(
                event_id="evt-workspace-observation-lifecycle-before",
                event_type=EventType.WORKSPACE_REVISION_OBSERVED,
                phase="before",
            )
            before_published = WorkspaceObservationLifecycle.model_validate(
                {
                    **artifact_published.model_dump(mode="json"),
                    "before_state": WorkspaceObservationEvidenceState.PUBLISHED.value,
                    "before_observation_id": before_event.id,
                    "artifacts": [
                        artifact_intent.model_copy(
                            update={"state": WorkspaceObservationArtifactState.REFERENCED}
                        ).model_dump(mode="json")
                    ],
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=artifact_published,
                current=before_published,
                phase="before-evidence",
                events=(before_event,),
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: before_published}

            after_event = observation_event(
                event_id="evt-workspace-observation-lifecycle-after",
                event_type=EventType.WORKSPACE_REVISION_OBSERVED,
                phase="after",
            )
            after_captured = WorkspaceObservationLifecycle.model_validate(
                {
                    **before_published.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.AFTER_CAPTURED.value,
                    "after_state": WorkspaceObservationEvidenceState.PUBLISHED.value,
                    "after_observation_id": after_event.id,
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=before_published,
                current=after_captured,
                phase="after-capture",
                events=(after_event,),
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: after_captured}

            delta_event = observation_event(
                event_id="evt-workspace-observation-lifecycle-delta",
                event_type=EventType.WORKSPACE_MUTATION_RECORDED,
                status="changed",
            )
            delta_published = WorkspaceObservationLifecycle.model_validate(
                {
                    **after_captured.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.DELTA_PUBLISHED.value,
                    "delta_state": WorkspaceObservationEvidenceState.PUBLISHED.value,
                    "mutation_event_id": delta_event.id,
                    "mutation_event_digest": workspace_observation_event_digest(delta_event),
                }
            )
            conflicting_delta_event = delta_event.model_copy(
                update={
                    "payload": {
                        **delta_event.payload,
                        "before_observation_id": "evt-conflicting-before-observation",
                    }
                },
                deep=True,
            )
            conflicting_delta = delta_published.model_copy(
                update={
                    "mutation_event_digest": workspace_observation_event_digest(
                        conflicting_delta_event
                    )
                }
            )
            with pytest.raises(
                ValueError,
                match=r"intent\.before_observation_id",
            ):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=after_captured,
                    current=conflicting_delta,
                    phase="delta-publication",
                    events=(conflicting_delta_event,),
                )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=after_captured,
                current=delta_published,
                phase="delta-publication",
                events=(delta_event,),
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: delta_published}

            await store.release_run_fence(session_id)
            replacement_owner = await store.fence_run_and_transform_checkpoint(
                session_id,
                statuses={SessionStatus.PENDING},
                checkpoint_transform=lambda _session, checkpoint: checkpoint or {},
            )
            terminal_event = Event(
                id="evt-workspace-observation-lifecycle-terminal",
                type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                session_id=session_id,
                agent_name=intent.agent_name,
                environment_name=intent.environment_name,
                tool_name=intent.tool_name,
                payload={
                    "window_id": intent.window_id,
                    "session_run_epoch": intent.source_run_epoch,
                    "binding_generation_id": intent.binding_generation_id,
                    "workspace_id": intent.workspace_id,
                    "observer": intent.observer,
                    "artifact_store_id": intent.artifact_store_id,
                    "tool_call_id": intent.tool_call_id,
                    "model_step_id": intent.model_step_id,
                    "model_attempt_id": intent.model_attempt_id,
                    "tool_round_id": intent.tool_round_id,
                    "model_step": intent.model_step,
                    "before_observation_id": delta_published.before_observation_id,
                    "after_observation_id": delta_published.after_observation_id,
                    "tool_outcome_event_id": delta_published.tool_outcome_event_id,
                    "tool_outcome_event_digest": delta_published.tool_outcome_event_digest,
                    "mutation_event_id": delta_published.mutation_event_id,
                    "mutation_event_digest": delta_published.mutation_event_digest,
                    "revision_before_artifact_id": artifact_intent.artifact_id,
                    "revision_before_artifact_sha256": artifact_intent.sha256,
                    "revision_before_artifact_size_bytes": artifact_intent.size_bytes,
                    "revision_before_artifact_state": "referenced",
                    "referenced_artifact_count": 1,
                    "failed_artifact_count": 0,
                    "status": "incomplete",
                    "detail_code": "worker_lost_before_workspace_observation_completed",
                },
            )
            with pytest.raises(SessionRunFenced):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=delta_published,
                    current=None,
                    phase="terminal",
                    terminal_status=WorkspaceObservationTerminalStatus.INCOMPLETE,
                    terminal_detail_code="worker_lost_before_workspace_observation_completed",
                    terminal_artifacts=delta_published.artifacts,
                    events=(terminal_event,),
                )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: delta_published}

            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=replacement_owner,
                previous=delta_published,
                current=None,
                phase="terminal",
                terminal_status=WorkspaceObservationTerminalStatus.INCOMPLETE,
                terminal_detail_code="worker_lost_before_workspace_observation_completed",
                terminal_artifacts=delta_published.artifacts,
                events=(terminal_event,),
            )
            store = await _reopen_store(session_store_case, store)
            assert (
                workspace_observations_from_checkpoint(await store.load_checkpoint(session_id))
                == {}
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("crash_phase", "expected_status", "tool_ran", "recovery_repairs_observation"),
    [
        ("intent", "ambiguous", False, True),
        ("before-capture", "ambiguous", False, True),
        ("terminal-stage", "incomplete", True, True),
        ("tool-outcome", "incomplete", True, True),
        ("artifact-revision-after-intent", "incomplete", True, True),
        ("artifact-revision-after-published", "incomplete", True, True),
        ("before-evidence", "incomplete", True, True),
        ("after-capture", "incomplete", True, True),
        ("delta-publication", "complete", True, True),
        ("terminal", "complete", True, False),
    ],
)
def test_session_store_conformance_recovers_workspace_observation_through_public_runtime(
    session_store_case,
    crash_phase: str,
    expected_status: str,
    tool_ran: bool,
    recovery_repairs_observation: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        workspace_root = session_store_case[1] / f"workspace-observation-runtime-{crash_phase}"
        workspace_root.mkdir(exist_ok=True)
        artifact_root = session_store_case[1] / f"workspace-observation-artifacts-{crash_phase}"
        try:
            session_id = f"workspace-observation-runtime-{crash_phase}-{session_store_case[0]}"
            first_provider = _WorkspaceObservationRecoveryProvider()
            first_tool = _WorkspaceObservationRecoveryTool()
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(first_provider, default=True)
            first_app.register_environment(
                Environment(
                    _workspace_observation_recovery_environment_spec(),
                    workspace=LocalWorkspace(
                        workspace_root,
                        workspace_id="workspace-observation-runtime",
                    ),
                    artifact_store=LocalArtifactStore(
                        artifact_root,
                        store_id="workspace-observation-artifacts",
                    ),
                    binding=DeterministicWorkspaceBinding(),
                ),
                default=True,
            )
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[first_tool],
            )

            original_publish = store.publish_runtime_publication
            original_transform = store.transform_checkpoint
            lost_process = False

            async def publish_then_lose_process(session_id: str, **kwargs):
                nonlocal lost_process
                result = await original_publish(session_id, **kwargs)
                if (
                    not lost_process
                    and kwargs["request"].kind == "workspace-observation"
                    and kwargs["request"].intent.get("phase") == crash_phase
                ):
                    lost_process = True
                    raise _SimulatedProcessLoss(crash_phase)
                return result

            async def transform_then_lose_process(session_id: str, transform):
                nonlocal lost_process
                result = await original_transform(session_id, transform)
                if crash_phase == "terminal-stage" and not lost_process:
                    checkpoint = await store.load_checkpoint(session_id)
                    observations = (
                        None if checkpoint is None else checkpoint.get("workspace_observations")
                    )
                    pending_round = (
                        None if checkpoint is None else checkpoint.get("pending_tool_round")
                    )
                    if (
                        type(observations) is dict
                        and len(observations) == 1
                        and next(iter(observations.values())).get("phase") == "before_captured"
                        and type(pending_round) is dict
                        and pending_round.get("staged_terminals")
                    ):
                        lost_process = True
                        raise _SimulatedProcessLoss(crash_phase)
                return result

            store.publish_runtime_publication = (  # type: ignore[method-assign]
                publish_then_lose_process
            )
            store.transform_checkpoint = transform_then_lose_process  # type: ignore[method-assign]
            with pytest.raises(_SimulatedProcessLoss, match=crash_phase):
                _ = [
                    event
                    async for event in first_app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id=session_id,
                            messages=[Message.text("user", "write once")],
                        )
                    )
                ]
            assert lost_process is True
            assert first_provider.requests == 1
            assert first_tool.calls == (1 if tool_ran else 0)
            assert (workspace_root / "recovered.txt").exists() is tool_ran

            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]
            store.transform_checkpoint = original_transform  # type: ignore[method-assign]
            await store.release_run_fence(session_id)
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            store = await _reopen_store(session_store_case, store)

            recovery_provider = _WorkspaceObservationRecoveryProvider()
            recovery_tool = _WorkspaceObservationRecoveryTool()
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_environment(
                Environment(
                    _workspace_observation_recovery_environment_spec(),
                    workspace=LocalWorkspace(
                        workspace_root,
                        workspace_id="workspace-observation-runtime",
                    ),
                    artifact_store=LocalArtifactStore(
                        artifact_root,
                        store_id="workspace-observation-artifacts",
                    ),
                    binding=DeterministicWorkspaceBinding(),
                ),
                default=True,
            )
            recovery_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[recovery_tool],
            )
            competing_provider = _WorkspaceObservationRecoveryProvider()
            competing_tool = _WorkspaceObservationRecoveryTool()
            competing_app = CayuApp(session_store=store, enable_logging=False)
            competing_app.register_provider(competing_provider, default=True)
            competing_app.register_environment(
                Environment(
                    _workspace_observation_recovery_environment_spec(),
                    workspace=LocalWorkspace(
                        workspace_root,
                        workspace_id="workspace-observation-runtime",
                    ),
                    artifact_store=LocalArtifactStore(
                        artifact_root,
                        store_id="workspace-observation-artifacts",
                    ),
                    binding=DeterministicWorkspaceBinding(),
                ),
                default=True,
            )
            competing_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[competing_tool],
            )
            recoveries = await asyncio.gather(
                recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                ),
                competing_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                ),
            )
            durable = await store.query_events(EventQuery(session_id=session_id))
            finalized = [
                record.event
                for record in durable
                if record.event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
            ]
            assert len(finalized) == 1
            assert finalized[0].payload["status"] == expected_status
            if crash_phase == "artifact-revision-after-intent":
                assert finalized[0].payload["revision_after_artifact_state"] == "intent"
            elif crash_phase == "artifact-revision-after-published":
                assert finalized[0].payload["revision_after_artifact_state"] == "orphaned"
            assert sum(
                IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in recovery.actions
                for recovery in recoveries
            ) == (1 if recovery_repairs_observation else 0)
            assert recovery_provider.requests == 0
            assert competing_provider.requests == 0
            assert recovery_tool.calls == 0
            assert competing_tool.calls == 0
            assert (
                workspace_observations_from_checkpoint(await store.load_checkpoint(session_id))
                == {}
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("publication_process_signal", [False, True])
def test_session_store_conformance_workspace_observation_commit_then_cancel_fans_out(
    session_store_case,
    publication_process_signal: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"workspace-observation-cancel-{session_store_case[0]}"
            session = await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[],
                ),
                identity=_identity(),
            )
            sink = InMemoryEventSink()
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(sink,),
            )
            lifecycle = WorkspaceObservationLifecycle(
                session_id=session_id,
                window_id="wmut-cancel-after-commit",
                source_run_epoch=session.run_epoch,
                binding_generation_id="wbind-cancel-after-commit",
                workspace_id="workspace-cancel-after-commit",
                observer="ConformanceWorkspaceBinding",
                observer_authority="configured",
                agent_name="assistant",
                environment_name="local",
                tool_name="mutate",
                tool_call_id="call-workspace-cancel",
                model_step_id="mstep-workspace-cancel",
                model_attempt_id="matt-workspace-cancel",
                tool_round_id="tround-workspace-cancel",
                model_step=1,
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=None,
                current=lifecycle,
                phase="intent",
                intent_admission=_admit_test_workspace_observation_intent(lifecycle),
            )
            terminal = Event(
                id="evt-workspace-observation-cancel-terminal",
                type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                session_id=session_id,
                agent_name=lifecycle.agent_name,
                environment_name=lifecycle.environment_name,
                tool_name=lifecycle.tool_name,
                payload={
                    "window_id": lifecycle.window_id,
                    "session_run_epoch": lifecycle.source_run_epoch,
                    "binding_generation_id": lifecycle.binding_generation_id,
                    "workspace_id": lifecycle.workspace_id,
                    "observer": lifecycle.observer,
                    "artifact_store_id": None,
                    "tool_call_id": lifecycle.tool_call_id,
                    "model_step_id": lifecycle.model_step_id,
                    "model_attempt_id": lifecycle.model_attempt_id,
                    "tool_round_id": lifecycle.tool_round_id,
                    "model_step": lifecycle.model_step,
                    "referenced_artifact_count": 0,
                    "failed_artifact_count": 0,
                    "status": "ambiguous",
                    "detail_code": "worker_lost_before_tool_outcome_was_durable",
                },
            )
            original_publish = store.publish_runtime_publication
            committed = asyncio.Event()
            release_acknowledgement = asyncio.Event()
            process_signal_raised = False

            async def commit_then_hold_acknowledgement(session_id: str, **kwargs):
                nonlocal process_signal_raised
                result = await original_publish(session_id, **kwargs)
                if (
                    kwargs["request"].intent.get("phase") == "terminal"
                    and not process_signal_raised
                ):
                    committed.set()
                    await release_acknowledgement.wait()
                    if publication_process_signal:
                        process_signal_raised = True
                        raise SystemExit(17)
                return result

            store.publish_runtime_publication = (  # type: ignore[method-assign]
                commit_then_hold_acknowledgement
            )
            publishing = asyncio.create_task(
                publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=lifecycle,
                    current=None,
                    phase="terminal",
                    terminal_status=WorkspaceObservationTerminalStatus.AMBIGUOUS,
                    terminal_detail_code="worker_lost_before_tool_outcome_was_durable",
                    events=(terminal,),
                )
            )
            await committed.wait()
            publishing.cancel("workspace observation acknowledgement cancelled")
            release_acknowledgement.set()
            if publication_process_signal:
                with pytest.raises(BaseExceptionGroup) as raised_group:
                    await publishing
                assert len(raised_group.value.exceptions) == 2
                process_signal, cancellation = raised_group.value.exceptions
                assert isinstance(process_signal, SystemExit)
                assert process_signal.args == (17,)
                assert isinstance(cancellation, asyncio.CancelledError)
                assert cancellation.args == ("workspace observation acknowledgement cancelled",)
                assert publishing.cancelling() == 1
                assert publishing.cancelled() is False
            else:
                with pytest.raises(asyncio.CancelledError) as raised:
                    await publishing
                assert raised.value.args == ("workspace observation acknowledgement cancelled",)
                assert publishing.cancelling() == 1
                assert publishing.cancelled() is True
            assert [event.type for event in sink.events] == [
                EventType.WORKSPACE_OBSERVATION_FINALIZED
            ]
            assert (
                workspace_observations_from_checkpoint(await store.load_checkpoint(session_id))
                == {}
            )
            receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"workspace-observation:{lifecycle.window_id}:terminal",
            )
            assert receipt is not None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_event_projection_preserves_private_authority(
    session_store_case,
) -> None:
    class RecordingSink(EventSink):
        def __init__(self) -> None:
            self.events: list[Event] = []

        async def emit(self, event: Event) -> None:
            self.events.append(event.model_copy(deep=True))

    async def run() -> None:
        store = await _open_store(
            session_store_case,
            public_authority_alias_codec=_public_authority_alias_codec(),
        )
        try:
            session_id = f"eventprojection{session_store_case[0]}"
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[],
                ),
                identity=_identity(),
            )
            sink = RecordingSink()
            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                enable_logging=False,
                secret_redactor=SecretRedactor(["-", "step", "legacycanary", "explicitsecret"]),
            )
            emitted = await app.emit_event(
                Event(
                    type=EventType.MODEL_STARTED,
                    session_id=session_id,
                    payload={"step": 1},
                )
            )
            with pytest.raises(ValueError, match=r"event\.event_id"):
                await app.emit_event(
                    Event(
                        id="caller-explicitsecret-event",
                        type=EventType.MODEL_STARTED,
                        session_id=session_id,
                        payload={"step": 2},
                    )
                )
            with pytest.raises(ValueError, match=r"event\.payload\.tool_call_id"):
                await app.emit_event(
                    Event(
                        type=EventType.TOOL_CALL_STARTED,
                        session_id=session_id,
                        payload={"tool_call_id": "legacycanary"},
                    )
                )
            legacy = Event(
                type="custom.legacycanary",
                session_id=session_id,
                payload={"legacycanary": "legacycanary"},
            )
            await store.append_event(session_id, legacy)
            recovered = await app.recover_persisted_event_side_effects()
            records = await store.query_events(EventQuery(session_id=session_id))

            assert public_event_sequence(emitted.id) == records[0].sequence
            assert [record.event.id for record in records[1:]] == [legacy.id]
            assert [event.id for event in sink.events] == [
                public_event_id(records[0].sequence),
                public_event_id(records[1].sequence),
            ]
            assert recovered[0].id == public_event_id(records[1].sequence)
            assert sink.events[0].payload == {"step": 1}
            assert sink.events[1].type == REDACTED_CUSTOM_EVENT_TYPE
            assert "legacycanary" not in repr(sink.events[1].model_dump(mode="json"))
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_stale_approval_cannot_claim_repaused_session(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"approval-repause-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "approve")],
                ),
                identity=_identity(),
            )
            await store.update_status(session.id, SessionStatus.INTERRUPTED)
            checkpoint_a, approval_a = _approval_checkpoint("a")
            checkpoint_b, approval_b = _approval_checkpoint("b")
            await store.checkpoint(session.id, checkpoint_a)

            observed = _pending_approval_for_atomic_claim(
                await store.load_checkpoint(session.id),
                approval_id=approval_a.approval_id,
                tool_round_id=approval_a.tool_round_id,
                gating_tool_call_id=approval_a.tool_call_id,
                redactor=SecretRedactor(),
            )
            assert observed == approval_a

            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
            )
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
                checkpoint_transform=lambda _session, _checkpoint: checkpoint_b,
            )
            repaused = await store.load(session.id)
            assert repaused is not None
            repaused_epoch = repaused.run_epoch

            def stale_claim(
                _session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any] | None:
                _pending_approval_for_atomic_claim(
                    checkpoint,
                    approval_id=approval_a.approval_id,
                    tool_round_id=approval_a.tool_round_id,
                    gating_tool_call_id=approval_a.tool_call_id,
                    redactor=SecretRedactor(),
                )
                return checkpoint

            with pytest.raises(
                ValueError,
                match="identity does not match the current pending approval",
            ):
                await store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={SessionStatus.INTERRUPTED},
                    to_status=SessionStatus.RUNNING,
                    checkpoint_transform=stale_claim,
                )

            after = await store.load(session.id)
            assert after is not None
            assert after.status is SessionStatus.INTERRUPTED
            assert after.run_epoch == repaused_epoch
            assert await store.load_checkpoint(session.id) == checkpoint_b
            assert (
                _pending_approval_for_atomic_claim(
                    await store.load_checkpoint(session.id),
                    approval_id=approval_b.approval_id,
                    tool_round_id=approval_b.tool_round_id,
                    gating_tool_call_id=approval_b.tool_call_id,
                    redactor=SecretRedactor(),
                )
                == approval_b
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_approval_publication_is_atomic_across_restart(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"approval-publication-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "approve")],
                ),
                identity=_identity(),
            )
            paired_checkpoint, approval = _approval_checkpoint("published")
            raw_round = tool_round_recovery.PendingToolRound(
                tool_round_id=approval.tool_round_id,
                model_step_id=approval.model_step_id,
                model_attempt_id=approval.model_attempt_id,
                agent_name="assistant",
                tool_calls=[
                    PendingToolCallApproval(
                        tool_call_id=approval.tool_call_id,
                        tool_name=approval.tool_name,
                        arguments=approval.arguments,
                    )
                ],
                policy_context_version=1,
            )
            raw_checkpoint = {
                tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY: (
                    raw_round.model_dump(mode="json")
                )
            }
            await store.checkpoint(session.id, raw_checkpoint)
            checkpoint_event = Event(
                id="approval-checkpointed",
                type=EventType.SESSION_CHECKPOINTED,
                session_id=session.id,
                payload={
                    "checkpoint": approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
                    "approval_id": approval.approval_id,
                    "tool_call_id": approval.tool_call_id,
                    "tool_round_id": approval.tool_round_id,
                    "model_step_id": approval.model_step_id,
                    "model_attempt_id": approval.model_attempt_id,
                },
            )
            approval_event = Event(
                id="approval-requested",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id=session.id,
                tool_name=approval.tool_name,
                payload={
                    "approval_id": approval.approval_id,
                    "tool_call_id": approval.tool_call_id,
                    "tool_round_id": approval.tool_round_id,
                    "model_step_id": approval.model_step_id,
                    "model_attempt_id": approval.model_attempt_id,
                    "approval": approval.model_dump(mode="json"),
                },
            )
            await store.append_event(
                session.id,
                Event(
                    id=approval_event.id,
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id=session.id,
                ),
            )

            with pytest.raises(ValueError, match="Event already exists for session"):
                await store.publish_checkpoint_and_events(
                    session.id,
                    checkpoint_transform=lambda _session, _checkpoint: paired_checkpoint,
                    events=[checkpoint_event, approval_event],
                )
            assert await store.load_checkpoint(session.id) == raw_checkpoint
            assert [event.id for event in await store.load_events(session.id)] == [
                approval_event.id
            ]

            replacement_approval_event = approval_event.model_copy(
                update={"id": "approval-requested-committed"},
                deep=True,
            )
            await store.publish_checkpoint_and_events(
                session.id,
                checkpoint_transform=lambda _session, _checkpoint: paired_checkpoint,
                events=[checkpoint_event, replacement_approval_event],
            )
            store = await _reopen_store(session_store_case, store)
            assert await store.load_checkpoint(session.id) == paired_checkpoint
            assert [event.id for event in await store.load_events(session.id)] == [
                approval_event.id,
                checkpoint_event.id,
                replacement_approval_event.id,
            ]
            assert (
                _pending_approval_for_atomic_claim(
                    await store.load_checkpoint(session.id),
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    gating_tool_call_id=approval.tool_call_id,
                    redactor=SecretRedactor(),
                )
                == approval
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_legacy_approval_round_migrates_in_atomic_claim(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"legacy-approval-claim-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "approve")],
                ),
                identity=_identity(),
            )
            await store.update_status(session.id, SessionStatus.INTERRUPTED)
            paired_checkpoint, approval = _approval_checkpoint("legacy")
            legacy_checkpoint = {
                approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: (
                    paired_checkpoint[approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY]
                )
            }
            await store.checkpoint(session.id, legacy_checkpoint)
            claimed_approval: PendingToolApproval | None = None

            def claim_legacy(
                _session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any] | None:
                nonlocal claimed_approval
                claimed_approval = _pending_approval_for_atomic_claim(
                    checkpoint,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    gating_tool_call_id=approval.tool_call_id,
                    redactor=SecretRedactor(),
                )
                return _checkpoint_with_legacy_approval_round(
                    checkpoint,
                    approval=claimed_approval,
                    redactor=SecretRedactor(),
                )

            claimed = await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=claim_legacy,
            )
            assert claimed.status is SessionStatus.RUNNING
            assert claimed_approval == approval

            store = await _reopen_store(session_store_case, store)
            migrated = await store.load_checkpoint(session.id)
            assert approval_support.pending_approval_from_checkpoint(migrated) == approval
            migrated_round = tool_round_recovery.pending_tool_round_from_checkpoint(migrated)
            assert migrated_round is not None
            assert migrated_round.policy_state == "planned"
            assert migrated_round.policy_context_version == 1
            assert migrated_round.tool_round_id == approval.tool_round_id
            assert migrated_round.tool_calls == approval.tool_calls
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "first_decision",
    [ToolApprovalDecision.APPROVE, ToolApprovalDecision.DENY],
    ids=["approve-then-deny", "deny-then-approve"],
)
def test_session_store_conformance_pre_digest_approval_claim_fails_closed(
    session_store_case,
    first_decision: ToolApprovalDecision,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        binding = _ApprovalResolutionBinding()
        tool_calls: list[dict[str, Any]] = []
        session_id = f"approval-decision-intent-{first_decision.value}-{session_store_case[0]}"
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(_ApprovalRecoveryProvider(), default=True)
            first_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    binding=binding,
                ),
                default=True,
            )
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            first_events = [
                event
                async for event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            requested = next(
                event
                for event in first_events
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            assert binding.bind_calls == 1

            binding.fail_next = True
            first_request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=first_decision,
            )
            first_request_digest = approval_support.approval_resolution_request_digest(
                first_request
            )
            failed_resolution = [
                event async for event in first_app.resolve_tool_approval(first_request)
            ]
            assert failed_resolution[0].type is EventType.INTERACTION_RESUMED
            assert any(event.type is EventType.SESSION_RESUMED for event in failed_resolution)
            assert failed_resolution[-1].type is EventType.SESSION_INTERRUPTED
            assert binding.bind_calls == 2
            assert tool_calls == []

            checkpoint = await store.load_checkpoint(session_id)
            intent = approval_support.approval_resolution_intent_from_checkpoint(
                checkpoint,
                redactor=SecretRedactor(),
            )
            assert intent == approval_support.approval_resolution_intent_for(
                approval,
                decision=first_decision,
                resolution_request_digest=first_request_digest,
            )
            legacy_checkpoint = await store.load_checkpoint(session_id)
            assert legacy_checkpoint is not None
            legacy_intent = dict(
                legacy_checkpoint[approval_support.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY]
            )
            legacy_intent.pop("resolution_request_digest")
            await store.checkpoint(
                session_id,
                {
                    **legacy_checkpoint,
                    approval_support.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY: legacy_intent,
                },
            )

            store = await _reopen_store(session_store_case, store)
            retry_app = CayuApp(session_store=store, enable_logging=False)
            retry_app.register_provider(
                _ApprovalRecoveryProvider(complete_without_tools=True),
                default=True,
            )
            retry_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    binding=binding,
                ),
                default=True,
            )
            retry_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            opposite = (
                ToolApprovalDecision.DENY
                if first_decision is ToolApprovalDecision.APPROVE
                else ToolApprovalDecision.APPROVE
            )
            opposite_request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=opposite,
            )
            conflicting = [
                event async for event in retry_app.resolve_tool_approval(opposite_request)
            ]
            assert [event.type for event in conflicting] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert (
                "already claimed with a different resolution decision"
                in conflicting[-1].payload["error"]
            )
            assert binding.bind_calls == 2
            assert tool_calls == []
            interrupted = await store.load(session_id)
            assert interrupted is not None
            assert interrupted.status is SessionStatus.INTERRUPTED
            assert approval_support.approval_resolution_intent_from_checkpoint(
                await store.load_checkpoint(session_id),
                redactor=SecretRedactor(),
            ) == approval_support.approval_resolution_intent_for(
                approval,
                decision=first_decision,
                resolution_request_digest=None,
            )

            legacy_retry = [event async for event in retry_app.resolve_tool_approval(first_request)]
            assert [event.type for event in legacy_retry] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert "predates exact resolution request identity" in legacy_retry[-1].payload["error"]
            assert binding.bind_calls == 2
            assert tool_calls == []
            final_checkpoint = await store.load_checkpoint(session_id)
            assert approval_support.approval_resolution_intent_from_checkpoint(
                final_checkpoint,
                redactor=SecretRedactor(),
            ) == approval_support.approval_resolution_intent_for(
                approval,
                decision=first_decision,
                resolution_request_digest=None,
            )
            assert approval_support.pending_approval_from_checkpoint(final_checkpoint) == approval
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_approval_open_and_close_lost_ack_replay_exact_receipts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        calls: list[dict[str, Any]] = []
        session_id = f"approval-close-lost-ack-{session_store_case[0]}"
        try:
            original_publish = store.publish_runtime_publication
            lost_ack_kinds: set[str] = set()

            async def publish_then_lose_ack(session_id: str, **kwargs):
                result = await original_publish(session_id, **kwargs)
                kind = kwargs["request"].kind
                if kind in {"approval-open", "approval-close"} and kind not in lost_ack_kinds:
                    lost_ack_kinds.add(kind)
                    raise OSError(f"{kind} acknowledgement lost")
                return result

            store.publish_runtime_publication = publish_then_lose_ack  # type: ignore[method-assign]
            provider = _ApprovalRecoveryProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            approval = await _pending_approval_for_public_event(
                store,
                next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                ),
            )
            assert "approval-open" in lost_ack_kinds
            open_receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"approval-open:{approval.approval_id}",
            )
            assert open_receipt is not None
            assert open_receipt.kind == "approval-open"
            provider._complete_without_tools = True
            request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.DENY,
                reason="conformance denial",
            )
            resolved = [event async for event in app.resolve_tool_approval(request)]
            assert resolved[-1].type is EventType.SESSION_COMPLETED
            assert lost_ack_kinds == {"approval-open", "approval-close"}
            assert calls == []
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_tool_approval" not in checkpoint
            assert "pending_tool_round" not in checkpoint
            receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"approval-close:{approval.approval_id}",
            )
            assert receipt is not None
            assert receipt.kind == "approval-close"

            replayed = [event async for event in app.resolve_tool_approval(request)]
            private_replayed = [
                await _private_event_for_public_event(store, event) for event in replayed
            ]
            assert tuple(event.id for event in private_replayed) == receipt.appended_event_ids
            assert calls == []
            with pytest.raises(RuntimeError, match="conflicting identity or decision"):
                _ = [
                    event
                    async for event in app.resolve_tool_approval(
                        request.model_copy(update={"decision": ToolApprovalDecision.APPROVE})
                    )
                ]

            later_message = Message.text("user", "later recovered interaction input")
            later_interaction_id = "interaction-after-old-approval-receipt"
            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_source_messages=[later_message],
                continued_interaction_id=later_interaction_id,
                defer_interaction_source=True,
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.release_run_fence(session_id)

            stale_replay = [event async for event in app.resolve_tool_approval(request)]
            private_stale_replay = [
                await _private_event_for_public_event(store, event) for event in stale_replay
            ]
            assert tuple(event.id for event in private_stale_replay) == receipt.appended_event_ids
            deferred = await store.load_deferred_interaction_input(session_id)
            assert deferred is not None
            assert deferred.interaction_id == later_interaction_id
            assert deferred.source_messages == [later_message]
            assert later_message not in await store.load_transcript(session_id)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_approval_limit_close_replays_exact_request(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        calls: list[dict[str, Any]] = []
        session_id = f"approval-limit-close-replay-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_ApprovalLimitProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            approval = await _pending_approval_for_public_event(
                store,
                next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                ),
            )
            request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="approved within the configured limit",
                metadata={"ticket": "OPS-526"},
                resolved_by=ResolutionActor(subject="operator-1"),
                limits=RunLimits(max_total_tokens=1, scope="session"),
            )

            closed = [event async for event in app.resolve_tool_approval(request)]
            assert EventType.SESSION_LIMIT_REACHED in [event.type for event in closed]
            assert calls == []
            receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"approval-close:{approval.approval_id}",
            )
            assert receipt is not None
            assert receipt.intent["decision"] == "limit_reached"
            assert receipt.intent["requested_decision"] == request.decision.value
            assert receipt.intent["resolution_request_digest"] == (
                runtime_publication_checkpoint_value_digest(
                    request.model_dump(
                        mode="json",
                        include={
                            "decision",
                            "reason",
                            "metadata",
                            "resolved_by",
                        },
                    )
                )
            )

            replayed = [event async for event in app.resolve_tool_approval(request)]
            private_replayed = [
                await _private_event_for_public_event(store, event) for event in replayed
            ]
            assert tuple(event.id for event in private_replayed) == receipt.appended_event_ids
            assert calls == []

            with pytest.raises(RuntimeError, match="conflicting identity or decision"):
                _ = [
                    event
                    async for event in app.resolve_tool_approval(
                        request.model_copy(update={"decision": ToolApprovalDecision.DENY})
                    )
                ]
            assert calls == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_approval_event_ack_loss_rejects_request_drift(
    session_store_case,
) -> None:
    class FailFirstApprovedEventSink(EventSink):
        def __init__(self) -> None:
            self.failed = False

        async def emit(self, event: Event) -> None:
            if event.type is EventType.TOOL_CALL_APPROVED and not self.failed:
                self.failed = True
                raise _SimulatedProcessLoss()

    async def run() -> None:
        store = await _open_store(session_store_case)
        calls: list[dict[str, Any]] = []
        session_id = f"approval-event-ack-loss-{session_store_case[0]}"
        redactor = SecretRedactor(["resolution", "digest"])
        try:
            original_append = store.append_event
            lost_append_ack = False

            async def append_then_lose_approved_ack(session_id: str, event: Event) -> None:
                nonlocal lost_append_ack
                await original_append(session_id, event)
                if event.type is EventType.TOOL_CALL_APPROVED and not lost_append_ack:
                    lost_append_ack = True
                    raise OSError("approval event append acknowledgement lost")

            store.append_event = append_then_lose_approved_ack  # type: ignore[method-assign]
            sink = FailFirstApprovedEventSink()
            provider = _ApprovalRecoveryProvider()
            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                secret_redactor=redactor,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalMetadataTool(calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            approval = await _pending_approval_for_public_event(
                store,
                next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                ),
            )
            request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="approved under the original condition",
                metadata={
                    "oversized_audit_value": "x" * 20_000,
                    "condition": "limit-500",
                },
                resolved_by=ResolutionActor(
                    subject="operator-1",
                    claims={"approval_scope": "sensitive-scope"},
                ),
            )
            reordered_retry = request.model_copy(
                update={
                    "metadata": {
                        "condition": "limit-500",
                        "oversized_audit_value": "x" * 20_000,
                    }
                }
            )
            assert approval_support.approval_resolution_request_digest(
                reordered_retry
            ) == approval_support.approval_resolution_request_digest(request)

            with pytest.raises(_SimulatedProcessLoss):
                _ = [event async for event in app.resolve_tool_approval(request)]
            assert lost_append_ack is True
            assert sink.failed is True
            assert calls == []

            store = await _reopen_store(session_store_case, store)
            retry_provider = _ApprovalRecoveryProvider(complete_without_tools=True)
            retry_app = CayuApp(
                session_store=store,
                secret_redactor=redactor,
                enable_logging=False,
            )
            retry_app.register_provider(retry_provider, default=True)
            retry_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalMetadataTool(calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            recovered = await retry_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            assert recovered.actions == (IncompleteSessionRecoveryAction.PENDING_APPROVAL,)
            checkpoint = await store.load_checkpoint(session_id)
            intent = approval_support.approval_resolution_intent_from_checkpoint(
                checkpoint,
                redactor=redactor,
            )
            assert intent is not None
            assert intent.resolution_request_digest == (
                approval_support.approval_resolution_request_digest(request)
            )
            approved_events = [
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.TOOL_CALL_APPROVED
            ]
            assert len(approved_events) == 1
            approved_event_id = approved_events[0].id
            assert "resolution_request_digest" not in approved_events[0].payload
            assert approved_events[0].payload["resolved_by"] == {
                "subject": "operator-1",
                "tenant": None,
                "source": None,
            }
            assert "claims" not in approved_events[0].payload["resolved_by"]

            conflicting = [
                event
                async for event in retry_app.resolve_tool_approval(
                    reordered_retry.model_copy(
                        update={
                            "metadata": {
                                "condition": "limit-5000",
                                "oversized_audit_value": "x" * 20_000,
                            },
                            "resolved_by": ResolutionActor(subject="operator-2"),
                        }
                    )
                )
            ]
            assert [event.type for event in conflicting] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert "different" in conflicting[-1].payload["error"]
            assert "request" in conflicting[-1].payload["error"]
            assert calls == []
            still_interrupted = await store.load(session_id)
            assert still_interrupted is not None
            assert still_interrupted.status is SessionStatus.INTERRUPTED

            completed = [event async for event in retry_app.resolve_tool_approval(reordered_retry)]
            assert completed[-1].type is EventType.SESSION_COMPLETED
            assert calls == [
                {
                    "arguments": {"value": "must remain gated"},
                    "condition": "limit-500",
                }
            ]
            approved_events = [
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.TOOL_CALL_APPROVED
            ]
            assert len(approved_events) == 1
            assert approved_events[0].id == approved_event_id
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "historical_decision",
    [ToolApprovalDecision.APPROVE, ToolApprovalDecision.DENY],
    ids=["historical-approve", "historical-deny"],
)
def test_session_store_conformance_legacy_history_cannot_be_poisoned_by_retry(
    session_store_case,
    historical_decision: ToolApprovalDecision,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        binding = _ApprovalResolutionBinding()
        tool_calls: list[dict[str, Any]] = []
        session_id = f"approval-legacy-history-{historical_decision.value}-{session_store_case[0]}"
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(_ApprovalRecoveryProvider(), default=True)
            first_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    binding=binding,
                ),
                default=True,
            )
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            first_events = [
                event
                async for event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            requested = next(
                event
                for event in first_events
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            historical_payload: dict[str, Any] = {
                "model_step_id": approval.model_step_id,
                "model_attempt_id": approval.model_attempt_id,
                "tool_round_id": approval.tool_round_id,
                "approval_id": approval.approval_id,
                "tool_call_id": approval.tool_call_id,
            }
            historical_type = (
                EventType.TOOL_CALL_APPROVED
                if historical_decision is ToolApprovalDecision.APPROVE
                else EventType.TOOL_CALL_APPROVAL_DENIED
            )
            if historical_decision is ToolApprovalDecision.APPROVE:
                historical_payload.update(
                    {
                        "reason": "legacy approval reason",
                        "metadata": {"condition": "legacy-approved"},
                        "resolved_by": {
                            "subject": "legacy-operator",
                            "tenant": None,
                            "source": None,
                        },
                    }
                )
            else:
                historical_payload.update(
                    {
                        "approval_required": True,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session_id,
                            tool_round_id=approval.tool_round_id,
                            tool_call_id=approval.tool_call_id,
                            approval_id=approval.approval_id,
                        ),
                        "result": ToolResult(
                            content="historically denied",
                            is_error=True,
                        ).model_dump(mode="json"),
                    }
                )
            paused_session = await store.load(session_id)
            assert paused_session is not None
            await store.append_event(
                session_id,
                approval_support.resumed_event(
                    session=paused_session,
                    agent_name="assistant",
                    environment_name="approval-environment",
                    approval=approval,
                    decision=historical_decision,
                ),
            )
            await store.append_event(
                session_id,
                Event(
                    type=historical_type,
                    session_id=session_id,
                    agent_name="assistant",
                    environment_name="approval-environment",
                    tool_name=approval.tool_name,
                    payload=historical_payload,
                ),
            )
            assert (
                approval_support.approval_resolution_intent_from_checkpoint(
                    await store.load_checkpoint(session_id),
                    redactor=SecretRedactor(),
                )
                is None
            )

            store = await _reopen_store(session_store_case, store)
            retry_app = CayuApp(session_store=store, enable_logging=False)
            retry_app.register_provider(
                _ApprovalRecoveryProvider(complete_without_tools=True),
                default=True,
            )
            retry_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    binding=binding,
                ),
                default=True,
            )
            retry_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            matching_request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=historical_decision,
                reason=(
                    "legacy approval reason"
                    if historical_decision is ToolApprovalDecision.APPROVE
                    else None
                ),
                metadata=(
                    {"condition": "legacy-approved"}
                    if historical_decision is ToolApprovalDecision.APPROVE
                    else {}
                ),
                resolved_by=(
                    ResolutionActor(subject="legacy-operator")
                    if historical_decision is ToolApprovalDecision.APPROVE
                    else None
                ),
            )
            if historical_decision is ToolApprovalDecision.APPROVE:
                drifted = [
                    event
                    async for event in retry_app.resolve_tool_approval(
                        matching_request.model_copy(
                            update={
                                "metadata": {"condition": "changed"},
                                "resolved_by": ResolutionActor(subject="different-operator"),
                            }
                        )
                    )
                ]
                assert drifted[-1].type is EventType.SESSION_INTERRUPTED
                assert (
                    "prior durable resolution activity has no exact resolution request identity"
                    in drifted[-1].payload["error"]
                )
                assert tool_calls == []
                assert (
                    approval_support.approval_resolution_intent_from_checkpoint(
                        await store.load_checkpoint(session_id),
                        redactor=SecretRedactor(),
                    )
                    is None
                )

            conflicting_decision = (
                ToolApprovalDecision.DENY
                if historical_decision is ToolApprovalDecision.APPROVE
                else ToolApprovalDecision.APPROVE
            )
            conflicting = [
                event
                async for event in retry_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=conflicting_decision,
                    )
                )
            ]
            assert [event.type for event in conflicting] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert binding.bind_calls == 1
            assert tool_calls == []
            assert (
                approval_support.approval_resolution_intent_from_checkpoint(
                    await store.load_checkpoint(session_id),
                    redactor=SecretRedactor(),
                )
                is None
            )

            legacy_retry = [
                event async for event in retry_app.resolve_tool_approval(matching_request)
            ]
            assert [event.type for event in legacy_retry] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert (
                "prior durable resolution activity has no exact resolution request identity"
                in legacy_retry[-1].payload["error"]
            )
            assert binding.bind_calls == 1
            assert tool_calls == []
            final_checkpoint = await store.load_checkpoint(session_id)
            assert (
                approval_support.approval_resolution_intent_from_checkpoint(
                    final_checkpoint,
                    redactor=SecretRedactor(),
                )
                is None
            )
            assert approval_support.pending_approval_from_checkpoint(final_checkpoint) == approval
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_lossy_legacy_grant_cannot_authorize_pending_sibling(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        calls: list[dict[str, Any]] = []
        session_id = f"approval-lossy-legacy-mixed-{session_store_case[0]}"
        original_secret = "legacy-resolution-secret-original"
        changed_secret = "legacy-resolution-secret-changed"
        redactor = SecretRedactor([original_secret, changed_secret])
        try:
            provider = _ApprovalMixedRecoveryProvider()
            policy_calls: list[ToolPolicyDecision] = []
            app = CayuApp(
                session_store=store,
                secret_redactor=redactor,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalMetadataTool(calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run both protected tools")],
                    )
                )
            ]
            approval = await _pending_approval_for_public_event(
                store,
                next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                ),
            )
            assert [call.policy_decision for call in approval.tool_calls] == [
                ToolPolicyDecision.REQUIRE_APPROVAL.value,
                ToolPolicyDecision.ALLOW.value,
            ]
            assert calls == []

            paused_session = await store.load(session_id)
            assert paused_session is not None
            await store.append_event(
                session_id,
                approval_support.resumed_event(
                    session=paused_session,
                    agent_name="assistant",
                    environment_name=None,
                    approval=approval,
                    decision=ToolApprovalDecision.APPROVE,
                    resolved_by=ResolutionActor(subject="legacy-operator"),
                ),
            )
            bounded_metadata = approval_support.bounded_resolution_metadata_payload(
                {
                    "credential": original_secret,
                    "large": "x" * 20_000,
                },
                redactor=redactor,
            )
            assert bounded_metadata["metadata"]["credential"] == REDACTED_SECRET
            assert bounded_metadata["metadata_truncated"] is True
            await store.append_event(
                session_id,
                Event(
                    type=EventType.TOOL_CALL_APPROVED,
                    session_id=session_id,
                    agent_name="assistant",
                    tool_name=approval.tool_calls[0].tool_name,
                    payload={
                        "model_step_id": approval.model_step_id,
                        "model_attempt_id": approval.model_attempt_id,
                        "tool_round_id": approval.tool_round_id,
                        "approval_id": approval.approval_id,
                        "tool_call_id": approval.tool_calls[0].tool_call_id,
                        "reason": "legacy approval",
                        **bounded_metadata,
                        "resolved_by": {
                            "subject": "legacy-operator",
                            "tenant": None,
                            "source": None,
                        },
                    },
                ),
            )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                    tool_name=approval.tool_calls[0].tool_name,
                    payload={
                        "model_step_id": approval.model_step_id,
                        "model_attempt_id": approval.model_attempt_id,
                        "tool_round_id": approval.tool_round_id,
                        "approval_id": approval.approval_id,
                        "tool_call_id": approval.tool_calls[0].tool_call_id,
                        "result": ToolResult(content="historically completed").model_dump(
                            mode="json"
                        ),
                    },
                ),
            )

            store = await _reopen_store(session_store_case, store)
            retry_app = CayuApp(
                session_store=store,
                secret_redactor=redactor,
                enable_logging=False,
            )
            retry_app.register_provider(
                _ApprovalMixedRecoveryProvider(complete_without_tools=True),
                default=True,
            )
            retry_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalMetadataTool(calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )
            retry = [
                event
                async for event in retry_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                        reason="legacy approval",
                        metadata={
                            "credential": changed_secret,
                            "large": "x" * 20_000,
                        },
                        resolved_by=ResolutionActor(subject="legacy-operator"),
                    )
                )
            ]
            assert [event.type for event in retry] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert (
                "prior durable resolution activity has no exact resolution request identity"
                in retry[-1].payload["error"]
            )
            assert calls == []
            checkpoint = await store.load_checkpoint(session_id)
            assert (
                approval_support.pending_approval_from_checkpoint(
                    checkpoint,
                    redactor=redactor,
                )
                == approval
            )
            assert (
                approval_support.approval_resolution_intent_from_checkpoint(
                    checkpoint,
                    redactor=redactor,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "legacy_unversioned",
    [False, True],
    ids=["versioned-raw-round", "legacy-unversioned-raw-round"],
)
@pytest.mark.parametrize(
    "resolution_decision",
    [ToolApprovalDecision.APPROVE, ToolApprovalDecision.DENY],
    ids=["approve", "deny"],
)
def test_session_store_conformance_ambiguous_policy_recovery_remains_gated(
    session_store_case,
    legacy_unversioned: bool,
    resolution_decision: ToolApprovalDecision,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        policy_calls: list[ToolPolicyDecision] = []
        tool_calls: list[dict[str, Any]] = []
        session_id = (
            f"ambiguous-policy-{legacy_unversioned}-"
            f"{resolution_decision.value}-{session_store_case[0]}"
        )
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(_ApprovalRecoveryProvider(), default=True)
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )

            async def lose_approval_publication(**_kwargs) -> None:
                raise _SimulatedProcessLoss(
                    "process stopped after policy evaluation and before publication"
                )

            first_app._tool_round_executor.checkpoint_pending_tool_approval = (
                lose_approval_publication
            )
            with pytest.raises(
                _SimulatedProcessLoss,
                match="after policy evaluation",
            ):
                async for _event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "use the protected tool")],
                    )
                ):
                    pass

            assert policy_calls == [ToolPolicyDecision.REQUIRE_APPROVAL]
            raw_checkpoint = await store.load_checkpoint(session_id)
            assert raw_checkpoint is not None
            assert "pending_tool_approval" not in raw_checkpoint
            raw_round = dict(raw_checkpoint["pending_tool_round"])
            assert raw_round["policy_state"] == "unplanned"
            if legacy_unversioned:
                raw_round.pop("policy_state")
                raw_round.pop("policy_context_version")
                raw_checkpoint = dict(raw_checkpoint)
                raw_checkpoint["pending_tool_round"] = raw_round
                await store.checkpoint(session_id, raw_checkpoint)
            await store.release_run_fence(session_id)
            await store.update_status(session_id, SessionStatus.INTERRUPTED)

            store = await _reopen_store(session_store_case, store)
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_provider = _ApprovalRecoveryProvider(complete_without_tools=True)
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )
            deferred_message = Message.text(
                "user",
                "continue after the recovered approval",
            )
            resume_events = [
                event
                async for event in recovery_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[deferred_message],
                    )
                )
            ]

            assert policy_calls == [ToolPolicyDecision.REQUIRE_APPROVAL]
            assert tool_calls == []
            assert recovery_provider.requests == []
            assert any(
                event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for event in resume_events
            )
            recovered_session = await store.load(session_id)
            assert recovered_session is not None
            assert recovered_session.status is SessionStatus.INTERRUPTED
            recovered_checkpoint = await store.load_checkpoint(session_id)
            assert recovered_checkpoint is not None
            approval = PendingToolApproval.model_validate(
                recovered_checkpoint["pending_tool_approval"]
            )
            planned_round = tool_round_recovery.PendingToolRound.model_validate(
                recovered_checkpoint["pending_tool_round"]
            )
            assert planned_round.policy_state == "planned"
            assert planned_round.policy_context_version == 1
            assert planned_round.deferred_messages == [deferred_message]
            assert approval.tool_round_id == planned_round.tool_round_id
            assert approval.tool_call_id == "call_policy_recovery"
            assert approval.tool_calls[0].policy_evidence == "ambiguous"
            assert approval.tool_calls[0].policy_decision is None
            assert approval.metadata == {
                "recovered": True,
                "policy_evaluation": "ambiguous",
            }
            requested = [
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            ]
            assert len(requested) == 1
            assert requested[0].payload["approval_id"] == approval.approval_id
            assert requested[0].payload["recovered"] is True

            resolution_events = [
                event
                async for event in recovery_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=resolution_decision,
                    )
                )
            ]
            assert tool_calls == []
            if resolution_decision is ToolApprovalDecision.APPROVE:
                ambiguous_blocks = [
                    event
                    for event in resolution_events
                    if event.type is EventType.TOOL_CALL_BLOCKED
                    and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
                ]
                assert len(ambiguous_blocks) == 1
                assert (
                    ambiguous_blocks[0].payload["requested_decision"]
                    == ToolApprovalDecision.APPROVE.value
                )
            assert resolution_events[-1].type is EventType.SESSION_COMPLETED
            assert len(recovery_provider.requests) == 1
            assert [message.role.value for message in recovery_provider.requests[0].messages] == [
                "user",
                "assistant",
                "tool",
                "user",
            ]
            assert recovery_provider.requests[0].messages[-1] == deferred_message
            resolved_session = await store.load(session_id)
            assert resolved_session is not None
            assert resolved_session.status is SessionStatus.COMPLETED
            resolved_transcript = await store.load_transcript(session_id)
            assert [message.role.value for message in resolved_transcript] == [
                "user",
                "assistant",
                "tool",
                "user",
                "assistant",
            ]
            assert resolved_transcript.count(deferred_message) == 1
            assert await store.load_deferred_interaction_input(session_id) is None
            resolved_checkpoint = await store.load_checkpoint(session_id)
            assert "pending_tool_approval" not in (resolved_checkpoint or {})
            assert "pending_tool_round" not in (resolved_checkpoint or {})
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "lost_outcome",
    ["deny", "exception"],
    ids=["lost-deny", "lost-policy-exception"],
)
def test_session_store_conformance_lost_policy_authority_never_becomes_executable(
    session_store_case,
    lost_outcome: str,
) -> None:
    """A missing durable outcome is not authorization, even after approval."""

    class LostOutcomePolicy(ToolPolicy):
        def __init__(self, calls: list[str]) -> None:
            self._calls = calls

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name=f"tests:lost-outcome-policy:{lost_outcome}",
                behavior_version="1",
                implementation_version="1",
            )

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            self._calls.append(lost_outcome)
            if lost_outcome == "exception":
                raise RuntimeError("policy backend failed before durable publication")
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="hard denial that must never be downgraded",
            )

    async def run() -> None:
        store = await _open_store(session_store_case)
        policy_calls: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        session_id = f"lost-{lost_outcome}-{session_store_case[0]}"
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(_ApprovalRecoveryProvider(), default=True)
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=LostOutcomePolicy(policy_calls),
            )

            if lost_outcome == "deny":

                async def lose_policy_plan_publication(**_kwargs) -> None:
                    raise _SimulatedProcessLoss(
                        "process stopped after denial and before publication"
                    )

                first_app._tool_round_executor.checkpoint_tool_round_policy_plan = (
                    lose_policy_plan_publication
                )
                with pytest.raises(_SimulatedProcessLoss, match="after denial"):
                    async for _event in first_app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "use the protected tool")],
                        )
                    ):
                        pass
                await store.release_run_fence(session_id)
                await store.update_status(session_id, SessionStatus.INTERRUPTED)
            else:
                events = [
                    event
                    async for event in first_app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "use the protected tool")],
                        )
                    )
                ]
                assert events[-1].type is EventType.SESSION_FAILED

            raw_checkpoint = await store.load_checkpoint(session_id)
            assert raw_checkpoint is not None
            assert "pending_tool_approval" not in raw_checkpoint
            assert raw_checkpoint["pending_tool_round"]["policy_state"] == "unplanned"
            assert policy_calls == [lost_outcome]
            assert tool_calls == []

            store = await _reopen_store(session_store_case, store)
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_provider = _ApprovalRecoveryProvider(complete_without_tools=True)
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=LostOutcomePolicy(policy_calls),
            )

            resume_events = [
                event
                async for event in recovery_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "recover safely")],
                    )
                )
            ]
            requested = next(
                event
                for event in resume_events
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            assert approval.tool_calls[0].policy_evidence == "ambiguous"
            assert approval.tool_calls[0].policy_decision is None

            resolution_events = [
                event
                async for event in recovery_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
            assert policy_calls == [lost_outcome]
            assert tool_calls == []
            ambiguous_block = next(
                event for event in resolution_events if event.type is EventType.TOOL_CALL_BLOCKED
            )
            assert ambiguous_block.payload["blocked_by"] == "policy_evaluation_ambiguous"
            assert not any(
                event.type in {EventType.TOOL_CALL_APPROVED, EventType.TOOL_CALL_STARTED}
                for event in resolution_events
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_registration_drift_rejects_paused_call(
    session_store_case,
) -> None:
    class MixedProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.tool_call(
                id="call_late",
                name="late_effect",
                arguments={"value": "must not execute"},
            )
            yield ModelStreamEvent.tool_call(
                id="call_approval",
                name="stateful_effect",
                arguments={"value": "approved effect"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    class LateEffectTool(Tool):
        spec = ToolSpec(
            name="late_effect",
            description="A tool registered only after the round paused.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self, calls: list[dict[str, Any]]) -> None:
            super().__init__()
            self._calls = calls

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx
            self._calls.append(dict(args))
            return ToolResult(content="unexpected")

    async def run() -> None:
        store = await _open_store(session_store_case)
        protected_calls: list[dict[str, Any]] = []
        late_calls: list[dict[str, Any]] = []
        policy_calls: list[ToolPolicyDecision] = []
        session_id = f"registration-drift-{session_store_case[0]}"
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(MixedProvider(), default=True)
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(protected_calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )
            events = [
                event
                async for event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the mixed round")],
                    )
                )
            ]
            requested = next(
                event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            assert [call.policy_evidence for call in approval.tool_calls] == [
                "unregistered",
                "authoritative",
            ]

            store = await _reopen_store(session_store_case, store)
            resumed_app = CayuApp(session_store=store, enable_logging=False)
            resumed_provider = _ApprovalRecoveryProvider(complete_without_tools=True)
            resumed_app.register_provider(resumed_provider, default=True)
            resumed_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[
                    LateEffectTool(late_calls),
                    _ApprovalRecoveryTool(protected_calls),
                ],
                tool_policy=AllowAllToolPolicy(),
            )
            with pytest.raises(ExecutionProfileMismatchError) as raised:
                _ = [
                    event
                    async for event in resumed_app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=session_id,
                            approval_id=approval.approval_id,
                            tool_round_id=approval.tool_round_id,
                            tool_call_id=approval.tool_call_id,
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                ]

            assert raised.value.changed_component_classes == (
                ExecutionProfileComponentClass.DIRECT_TOOLS,
                ExecutionProfileComponentClass.EFFECT_AUTHORITY,
                ExecutionProfileComponentClass.EXECUTION_POLICIES,
                ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
                ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
            )
            assert late_calls == []
            assert protected_calls == []
            assert resumed_provider.requests == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_store_wide_budget_reservation_reuse(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            first = await store.create(
                RunRequest(
                    session_id=f"reservation-identity-first-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "first")],
                ),
                identity=_identity(),
            )
            second = await store.create(
                RunRequest(
                    session_id=f"reservation-identity-second-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "second")],
                ),
                identity=_identity(),
            )
            reservation_id = "bres_shared_conformance_identity"
            await store.append_event(
                first.id,
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id=first.id,
                    payload={"reservation_id": reservation_id},
                ),
            )

            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.append_event(
                    second.id,
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id=second.id,
                        payload={"reservation_id": reservation_id},
                    ),
                )

            assert [event.type for event in await store.load_events(first.id)] == [
                EventType.BUDGET_RESERVED
            ]
            assert await store.load_events(second.id) == []

            concurrent_sessions = [
                await store.create(
                    RunRequest(
                        session_id=(f"reservation-identity-race-{index}-{session_store_case[0]}"),
                        agent_name="assistant",
                        messages=[Message.text("user", f"race {index}")],
                    ),
                    identity=_identity(),
                )
                for index in range(2)
            ]
            concurrent_reservation_id = "bres_shared_conformance_race_identity"
            results = await asyncio.gather(
                *(
                    store.append_event(
                        session.id,
                        Event(
                            type=EventType.BUDGET_RESERVED,
                            session_id=session.id,
                            payload={"reservation_id": concurrent_reservation_id},
                        ),
                    )
                    for session in concurrent_sessions
                ),
                return_exceptions=True,
            )

            assert sum(result is None for result in results) == 1
            conflicts = [
                result
                for result in results
                if isinstance(result, BudgetReservationIdentityConflict)
            ]
            assert len(conflicts) == 1
            concurrent_event_counts = [
                len(await store.load_events(session.id)) for session in concurrent_sessions
            ]
            assert sum(concurrent_event_counts) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_retains_reservation_identity_after_session_deletion(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            first = await store.create(
                RunRequest(
                    session_id=f"reservation-delete-first-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "first")],
                ),
                identity=_identity(),
            )
            reservation_id = "bres_shared_conformance_deleted_session"
            reserved = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=first.id,
                payload={"reservation_id": reservation_id},
            )
            await store.append_event(
                first.id,
                reserved,
            )
            reserved_claim = await store.claim_persisted_event_side_effect(
                session_id=first.id,
                event_id=reserved.id,
            )
            assert reserved_claim is not None
            await store.mark_persisted_event_side_effect_delivered(reserved_claim)
            released = Event(
                type=EventType.BUDGET_RESERVATION_RELEASED,
                session_id=first.id,
                payload={"reservation_id": reservation_id},
            )
            await store.append_event(first.id, released)
            released_claim = await store.claim_persisted_event_side_effect(
                session_id=first.id,
                event_id=released.id,
            )
            assert released_claim is not None
            await store.mark_persisted_event_side_effect_delivered(released_claim)
            await store.delete_session(first.id)
            store = await _reopen_store(session_store_case, store)

            second = await store.create(
                RunRequest(
                    session_id=f"reservation-delete-second-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "second")],
                ),
                identity=_identity(),
            )
            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.append_event(
                    second.id,
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id=second.id,
                        payload={"reservation_id": reservation_id},
                    ),
                )
            assert await store.load_events(second.id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_claims_reservation_publication_idempotently(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"reservation-claim-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "claim")],
                ),
                identity=_identity(),
            )
            event = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": "bres_shared_conformance_claim"},
            )
            await store.claim_budget_reservation_identity(
                reservation_id="bres_shared_conformance_claim",
                publication_session_id=session.id,
                publication_id=event.id,
            )
            await store.claim_budget_reservation_identity(
                reservation_id="bres_shared_conformance_claim",
                publication_session_id=session.id,
                publication_id=event.id,
            )
            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.claim_budget_reservation_identity(
                    reservation_id="bres_shared_conformance_claim",
                    publication_session_id=session.id,
                    publication_id="evt_conflicting_reservation_publication",
                )
            conflicting_session = await store.create(
                RunRequest(
                    session_id=f"reservation-claim-conflict-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "conflict")],
                ),
                identity=_identity(),
            )
            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.claim_budget_reservation_identity(
                    reservation_id="bres_shared_conformance_claim",
                    publication_session_id=conflicting_session.id,
                    publication_id=event.id,
                )
            with pytest.raises(KeyError, match="Session not found"):
                await store.claim_budget_reservation_identity(
                    reservation_id="bres_shared_conformance_missing_session",
                    publication_session_id="sess_missing_reservation_publication",
                    publication_id=event.id,
                )

            await store.append_event(session.id, event)
            assert [stored.id for stored in await store.load_events(session.id)] == [event.id]

            concurrent_results = await asyncio.gather(
                *(
                    store.claim_budget_reservation_identity(
                        reservation_id="bres_shared_conformance_concurrent_claim",
                        publication_session_id=session.id,
                        publication_id=publication_id,
                    )
                    for publication_id in (
                        "evt_concurrent_claim_first",
                        "evt_concurrent_claim_second",
                    )
                ),
                return_exceptions=True,
            )
            assert sum(result is None for result in concurrent_results) == 1
            assert (
                sum(
                    isinstance(result, BudgetReservationIdentityConflict)
                    for result in concurrent_results
                )
                == 1
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_classifies_exact_budget_event_replay_as_duplicate(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"reservation-event-replay-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "publish")],
                ),
                identity=_identity(),
            )
            event = Event(
                id="evt_shared_conformance_reservation_replay",
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": "bres_shared_conformance_reservation_replay"},
            )

            await store.append_event(session.id, event)

            with pytest.raises(ValueError, match="Event already exists for session"):
                await store.append_event(session.id, event)

            assert [stored.id for stored in await store.load_events(session.id)] == [event.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_enforces_reservation_claims_in_atomic_publication(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"reservation-atomic-publication-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "publish")],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session.id, {"state": "before"})

            reservation_id = "bres_shared_conformance_atomic_publication"
            rightful_event = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": reservation_id},
            )
            conflicting_event = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": reservation_id},
            )
            await store.claim_budget_reservation_identity(
                reservation_id=reservation_id,
                publication_session_id=session.id,
                publication_id=rightful_event.id,
            )

            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.publish_checkpoint_and_events(
                    session.id,
                    checkpoint_transform=lambda _session, _checkpoint: {"state": "conflicting"},
                    events=[conflicting_event],
                )

            assert await store.load_checkpoint(session.id) == {"state": "before"}
            assert await store.load_events(session.id) == []

            await store.publish_checkpoint_and_events(
                session.id,
                checkpoint_transform=lambda _session, _checkpoint: {"state": "published"},
                events=[rightful_event],
            )

            assert await store.load_checkpoint(session.id) == {"state": "published"}
            assert [event.id for event in await store.load_events(session.id)] == [
                rightful_event.id
            ]
            with pytest.raises(ValueError, match="Event already exists for session"):
                await store.publish_checkpoint_and_events(
                    session.id,
                    checkpoint_transform=lambda _session, _checkpoint: {"state": "duplicate"},
                    events=[rightful_event],
                )

            assert await store.load_checkpoint(session.id) == {"state": "published"}
            assert [event.id for event in await store.load_events(session.id)] == [
                rightful_event.id
            ]
            await store.claim_budget_reservation_identity(
                reservation_id=reservation_id,
                publication_session_id=session.id,
                publication_id=rightful_event.id,
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fences_stale_reservation_claims(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        replacement_ready = asyncio.Event()
        allow_replacement_claim = asyncio.Event()
        try:
            created = await store.create(
                RunRequest(
                    session_id=f"reservation-claim-fence-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "claim")],
                ),
                identity=_identity(),
            )
            owned = await store.transition_status(
                created.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )

            async def replace_owner_and_claim() -> Session:
                replacement = await store.fence_stalled_run(
                    created.id,
                    statuses={SessionStatus.RUNNING},
                    inactive_before=datetime.max.replace(tzinfo=UTC),
                )
                assert replacement is not None
                assert replacement.run_epoch == owned.run_epoch + 1
                replacement_ready.set()
                await allow_replacement_claim.wait()
                await store.claim_budget_reservation_identity(
                    reservation_id="bres_shared_conformance_fenced_claim",
                    publication_session_id=created.id,
                    publication_id="evt_replacement_reservation_publication",
                )
                await store.release_run_fence(created.id)
                return replacement

            replacement_task = asyncio.create_task(
                replace_owner_and_claim(),
                context=contextvars.Context(),
            )
            await asyncio.wait_for(replacement_ready.wait(), timeout=5)
            try:
                with pytest.raises(
                    SessionRunFenced,
                    match="Session run epoch no longer owns",
                ):
                    await store.claim_budget_reservation_identity(
                        reservation_id="bres_shared_conformance_fenced_claim",
                        publication_session_id=created.id,
                        publication_id="evt_stale_reservation_publication",
                    )
            finally:
                allow_replacement_claim.set()
            await replacement_task
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_inspection_uses_tolerant_usage_aggregates(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"inspection-usage-{session_store_case[0]}"
        timestamp = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "inspect")],
                ),
                identity=_identity(),
            )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    timestamp=timestamp,
                    payload={
                        "usage_metrics": {
                            "provider_name": " fake ",
                            "model": "valid-model",
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                            "reasoning_output_tokens": "not-an-integer",
                            "cache": {
                                "read_tokens": 4,
                                "write_tokens": -1,
                            },
                        }
                    },
                ),
            )

            inspection = await store.inspect_summary(session_id)
            native = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=timestamp - timedelta(seconds=1),
                    end_at=timestamp + timedelta(seconds=1),
                )
            )

            assert inspection.model_calls == native.totals.model_steps == 1
            assert inspection.model_calls_with_usage == 1
            assert inspection.model_calls_with_usage == native.totals.model_steps_with_usage
            assert inspection.usage.usage == native.totals.usage
            assert inspection.usage.provider_names == []
            assert inspection.usage.models == ["valid-model"]
            assert inspection.usage.usage.input_tokens == 7
            assert inspection.usage.usage.output_tokens == 3
            assert inspection.usage.usage.reasoning_output_tokens == 0
            assert inspection.usage.usage.cache.read_tokens == 4
            assert inspection.usage.usage.cache.write_tokens == 0
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_usage_normalization_failure_is_authoritative(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"normalization-failed-usage-{session_store_case[0]}"
        timestamp = datetime(2026, 7, 20, 13, 0, tzinfo=UTC)
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "inspect")],
                ),
                identity=_identity(),
            )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    timestamp=timestamp,
                    payload={
                        "provider_name": "raw-provider",
                        "model": "raw-model",
                        "usage_normalization_failed": True,
                        "usage": {
                            "input_tokens": 70,
                            "output_tokens": 30,
                            "total_tokens": 100,
                        },
                        "usage_metrics": {
                            "provider_name": "normalized-provider",
                            "model": "normalized-model",
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                        },
                    },
                ),
            )

            inspection = await store.inspect_summary(session_id)
            native = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=timestamp - timedelta(seconds=1),
                    end_at=timestamp + timedelta(seconds=1),
                    include_pricing_inputs=True,
                )
            )
            cost = estimate_usage_rollup_cost(
                native,
                PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="normalized-provider",
                            model="normalized-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
            )

            assert inspection.model_calls == native.totals.model_steps == 1
            assert inspection.model_calls_with_usage == 0
            assert native.totals.model_steps_with_usage == 0
            assert inspection.usage.provider_names == []
            assert inspection.usage.models == []
            assert inspection.usage.usage.total_tokens == 0
            assert native.totals.usage.total_tokens == 0
            assert native.pricing_input_group_count == 1
            assert native.pricing_inputs[0].metrics is None
            assert native.pricing_inputs[0].occurrences == 1
            assert cost.priced_model_steps == 0
            assert cost.unpriced_model_steps == 1
            assert cost.unpriced_reasons[0].reason == (
                "model.completed event has no valid normalized usage metrics"
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_inspection_fails_closed_for_malformed_budget_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        try:
            for index, case in enumerate(
                (
                    "invalid-attempt",
                    "whitespace-reservation",
                    "invalid-amount",
                    "outstanding",
                    "invalid-pricing",
                ),
                start=1,
            ):
                session_id = f"inspection-budget-{case}-{store_kind}"
                budget_limit_id = f"blim_{index:064x}"
                identity = {
                    "model_step_id": f"mstep_{index:032x}",
                    "model_attempt_id": f"matt_{index:032x}",
                }
                reservation_id = (
                    " reservation-with-whitespace "
                    if case == "whitespace-reservation"
                    else f"reservation-{case}"
                )
                reservation_identity = (
                    {**identity, "model_attempt_id": "bad"}
                    if case == "invalid-attempt"
                    else identity
                )

                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "inspect")],
                    ),
                    identity=_identity(),
                )
                await store.append_event(
                    session_id,
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id=session_id,
                        payload={
                            "reservation_id": reservation_id,
                            "budget_limit_id": budget_limit_id,
                            **reservation_identity,
                            "scope": "session",
                            "key": session_id,
                            "window": "all_time",
                            "currency": "USD",
                            "maximum": "1",
                            "action": "interrupt",
                        },
                    ),
                )
                if case not in {"invalid-attempt", "outstanding"}:
                    pricing: object = (
                        {"provider_name": 42}
                        if case == "invalid-pricing"
                        else {
                            "provider_name": "fake",
                            "model": "model",
                            "match": "exact",
                            "provenance": {
                                "source": "application",
                                "url": "application://test-price-book",
                                "as_of": "2026-07-25",
                            },
                            "effective_from": None,
                            "effective_through": None,
                            "tier_max_input_tokens": None,
                        }
                    )
                    await store.append_event(
                        session_id,
                        Event(
                            type=EventType.BUDGET_RECONCILED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "budget_limit_id": budget_limit_id,
                                **identity,
                                "actual_amount": (
                                    "not-a-number" if case == "invalid-amount" else "0.25"
                                ),
                                "pricing": pricing,
                            },
                        ),
                    )

                inspection = await store.inspect_summary(session_id)

                assert inspection.budget.cost_state == "partial"
                assert inspection.budget.amount is None
                assert inspection.budget.currency is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_inspection_preserves_budget_event_order(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        pricing = {
            "provider_name": "fake",
            "model": "model",
            "match": "exact",
            "provenance": {
                "source": "application",
                "url": "application://test-price-book",
                "as_of": "2026-08-10",
            },
            "effective_from": None,
            "effective_through": None,
            "tier_max_input_tokens": None,
        }
        try:
            case_index = 0
            for terminal_kind in ("reconciled", "released"):
                for reordered in (False, True):
                    case_index += 1
                    order_name = "reordered" if reordered else "ordered"
                    session_id = f"inspection-budget-{terminal_kind}-{order_name}-{store_kind}"
                    budget_limit_id = f"blim_{case_index:064x}"
                    reservation_id = f"reservation-{terminal_kind}-{case_index}"
                    identity = {
                        "model_step_id": f"mstep_{case_index:032x}",
                        "model_attempt_id": f"matt_{case_index:032x}",
                    }
                    await store.create(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "inspect")],
                        ),
                        identity=_identity(),
                    )
                    reservation = Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id=session_id,
                        payload={
                            "reservation_id": reservation_id,
                            "budget_limit_id": budget_limit_id,
                            **identity,
                            "scope": "session",
                            "key": None,
                            "window": "all_time",
                            "currency": "USD",
                            "maximum": "1",
                            "action": "interrupt",
                        },
                    )
                    if terminal_kind == "reconciled":
                        terminal = Event(
                            type=EventType.BUDGET_RECONCILED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "settlement_kind": "completed",
                                "budget_limit_id": budget_limit_id,
                                **identity,
                                "actual_amount": "0.25",
                                "pricing": pricing,
                            },
                        )
                        model_terminal = Event(
                            type=EventType.MODEL_COMPLETED,
                            session_id=session_id,
                            payload=identity,
                        )
                        expected_amount = "0.25"
                    else:
                        terminal = Event(
                            type=EventType.BUDGET_RESERVATION_RELEASED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "settlement_kind": "released",
                                "budget_limit_id": budget_limit_id,
                                **identity,
                            },
                        )
                        model_terminal = None
                        expected_amount = "0"
                    events = [terminal, reservation] if reordered else [reservation, terminal]
                    if model_terminal is not None:
                        events.append(model_terminal)
                    await store.append_events(session_id, events)

                    inspection = await store.inspect_summary(session_id)

                    if reordered:
                        assert inspection.budget.cost_state == "partial"
                        assert inspection.budget.amount is None
                        assert inspection.budget.currency is None
                    else:
                        assert inspection.budget.cost_state == "priced"
                        assert inspection.budget.amount == expected_amount
                        assert inspection.budget.currency == "USD"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_inspection_requires_exact_model_budget_join(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        pricing = {
            "provider_name": "fake",
            "model": "model",
            "match": "exact",
            "provenance": {
                "source": "application",
                "url": "application://test-price-book",
                "as_of": "2026-07-27",
            },
            "effective_from": None,
            "effective_through": None,
            "tier_max_input_tokens": None,
        }
        try:
            for index, case in enumerate(("matching", "conflicting", "missing"), start=1):
                session_id = f"inspection-model-budget-{case}-{store_kind}"
                model_step_id = f"mstep_{index:032x}"
                model_attempt_id = f"matt_{index:032x}"
                budget_step_id = (
                    f"mstep_{index + 10:032x}" if case == "conflicting" else model_step_id
                )
                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "inspect")],
                    ),
                    identity=_identity(),
                )
                events = []
                if case != "missing":
                    events.append(
                        Event(
                            type=EventType.MODEL_COMPLETED,
                            session_id=session_id,
                            payload={
                                "model_step_id": model_step_id,
                                "model_attempt_id": model_attempt_id,
                                "usage_metrics": {
                                    "input_tokens": 7,
                                    "output_tokens": 3,
                                    "total_tokens": 10,
                                },
                            },
                        )
                    )
                events.extend(
                    [
                        Event(
                            type=EventType.BUDGET_RESERVED,
                            session_id=session_id,
                            payload={
                                "reservation_id": f"reservation-{case}",
                                "budget_limit_id": f"blim_{index:064x}",
                                "model_step_id": budget_step_id,
                                "model_attempt_id": model_attempt_id,
                                "scope": "session",
                                "key": None,
                                "window": "all_time",
                                "currency": "USD",
                                "maximum": "1",
                                "action": "interrupt",
                            },
                        ),
                        Event(
                            type=EventType.BUDGET_RECONCILED,
                            session_id=session_id,
                            payload={
                                "reservation_id": f"reservation-{case}",
                                "settlement_kind": "completed",
                                "budget_limit_id": f"blim_{index:064x}",
                                "model_step_id": budget_step_id,
                                "model_attempt_id": model_attempt_id,
                                "actual_amount": "0.25",
                                "pricing": pricing,
                            },
                        ),
                    ]
                )
                await store.append_events(session_id, events)

                inspection = await store.inspect_summary(session_id)

                if case == "matching":
                    assert inspection.budget.cost_state == "priced"
                    assert inspection.budget.amount == "0.25"
                    assert inspection.budget.currency == "USD"
                else:
                    assert inspection.budget.cost_state == "partial"
                    assert inspection.budget.amount is None
                    assert inspection.budget.currency is None
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("invalid_text", ["100\x00", "\ud800"], ids=["nul", "surrogate"])
@pytest.mark.parametrize(
    "invalid_primary_counter",
    [True, False],
    ids=["invalid-primary", "invalid-extra-field"],
)
@pytest.mark.parametrize("with_reservation", [False, True], ids=["strict", "reservation"])
def test_session_store_conformance_preserves_undurable_completion_spend(
    session_store_case,
    invalid_text: str,
    invalid_primary_counter: bool,
    with_reservation: bool,
) -> None:
    class MalformedUsageProvider(ModelProvider):
        name = "renamed-openai"
        usage_dialect = UsageDialect.OPENAI

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            del request
            self.calls += 1
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed(
                {
                    "provider_name": "provider-controlled-spoof",
                    "model": "gpt-test",
                    "usage": (
                        {
                            "input_tokens": invalid_text,
                            "output_tokens": 1,
                        }
                        if invalid_primary_counter
                        else {
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "provider_note": invalid_text,
                        }
                    ),
                }
            )

    async def run() -> None:
        store = await _open_store(session_store_case)
        provider = MalformedUsageProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="gpt-test",
                    input_per_million=Decimal("10"),
                    output_per_million=Decimal("10"),
                ),
            )
        )
        reservation = (
            BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            )
            if with_reservation
            else None
        )
        maximum = Decimal("10") if with_reservation else Decimal("100")
        policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=maximum,
                    pricing=pricing,
                    reservation=reservation,
                ),
            )
        )

        def build_app(current_store: SessionStore) -> CayuApp:
            app = CayuApp(
                session_store=current_store,
                budget_policy=policy,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
            return app

        store_kind = session_store_case[0]
        case_suffix = f"{store_kind}-{with_reservation}-{invalid_primary_counter}"
        first_session_id = f"undurable-completion-{case_suffix}-first"
        second_session_id = f"undurable-completion-{case_suffix}-second"
        try:
            app = build_app(store)
            first_events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=first_session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "first")],
                    )
                )
            ]
            completed = next(
                event for event in first_events if event.type == EventType.MODEL_COMPLETED
            )
            assert "usage" not in completed.payload
            assert "usage_metrics" not in completed.payload
            assert completed.payload["usage_normalization_failed"] is True
            assert completed.payload["usage_unavailable_reason"] == (
                "invalid model completion usage telemetry"
            )
            assert completed.payload["provider_name"] == provider.name

            cost = await app.get_session_cost(first_session_id, pricing)
            assert cost.model_steps == 1
            assert cost.priced_model_steps == 0
            assert cost.unpriced_model_steps == 1
            assert cost.line_items[0].provider_name == provider.name
            if with_reservation:
                reconciliation = next(
                    event for event in first_events if event.type == EventType.BUDGET_RECONCILED
                )
                assert reconciliation.payload["actual_amount"] == "10"
                assert reconciliation.payload["reason"] == (
                    "model completed without priced usage; charged reserved amount"
                )

            store = await _reopen_store(session_store_case, store)
            restarted_app = build_app(store)
            second_events = [
                event
                async for event in restarted_app.run(
                    RunRequest(
                        session_id=second_session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "second")],
                    )
                )
            ]
            assert provider.calls == 1
            assert EventType.MODEL_STARTED not in {event.type for event in second_events}
            assert EventType.BUDGET_LIMIT_REACHED in {event.type for event in second_events}
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_usage_for_invalid_provider_state(
    session_store_case,
) -> None:
    class InvalidProviderStateProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            del request
            self.calls += 1
            yield ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                    },
                    "provider_state": {},
                }
            )

    async def assert_durable_result(
        store: SessionStore,
        session_id: str,
    ) -> None:
        events = await store.load_events(session_id)
        completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        assert completed.payload["usage_metrics"]["input_tokens"] == 7
        assert completed.payload["usage_metrics"]["output_tokens"] == 3
        assert completed.payload["usage_metrics"]["total_tokens"] == 10
        assert completed.payload["completion_outcome"] == "invalid_transcript_state"
        assert completed.payload["completion_error"]["provider_error_code"] == (
            "invalid_model_completion_transcript"
        )
        assert completed.payload["transcript_cursor"] == 1
        assert "provider_state" not in completed.payload

        app = CayuApp(session_store=store, enable_logging=False)
        usage = await app.get_session_usage(session_id)
        assert usage.model_steps == 1
        assert usage.usage.input_tokens == 7
        assert usage.usage.output_tokens == 3
        assert usage.usage.total_tokens == 10
        assert await store.load_transcript(session_id) == [Message.text("user", "hello")]
        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.FAILED

    async def run() -> None:
        store = await _open_store(session_store_case)
        provider = InvalidProviderStateProvider()
        session_id = f"invalid-provider-state-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                    )
                )
            ]

            assert provider.calls == 1
            assert EventType.MODEL_RETRY not in {event.type for event in events}
            assert EventType.MODEL_ERROR not in {event.type for event in events}
            assert events[-1].type == EventType.SESSION_FAILED
            await assert_durable_result(store, session_id)

            store = await _reopen_store(session_store_case, store)
            await assert_durable_result(store, session_id)
        finally:
            await _close_store(store)

    asyncio.run(run())


def _assert_durable_error(exc: BaseException, code: str) -> None:
    durable_error = extract_durable_value_error(exc)
    assert durable_error is not None
    assert durable_error.code == code


def _portable_number_probe() -> dict[str, int | float]:
    return {
        "ordinary": 1.0,
        "negative_zero": -0.0,
        "large": 1e18,
        "minimum": float(-(2**63)),
        "fractional": 1e-7,
    }


def _assert_portable_number_probe(value: dict[str, Any]) -> None:
    assert value == {
        "ordinary": 1,
        "negative_zero": 0,
        "large": 1_000_000_000_000_000_000,
        "minimum": -(2**63),
        "fractional": 1e-7,
    }
    assert type(value["ordinary"]) is int
    assert type(value["negative_zero"]) is int
    assert type(value["large"]) is int
    assert type(value["minimum"]) is int
    assert type(value["fractional"]) is float


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: CompactSessionRequest(
            session_id="sess_1",
            idempotency_key="compact_1",
            expected_run_epoch=0,
            expected_transcript_cursor=0,
            instructions=value,
        ),
        lambda value: EnqueueSessionMessageRequest(
            session_id="sess_1",
            idempotency_key="queue_1",
            content=value,
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        ),
        lambda value: SessionQueuedMessage(
            queue_id="queue_1",
            session_id="sess_1",
            idempotency_key="queue_1",
            content=value,
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            status=SessionMessageQueueStatus.QUEUED,
            ordering_key=1,
            accepted_run_epoch=0,
            accepted_transcript_cursor=0,
            accepted_event_id="event_1",
            accepted_at=datetime.now(UTC),
        ),
    ],
)
def test_durable_queue_and_compaction_validation_does_not_echo_rejected_input(factory) -> None:
    secret = "workload-secret-value\x00"

    with pytest.raises(ValidationError) as raised:
        factory(secret)

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == "nul_character"
    assert "workload-secret-value" not in str(raised.value)


def test_session_store_conformance_revalidates_all_mutable_durable_inputs_atomically(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            poisoned_create = RunRequest(
                agent_name="assistant",
                session_id="sess_poisoned_create",
                messages=[
                    Message.tool_call(
                        tool_call_id="call_create",
                        tool_name="echo",
                        arguments={"safe": True},
                    )
                ],
            )
            create_part = poisoned_create.messages[0].content[0]
            assert isinstance(create_part, ToolCallPart)
            create_part.arguments["bad"] = float("nan")
            with pytest.raises((DurableValueError, ValidationError)) as invalid_create:
                await store.create(poisoned_create, identity=_identity())
            _assert_durable_error(invalid_create.value, "non_finite_number")
            assert await store.load("sess_poisoned_create") is None

            session_id = "sess_portable_revalidation"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                    metadata={"stable": True},
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, {"stable": True})

            good_event = Event(
                id="portable-good-event",
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                payload={"safe": True},
            )
            poisoned_event = Event(
                id="portable-bad-event",
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                payload={"safe": True},
            )
            poisoned_event.payload["bad"] = float("inf")
            with pytest.raises((DurableValueError, ValidationError)) as invalid_event:
                await store.append_events(session_id, [good_event, poisoned_event])
            _assert_durable_error(invalid_event.value, "non_finite_number")
            assert await store.load_events(session_id) == []

            good_message = Message.text("assistant", "safe")
            poisoned_message = Message.tool_call(
                tool_call_id="call_transcript",
                tool_name="echo",
                arguments={"safe": True},
            )
            transcript_part = poisoned_message.content[0]
            assert isinstance(transcript_part, ToolCallPart)
            transcript_part.arguments["bad"] = "value\ud800"
            with pytest.raises((DurableValueError, ValidationError)) as invalid_transcript:
                await store.append_transcript_messages(
                    session_id,
                    [good_message, poisoned_message],
                )
            _assert_durable_error(invalid_transcript.value, "unicode_surrogate")
            assert await store.load_transcript(session_id) == []

            with pytest.raises(DurableValueError) as invalid_checkpoint:
                await store.checkpoint(
                    session_id,
                    {"stable": False, "nested": {"bad": MAX_DURABLE_JSON_INTEGER + 1}},
                )
            assert invalid_checkpoint.value.code == "integer_out_of_range"
            assert await store.load_checkpoint(session_id) == {"stable": True}

            with pytest.raises(DurableValueError) as invalid_metadata:
                await store.update_metadata(session_id, {"bad": "value\x00"})
            assert invalid_metadata.value.code == "nul_character"
            loaded = await store.load(session_id)
            assert loaded is not None
            assert loaded.metadata == {"stable": True}

            queue_request = EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="portable-queue",
                content="safe",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            queue_request.content = "poisoned\x00content"
            with pytest.raises((DurableValueError, ValidationError)) as invalid_queue:
                await store.enqueue_session_message(queue_request)
            _assert_durable_error(invalid_queue.value, "nul_character")
            assert await store.load_events(session_id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("invalid_text", "code"),
    [
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ],
)
def test_session_store_conformance_rejects_nonportable_identifiers_and_query_text(
    session_store_case,
    invalid_text: str,
    code: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_portable_text_boundary"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                ),
                identity=_identity(),
            )

            with pytest.raises(DurableValueError) as invalid_identifier:
                await store.update_metadata(invalid_text, {"mutated": True})
            assert invalid_identifier.value.code == code
            assert "workload-secret-value" not in str(invalid_identifier.value)

            forged_query = SessionQuery(q="safe")
            forged_query.q = invalid_text
            with pytest.raises(ValidationError) as invalid_query:
                await store.list_sessions(forged_query)
            _assert_durable_error(invalid_query.value, code)
            assert "workload-secret-value" not in str(invalid_query.value)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_forged_out_of_range_query_cursors(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_query = SessionQuery()
            session_query.offset = MAX_DURABLE_JSON_INTEGER + 1
            with pytest.raises(ValidationError):
                await store.list_sessions(session_query)

            event_query = EventQuery(session_id="sess_portable_integer_boundary")
            event_query.after_sequence = MAX_DURABLE_JSON_INTEGER + 1
            with pytest.raises(ValidationError):
                await store.query_events(event_query)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_exact_portable_number_representation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_portable_numbers"
            request = RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create")],
                metadata={"numbers": {"safe": True}},
            )
            request.metadata["numbers"] = _portable_number_probe()
            await store.create(request, identity=_identity())

            event = Event(
                id="portable-number-event",
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                payload={"numbers": {"safe": True}},
            )
            event.payload["numbers"] = _portable_number_probe()
            await store.append_events(session_id, [event])

            message = Message.tool_call(
                tool_call_id="portable-number-call",
                tool_name="echo",
                arguments={"numbers": {"safe": True}},
            )
            message_part = message.content[0]
            assert isinstance(message_part, ToolCallPart)
            message_part.arguments["numbers"] = _portable_number_probe()
            await store.append_transcript_messages(session_id, [message])
            await store.checkpoint(session_id, {"numbers": _portable_number_probe()})

            store = await _reopen_store(session_store_case, store)
            loaded = await store.load(session_id)
            assert loaded is not None
            _assert_portable_number_probe(loaded.metadata["numbers"])

            events = await store.load_events(session_id)
            assert len(events) == 1
            _assert_portable_number_probe(events[0].payload["numbers"])

            transcript = await store.load_transcript(session_id)
            assert len(transcript) == 1
            loaded_part = transcript[0].content[0]
            assert isinstance(loaded_part, ToolCallPart)
            _assert_portable_number_probe(loaded_part.arguments["numbers"])

            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            _assert_portable_number_probe(checkpoint["numbers"])
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_sqlite_jsonl_to_postgres_preserves_exact_portable_number_representation(
    tmp_path,
    conformance_postgres_dsn,
) -> None:
    async def run() -> None:
        session_id = "sess_sqlite_postgres_portable_numbers"
        sqlite_store = SQLiteSessionStore(tmp_path / "portable-export.sqlite")
        try:
            await sqlite_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "export")],
                    metadata={"numbers": _portable_number_probe()},
                ),
                identity=_identity(),
            )
            await sqlite_store.append_events(
                session_id,
                [
                    Event(
                        id="sqlite-postgres-portable-event",
                        type=EventType.SESSION_STARTED,
                        session_id=session_id,
                        payload={"numbers": _portable_number_probe()},
                    )
                ],
            )
            await sqlite_store.append_transcript_messages(
                session_id,
                [
                    Message.tool_call(
                        tool_call_id="sqlite-postgres-portable-call",
                        tool_name="echo",
                        arguments={"numbers": _portable_number_probe()},
                    )
                ],
            )
            await sqlite_store.checkpoint(
                session_id,
                {"numbers": _portable_number_probe()},
            )

            stream = io.StringIO()
            assert await export_sessions(sqlite_store, stream=stream) == 1
            imported_records = list(import_sessions(io.StringIO(stream.getvalue())))
            assert len(imported_records) == 1
            imported = imported_records[0]
        finally:
            await _close_store(sqlite_store)

        _assert_portable_number_probe(imported.session.metadata["numbers"])
        _assert_portable_number_probe(imported.events[0].payload["numbers"])
        imported_part = imported.transcript[0].content[0]
        assert isinstance(imported_part, ToolCallPart)
        _assert_portable_number_probe(imported_part.arguments["numbers"])
        assert imported.checkpoint is not None
        _assert_portable_number_probe(imported.checkpoint["numbers"])

        await _reset_postgres_data(conformance_postgres_dsn)
        postgres_store = _new_postgres_store(conformance_postgres_dsn)
        try:
            source = imported.session
            await postgres_store.create(
                RunRequest(
                    agent_name=source.agent_name,
                    session_id=source.id,
                    parent_session_id=source.parent_session_id,
                    causal_budget_id=source.causal_budget_id,
                    target=ModelTarget(
                        provider_name=source.provider_name,
                        model=source.model,
                    ),
                    environment_name=source.environment_name,
                    messages=[],
                    labels=source.labels,
                    metadata=source.metadata,
                ),
                identity=SessionIdentity(
                    provider_name=source.provider_name,
                    model=source.model,
                    runtime_name=source.runtime_name,
                    runtime_version=source.runtime_version,
                ),
            )
            await postgres_store.append_events(source.id, imported.events)
            await postgres_store.append_transcript_messages(source.id, imported.transcript)
            await postgres_store.checkpoint(source.id, imported.checkpoint)

            postgres_store = await _reopen_store(
                ("postgres", tmp_path, conformance_postgres_dsn),
                postgres_store,
            )
            restored = await postgres_store.load(source.id)
            assert restored is not None
            _assert_portable_number_probe(restored.metadata["numbers"])
            restored_events = await postgres_store.load_events(source.id)
            _assert_portable_number_probe(restored_events[0].payload["numbers"])
            restored_transcript = await postgres_store.load_transcript(source.id)
            restored_part = restored_transcript[0].content[0]
            assert isinstance(restored_part, ToolCallPart)
            _assert_portable_number_probe(restored_part.arguments["numbers"])
            restored_checkpoint = await postgres_store.load_checkpoint(source.id)
            assert restored_checkpoint is not None
            _assert_portable_number_probe(restored_checkpoint["numbers"])
        finally:
            await _close_store(postgres_store)

    asyncio.run(run())


_RUNTIME_PUBLICATION_PREFIX = "__cayu_runtime_publication_v1__:"
_MODEL_COMPLETION_STAGE_PREFIX = "__cayu_model_completion_stage_v1__:"


class _IterationCountingEventIds:
    def __init__(self, values: set[str]) -> None:
        self._values = values
        self.iterated_values = 0
        self.membership_checks = 0
        self.add_calls = 0

    def __contains__(self, value: object) -> bool:
        self.membership_checks += 1
        return value in self._values

    def __iter__(self) -> Iterator[str]:
        for value in self._values:
            self.iterated_values += 1
            yield value

    def __len__(self) -> int:
        return len(self._values)

    def add(self, value: str) -> None:
        self.add_calls += 1
        self._values.add(value)


def _runtime_publication_key(publication_id: str) -> str:
    return _RUNTIME_PUBLICATION_PREFIX + sha256(publication_id.encode()).hexdigest()


def _model_completion_stage_key(stage_id: str, *, terminal: bool) -> str:
    phase = "completed" if terminal else "prepared"
    return _MODEL_COMPLETION_STAGE_PREFIX + phase + ":" + sha256(stage_id.encode()).hexdigest()


def _model_completion_winner_key(logical_step_id: str) -> str:
    return _MODEL_COMPLETION_STAGE_PREFIX + "winner:" + sha256(logical_step_id.encode()).hexdigest()


def _model_completion_abandonment_key(stage_id: str) -> str:
    return _MODEL_COMPLETION_STAGE_PREFIX + "abandoned:" + sha256(stage_id.encode()).hexdigest()


_MODEL_COMPLETION_ACTIVE_KEY = _MODEL_COMPLETION_STAGE_PREFIX + "active"


def _assistant_model_completion_publication(
    *,
    session_id: str,
    stage_id: str,
    logical_step_id: str,
    intent: dict[str, Any],
    completion_event_id: str,
    source_transcript_cursor: int,
    assistant_message: Message | None,
    classification: dict[str, Any] | None = None,
    event_payload: dict[str, Any] | None = None,
    reservation_ids: tuple[str, ...] = (),
    include_classification: bool = True,
) -> RuntimePublicationRequest:
    if classification is None:
        assert assistant_message is not None
        classification = {"type": "final"}
    transcript_end_cursor = source_transcript_cursor + int(assistant_message is not None)
    completion_payload = {} if event_payload is None else dict(event_payload)
    completion_timestamp = datetime.now(UTC)
    if reservation_ids:
        identity_digest = sha256(stage_id.encode()).hexdigest()
        model_step_id = f"mstep_{identity_digest[:32]}"
        model_attempt_id = f"matt_{identity_digest[16:48]}"
        completion_payload.update(
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
        )
        completion_payload[MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY] = [
            budget_reconciliation_payload(
                BudgetReconciliation(
                    reservation_id=reservation_id,
                    settlement_id=budget_settlement_id(reservation_id),
                    settlement_kind="completed",
                    budget_limit_id=f"blim_{sha256(reservation_id.encode()).hexdigest()}",
                    model_step_id=model_step_id,
                    model_attempt_id=model_attempt_id,
                    status="reconciled",
                    reserved_amount=Decimal("1"),
                    actual_amount=Decimal("0"),
                    released_amount=Decimal("1"),
                    reason="model completed",
                    settled_at=completion_timestamp,
                )
            )
            for reservation_id in reservation_ids
        ]
    if include_classification:
        completion_payload["step_classification"] = classification
    completion_payload["transcript_cursor"] = transcript_end_cursor
    pointer = ModelStepPublicationCheckpoint(
        logical_step_id=logical_step_id,
        stage_id=stage_id,
        source_transcript_cursor=source_transcript_cursor,
        transcript_end_cursor=transcript_end_cursor,
        completion_event_id=completion_event_id,
        classification=classification,
        assistant_message_published=assistant_message is not None,
        tool_round_id=None,
    )
    return RuntimePublicationRequest(
        publication_id=logical_step_id,
        kind="model-step",
        intent=intent,
        mutation=runtime_publication_checkpoint_mutation(
            None,
            {LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: pointer.model_dump(mode="json")},
        ),
        transcript_messages=() if assistant_message is None else (assistant_message,),
        events=(
            Event(
                id=completion_event_id,
                type=EventType.MODEL_COMPLETED,
                timestamp=completion_timestamp,
                session_id=session_id,
                payload=completion_payload,
            ),
        ),
    )


def _model_completion_publication_checkpoint(
    publication: RuntimePublicationRequest,
) -> dict[str, Any]:
    (operation,) = publication.mutation.operations
    assert operation.action == "set"
    assert operation.key == LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY
    return {operation.key: operation.value}


def test_session_store_conformance_model_completion_accepts_root_schema_stamp(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_root_schema"
            logical_step_id = "model-step:root-schema"
            stage_id = f"{logical_step_id}:attempt-0"
            intent = {"logical_step": logical_step_id}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.checkpoint(
                session_id,
                {
                    CHECKPOINT_SCHEMA_VERSION_KEY: (CURRENT_CHECKPOINT_SCHEMA_VERSION),
                },
            )
            await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent=intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            valid = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                intent=intent,
                completion_event_id="root-schema-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative"),
            )
            publication = RuntimePublicationRequest(
                publication_id=valid.publication_id,
                kind=valid.kind,
                intent=valid.intent,
                mutation=RuntimePublicationMutation(
                    operations=(
                        *valid.mutation.operations,
                        RuntimePublicationCheckpointOperation(
                            key=CHECKPOINT_SCHEMA_VERSION_KEY,
                            expected_value_digest=(
                                runtime_publication_checkpoint_value_digest(
                                    CURRENT_CHECKPOINT_SCHEMA_VERSION,
                                )
                            ),
                            action="set",
                            value=CURRENT_CHECKPOINT_SCHEMA_VERSION,
                        ),
                    )
                ),
                transcript_messages=valid.transcript_messages,
                events=valid.events,
                referenced_events=valid.referenced_events,
            )

            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )

            assert completed.stage.state == "completed"
        finally:
            await _close_store(store)

    asyncio.run(run())


async def _load_raw_session_operation_record(
    case,
    store: SessionStore,
    *,
    session_id: str,
    storage_key: str,
) -> dict[str, Any] | None:
    store_kind, _tmp_path, postgres_dsn = case
    if store_kind == "memory":
        assert isinstance(store, InMemorySessionStore)
        record = store._session_operation_records[session_id].get(storage_key)
        return None if record is None else json.loads(json.dumps(record))
    if store_kind == "sqlite":
        assert isinstance(store, SQLiteSessionStore)

        def query(connection):
            row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, storage_key),
            ).fetchone()
            return None if row is None else json.loads(row["record_json"])

        return await store._run_read(query)

    import psycopg

    async with (
        await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
        connection.cursor() as cursor,
    ):
        await cursor.execute(
            "SELECT record FROM cayu_session_operations "
            "WHERE session_id = %s AND idempotency_key = %s",
            (session_id, storage_key),
        )
        row = await cursor.fetchone()
        return None if row is None else row[0]


async def _set_raw_session_operation_record(
    case,
    store: SessionStore,
    *,
    session_id: str,
    storage_key: str,
    record: dict[str, Any],
) -> None:
    store_kind, _tmp_path, postgres_dsn = case
    updated_at = datetime.now(UTC)
    if store_kind == "memory":
        assert isinstance(store, InMemorySessionStore)
        store._session_operation_records[session_id][storage_key] = json.loads(json.dumps(record))
        return
    if store_kind == "sqlite":
        assert isinstance(store, SQLiteSessionStore)

        def statement(connection) -> None:
            connection.execute(
                "INSERT INTO cayu_session_operations "
                "(session_id, idempotency_key, record_json, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                "record_json = excluded.record_json, updated_at = excluded.updated_at",
                (session_id, storage_key, json.dumps(record), updated_at.isoformat()),
            )
            connection.commit()

        await store._run_write(statement)
        return

    import psycopg

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO cayu_session_operations "
                "(session_id, idempotency_key, record, updated_at) "
                "VALUES (%s, %s, %s::jsonb, %s) "
                "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                "record = EXCLUDED.record, updated_at = EXCLUDED.updated_at",
                (session_id, storage_key, json.dumps(record), updated_at),
            )
        await connection.commit()


async def _delete_raw_session_operation_record(
    case,
    store: SessionStore,
    *,
    session_id: str,
    storage_key: str,
) -> None:
    store_kind, _tmp_path, postgres_dsn = case
    if store_kind == "memory":
        assert isinstance(store, InMemorySessionStore)
        store._session_operation_records[session_id].pop(storage_key, None)
        return
    if store_kind == "sqlite":
        assert isinstance(store, SQLiteSessionStore)

        def statement(connection) -> None:
            connection.execute(
                "DELETE FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
                (session_id, storage_key),
            )
            connection.commit()

        await store._run_write(statement)
        return

    import psycopg

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s",
                (session_id, storage_key),
            )
        await connection.commit()


async def _replace_runtime_publication_record(
    case,
    store: SessionStore,
    *,
    session_id: str,
    publication_id: str,
    record: Any,
) -> None:
    store_kind, _tmp_path, postgres_dsn = case
    storage_key = _runtime_publication_key(publication_id)
    if store_kind == "memory":
        assert isinstance(store, InMemorySessionStore)
        store._session_operation_records[session_id][storage_key] = record
        return
    if store_kind == "sqlite":
        assert isinstance(store, SQLiteSessionStore)

        def statement(connection) -> None:
            connection.execute(
                "UPDATE cayu_session_operations SET record_json = ? "
                "WHERE session_id = ? AND idempotency_key = ?",
                (json.dumps(record), session_id, storage_key),
            )
            connection.commit()

        await store._run_write(statement)
        return

    import psycopg

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE cayu_session_operations SET record = %s::jsonb "
                "WHERE session_id = %s AND idempotency_key = %s",
                (json.dumps(record), session_id, storage_key),
            )
        await connection.commit()


async def _replace_model_completion_stage_record(
    case,
    store: SessionStore,
    *,
    session_id: str,
    stage_id: str,
    terminal: bool,
    record: Any,
) -> None:
    store_kind, _tmp_path, postgres_dsn = case
    storage_key = _model_completion_stage_key(stage_id, terminal=terminal)
    if store_kind == "memory":
        assert isinstance(store, InMemorySessionStore)
        store._session_operation_records[session_id][storage_key] = record
        return
    if store_kind == "sqlite":
        assert isinstance(store, SQLiteSessionStore)

        def statement(connection) -> None:
            connection.execute(
                "UPDATE cayu_session_operations SET record_json = ? "
                "WHERE session_id = ? AND idempotency_key = ?",
                (json.dumps(record), session_id, storage_key),
            )
            connection.commit()

        await store._run_write(statement)
        return

    import psycopg

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE cayu_session_operations SET record = %s::jsonb "
                "WHERE session_id = %s AND idempotency_key = %s",
                (json.dumps(record), session_id, storage_key),
            )
        await connection.commit()


async def _corrupt_runtime_publication_material(
    case,
    store: SessionStore,
    *,
    session_id: str,
    corruption: str,
    event_id: str,
) -> None:
    store_kind, _tmp_path, postgres_dsn = case
    if store_kind == "memory":
        assert isinstance(store, InMemorySessionStore)
        if corruption == "transcript-attribution":
            store._transcript_interaction_ids[session_id][0] = "interaction-drifted"
            return
        if corruption == "transcript":
            store._transcripts[session_id][0] = Message.text("assistant", "drifted")
            return
        if corruption in {"event", "reference-content"}:
            corrupted_event = next(
                event.model_copy(update={"payload": {"drifted": True}}, deep=True)
                for event in store._events[session_id]
                if event.id == event_id
            )
            store._events[session_id] = [
                corrupted_event if event.id == event_id else event
                for event in store._events[session_id]
            ]
            store._event_records_by_id[(session_id, event_id)].event = corrupted_event
            return
        if corruption == "reference":
            store._events[session_id] = [
                event for event in store._events[session_id] if event.id != event_id
            ]
            store._event_ids[session_id].discard(event_id)
            store._event_records_by_id.pop((session_id, event_id))
            return
        raise AssertionError(f"Unsupported corruption: {corruption}")

    if store_kind == "sqlite":
        assert isinstance(store, SQLiteSessionStore)

        def statement(connection) -> None:
            if corruption == "transcript-attribution":
                connection.execute(
                    "UPDATE cayu_transcript_messages SET interaction_id = ? "
                    "WHERE sequence = (SELECT MIN(sequence) FROM cayu_transcript_messages "
                    "WHERE session_id = ?)",
                    ("interaction-drifted", session_id),
                )
            elif corruption == "transcript":
                connection.execute(
                    "UPDATE cayu_transcript_messages SET message_json = ? "
                    "WHERE sequence = (SELECT MIN(sequence) FROM cayu_transcript_messages "
                    "WHERE session_id = ?)",
                    (
                        json.dumps(Message.text("assistant", "drifted").model_dump(mode="json")),
                        session_id,
                    ),
                )
            elif corruption in {"event", "reference-content"}:
                connection.execute(
                    "UPDATE cayu_events SET payload_json = ? WHERE session_id = ? AND event_id = ?",
                    (json.dumps({"drifted": True}), session_id, event_id),
                )
            elif corruption == "reference":
                connection.execute(
                    "DELETE FROM cayu_persisted_event_side_effects "
                    "WHERE session_id = ? AND event_id = ?",
                    (session_id, event_id),
                )
                connection.execute(
                    "DELETE FROM cayu_events WHERE session_id = ? AND event_id = ?",
                    (session_id, event_id),
                )
            else:
                raise AssertionError(f"Unsupported corruption: {corruption}")
            connection.commit()

        await store._run_write(statement)
        return

    import psycopg

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            if corruption == "transcript-attribution":
                await cursor.execute(
                    "UPDATE cayu_transcript_messages SET interaction_id = %s "
                    "WHERE sequence = (SELECT MIN(sequence) FROM cayu_transcript_messages "
                    "WHERE session_id = %s)",
                    ("interaction-drifted", session_id),
                )
            elif corruption == "transcript":
                await cursor.execute(
                    "UPDATE cayu_transcript_messages SET message = %s::jsonb "
                    "WHERE sequence = (SELECT MIN(sequence) FROM cayu_transcript_messages "
                    "WHERE session_id = %s)",
                    (
                        json.dumps(Message.text("assistant", "drifted").model_dump(mode="json")),
                        session_id,
                    ),
                )
            elif corruption in {"event", "reference-content"}:
                await cursor.execute(
                    "UPDATE cayu_events SET payload = %s::jsonb, "
                    "event = jsonb_set(event, '{payload}', %s::jsonb) "
                    "WHERE session_id = %s AND event_id = %s",
                    (
                        json.dumps({"drifted": True}),
                        json.dumps({"drifted": True}),
                        session_id,
                        event_id,
                    ),
                )
            elif corruption == "reference":
                await cursor.execute(
                    "DELETE FROM cayu_persisted_event_side_effects "
                    "WHERE session_id = %s AND event_id = %s",
                    (session_id, event_id),
                )
                await cursor.execute(
                    "DELETE FROM cayu_events WHERE session_id = %s AND event_id = %s",
                    (session_id, event_id),
                )
            else:
                raise AssertionError(f"Unsupported corruption: {corruption}")
        await connection.commit()


def test_session_store_conformance_preserves_only_safe_bedrock_aggregate_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_bedrock_aggregate_evidence"
            start = datetime(2026, 7, 1, tzinfo=UTC)
            invoked_model = "global.anthropic.claude-sonnet-4-6"
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "price this run")],
                ),
                identity=_identity(),
            )

            def identity_for_region(region: str) -> BillingIdentity:
                completed = completed_bedrock_billing_identity(
                    bedrock_billing_identity(
                        invoked_model=invoked_model,
                        source_region=region,
                        resource_type="inference_profile",
                        profile_scope="global",
                        requested_service_tier="default",
                    ),
                    effective_service_tier="default",
                )
                return BillingIdentity(
                    provider_name=completed.provider_name,
                    resource_id=completed.resource_id,
                    request_evidence={
                        **completed.request_evidence,
                        "customer_secret": "must-not-cross-the-aggregate-boundary",
                    },
                    completion_evidence={
                        **completed.completion_evidence,
                        "provider_trace": "must-also-remain-redacted",
                    },
                    pricing_contexts=completed.pricing_contexts,
                )

            nested_identity = identity_for_region("us-east-1")
            root_identity = identity_for_region("us-west-2")
            nested_metrics = UsageMetrics(
                provider_name="bedrock",
                model=invoked_model,
                billing_identity=nested_identity,
                input_tokens=1,
                total_tokens=1,
            )
            root_metrics = UsageMetrics(
                provider_name="bedrock",
                model=invoked_model,
                input_tokens=1,
                total_tokens=1,
            )
            await store.append_events(
                session_id,
                [
                    Event(
                        id="bedrock-nested-aggregate-evidence",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start,
                        payload={"usage_metrics": nested_metrics.model_dump(mode="json")},
                    ),
                    Event(
                        id="bedrock-root-aggregate-evidence",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start + timedelta(minutes=1),
                        payload={
                            "usage_metrics": root_metrics.model_dump(mode="json"),
                            "billing_identity": root_identity.model_dump(mode="json"),
                        },
                    ),
                ],
            )

            result = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=start,
                    end_at=start + timedelta(days=1),
                    include_pricing_inputs=True,
                )
            )

            assert result.pricing_inputs_accuracy.kind == "exact"
            assert len(result.pricing_inputs) == 2
            projected_by_region: dict[str, BillingIdentity] = {}
            for item in result.pricing_inputs:
                assert item.metrics is not None
                projected = item.metrics.billing_identity
                assert projected is not None
                region = projected.request_evidence.get("source_region")
                assert region is not None
                projected_by_region[region] = projected
            assert set(projected_by_region) == {"us-east-1", "us-west-2"}
            for region, projected in projected_by_region.items():
                assert projected.request_evidence == {
                    "source_region": region,
                    "resource_type": "inference_profile",
                    "profile_scope": "global",
                    "requested_service_tier": "default",
                }
                assert projected.completion_evidence == {"effective_service_tier": "default"}
                assert "customer_secret" not in projected.request_evidence
                assert "provider_trace" not in projected.completion_evidence
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_gemini_thinking_usage(
    session_store_case,
) -> None:
    class GeminiUsageProvider(ModelProvider):
        name = "gemini"
        usage_dialect = UsageDialect.GEMINI

        async def stream(self, request):
            del request
            yield ModelStreamEvent.text_delta("OK.")
            yield ModelStreamEvent.completed(
                {
                    "model": "gemini-3.5-flash",
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 60,
                    },
                }
            )

    async def assert_usage(store: SessionStore, session_id: str) -> None:
        inspection = await store.inspect_summary(session_id)
        assert inspection.model_calls == 1
        assert inspection.model_calls_with_usage == 1
        assert inspection.usage.provider_names == ["gemini"]
        assert inspection.usage.models == ["gemini-3.5-flash"]
        assert inspection.usage.usage.input_tokens == 5
        assert inspection.usage.usage.output_tokens == 55
        assert inspection.usage.usage.reasoning_output_tokens == 53
        assert inspection.usage.usage.total_tokens == 60

    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"gemini-thinking-usage-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(GeminiUsageProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="gemini-3.5-flash"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "Reply with OK.")],
                    )
                )
            ]
            completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
            assert "usage_normalization_failed" not in completed.payload
            await assert_usage(store, session_id)

            store = await _reopen_store(session_store_case, store)
            await assert_usage(store, session_id)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_context_usage_pages_past_compaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"context-usage-pagination-{session_store_case[0]}"
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=_identity(),
            )
            await store.append_events(
                session_id,
                [
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        payload={
                            "model": "fake-model",
                            "transcript_cursor": 2,
                            "usage": {
                                "input_tokens": 8,
                                "output_tokens": 2,
                                "total_tokens": 10,
                            },
                        },
                    ),
                    *[
                        Event(
                            type=EventType.MODEL_COMPLETED,
                            session_id=session_id,
                            payload={
                                **(
                                    {"purpose": "context_compaction"}
                                    if index % 3 == 0
                                    else (
                                        {"purpose": "future_auxiliary_call"}
                                        if index % 3 == 1
                                        else {}
                                    )
                                ),
                                "model": "summary-model",
                                "usage": {
                                    "input_tokens": index,
                                    "output_tokens": 1,
                                    "total_tokens": index + 1,
                                },
                            },
                        )
                        for index in range(1, 102)
                    ],
                ],
            )

            usage = await model_step_executor_module._context_usage_state_for_session(
                session_store=store,
                session_id=session_id,
            )

            assert usage.last_input_tokens == 8
            assert usage.last_output_tokens == 2
            assert usage.last_total_tokens == 10
            assert usage.last_transcript_cursor == 2
            assert usage.last_model == "fake-model"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_explicit_compaction_operation(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            secret = "explicit-conformance-transcript-secret"
            compactor = _ConformanceCompactor()
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            transcript = [
                Message.text("user", f"old request containing {secret}"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
                Message.text("assistant", "current answer"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            await store.append_event(
                created.id,
                Event(
                    type=EventType.INTERACTION_COMPLETED,
                    session_id=created.id,
                    interaction_id="interaction-deleted-with-session",
                ),
            )
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-conformance-1",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            first = [event async for event in app.compact_session(request)]
            store = await _reopen_store(session_store_case, store)
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            replay = [event async for event in app.compact_session(request)]
            assert [event.id for event in replay] == [event.id for event in first]
            assert compactor.calls == 1
            assert secret not in json.dumps(
                [message.model_dump(mode="json") for message in compactor.requests[0].messages]
            )
            assert REDACTED_SECRET in json.dumps(
                [message.model_dump(mode="json") for message in compactor.requests[0].messages]
            )
            assert await store.load_transcript(created.id) == transcript
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["summary"] == "summary-1"
            assert "session_operations" not in checkpoint
            completed_operation = await store.load_session_operation(
                created.id,
                request.idempotency_key,
            )
            assert completed_operation is not None
            assert completed_operation["status"] == "completed"

            with pytest.raises(ValueError, match="transcript cursor is stale"):
                async for _event in app.compact_session(
                    request.model_copy(
                        update={
                            "idempotency_key": "compact-stale",
                            "expected_transcript_cursor": len(transcript) - 1,
                        }
                    )
                ):
                    pass

            tail = [
                Message.text("user", "later request"),
                Message.text("assistant", "later answer"),
            ]
            await store.append_transcript_messages(created.id, tail)
            failed_request = request.model_copy(
                update={
                    "idempotency_key": "compact-failure",
                    "expected_transcript_cursor": len(transcript) + len(tail),
                }
            )
            compactor.fail_next = True
            with pytest.raises(RuntimeError, match="conformance compactor failed"):
                async for _event in app.compact_session(failed_request):
                    pass
            assert compactor.calls == 2
            failed_operation = await store.load_session_operation(
                created.id,
                failed_request.idempotency_key,
            )
            assert failed_operation is not None
            assert failed_operation["status"] == "failed"

            retry = [
                event
                async for event in app.compact_session(
                    failed_request.model_copy(update={"idempotency_key": "compact-retry"})
                )
            ]
            assert retry[-1].type == EventType.SESSION_CHECKPOINTED
            assert compactor.calls == 3
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert "session_operations" not in checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_explicit_compaction_projects_legacy_summary(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            secret = "explicit-conformance-legacy-summary-secret"
            legacy_summary = f"legacy summary containing {secret}"
            compactor = _ConformanceCompactor()
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            session_id = f"sess_compaction_legacy_summary_{session_store_case[0]}"
            created = await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "newer request"),
                Message.text("assistant", "newer answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            await store.checkpoint(
                created.id,
                {
                    "context_compaction": {
                        "version": 2,
                        "summary": legacy_summary,
                        "compacted_transcript_cursor": 2,
                        "metadata": {"compactor": "legacy", "mode": "deterministic"},
                    }
                },
            )
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-legacy-summary",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            first = [event async for event in app.compact_session(request)]
            store = await _reopen_store(session_store_case, store)
            replay_app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            replay = [event async for event in replay_app.compact_session(request)]

            assert [event.id for event in replay] == [event.id for event in first]
            assert compactor.calls == 1
            assert compactor.requests[0].existing_summary == (
                f"legacy summary containing {REDACTED_SECRET}"
            )
            assert secret not in json.dumps(compactor.requests[0].model_dump(mode="json"))
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["summary"] == (
                f"legacy summary containing {REDACTED_SECRET}|summary-1"
            )
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("authority_position", ("first", "middle", "final"))
@pytest.mark.parametrize(
    "authority_kind",
    ("file_attachment", "runtime_projection_artifact_id", "runtime_projection_store_id"),
)
def test_session_store_conformance_explicit_compaction_rejects_secret_authority(
    session_store_case,
    authority_position: str,
    authority_kind: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            secret = (
                "deadbeef"
                if authority_kind == "runtime_projection_artifact_id"
                else "explicit-conformance-authority-secret"
            )
            compactor = _ConformanceCompactor()
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_authority_conformance",
                    messages=[],
                ),
                identity=_identity(),
            )
            if authority_kind == "file_attachment":
                artifact = file_attachment(
                    artifact_id=f"artifact-{secret}-suffix",
                    kind="image",
                    filename="safe.png",
                    content_type="image/png",
                    size_bytes=1,
                )
            else:
                artifact_id = (
                    f"art_{secret}{'a' * 24}"
                    if authority_kind == "runtime_projection_artifact_id"
                    else f"art_{'a' * 32}"
                )
                artifact = {
                    "type": "cayu.tool_result_artifact.v1",
                    "artifact_id": artifact_id,
                    "store_id": (
                        "artifacts"
                        if authority_kind == "runtime_projection_artifact_id"
                        else f"store-{secret}-suffix"
                    ),
                    "filename": f"tool-result-{artifact_id}.txt",
                    "content_type": "text/plain; charset=utf-8",
                    "size_bytes": 1,
                    "sha256": "b" * 64,
                    "scope": "session",
                    "session_id_sha256": "c" * 64,
                    "readback_max_bytes": 64,
                    "projection_authority": "cayu.tool_result_projection.v1",
                }
            authority_message = Message(
                role=MessageRole.TOOL,
                content=(
                    ToolResultPart(
                        tool_call_id="safe-call",
                        tool_name="safe_tool",
                        artifacts=[artifact],
                    ),
                ),
            )
            safe_messages = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            authority_index = {
                "first": 0,
                "middle": 1,
                "final": len(safe_messages),
            }[authority_position]
            transcript = list(safe_messages)
            transcript.insert(authority_index, authority_message)
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            before_events = await store.query_events(EventQuery(session_id=created.id, limit=100))
            before_checkpoint = await store.load_checkpoint(created.id)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key=f"reject-{authority_kind}-authority-conformance",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            with pytest.raises(
                ValueError,
                match="workload secret in execution authority",
            ):
                async for _event in app.compact_session(request):
                    pass

            current = await store.load(created.id)
            assert current is not None
            assert current.status == completed.status
            assert current.run_epoch == completed.run_epoch
            assert await store.load_transcript(created.id) == transcript
            assert await store.load_checkpoint(created.id) == before_checkpoint
            assert (
                await store.query_events(EventQuery(session_id=created.id, limit=100))
                == before_events
            )
            assert (
                await store.load_session_operation(
                    created.id,
                    request.idempotency_key,
                )
                is None
            )
            assert compactor.calls == 0
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_partial_compaction_cursor_survives_reopen(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=_ConformancePartialCompactor(), max_user_turns=1
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_partial_coverage_conformance",
                    messages=[],
                ),
                identity=_identity(),
            )
            transcript = [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="partial-coverage-1",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
            events = [event async for event in app.compact_session(request)]
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
            assert await store.load_transcript(created.id) == transcript
            reopened = await _reopen_store(session_store_case, store)
            store = reopened
            assert (await store.load_checkpoint(created.id))["context_compaction"][
                "compacted_transcript_cursor"
            ] == 1
            assert any(event.type == EventType.SESSION_CHECKPOINTED for event in events)

            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=_ConformancePartialCompactor(), max_user_turns=1
                ),
            )
            second = request.model_copy(update={"idempotency_key": "partial-coverage-2"})
            second_events = [event async for event in replay_app.compact_session(second)]
            second_checkpoint = await store.load_checkpoint(created.id)
            assert second_checkpoint is not None
            assert second_checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2
            assert await store.load_transcript(created.id) == transcript
            assert any(event.type == EventType.SESSION_CHECKPOINTED for event in second_events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_in_memory_event_append_does_not_scan_existing_event_ids() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "sess_event_id_membership"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "append without scanning history")],
            ),
            identity=_identity(),
        )
        existing_events = [
            Event(
                id=f"existing-event-{index}",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
            )
            for index in range(32)
        ]
        await store.append_events(session_id, existing_events)

        tracked_ids = _IterationCountingEventIds(store._event_ids[session_id])
        store._event_ids[session_id] = cast("set[str]", tracked_ids)
        appended = Event(
            id="new-event",
            type=EventType.MODEL_COMPLETED,
            session_id=session_id,
        )
        await store.append_event(session_id, appended)

        assert tracked_ids.iterated_values == 0
        assert tracked_ids.membership_checks == 1
        assert tracked_ids.add_calls == 1
        assert [event.id for event in await store.load_events(session_id)] == [
            *(event.id for event in existing_events),
            appended.id,
        ]

    asyncio.run(run())


def test_session_store_conformance_cancelled_partial_compaction_publishes_no_cursor(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            compactor = _ConformancePartialCancellationCompactor()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_partial_cancel_conformance",
                    messages=[],
                ),
                identity=_identity(),
            )
            transcript = [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="partial-cancel",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            async def collect() -> list[Event]:
                return [event async for event in app.compact_session(request)]

            task = asyncio.create_task(collect())
            await compactor.started.wait()
            task.cancel("cancel partial publication")
            with pytest.raises(asyncio.CancelledError, match="cancel partial publication"):
                await task

            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is None or "context_compaction" not in checkpoint
            assert await store.load_transcript(created.id) == transcript
            durable_events = [
                record.event
                for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
            ]
            assert EventType.CONTEXT_COMPACTION_COMPLETED not in {
                event.type for event in durable_events
            }
            assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable_events}
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_first_commit_and_stale_replay(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_commit"
            publication_id = "model-step:1"
            fixed_at = datetime(2026, 7, 22, 9, 0, tzinfo=UTC)
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "publish atomically")],
                ),
                identity=_identity(),
            )
            await store.publish_checkpoint_and_events(
                session_id,
                checkpoint_transform=lambda _session, _checkpoint: {
                    "phase": "before",
                    "nested": {"owned": True},
                },
                events=[],
            )
            referenced_event = Event(
                id="model-requested",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                timestamp=fixed_at,
                interaction_id="interaction-publication",
            )
            await store.append_event(session_id, referenced_event)
            before = await store.load(session_id)
            assert before is not None

            intent_source = {"logical_step": {"number": 1}}
            structured_source = {"stable": "original"}
            event_payload_source = {"usage": {"total_tokens": 3}}
            transcript_message = Message.tool_result(
                tool_call_id="structured-output-1",
                tool_name="final_output",
                content="complete",
                structured=structured_source,
            )
            completed_event = Event(
                id="model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                timestamp=fixed_at + timedelta(seconds=1),
                interaction_id="interaction-publication",
                payload=event_payload_source,
            )
            source_checkpoint = {
                "phase": "before",
                "nested": {"owned": True},
            }
            published_checkpoint = {
                "phase": "published",
                "model_step": 1,
                "budget_reservation_id": "reservation-1",
            }
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                interaction_id="interaction-publication",
                intent=intent_source,
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    published_checkpoint,
                ),
                transcript_messages=[transcript_message],
                events=[completed_event],
                referenced_events=[runtime_publication_event_reference(referenced_event)],
            )
            intent_source["logical_step"]["number"] = 99
            structured_source["stable"] = "caller-mutated"
            event_payload_source["usage"]["total_tokens"] = 99

            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert published.session.updated_at > before.updated_at
            assert published.session.updated_at == published.session.last_activity_at
            assert published.receipt.published_at == published.session.updated_at
            assert published.receipt.session_id == session_id
            assert published.receipt.publication_id == publication_id
            assert published.receipt.kind == "model-step"
            assert published.receipt.interaction_id == "interaction-publication"
            assert published.receipt.intent == {"logical_step": {"number": 1}}
            assert published.receipt.source_status is SessionStatus.PENDING
            assert published.receipt.source_run_epoch == 0
            assert published.receipt.transcript_start_cursor == 0
            assert published.receipt.transcript_end_cursor == 1
            assert published.receipt.appended_event_ids == (completed_event.id,)
            assert published.receipt.referenced_events == (
                runtime_publication_event_reference(referenced_event),
            )
            assert len(published.receipt.request_digest) == 64
            assert len(published.receipt.publication_digest) == 64
            assert await store.load_checkpoint(session_id) == published_checkpoint
            loaded_session = await store.load(session_id)
            assert loaded_session is not None
            assert "must_not_persist" not in loaded_session.metadata
            loaded_transcript = await store.load_transcript(session_id)
            assert loaded_transcript == [transcript_message]
            assert loaded_transcript[0].content[0].structured == {"stable": "original"}
            loaded_events = await store.load_events(session_id)
            assert [event.id for event in loaded_events] == [
                referenced_event.id,
                completed_event.id,
            ]
            assert loaded_events[-1].payload == {"usage": {"total_tokens": 3}}
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=session_id,
                event_id=completed_event.id,
            )
            assert delivery is not None
            assert delivery.status is PersistedEventSideEffectStatus.PENDING
            assert (
                await store.load_runtime_publication_receipt(session_id, publication_id)
                == published.receipt
            )

            request.intent["logical_step"]["number"] = 500
            request.transcript_messages[0].content[0].structured["stable"] = "changed"
            request.events[0].payload["usage"]["total_tokens"] = 500
            assert (await store.load_transcript(session_id))[0].content[0].structured == {
                "stable": "original"
            }
            assert (await store.load_events(session_id))[-1].payload == {
                "usage": {"total_tokens": 3}
            }
            assert (
                await store.load_runtime_publication_receipt(session_id, publication_id)
            ).intent == {"logical_step": {"number": 1}}

            equivalent_request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                interaction_id="interaction-publication",
                intent={"logical_step": {"number": 1}},
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    published_checkpoint,
                ),
                transcript_messages=[transcript_message],
                events=[completed_event],
                referenced_events=[runtime_publication_event_reference(referenced_event)],
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            assert running.run_epoch == 1
            await store.release_run_fence(session_id)
            await store.update_status(session_id, SessionStatus.COMPLETED)
            later_message = Message.text("user", "state advanced after commit")
            await store.append_transcript_messages(session_id, [later_message])
            store = await _reopen_store(session_store_case, store)

            before_replay_session = await store.load(session_id)
            before_replay_checkpoint = await store.load_checkpoint(session_id)
            before_replay_transcript = await store.load_transcript(session_id)
            before_replay_events = await store.load_events(session_id)
            replayed = await store.publish_runtime_publication(
                session_id,
                request=equivalent_request,
                expected_statuses={SessionStatus.COMPLETED},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=2,
            )
            assert replayed.replayed is True
            assert replayed.session == before_replay_session
            assert replayed.receipt == published.receipt
            assert await store.load(session_id) == before_replay_session
            assert await store.load_checkpoint(session_id) == before_replay_checkpoint
            assert await store.load_transcript(session_id) == before_replay_transcript
            assert await store.load_events(session_id) == before_replay_events

            conflicting_request = equivalent_request.model_copy(
                update={"intent": {"logical_step": {"number": 2}}}
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="different request",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=conflicting_request,
                    expected_statuses={SessionStatus.COMPLETED},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=2,
                )
            conflicting_attribution = equivalent_request.model_copy(
                update={
                    "interaction_id": "interaction-other",
                    "events": [
                        event.model_copy(
                            update={"interaction_id": "interaction-other"},
                            deep=True,
                        )
                        for event in equivalent_request.events
                    ],
                },
                deep=True,
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="different request",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=conflicting_attribution,
                    expected_statuses={SessionStatus.COMPLETED},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=2,
                )
            assert await store.load(session_id) == before_replay_session
            assert await store.load_checkpoint(session_id) == before_replay_checkpoint
            assert await store.load_transcript(session_id) == before_replay_transcript
            assert await store.load_events(session_id) == before_replay_events
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_sanitizes_caller_authority(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            parent_session_id = f"publication-origin-parent-{session_store_case[0]}"
            child_session_id = f"publication-origin-child-{session_store_case[0]}"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=parent_session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=child_session_id,
                    parent_session_id=parent_session_id,
                    messages=[],
                ),
                identity=_identity(),
            )

            caller_authored_origin = Event(
                id="caller-authored-session-started",
                type=EventType.SESSION_STARTED,
                session_id=child_session_id,
                payload={
                    "parent_session_id": parent_session_id,
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                        "v1:0:1:original:text:sha256:" + "0" * 64
                    ),
                },
            )
            request_template = RuntimePublicationRequest(
                publication_id="caller-authored-origin-publication",
                kind="model-step",
                intent={"purpose": "origin-lineage-conformance"},
                mutation=RuntimePublicationMutation(),
                transcript_messages=(),
                events=(),
            )
            # Bypass the request validator deliberately. publish_runtime_publication()
            # must reconstruct and sanitize the request before digesting or storing it.
            bypassed_request = request_template.model_copy(
                update={"events": (caller_authored_origin,)},
            )
            assert bypassed_request.events[0].payload["parent_session_id"] == parent_session_id

            published = await store.publish_runtime_publication(
                child_session_id,
                request=bypassed_request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert published.receipt.appended_event_ids == (caller_authored_origin.id,)

            store = await _reopen_store(session_store_case, store)
            records = await store.query_events(
                EventQuery(
                    session_id=child_session_id,
                    event_type=EventType.SESSION_STARTED,
                    limit=2,
                )
            )
            assert len(records) == 1
            assert "parent_session_id" not in records[0].event.payload
            assert SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY not in records[0].event.payload

            equivalent_request = RuntimePublicationRequest(
                publication_id="caller-authored-origin-publication",
                kind="model-step",
                intent={"purpose": "origin-lineage-conformance"},
                mutation=RuntimePublicationMutation(),
                transcript_messages=(),
                events=(caller_authored_origin,),
            )
            assert "parent_session_id" not in equivalent_request.events[0].payload
            assert (
                SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY
                not in equivalent_request.events[0].payload
            )
            replayed = await store.publish_runtime_publication(
                child_session_id,
                request=equivalent_request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reclaimed_partial_publication_has_one_prefix(
    session_store_case,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        store = await _open_store(session_store_case)
        try:
            compactor = _ConformancePartialOverlapCompactor()

            def configured_app(now: datetime) -> CayuApp:
                app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now)
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    context_policy=CheckpointCompactionContextPolicy(
                        compactor=compactor,
                        max_user_turns=1,
                    ),
                )
                return app

            first_app = configured_app(accepted_at)
            reclaimed_app = configured_app(accepted_at + timedelta(minutes=6))
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_partial_reclaim_conformance",
                    messages=[],
                ),
                identity=_identity(),
            )
            transcript = [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="partial-reclaim",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
                requested_by=ResolutionActor(subject="operator-a"),
            )

            async def collect(app: CayuApp, requested_by: str) -> list[Event]:
                attempted = request.model_copy(
                    update={"requested_by": ResolutionActor(subject=requested_by)}
                )
                return [event async for event in app.compact_session(attempted)]

            first = asyncio.create_task(collect(first_app, "operator-a"))
            await compactor.started[0].wait()
            reclaimed = asyncio.create_task(collect(reclaimed_app, "operator-b"))
            await compactor.started[1].wait()
            compactor.release[1].set()
            reclaimed_events = await reclaimed
            compactor.release[0].set()
            with pytest.raises(RuntimeError, match="superseded"):
                await first

            store = await _reopen_store(session_store_case, store)
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
            assert checkpoint["context_compaction"]["summary"] == "partial-1"
            assert await store.load_transcript(created.id) == transcript
            assert (
                sum(
                    event.type == EventType.CONTEXT_COMPACTION_COMPLETED
                    for event in reclaimed_events
                )
                == 1
            )
            durable_events = [
                record.event
                for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
            ]
            assert (
                sum(event.type == EventType.SESSION_CHECKPOINTED for event in durable_events) == 1
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_runtime_publication_request_rejects_id_only_event_references() -> None:
    with pytest.raises(ValueError, match="referenced_event_ids"):
        RuntimePublicationRequest.model_validate(
            {
                "publication_id": "id-only-reference",
                "kind": "tool-round",
                "intent": {},
                "transcript_messages": [],
                "events": [],
                "referenced_event_ids": ["unbound-event-id"],
            }
        )


@pytest.mark.parametrize("failure", ["missing", "wrong-digest", "wrong-content"])
def test_session_store_conformance_runtime_publication_rejects_unbound_event_references(
    session_store_case,
    failure: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_runtime_publication_reference_{failure}"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            durable_event = Event(
                id="referenced-terminal-event",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                payload={"tool_call_id": "call-1", "result": "authoritative"},
            )
            if failure != "missing":
                await store.append_event(session_id, durable_event)

            reference_source = durable_event
            if failure == "missing":
                reference_source = durable_event.model_copy(
                    update={"id": "event-that-was-never-persisted"},
                    deep=True,
                )
            elif failure == "wrong-content":
                reference_source = durable_event.model_copy(
                    update={"payload": {"tool_call_id": "call-1", "result": "forged"}},
                    deep=True,
                )
            reference = runtime_publication_event_reference(reference_source)
            if failure == "wrong-digest":
                reference = reference.model_copy(update={"event_digest": "0" * 64})

            round_identity = _publication_tool_round_identity(f"reference-{failure}")
            round_id = round_identity["tool_round_id"]
            request = RuntimePublicationRequest(
                publication_id=f"tool-round:{round_id}",
                kind="tool-round",
                intent={
                    "round_id": round_id,
                    **round_identity,
                    "tool_call_ids": ["call-1"],
                },
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "must-not-publish"},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id="call-1",
                        tool_name="lookup",
                        content="must not publish",
                        **round_identity,
                    )
                ],
                events=[],
                referenced_events=[reference],
            )

            expected_error = "not durable" if failure == "missing" else "does not match"
            with pytest.raises(ValueError, match=expected_error):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                )
            assert await store.load_checkpoint(session_id) is None
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == (
                [] if failure == "missing" else [durable_event]
            )
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_reference_order_is_durable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_reference_order"
            publication_id = "model-step:ordered-references"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            first = Event(
                id="ordered-reference-first",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                payload={"tool_call_id": "call-1", "order": 1},
            )
            second = Event(
                id="ordered-reference-second",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                payload={"tool_call_id": "call-2", "order": 2},
            )
            await store.append_event(session_id, first)
            await store.append_event(session_id, second)
            references = (
                runtime_publication_event_reference(first),
                runtime_publication_event_reference(second),
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"tool_call_ids": ["call-1", "call-2"]},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[],
                events=[],
                referenced_events=references,
            )

            published = await store.publish_runtime_publication(
                session_id,
                request=request,
            )
            assert published.replayed is False
            assert published.receipt.referenced_events == references

            store = await _reopen_store(session_store_case, store)
            replayed = await store.publish_runtime_publication(
                session_id,
                request=request,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt

            reordered_request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind=request.kind,
                intent=request.intent,
                mutation=request.mutation,
                transcript_messages=request.transcript_messages,
                events=request.events,
                referenced_events=tuple(reversed(references)),
            )
            with pytest.raises(SessionRuntimePublicationConflict, match="different request"):
                await store.publish_runtime_publication(
                    session_id,
                    request=reordered_request,
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_recovers_commit_then_cancel(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_commit_then_cancel"
            round_identity = _publication_tool_round_identity("commit-then-cancel")
            round_id = round_identity["tool_round_id"]
            publication_id = f"tool-round:{round_id}"
            pending_round = {
                **round_identity,
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "lookup",
                        "arguments": {},
                    }
                ],
            }
            source_checkpoint = {"pending_tool_round": pending_round}
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "recover the ambiguous commit")],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, source_checkpoint)
            started_event = Event(
                id="commit-then-cancel-started",
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **round_identity,
                    "tool_call_id": "call-1",
                    "idempotency_key": "tool-call:call-1",
                },
            )
            await store.append_event(session_id, started_event)
            completed_result = {
                "content": "completed once",
                "structured": None,
                "artifacts": [],
                "is_error": False,
            }
            completed_event = Event(
                id="commit-then-cancel-event",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **round_identity,
                    "tool_call_id": "call-1",
                    "idempotency_key": "tool-call:call-1",
                    "result": completed_result,
                },
            )
            await store.append_event(session_id, completed_event)
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="tool-round",
                intent={
                    "schema_version": 1,
                    "round_id": round_id,
                    **round_identity,
                    "tool_call_ids": ["call-1"],
                    "pending_round_digest": runtime_publication_checkpoint_value_digest(
                        pending_round
                    ),
                },
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id="call-1",
                        tool_name="lookup",
                        content="completed once",
                        **round_identity,
                    )
                ],
                events=[],
                referenced_events=[
                    runtime_publication_event_reference(started_event),
                    runtime_publication_event_reference(completed_event),
                ],
            )

            original_atomic_publish = store._publish_runtime_publication_atomic
            publication_committed = asyncio.Event()
            hold_acknowledgement = asyncio.Event()

            async def commit_then_hold_acknowledgement(
                prepared: Any,
            ) -> Any:
                result = await original_atomic_publish(prepared)
                assert result.replayed is False
                publication_committed.set()
                await hold_acknowledgement.wait()
                return result

            with monkeypatch.context() as publication_patch:
                publication_patch.setattr(
                    store,
                    "_publish_runtime_publication_atomic",
                    commit_then_hold_acknowledgement,
                )
                publishing = asyncio.create_task(
                    store.publish_runtime_publication(
                        session_id,
                        request=request,
                        expected_statuses={SessionStatus.PENDING},
                        expected_run_epoch=0,
                        expected_transcript_cursor=0,
                    )
                )
                await asyncio.wait_for(publication_committed.wait(), timeout=5)
                publishing.cancel("runtime publication acknowledgement lost")
                with pytest.raises(asyncio.CancelledError) as cancellation:
                    await publishing
                assert cancellation.value.args == ("runtime publication acknowledgement lost",)

            committed_receipt = await store.load_runtime_publication_receipt(
                session_id,
                publication_id,
            )
            assert committed_receipt is not None
            assert await store.load_checkpoint(session_id) == {}
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == [started_event, completed_event]
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=session_id,
                event_id=completed_event.id,
            )
            assert delivery is not None
            assert delivery.status is PersistedEventSideEffectStatus.PENDING

            store = await _reopen_store(session_store_case, store)
            before_replay_session = await store.load(session_id)
            before_replay_checkpoint = await store.load_checkpoint(session_id)
            before_replay_transcript = await store.load_transcript(session_id)
            before_replay_events = await store.load_events(session_id)

            replayed = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert replayed.replayed is True
            assert replayed.receipt == committed_receipt
            assert replayed.session == before_replay_session
            assert await store.load(session_id) == before_replay_session
            assert await store.load_checkpoint(session_id) == before_replay_checkpoint
            assert await store.load_transcript(session_id) == before_replay_transcript
            assert await store.load_events(session_id) == before_replay_events
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_tool_round_rejects_omitted_terminal_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_tool_round_omitted_terminal"
            round_identity = _publication_tool_round_identity("round-with-conflicting-terminal")
            round_id = round_identity["tool_round_id"]
            pending_round = {
                **round_identity,
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "lookup",
                        "arguments": {},
                    }
                ],
            }
            source_checkpoint = {"pending_tool_round": pending_round}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            await store.checkpoint(session_id, source_checkpoint)
            lifecycle_payload = {
                **round_identity,
                "tool_call_id": "call-1",
                "idempotency_key": "tool-call:call-1",
            }
            started = Event(
                id="omitted-terminal-started",
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                tool_name="lookup",
                payload=lifecycle_payload,
            )
            authoritative_result = {
                "content": "first terminal",
                "structured": None,
                "artifacts": [],
                "is_error": False,
            }
            first_terminal = Event(
                id="omitted-terminal-first",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={**lifecycle_payload, "result": authoritative_result},
            )
            second_terminal = Event(
                id="omitted-terminal-second",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **lifecycle_payload,
                    "result": {
                        **authoritative_result,
                        "content": "contradictory second terminal",
                    },
                },
            )
            await store.append_events(
                session_id,
                [started, first_terminal, second_terminal],
            )
            lifecycle_events = await store.load_tool_round_lifecycle_events(
                session_id,
                ["call-1"],
            )
            assert [event.id for event in lifecycle_events] == [
                started.id,
                first_terminal.id,
                second_terminal.id,
            ]
            request = RuntimePublicationRequest(
                publication_id=f"tool-round:{round_id}",
                kind="tool-round",
                intent={
                    "round_id": round_id,
                    **round_identity,
                    "tool_call_ids": ["call-1"],
                },
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id="call-1",
                        tool_name="lookup",
                        content="first terminal",
                        **round_identity,
                    )
                ],
                events=[],
                referenced_events=[
                    runtime_publication_event_reference(started),
                    runtime_publication_event_reference(first_terminal),
                ],
            )

            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="bind every durable lifecycle event",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_transcript_cursor=0,
                )

            assert await store.load_checkpoint(session_id) == source_checkpoint
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == [
                started,
                first_terminal,
                second_terminal,
            ]
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_structured_output_tool_round_auxiliary_publication(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_structured_output_tool_round_publication"
            round_identity = _publication_tool_round_identity("structured-output-round")
            round_id = round_identity["tool_round_id"]
            tool_call_id = "structured-output-call"
            spec = StructuredOutputSpec(
                name="answer",
                json_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                max_retries=2,
            )
            pending_round = {
                **round_identity,
                "agent_name": "assistant",
                "environment_name": None,
                "task_id": None,
                "source_model_step_id": "model-step:structured-output",
                "source_transcript_cursor": 0,
                "model_step": 2,
                "structured_output_attempt": 1,
                "structured_output_validation": {
                    "valid": True,
                    "output": {"answer": "ok"},
                    "errors": [],
                },
                "tool_calls": [
                    {
                        "tool_call_id": tool_call_id,
                        "tool_name": STRUCTURED_OUTPUT_TOOL_NAME,
                        "arguments": {"output": {"answer": "ok"}},
                    }
                ],
                "structured_output": spec.model_dump(mode="json"),
            }
            source_checkpoint = {"pending_tool_round": pending_round}
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, source_checkpoint)
            result_payload = {
                "content": "Structured output accepted.",
                "structured": {"output": {"answer": "ok"}},
                "artifacts": [],
                "is_error": False,
            }
            terminal = Event(
                id="structured-output-terminal",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                agent_name="assistant",
                tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
                payload={
                    **round_identity,
                    "tool_call_id": tool_call_id,
                    "idempotency_key": "structured-output-idempotency",
                    "result": result_payload,
                },
            )
            await store.append_event(session_id, terminal)
            validating = Event(
                id="structured-output-validating",
                type=EventType.STRUCTURED_OUTPUT_VALIDATING,
                session_id=session_id,
                agent_name="assistant",
                payload={
                    **round_identity,
                    "name": spec.name,
                    "strategy": "tool",
                    "step": 2,
                    "attempt": 1,
                    "max_retries": spec.max_retries,
                },
            )
            validated = Event(
                id="structured-output-validated",
                type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                session_id=session_id,
                agent_name="assistant",
                payload={
                    **round_identity,
                    "name": spec.name,
                    "step": 2,
                    "attempt": 1,
                    "max_retries": spec.max_retries,
                    "valid": True,
                    "errors": [],
                    "output": {"answer": "ok"},
                },
            )
            request = RuntimePublicationRequest(
                publication_id=f"tool-round:{round_id}",
                kind="tool-round",
                intent={
                    "round_id": round_id,
                    **round_identity,
                    "tool_call_ids": [tool_call_id],
                    "auxiliary": {
                        "schema_version": 1,
                        "kind": "structured-output-validation",
                        "step": 2,
                        "attempt": 1,
                        "valid": True,
                        "retry_scheduled": False,
                        "event_ids": [validating.id, validated.id],
                    },
                },
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id=tool_call_id,
                        tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
                        content=result_payload["content"],
                        structured=result_payload["structured"],
                        **round_identity,
                    )
                ],
                events=[validating, validated],
                referenced_events=[
                    runtime_publication_event_reference(terminal),
                ],
            )

            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=created.run_epoch,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {}
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == [
                terminal,
                validating,
                validated,
            ]

            store = await _reopen_store(session_store_case, store)
            replayed = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=created.run_epoch,
                expected_transcript_cursor=0,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt
            assert await store.load_events(session_id) == [
                terminal,
                validating,
                validated,
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def _structured_output_coherence_publication(
    *,
    session_id: str,
    case: str,
) -> tuple[dict[str, Any], Event, RuntimePublicationRequest]:
    round_identity = _publication_tool_round_identity(f"structured-output-coherence-{case}")
    round_id = round_identity["tool_round_id"]
    tool_call_id = f"structured-output-call-{case}"
    spec = StructuredOutputSpec(
        name="answer",
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        max_retries=2,
    )
    claimed_valid = case in {"output_mismatch", "valid_with_errors"}
    call_output: dict[str, Any] = (
        {"answer": 7} if case == "failed_retry_disagreement" else {"answer": "ok"}
    )
    primary_errors = [
        {
            "path": "$.answer",
            "message": "7 is not of type 'string'",
            "schema_path": "$.properties.answer.type",
        }
    ]
    source_checkpoint = {
        "pending_tool_round": {
            **round_identity,
            "agent_name": "assistant",
            "environment_name": None,
            "task_id": None,
            "source_model_step_id": f"model-step:{case}",
            "source_transcript_cursor": 0,
            "model_step": 2,
            "structured_output_attempt": 1,
            "structured_output_validation": (
                {
                    "valid": True,
                    "output": {"answer": "ok"},
                    "errors": [],
                }
                if claimed_valid or case == "authoritative_validity"
                else {
                    "valid": False,
                    "output": None,
                    "errors": primary_errors,
                }
            ),
            "tool_calls": [
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": STRUCTURED_OUTPUT_TOOL_NAME,
                    "arguments": {"output": call_output},
                }
            ],
            "structured_output": spec.model_dump(mode="json"),
        }
    }
    result_payload = (
        {
            "content": "Structured output accepted.",
            "structured": {"output": {"answer": "ok"}},
            "artifacts": [],
            "is_error": False,
        }
        if claimed_valid
        else {
            "content": "Structured output rejected.",
            "structured": {"structured_output_errors": primary_errors},
            "artifacts": [],
            "is_error": True,
        }
    )
    terminal = Event(
        id=f"structured-output-terminal-{case}",
        type=(EventType.TOOL_CALL_COMPLETED if claimed_valid else EventType.TOOL_CALL_FAILED),
        session_id=session_id,
        agent_name="assistant",
        tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
        payload={
            **round_identity,
            "tool_call_id": tool_call_id,
            "idempotency_key": f"structured-output-idempotency-{case}",
            "result": result_payload,
        },
    )
    common_payload = {
        **round_identity,
        "name": spec.name,
        "step": 2,
        "attempt": 1,
        "max_retries": spec.max_retries,
    }
    validating = Event(
        id=f"structured-output-validating-{case}",
        type=EventType.STRUCTURED_OUTPUT_VALIDATING,
        session_id=session_id,
        agent_name="assistant",
        payload={**common_payload, "strategy": "tool"},
    )
    outcome = Event(
        id=f"structured-output-outcome-{case}",
        type=(
            EventType.STRUCTURED_OUTPUT_VALIDATED
            if claimed_valid
            else EventType.STRUCTURED_OUTPUT_FAILED
        ),
        session_id=session_id,
        agent_name="assistant",
        payload={
            **common_payload,
            "valid": claimed_valid,
            "errors": (primary_errors if not claimed_valid or case == "valid_with_errors" else []),
            **(
                {
                    "output": (
                        {"answer": "event-only"} if case == "output_mismatch" else {"answer": "ok"}
                    )
                }
                if claimed_valid
                else {}
            ),
        },
    )
    auxiliary_events = [validating, outcome]
    retry_scheduled = case == "failed_retry_disagreement"
    if retry_scheduled:
        auxiliary_events.append(
            Event(
                id=f"structured-output-retry-{case}",
                type=EventType.STRUCTURED_OUTPUT_RETRY,
                session_id=session_id,
                agent_name="assistant",
                payload={
                    **common_payload,
                    "valid": False,
                    "errors": [
                        {
                            "path": "$.answer",
                            "message": "retry disagrees with failed outcome",
                            "schema_path": "$.properties.answer.type",
                        }
                    ],
                },
            )
        )

    request = RuntimePublicationRequest(
        publication_id=f"tool-round:{round_id}",
        kind="tool-round",
        intent={
            "round_id": round_id,
            **round_identity,
            "tool_call_ids": [tool_call_id],
            "auxiliary": {
                "schema_version": 1,
                "kind": "structured-output-validation",
                "step": 2,
                "attempt": 1,
                "valid": claimed_valid,
                "retry_scheduled": retry_scheduled,
                "event_ids": [event.id for event in auxiliary_events],
            },
        },
        mutation=runtime_publication_checkpoint_mutation(
            source_checkpoint,
            {},
        ),
        transcript_messages=[
            Message.tool_result(
                tool_call_id=tool_call_id,
                tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
                content=result_payload["content"],
                structured=result_payload["structured"],
                is_error=result_payload["is_error"],
                **round_identity,
            )
        ],
        events=auxiliary_events,
        referenced_events=[runtime_publication_event_reference(terminal)],
    )
    return source_checkpoint, terminal, request


@pytest.mark.parametrize(
    ("case", "error_match"),
    [
        (
            "output_mismatch",
            "outcome conflicts with its authoritative validation",
        ),
        (
            "valid_with_errors",
            "valid structured-output event requires output and no errors",
        ),
        (
            "failed_retry_disagreement",
            "outcome and retry events contain conflicting results",
        ),
        (
            "authoritative_validity",
            "outcome conflicts with its authoritative validation",
        ),
    ],
)
def test_session_store_conformance_rejects_structured_output_auxiliary_incoherence(
    session_store_case,
    case: str,
    error_match: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_structured_output_coherence_{case}"
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            source_checkpoint, terminal, request = _structured_output_coherence_publication(
                session_id=session_id,
                case=case,
            )
            await store.checkpoint(session_id, source_checkpoint)
            await store.append_event(session_id, terminal)

            with pytest.raises(ValueError, match=error_match):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.PENDING},
                    expected_run_epoch=created.run_epoch,
                    expected_transcript_cursor=0,
                )

            assert await store.load_checkpoint(session_id) == source_checkpoint
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == [terminal]
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_tool_round_scopes_reused_call_id_by_round(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_tool_round_reused_call_id"
            current_identity = _publication_tool_round_identity("current-round")
            old_identity = _publication_tool_round_identity("old-round")
            round_id = current_identity["tool_round_id"]
            tool_call_id = "reused-call"
            source_checkpoint = {
                "pending_tool_round": {
                    **current_identity,
                    "agent_name": "assistant",
                    "tool_calls": [
                        {
                            "tool_call_id": tool_call_id,
                            "tool_name": "lookup",
                            "arguments": {},
                        }
                    ],
                }
            }
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, source_checkpoint)
            result_payload = {
                "content": "current result",
                "structured": None,
                "artifacts": [],
                "is_error": False,
            }
            old_terminal = Event(
                id="reused-call-old-terminal",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **old_identity,
                    "tool_call_id": tool_call_id,
                    "idempotency_key": "old-idempotency",
                    "result": {
                        **result_payload,
                        "content": "old result",
                    },
                },
            )
            current_terminal = Event(
                id="reused-call-current-terminal",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **current_identity,
                    "tool_call_id": tool_call_id,
                    "idempotency_key": "current-idempotency",
                    "result": result_payload,
                },
            )
            await store.append_events(
                session_id,
                [old_terminal, current_terminal],
            )
            assert [
                event.id
                for event in await store.load_tool_round_lifecycle_events(
                    session_id,
                    [tool_call_id],
                )
            ] == [old_terminal.id, current_terminal.id]
            assert [
                event.id
                for event in await store.load_tool_round_lifecycle_events_for_round(
                    session_id,
                    [tool_call_id],
                    tool_round_identity=ToolRoundIdentity.model_validate(current_identity),
                )
            ] == [current_terminal.id]
            request = RuntimePublicationRequest(
                publication_id=f"tool-round:{round_id}",
                kind="tool-round",
                intent={
                    "round_id": round_id,
                    **current_identity,
                    "tool_call_ids": [tool_call_id],
                },
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id=tool_call_id,
                        tool_name="lookup",
                        content=result_payload["content"],
                        **current_identity,
                    )
                ],
                events=[],
                referenced_events=[
                    runtime_publication_event_reference(current_terminal),
                ],
            )

            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )

            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {}
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == [
                old_terminal,
                current_terminal,
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_round_lookup_retains_ambiguous_reused_id_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_tool_round_ambiguous_reused_call_id"
            tool_call_id = "reused-call"
            old_identity = _publication_tool_round_identity("ambiguous-old-round")
            current_identity = _publication_tool_round_identity("ambiguous-current-round")
            conflicting_identity = _publication_tool_round_identity("ambiguous-conflicting-round")
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(
                session_id,
                {
                    "pending_tool_round": {
                        **current_identity,
                        "agent_name": "assistant",
                        "tool_calls": [
                            {
                                "tool_call_id": tool_call_id,
                                "tool_name": "lookup",
                                "arguments": {},
                            }
                        ],
                    }
                },
            )
            events = [
                Event(
                    id="reused-call-old-valid",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        **old_identity,
                        "tool_call_id": tool_call_id,
                    },
                ),
                Event(
                    id="reused-call-missing-round",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        "tool_call_id": tool_call_id,
                        "model_step_id": current_identity["model_step_id"],
                        "model_attempt_id": current_identity["model_attempt_id"],
                    },
                ),
                Event(
                    id="reused-call-padded-round",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        "tool_call_id": tool_call_id,
                        "model_step_id": current_identity["model_step_id"],
                        "model_attempt_id": current_identity["model_attempt_id"],
                        "tool_round_id": " padded-round ",
                    },
                ),
                Event(
                    id="reused-call-malformed-model-identity",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        "tool_call_id": tool_call_id,
                        "model_step_id": "malformed-model-step",
                        "model_attempt_id": conflicting_identity["model_attempt_id"],
                        "tool_round_id": conflicting_identity["tool_round_id"],
                    },
                ),
                Event(
                    id="reused-call-conflicting-round",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        **current_identity,
                        "tool_round_id": conflicting_identity["tool_round_id"],
                        "tool_call_id": tool_call_id,
                    },
                ),
                Event(
                    id="reused-call-current-valid",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        **current_identity,
                        "tool_call_id": tool_call_id,
                    },
                ),
            ]
            await store.append_events(session_id, events)

            scoped = await store.load_tool_round_lifecycle_events_for_round(
                session_id,
                [tool_call_id],
                tool_round_identity=ToolRoundIdentity.model_validate(current_identity),
            )

            assert [event.id for event in scoped] == [
                "reused-call-missing-round",
                "reused-call-padded-round",
                "reused-call-malformed-model-identity",
                "reused-call-conflicting-round",
                "reused-call-current-valid",
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_reopens_and_promotes_exactly_once(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_lifecycle"
            stage_id = "model-step:run-1:cursor-0"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "stage the provider completion")],
                ),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            stage_request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=stage_id,
                dispatch_ordinal=0,
                intent={"logical_step": 1, "request_fingerprint": "request-1"},
                reservation_ids=["reservation-app", "reservation-session"],
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=stage_request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert prepared.replayed is False
            assert prepared.dispatch_authorized is True
            assert prepared.stage.state == "in_flight"
            assert prepared.stage.source_status is SessionStatus.RUNNING
            assert prepared.stage.source_run_epoch == running.run_epoch
            assert prepared.stage.source_transcript_cursor == 0
            assert prepared.stage.reservation_ids == (
                "reservation-app",
                "reservation-session",
            )
            assert prepared.stage.publication is None
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == prepared.stage
            assert active.marker_digest
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []

            preparation_replay = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=stage_id,
                    dispatch_ordinal=0,
                    intent={"logical_step": 1, "request_fingerprint": "request-1"},
                    reservation_ids=["reservation-app", "reservation-session"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert preparation_replay.replayed is True
            assert preparation_replay.dispatch_authorized is False
            assert preparation_replay.stage == prepared.stage

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1, "request_fingerprint": "request-1"},
                completion_event_id="staged-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "durably staged answer"),
                event_payload={
                    "step": 1,
                    "usage": {"input_tokens": 4, "output_tokens": 3},
                },
                reservation_ids=("reservation-app", "reservation-session"),
            )
            before_completion_session = await store.load(session_id)
            assert before_completion_session is not None
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert completed.replayed is False
            assert completed.dispatch_authorized is False
            assert completed.stage.state == "completed"
            assert completed.stage.publication == publication
            assert completed.stage.completed_at is not None
            assert completed.stage.completed_at >= completed.stage.prepared_at
            after_completion_session = await store.load(session_id)
            assert after_completion_session is not None
            assert after_completion_session.updated_at == completed.stage.completed_at
            assert after_completion_session.last_activity_at == completed.stage.completed_at
            assert (
                after_completion_session.last_activity_at
                > before_completion_session.last_activity_at
            )
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == completed.stage

            completion_replay = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=RuntimePublicationRequest(
                    publication_id=publication.publication_id,
                    kind=publication.kind,
                    intent=publication.intent,
                    mutation=publication.mutation,
                    transcript_messages=publication.transcript_messages,
                    events=publication.events,
                    referenced_events=publication.referenced_events,
                ),
            )
            assert completion_replay.replayed is True
            assert completion_replay.dispatch_authorized is False
            assert completion_replay.stage == completed.stage
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="must be published before another dispatch",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id=f"{stage_id}:retry-after-terminal",
                        logical_step_id=stage_id,
                        dispatch_ordinal=1,
                        intent={"logical_step": 1, "request_fingerprint": "request-2"},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []

            store = await _reopen_store(session_store_case, store)
            assert await store.load_model_completion_stage(session_id, stage_id) == completed.stage

            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False
            assert promoted.receipt.publication_id == stage_id
            assert promoted.receipt.appended_event_ids == ("staged-model-completed",)
            assert promoted.receipt.transcript_start_cursor == 0
            assert promoted.receipt.transcript_end_cursor == 1
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_active_model_completion_stage(session_id) is None
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=session_id,
                event_id="staged-model-completed",
            )
            assert delivery is not None
            assert delivery.status is PersistedEventSideEffectStatus.PENDING

            store = await _reopen_store(session_store_case, store)
            later_preparation = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id="model-step:later-logical-step:attempt-0",
                    logical_step_id="model-step:later-logical-step",
                    dispatch_ordinal=0,
                    intent={"logical_step": 2, "request_fingerprint": "request-2"},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=1,
            )
            assert later_preparation.dispatch_authorized is True
            later_active = await store.load_active_model_completion_stage(session_id)
            assert later_active is not None
            assert later_active.stage == later_preparation.stage
            store = await _reopen_store(session_store_case, store)
            before_replay_session = await store.load(session_id)
            replayed = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch + 100,
            )
            assert replayed.replayed is True
            assert replayed.receipt == promoted.receipt
            assert replayed.session == before_replay_session
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
            assert await store.load_model_completion_stage(session_id, stage_id) == completed.stage
            assert await store.load_active_model_completion_stage(session_id) == later_active
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_bounds_model_completion_recovery_context(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_recovery_context_bound"
            stage_id = "model-step:recovery-context-bound"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            recovery_context = {
                "schema_version": 1,
                "request_metadata": {"payload": "x" * 1024},
            }
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=stage_id,
                    dispatch_ordinal=0,
                    intent={"recovery_context": recovery_context},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert prepared.stage.intent["recovery_context"] == recovery_context
            assert (await store.load_model_completion_stage(session_id, stage_id)) == (
                prepared.stage
            )

            with pytest.raises(ValidationError, match="exceeds the durable byte limit"):
                ModelCompletionStageRequest(
                    stage_id="model-step:oversized-recovery-context",
                    logical_step_id="model-step:oversized-recovery-context",
                    dispatch_ordinal=1,
                    intent={
                        "recovery_context": {
                            "payload": "x" * MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES
                        }
                    },
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_abandonment_rejects_wrong_digest_and_terminal(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_abandonment_refusal"
            stage_id = "model-step:abandonment-refusal:attempt-0"
            logical_step_id = "model-step:abandonment-refusal"
            intent = {"logical_step": "abandonment-refusal", "request_fingerprint": "request-1"}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent=intent,
                    reservation_ids=("reservation-1",),
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="preparation digest is stale",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest="0" * 64,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == prepared.stage
            assert (
                await _load_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_abandonment_key(stage_id),
                )
                is None
            )
            with pytest.raises(SessionRunFenced, match="stage run epoch is stale"):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=prepared.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch + 1,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage
            assert await store.load_active_model_completion_stage(session_id) == active

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                intent=intent,
                completion_event_id="abandonment-refusal-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "terminal evidence"),
                reservation_ids=("reservation-1",),
            )
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="terminal model-completion stage cannot be abandoned",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=prepared.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == completed.stage
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == completed.stage
            assert (
                await _load_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_abandonment_key(stage_id),
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("publication_state", ["winner", "receipt"])
def test_session_store_conformance_model_completion_abandonment_rejects_publication_state(
    session_store_case,
    publication_state: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_model_completion_abandonment_{publication_state}"
            stage_id = f"model-step:abandonment-{publication_state}:attempt-0"
            logical_step_id = f"model-step:abandonment-{publication_state}"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent={"logical_step": publication_state},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            publication_storage_key = (
                _model_completion_winner_key(logical_step_id)
                if publication_state == "winner"
                else _runtime_publication_key(logical_step_id)
            )
            await _set_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=publication_storage_key,
                record={"malformed_but_durable": publication_state},
            )

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="durable publication state cannot be abandoned",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=prepared.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == prepared.stage
            assert (
                await _load_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_abandonment_key(stage_id),
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_abandonment_replays_and_reprepares_safely(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_abandonment_replay"
            stage_id = "model-step:abandonment-replay:attempt-0"
            logical_step_id = "model-step:abandonment-replay"
            request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                intent={"logical_step": "abandonment-replay", "request_fingerprint": "exact"},
                reservation_ids=("reservation-1", "reservation-2"),
            )
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            first = await store.prepare_model_completion_stage(
                session_id,
                request=request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            first_active = await store.load_active_model_completion_stage(session_id)
            assert first_active is not None

            abandoned = await store.abandon_model_completion_stage(
                session_id,
                stage_id=stage_id,
                preparation_digest=first.stage.preparation_digest,
                expected_run_epoch=running.run_epoch,
            )
            assert abandoned.replayed is False
            assert abandoned.abandonment.session_id == session_id
            assert abandoned.abandonment.stage_id == stage_id
            assert abandoned.abandonment.logical_step_id == logical_step_id
            assert abandoned.abandonment.preparation_request_digest == (
                first.stage.preparation_request_digest
            )
            assert abandoned.abandonment.preparation_digest == first.stage.preparation_digest
            assert abandoned.abandonment.active_marker_digest == first_active.marker_digest
            assert abandoned.abandonment.source_run_epoch == running.run_epoch
            assert await store.load_model_completion_stage(session_id, stage_id) is None
            assert await store.load_active_model_completion_stage(session_id) is None
            tombstone = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=_model_completion_abandonment_key(stage_id),
            )
            assert tombstone is not None
            assert tombstone["preparation_digest"] == first.stage.preparation_digest
            assert tombstone["record_digest"] == abandoned.abandonment.abandonment_digest

            store = await _reopen_store(session_store_case, store)
            before_replay = await store.load(session_id)
            replayed = await store.abandon_model_completion_stage(
                session_id,
                stage_id=stage_id,
                preparation_digest=first.stage.preparation_digest,
                expected_run_epoch=running.run_epoch,
            )
            assert replayed.replayed is True
            assert replayed.abandonment == abandoned.abandonment
            assert await store.load(session_id) == before_replay
            assert await store.load_model_completion_stage(session_id, stage_id) is None
            assert await store.load_active_model_completion_stage(session_id) is None

            conflicting_request = request.model_copy(
                update={"intent": {"logical_step": "abandonment-replay", "changed": True}},
                deep=True,
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="only be reused for its exact request",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=conflicting_request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) is None

            second = await store.prepare_model_completion_stage(
                session_id,
                request=request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert second.replayed is False
            assert second.dispatch_authorized is True
            assert second.stage.preparation_digest != first.stage.preparation_digest
            second_active = await store.load_active_model_completion_stage(session_id)
            assert second_active is not None
            assert second_active.stage == second.stage

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="preparation digest is stale",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=first.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == second.stage
            assert await store.load_active_model_completion_stage(session_id) == second_active
            unchanged_tombstone = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=_model_completion_abandonment_key(stage_id),
            )
            assert unchanged_tombstone == tombstone

            second_abandonment = await store.abandon_model_completion_stage(
                session_id,
                stage_id=stage_id,
                preparation_digest=second.stage.preparation_digest,
                expected_run_epoch=running.run_epoch,
            )
            assert second_abandonment.replayed is False
            assert second_abandonment.abandonment.preparation_digest == (
                second.stage.preparation_digest
            )
            assert await store.load_model_completion_stage(session_id, stage_id) is None
            assert await store.load_active_model_completion_stage(session_id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["missing-active", "malformed-active", "forged-winner"])
def test_session_store_conformance_model_completion_preparation_replay_fails_closed(
    session_store_case,
    corruption: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_model_completion_prepare_replay_{corruption}"
            logical_step_id = f"model-step:prepare-replay-{corruption}"
            stage_id = f"{logical_step_id}:attempt-0"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                intent={"logical_step": logical_step_id},
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert prepared.dispatch_authorized is True

            if corruption == "missing-active":
                await _delete_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
                )
            elif corruption == "malformed-active":
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
                    record={"malformed": True},
                )
            else:
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_winner_key(logical_step_id),
                    record={"forged": True},
                )
            store = await _reopen_store(session_store_case, store)

            with pytest.raises(SessionModelCompletionStageConflict):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            durable_stage = await store.load_model_completion_stage(session_id, stage_id)
            assert durable_stage == prepared.stage
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
            assert await store.load_runtime_publication_receipt(session_id, logical_step_id) is None
            if corruption == "malformed-active":
                before_completion = await store.load(session_id)
                assert before_completion is not None
                terminal = await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=_assistant_model_completion_publication(
                        session_id=session_id,
                        stage_id=stage_id,
                        logical_step_id=logical_step_id,
                        intent=request.intent,
                        completion_event_id="malformed-active-terminal-evidence",
                        source_transcript_cursor=0,
                        assistant_message=Message.text(
                            "assistant",
                            "retain terminal evidence",
                        ),
                    ),
                )
                assert terminal.stage.state == "completed"
                after_completion = await store.load(session_id)
                assert after_completion is not None
                assert after_completion.updated_at == terminal.stage.completed_at
                assert after_completion.updated_at > before_completion.updated_at
                assert after_completion.last_activity_at == before_completion.last_activity_at
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    [
        "winner-active",
        "malformed-active",
        "malformed-winner",
        "missing-terminal",
        "missing-receipt",
        "missing-winner",
    ],
)
def test_session_store_conformance_model_completion_publication_state_fails_closed(
    session_store_case,
    corruption: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_model_completion_publication_state_{corruption}"
            logical_step_id = f"model-step:publication-state-{corruption}"
            stage_id = f"{logical_step_id}:attempt-0"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            stage_request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                intent={"logical_step": logical_step_id},
            )
            await store.prepare_model_completion_stage(
                session_id,
                request=stage_request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            active_record = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
            )
            assert active_record is not None
            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                intent=stage_request.intent,
                completion_event_id=f"publication-state-event-{corruption}",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative answer"),
            )
            await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False

            if corruption == "winner-active":
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
                    record=active_record,
                )
            elif corruption == "malformed-active":
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
                    record={"malformed": True},
                )
            elif corruption == "malformed-winner":
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_winner_key(logical_step_id),
                    record={"malformed": True},
                )
            elif corruption == "missing-terminal":
                await _delete_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_stage_key(stage_id, terminal=True),
                )
            elif corruption == "missing-receipt":
                await _delete_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_runtime_publication_key(logical_step_id),
                )
            else:
                await _delete_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_winner_key(logical_step_id),
                )
            store = await _reopen_store(session_store_case, store)

            with pytest.raises(SessionModelCompletionStageConflict):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=stage_request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )

            expected_error = (
                (SessionModelCompletionStageConflict, SessionModelCompletionStageIncomplete)
                if corruption == "missing-terminal"
                else SessionModelCompletionStageConflict
            )
            with pytest.raises(expected_error):
                await store.promote_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    expected_run_epoch=running.run_epoch + 100,
                )
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_preserves_non_turn_completion(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_non_turn"
            stage_id = "model-step:contentless-attempt"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            intent = {"logical_step": 1, "attempt": 1}
            await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=stage_id,
                    dispatch_ordinal=0,
                    intent=intent,
                    reservation_ids=["reservation-contentless-attempt"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent=intent,
                completion_event_id="contentless-model-completed",
                source_transcript_cursor=0,
                assistant_message=None,
                classification={
                    "type": "invalid",
                    "reason": "assistant produced no content",
                },
                event_payload={
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 0},
                },
                reservation_ids=("reservation-contentless-attempt",),
            )
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert completed.stage.state == "completed"

            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False
            assert promoted.receipt.transcript_start_cursor == 0
            assert promoted.receipt.transcript_end_cursor == 0
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == list(publication.events)

            store = await _reopen_store(session_store_case, store)
            replayed = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch + 1,
            )
            assert replayed.replayed is True
            assert replayed.receipt == promoted.receipt
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == list(publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("missing", "atomically publish exactly"),
        ("stage-id", "pointer conflicts"),
        ("source-boundary", "pointer conflicts"),
        ("completion-event", "pointer conflicts"),
        ("classification", "pointer conflicts"),
    ],
)
def test_session_store_conformance_model_completion_stage_rejects_invalid_publication_pointer(
    session_store_case,
    corruption: str,
    expected_error: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_model_completion_pointer_{corruption}"
            logical_step_id = f"model-step:pointer-{corruption}"
            stage_id = f"{logical_step_id}:attempt-0"
            intent = {"logical_step": logical_step_id}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent=intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            valid = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                intent=intent,
                completion_event_id=f"pointer-completed-{corruption}",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative"),
            )
            if corruption == "missing":
                mutation = runtime_publication_checkpoint_mutation(None, None)
            else:
                (pointer_operation,) = valid.mutation.operations
                assert type(pointer_operation.value) is dict
                pointer = dict(pointer_operation.value)
                if corruption == "stage-id":
                    pointer["stage_id"] = f"{stage_id}:forged"
                elif corruption == "source-boundary":
                    pointer["source_transcript_cursor"] = 1
                    pointer["transcript_end_cursor"] = 2
                elif corruption == "completion-event":
                    pointer["completion_event_id"] = "forged-model-completed"
                else:
                    pointer["classification"] = {
                        "type": "final",
                        "reason": "forged",
                    }
                mutation = runtime_publication_checkpoint_mutation(
                    None,
                    {LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: pointer},
                )
            publication = RuntimePublicationRequest(
                publication_id=valid.publication_id,
                kind=valid.kind,
                intent=valid.intent,
                mutation=mutation,
                transcript_messages=valid.transcript_messages,
                events=valid.events,
                referenced_events=valid.referenced_events,
            )

            with pytest.raises(ValueError, match=expected_error):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=publication,
                )
            stage = await store.load_model_completion_stage(session_id, stage_id)
            assert stage is not None
            assert stage.state == "in_flight"
            assert await store.load_checkpoint(session_id) is None
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_preserves_purpose_validation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_compaction_purpose"
            stage_id = "context-compaction:attempt-0"
            logical_step_id = "context-compaction:logical-0"
            intent = {"operation_id": "compaction-0", "input_digest": "input-0"}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    purpose="context-compaction",
                    intent=intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            event = Event(
                id="context-compaction-model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                payload={"purpose": "context_compaction", "usage": {"input_tokens": 2}},
            )
            with pytest.raises(ValueError, match="kind must be 'context-compaction'"):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=RuntimePublicationRequest(
                        publication_id=logical_step_id,
                        kind="model-step",
                        intent=intent,
                        mutation=runtime_publication_checkpoint_mutation(None, None),
                        transcript_messages=[],
                        events=[event],
                    ),
                )
            publication = RuntimePublicationRequest(
                publication_id=logical_step_id,
                kind="context-compaction",
                intent=intent,
                mutation=runtime_publication_checkpoint_mutation(None, None),
                transcript_messages=[],
                events=[event],
            )
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert completed.stage.state == "completed"
            assert completed.stage.purpose == "context-compaction"
            assert completed.stage.publication == publication
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []

            assistant_session_id = "sess_model_completion_stage_assistant_purpose"
            assistant_stage_id = "model-step:assistant-purpose:attempt-0"
            assistant_logical_step_id = "model-step:assistant-purpose"
            assistant_intent = {"logical_step": "assistant-purpose"}
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=assistant_session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            assistant_running = await store.transition_status(
                assistant_session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                assistant_session_id,
                request=ModelCompletionStageRequest(
                    stage_id=assistant_stage_id,
                    logical_step_id=assistant_logical_step_id,
                    dispatch_ordinal=0,
                    purpose="assistant-turn",
                    intent=assistant_intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=assistant_running.run_epoch,
                expected_transcript_cursor=0,
            )
            with pytest.raises(
                ValueError,
                match="cannot carry context-compaction evidence",
            ):
                await store.complete_model_completion_stage(
                    assistant_session_id,
                    stage_id=assistant_stage_id,
                    publication=RuntimePublicationRequest(
                        publication_id=assistant_logical_step_id,
                        kind="model-step",
                        intent=assistant_intent,
                        mutation=runtime_publication_checkpoint_mutation(None, None),
                        transcript_messages=[Message.text("assistant", "not a compaction")],
                        events=[
                            Event(
                                id="assistant-purpose-model-completed",
                                type=EventType.MODEL_COMPLETED,
                                session_id=assistant_session_id,
                                payload={"purpose": "context_compaction"},
                            )
                        ],
                    ),
                )
            assistant_stage = await store.load_model_completion_stage(
                assistant_session_id,
                assistant_stage_id,
            )
            assert assistant_stage is not None
            assert assistant_stage.state == "in_flight"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_fences_conflicts_and_drift(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_conflicts"
            stage_id = "model-step:conflict"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=stage_id,
                dispatch_ordinal=0,
                intent={"logical_step": 1},
                reservation_ids=["reservation-1"],
            )

            with pytest.raises(SessionRunFenced, match="run epoch is stale"):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch + 1,
                    expected_transcript_cursor=0,
                )
            with pytest.raises(ValueError, match="transcript cursor is stale"):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=1,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) is None

            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different logical model completion",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id="model-step:unrelated",
                        logical_step_id="model-step:unrelated",
                        dispatch_ordinal=0,
                        intent={"logical_step": "unrelated"},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="advance dispatch_ordinal monotonically",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id="model-step:non-monotonic-retry",
                        logical_step_id=stage_id,
                        dispatch_ordinal=0,
                        intent={"logical_step": 1, "retry": True},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different request",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id=stage_id,
                        logical_step_id=stage_id,
                        dispatch_ordinal=0,
                        intent={"logical_step": 2},
                        reservation_ids=["reservation-1"],
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            with pytest.raises(SessionModelCompletionStageIncomplete, match="still in flight"):
                await store.promote_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    expected_run_epoch=running.run_epoch,
                )

            with pytest.raises(ValueError, match="requires a non-turn"):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=_assistant_model_completion_publication(
                        session_id=session_id,
                        stage_id=stage_id,
                        logical_step_id=stage_id,
                        intent={"logical_step": 1},
                        completion_event_id="unclassified-contentless-completion",
                        source_transcript_cursor=0,
                        assistant_message=None,
                        classification={"type": "invalid"},
                        reservation_ids=("reservation-1",),
                        include_classification=False,
                    ),
                )

            private_key = _model_completion_stage_key(stage_id, terminal=False)
            with pytest.raises(ValueError, match="reserved model-completion stage namespace"):
                await store.load_session_operation(session_id, private_key)
            with pytest.raises(ValueError, match="reserved model-completion stage namespace"):
                await store.publish_session_operation(
                    session_id,
                    idempotency_key=private_key,
                    operation_transform=lambda _session, _checkpoint, _record: (
                        SessionOperationPublication(checkpoint={})
                    ),
                    events=[],
                )

            wrong_intent = RuntimePublicationRequest(
                publication_id=stage_id,
                kind="model-step",
                intent={"logical_step": 2},
                mutation=runtime_publication_checkpoint_mutation(None, None),
                transcript_messages=[Message.text("assistant", "wrong")],
                events=[
                    Event(
                        id="wrong-intent-completion",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="intent conflicts",
            ):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=wrong_intent,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1},
                completion_event_id="authoritative-completion",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative"),
                reservation_ids=("reservation-1",),
            )
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            conflicting_publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1},
                completion_event_id="contradictory-completion",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "contradictory"),
                reservation_ids=("reservation-1",),
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different terminal material",
            ):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=conflicting_publication,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == completed.stage

            await _replace_model_completion_stage_record(
                session_store_case,
                store,
                session_id=session_id,
                stage_id=stage_id,
                terminal=True,
                record={"record_type": "corrupted-terminal"},
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="terminal model completion is malformed",
            ):
                await store.load_model_completion_stage(session_id, stage_id)

            cursor_session_id = "sess_model_completion_stage_cursor_fence"
            await store.create(
                RunRequest(agent_name="assistant", session_id=cursor_session_id, messages=[]),
                identity=_identity(),
            )
            cursor_running = await store.transition_status(
                cursor_session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                cursor_session_id,
                request=ModelCompletionStageRequest(
                    stage_id="model-step:cursor-fence",
                    logical_step_id="model-step:cursor-fence",
                    dispatch_ordinal=0,
                    intent={"logical_step": 1},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=cursor_running.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.append_transcript_messages(
                cursor_session_id,
                [Message.text("user", "concurrent input")],
            )
            cursor_publication = _assistant_model_completion_publication(
                session_id=cursor_session_id,
                stage_id="model-step:cursor-fence",
                logical_step_id="model-step:cursor-fence",
                intent={"logical_step": 1},
                completion_event_id="cursor-fenced-completion",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "stale"),
            )
            cursor_completion = await store.complete_model_completion_stage(
                cursor_session_id,
                stage_id="model-step:cursor-fence",
                publication=cursor_publication,
            )
            assert cursor_completion.stage.state == "completed"
            assert await store.load_events(cursor_session_id) == []
            with pytest.raises(ValueError, match="transcript cursor is stale"):
                await store.promote_model_completion_stage(
                    cursor_session_id,
                    stage_id="model-step:cursor-fence",
                    expected_run_epoch=cursor_running.run_epoch,
                )
            assert await store.load_checkpoint(cursor_session_id) is None

            epoch_session_id = "sess_model_completion_stage_epoch_fence"
            await store.create(
                RunRequest(agent_name="assistant", session_id=epoch_session_id, messages=[]),
                identity=_identity(),
            )
            first_epoch = await store.transition_status(
                epoch_session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                epoch_session_id,
                request=ModelCompletionStageRequest(
                    stage_id="model-step:epoch-fence",
                    logical_step_id="model-step:epoch-fence",
                    dispatch_ordinal=0,
                    intent={"logical_step": 1},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=first_epoch.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.release_run_fence(epoch_session_id)
            await store.update_status(epoch_session_id, SessionStatus.INTERRUPTED)
            second_epoch = await store.transition_status(
                epoch_session_id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
            )
            assert second_epoch.run_epoch > first_epoch.run_epoch
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different logical model completion",
            ):
                await store.prepare_model_completion_stage(
                    epoch_session_id,
                    request=ModelCompletionStageRequest(
                        stage_id="model-step:epoch-fence:retry",
                        logical_step_id="model-step:epoch-fence",
                        dispatch_ordinal=1,
                        intent={"logical_step": 1, "attempt": 2},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=second_epoch.run_epoch,
                    expected_transcript_cursor=0,
                )
            active_after_takeover = await store.load_active_model_completion_stage(epoch_session_id)
            assert active_after_takeover is not None
            assert active_after_takeover.stage.stage_id == "model-step:epoch-fence"
            epoch_publication = _assistant_model_completion_publication(
                session_id=epoch_session_id,
                stage_id="model-step:epoch-fence",
                logical_step_id="model-step:epoch-fence",
                intent={"logical_step": 1},
                completion_event_id="epoch-fenced-completion",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "recovered completion"),
            )
            before_epoch_completion = await store.load(epoch_session_id)
            assert before_epoch_completion is not None
            epoch_completion = await store.complete_model_completion_stage(
                epoch_session_id,
                stage_id="model-step:epoch-fence",
                publication=epoch_publication,
            )
            assert epoch_completion.stage.state == "completed"
            assert epoch_completion.dispatch_authorized is False
            assert epoch_completion.stage.source_run_epoch == first_epoch.run_epoch
            after_epoch_completion = await store.load(epoch_session_id)
            assert after_epoch_completion is not None
            assert after_epoch_completion.updated_at == epoch_completion.stage.completed_at
            assert after_epoch_completion.updated_at > before_epoch_completion.updated_at
            assert (
                after_epoch_completion.last_activity_at == before_epoch_completion.last_activity_at
            )
            assert await store.load_transcript(epoch_session_id) == []
            assert await store.load_events(epoch_session_id) == []
            with pytest.raises(SessionRunFenced, match="run epoch is stale"):
                await store.promote_model_completion_stage(
                    epoch_session_id,
                    stage_id="model-step:epoch-fence",
                    expected_run_epoch=first_epoch.run_epoch,
                )
            assert await store.load_checkpoint(epoch_session_id) is None

            recovered_promotion = await store.promote_model_completion_stage(
                epoch_session_id,
                stage_id="model-step:epoch-fence",
                expected_run_epoch=second_epoch.run_epoch,
            )
            assert recovered_promotion.replayed is False
            assert recovered_promotion.receipt.source_run_epoch == second_epoch.run_epoch
            assert await store.load_checkpoint(
                epoch_session_id
            ) == _model_completion_publication_checkpoint(epoch_publication)
            assert await store.load_transcript(epoch_session_id) == list(
                epoch_publication.transcript_messages
            )
            assert await store.load_events(epoch_session_id) == list(epoch_publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_late_superseded_completion_liveness(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_superseded_liveness"
            logical_step_id = "model-step:superseded-liveness"
            first_stage_id = f"{logical_step_id}:attempt-0"
            retry_stage_id = f"{logical_step_id}:attempt-1"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            first_request = ModelCompletionStageRequest(
                stage_id=first_stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                intent={"logical_step": logical_step_id, "attempt": 0},
            )
            first = await store.prepare_model_completion_stage(
                session_id,
                request=first_request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert first.dispatch_authorized is True
            retry = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=retry_stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=1,
                    intent={"logical_step": logical_step_id, "attempt": 1},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert retry.dispatch_authorized is True
            before_completion = await store.load(session_id)
            assert before_completion is not None

            late_completion = await store.complete_model_completion_stage(
                session_id,
                stage_id=first_stage_id,
                publication=_assistant_model_completion_publication(
                    session_id=session_id,
                    stage_id=first_stage_id,
                    logical_step_id=logical_step_id,
                    intent=first_request.intent,
                    completion_event_id="superseded-liveness-model-completed",
                    source_transcript_cursor=0,
                    assistant_message=Message.text("assistant", "late first attempt"),
                ),
            )
            assert late_completion.replayed is False
            assert late_completion.dispatch_authorized is False
            after_completion = await store.load(session_id)
            assert after_completion is not None
            assert after_completion.updated_at == late_completion.stage.completed_at
            assert after_completion.updated_at > before_completion.updated_at
            assert after_completion.last_activity_at == before_completion.last_activity_at
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == retry.stage

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="no longer the exact active dispatch",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=first_request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            assert await store.load_active_model_completion_stage(session_id) == active
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_allows_trusted_live_retry(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_live_retry"
            first_stage_id = "model-step:logical-1:attempt-1"
            retry_stage_id = "model-step:logical-1:attempt-2"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )

            first = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=first_stage_id,
                    logical_step_id="model-step:logical-1",
                    dispatch_ordinal=0,
                    intent={"logical_step": "logical-1", "attempt": 1},
                    reservation_ids=["reservation-attempt-1"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert first.stage.state == "in_flight"
            first_active = await store.load_active_model_completion_stage(session_id)
            assert first_active is not None
            assert first_active.stage == first.stage

            # A runtime that directly observed a terminal provider error may start
            # a new attempt. The old identity remains dispatch-ambiguous for
            # process-loss recovery and is never reused for provider dispatch.
            retry = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=retry_stage_id,
                    logical_step_id="model-step:logical-1",
                    dispatch_ordinal=1,
                    intent={"logical_step": "logical-1", "attempt": 2},
                    reservation_ids=["reservation-attempt-2"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert retry.stage.state == "in_flight"
            retry_active = await store.load_active_model_completion_stage(session_id)
            assert retry_active is not None
            assert retry_active.stage == retry.stage
            assert retry_active.stage.dispatch_ordinal == 1
            assert (
                await store.load_model_completion_stage(session_id, first_stage_id) == first.stage
            )

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=retry_stage_id,
                logical_step_id="model-step:logical-1",
                intent={"logical_step": "logical-1", "attempt": 2},
                completion_event_id="live-retry-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "retry succeeded"),
                reservation_ids=("reservation-attempt-2",),
            )
            await store.complete_model_completion_stage(
                session_id,
                stage_id=retry_stage_id,
                publication=publication,
            )
            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=retry_stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
            assert await store.load_active_model_completion_stage(session_id) is None
            durable_first = await store.load_model_completion_stage(
                session_id,
                first_stage_id,
            )
            assert durable_first is not None
            assert durable_first.state == "in_flight"

            late_first_completion = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=first_stage_id,
                logical_step_id="model-step:logical-1",
                intent={"logical_step": "logical-1", "attempt": 1},
                completion_event_id="late-first-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "late first response"),
                reservation_ids=("reservation-attempt-1",),
            )
            late_completion = await store.complete_model_completion_stage(
                session_id,
                stage_id=first_stage_id,
                publication=late_first_completion,
            )
            assert late_completion.stage.state == "completed"
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different model-completion winner",
            ):
                await store.promote_model_completion_stage(
                    session_id,
                    stage_id=first_stage_id,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_concurrent_replay(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_concurrency"
            stage_id = "model-step:concurrent"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=stage_id,
                dispatch_ordinal=0,
                intent={"logical_step": 1, "request_fingerprint": "shared"},
                reservation_ids=["reservation-shared"],
            )

            async def prepare():
                return await store.prepare_model_completion_stage(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )

            preparation_results = await asyncio.gather(prepare(), prepare())
            assert sorted(result.replayed for result in preparation_results) == [False, True]
            assert sorted(result.dispatch_authorized for result in preparation_results) == [
                False,
                True,
            ]
            assert preparation_results[0].stage == preparation_results[1].stage

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1, "request_fingerprint": "shared"},
                completion_event_id="concurrent-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "one authoritative answer"),
                reservation_ids=("reservation-shared",),
            )

            async def complete():
                return await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=publication,
                )

            completion_results = await asyncio.gather(complete(), complete())
            assert sorted(result.replayed for result in completion_results) == [False, True]
            assert all(result.dispatch_authorized is False for result in completion_results)
            assert completion_results[0].stage == completion_results[1].stage

            async def promote():
                return await store.promote_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    expected_run_epoch=running.run_epoch,
                )

            promotion_results = await asyncio.gather(promote(), promote())
            assert sorted(result.replayed for result in promotion_results) == [False, True]
            assert promotion_results[0].receipt == promotion_results[1].receipt
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_zero_message_model_retry_has_one_logical_winner(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_zero_message_model_retry_winner"
            logical_step_id = "model-step:zero-message-winner"
            stage_ids = (
                "model-step:zero-message:attempt-0",
                "model-step:zero-message:attempt-1",
            )
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            for ordinal, stage_id in enumerate(stage_ids):
                prepared = await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id=stage_id,
                        logical_step_id=logical_step_id,
                        dispatch_ordinal=ordinal,
                        intent={"logical_step": "zero-message", "attempt": ordinal},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            assert prepared.dispatch_authorized is True
            for ordinal, stage_id in enumerate(stage_ids):
                publication = _assistant_model_completion_publication(
                    session_id=session_id,
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    intent={"logical_step": "zero-message", "attempt": ordinal},
                    completion_event_id=f"zero-message-completed-{ordinal}",
                    source_transcript_cursor=0,
                    assistant_message=None,
                    classification={
                        "type": "invalid",
                        "reason": "assistant produced no content",
                    },
                )
                completed = await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=publication,
                )
                assert completed.dispatch_authorized is False

            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage.stage_id == stage_ids[1]
            assert active.stage.dispatch_ordinal == 1

            outcomes = await asyncio.gather(
                *(
                    store.promote_model_completion_stage(
                        session_id,
                        stage_id=stage_id,
                        expected_run_epoch=running.run_epoch,
                    )
                    for stage_id in stage_ids
                ),
                return_exceptions=True,
            )
            winners = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
            conflicts = [
                outcome
                for outcome in outcomes
                if isinstance(outcome, SessionModelCompletionStageConflict)
            ]
            assert len(winners) == 1
            assert len(conflicts) == 1
            winner = winners[0]
            assert winner.replayed is False
            assert winner.receipt.publication_id == logical_step_id
            assert winner.receipt.transcript_start_cursor == 0
            assert winner.receipt.transcript_end_cursor == 0
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_transcript(session_id) == []
            assert [event.id for event in await store.load_events(session_id)] == [
                "zero-message-completed-1"
            ]
            assert await store.load_active_model_completion_stage(session_id) is None

            replayed = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_ids[1],
                expected_run_epoch=running.run_epoch + 100,
            )
            assert replayed.replayed is True
            assert replayed.receipt == winner.receipt
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_recovers_lost_acknowledgements(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_lost_ack"
            stage_id = "model-step:lost-ack"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=stage_id,
                    dispatch_ordinal=0,
                    intent={"logical_step": 1},
                    reservation_ids=["reservation-lost-ack"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1},
                completion_event_id="lost-ack-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text(
                    "assistant",
                    "committed before cancellation",
                ),
                reservation_ids=("reservation-lost-ack",),
            )

            original_complete = store._complete_model_completion_stage_atomic
            completion_committed = asyncio.Event()
            hold_completion_ack = asyncio.Event()

            async def complete_then_hold_ack(prepared):
                result = await original_complete(prepared)
                assert result.replayed is False
                completion_committed.set()
                await hold_completion_ack.wait()
                return result

            with monkeypatch.context() as completion_patch:
                completion_patch.setattr(
                    store,
                    "_complete_model_completion_stage_atomic",
                    complete_then_hold_ack,
                )
                completing = asyncio.create_task(
                    store.complete_model_completion_stage(
                        session_id,
                        stage_id=stage_id,
                        publication=publication,
                    )
                )
                await asyncio.wait_for(completion_committed.wait(), timeout=5)
                completing.cancel("terminal stage acknowledgement lost")
                with pytest.raises(asyncio.CancelledError) as cancellation:
                    await completing
                assert cancellation.value.args == ("terminal stage acknowledgement lost",)

            completed_stage = await store.load_model_completion_stage(session_id, stage_id)
            assert completed_stage is not None
            assert completed_stage.state == "completed"
            store = await _reopen_store(session_store_case, store)
            terminal_replay = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert terminal_replay.replayed is True
            assert terminal_replay.stage == completed_stage

            original_promote = store._promote_model_completion_stage_atomic
            promotion_committed = asyncio.Event()
            hold_promotion_ack = asyncio.Event()

            async def promote_then_hold_ack(**kwargs):
                result = await original_promote(**kwargs)
                assert result.replayed is False
                promotion_committed.set()
                await hold_promotion_ack.wait()
                return result

            with monkeypatch.context() as promotion_patch:
                promotion_patch.setattr(
                    store,
                    "_promote_model_completion_stage_atomic",
                    promote_then_hold_ack,
                )
                promoting = asyncio.create_task(
                    store.promote_model_completion_stage(
                        session_id,
                        stage_id=stage_id,
                        expected_run_epoch=running.run_epoch,
                    )
                )
                await asyncio.wait_for(promotion_committed.wait(), timeout=5)
                promoting.cancel("promotion acknowledgement lost")
                with pytest.raises(asyncio.CancelledError) as cancellation:
                    await promoting
                assert cancellation.value.args == ("promotion acknowledgement lost",)

            committed_receipt = await store.load_runtime_publication_receipt(
                session_id,
                stage_id,
            )
            assert committed_receipt is not None
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)

            store = await _reopen_store(session_store_case, store)

            promotion_replay = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=0,
            )
            assert promotion_replay.replayed is True
            assert promotion_replay.receipt == committed_receipt
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    [
        "transcript",
        "transcript-attribution",
        "event",
        "reference",
        "reference-content",
    ],
)
def test_session_store_conformance_runtime_publication_replay_fails_closed_on_material_drift(
    session_store_case,
    corruption: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_runtime_publication_material_{corruption}"
            publication_id = f"model-step:material-{corruption}"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            referenced_event = Event(
                id=f"material-reference-{corruption}",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                interaction_id="interaction-material",
            )
            await store.append_event(session_id, referenced_event)
            appended_event = Event(
                id=f"material-completion-{corruption}",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                interaction_id="interaction-material",
                payload={"usage": {"input_tokens": 2, "output_tokens": 1}},
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                interaction_id="interaction-material",
                intent={"logical_step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[Message.text("assistant", "authoritative material")],
                events=[appended_event],
                referenced_events=[runtime_publication_event_reference(referenced_event)],
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False

            corrupted_event_id = (
                referenced_event.id
                if corruption in {"reference", "reference-content"}
                else appended_event.id
            )
            await _corrupt_runtime_publication_material(
                session_store_case,
                store,
                session_id=session_id,
                corruption=corruption,
                event_id=corrupted_event_id,
            )
            expected_error = {
                "transcript": "transcript segment conflicts",
                "transcript-attribution": "transcript attribution conflicts",
                "event": "event batch conflicts",
                "reference": "references a missing event",
                "reference-content": "referenced event content conflicts",
            }[corruption]
            with pytest.raises(SessionRuntimePublicationConflict, match=expected_error):
                await store.load_runtime_publication_receipt(session_id, publication_id)
            with pytest.raises(SessionRuntimePublicationConflict, match=expected_error):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.PENDING},
                    expected_run_epoch=0,
                    expected_transcript_cursor=0,
                )
            assert await store.load_checkpoint(session_id) == {"phase": "published"}
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_rejects_before_mutation_and_recovers(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_rollback"
            fixed_at = datetime(2026, 7, 22, 10, 0, tzinfo=UTC)
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "roll back invalid publication")],
                ),
                identity=_identity(),
            )
            await store.publish_checkpoint_and_events(
                session_id,
                checkpoint_transform=lambda _session, _checkpoint: {"phase": "before"},
                events=[],
            )
            before_session = await store.load(session_id)
            before_checkpoint = await store.load_checkpoint(session_id)
            before_transcript = await store.load_transcript(session_id)
            before_events = await store.load_events(session_id)

            malformed_request = RuntimePublicationRequest.model_construct(
                publication_id="malformed-mutation",
                kind="model-step",
                intent={"step": 1},
                mutation=RuntimePublicationMutation.model_construct(
                    operations=(
                        {
                            "key": "non_finite",
                            "expected_value_digest": None,
                            "action": "set",
                            "value": float("nan"),
                        },
                    )
                ),
                transcript_messages=(Message.text("assistant", "must not persist"),),
                events=(
                    Event(
                        id="malformed-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at,
                    ),
                ),
                referenced_events=(),
            )

            with pytest.raises(ValueError, match="mutation is malformed"):
                await store.publish_runtime_publication(
                    session_id,
                    request=malformed_request,
                )
            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == before_checkpoint
            assert await store.load_transcript(session_id) == before_transcript
            assert await store.load_events(session_id) == before_events
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    malformed_request.publication_id,
                )
                is None
            )

            stale_request = RuntimePublicationRequest(
                publication_id="stale-mutation",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "different-source"},
                    {"phase": "must-not-publish"},
                ),
                transcript_messages=[Message.text("assistant", "must not persist")],
                events=[
                    Event(
                        id="stale-mutation-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at,
                    )
                ],
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="checkpoint key changed",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=stale_request,
                )
            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == before_checkpoint
            assert await store.load_transcript(session_id) == before_transcript
            assert await store.load_events(session_id) == before_events

            malformed_boundary_request = RuntimePublicationRequest.model_construct(
                publication_id="malformed-request",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "before"},
                    {"phase": "wrong"},
                ),
                transcript_messages=(Message.text("assistant", "invalid boundary"),),
                events=(
                    Event(
                        id="invalid-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id="different-session",
                    ),
                ),
                referenced_events=(),
            )
            with pytest.raises(ValueError, match="does not match target session"):
                await store.publish_runtime_publication(
                    session_id,
                    request=malformed_boundary_request,
                )
            assert await store.load(session_id) == before_session

            malformed_reference = runtime_publication_event_reference(
                Event(
                    id="malformed-reference",
                    type=EventType.MODEL_STARTED,
                    session_id=session_id,
                )
            ).model_copy(update={"event_digest": "not-a-digest"})
            malformed_reference_request = RuntimePublicationRequest.model_construct(
                publication_id="malformed-reference-request",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "before"},
                    {"phase": "wrong"},
                ),
                transcript_messages=(),
                events=(),
                referenced_events=(malformed_reference,),
            )
            with pytest.raises(ValueError, match="malformed reference"):
                await store.publish_runtime_publication(
                    session_id,
                    request=malformed_reference_request,
                )
            assert await store.load(session_id) == before_session

            recovered_request = RuntimePublicationRequest(
                publication_id="recovered-mutation",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "before"},
                    {"phase": "recovered"},
                ),
                transcript_messages=[Message.text("assistant", "recovered")],
                events=[
                    Event(
                        id="recovered-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at + timedelta(seconds=1),
                    )
                ],
            )
            recovered = await store.publish_runtime_publication(
                session_id,
                request=recovered_request,
                expected_transcript_cursor=0,
            )
            assert recovered.replayed is False
            assert await store.load_checkpoint(session_id) == {"phase": "recovered"}
            assert await store.load_transcript(session_id) == [
                Message.text("assistant", "recovered")
            ]
            assert [event.id for event in await store.load_events(session_id)] == [
                "recovered-event"
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_serializes_concurrent_commits(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_concurrent"
            fixed_at = datetime(2026, 7, 22, 11, 0, tzinfo=UTC)
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "publish once concurrently")],
                ),
                identity=_identity(),
            )
            request = RuntimePublicationRequest(
                publication_id="shared-model-step",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"winner": "shared"},
                ),
                transcript_messages=[Message.text("assistant", "only once")],
                events=[
                    Event(
                        id="shared-model-completed",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at,
                    )
                ],
            )
            results = await asyncio.gather(
                *[
                    store.publish_runtime_publication(
                        session_id,
                        request=request,
                        expected_statuses={SessionStatus.PENDING},
                        expected_run_epoch=0,
                        expected_transcript_cursor=0,
                    )
                    for _ in range(8)
                ]
            )
            assert sum(not result.replayed for result in results) == 1
            assert sum(result.replayed for result in results) == 7
            assert len({result.receipt.publication_digest for result in results}) == 1
            assert await store.load_transcript(session_id) == [
                Message.text("assistant", "only once")
            ]
            assert [event.id for event in await store.load_events(session_id)] == [
                "shared-model-completed"
            ]

            fenced_session_id = "sess_runtime_publication_cursor_fence"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=fenced_session_id,
                    messages=[Message.text("user", "serialize different identities")],
                ),
                identity=_identity(),
            )
            assert await store.load_checkpoint(session_id) == {"winner": "shared"}
            contender_results = await asyncio.gather(
                *[
                    store.publish_runtime_publication(
                        fenced_session_id,
                        request=RuntimePublicationRequest(
                            publication_id=f"contender:{contender}",
                            kind="model-step",
                            intent={"round": contender},
                            mutation=runtime_publication_checkpoint_mutation(
                                None,
                                {"winner": contender},
                            ),
                            transcript_messages=[Message.text("assistant", contender)],
                            events=[
                                Event(
                                    id=f"event-{contender}",
                                    type=EventType.MODEL_COMPLETED,
                                    session_id=fenced_session_id,
                                    timestamp=fixed_at + timedelta(seconds=1),
                                )
                            ],
                        ),
                        expected_transcript_cursor=0,
                    )
                    for contender in ("a", "b")
                ],
                return_exceptions=True,
            )
            successes = [
                result for result in contender_results if not isinstance(result, BaseException)
            ]
            failures = [result for result in contender_results if isinstance(result, BaseException)]
            assert len(successes) == 1
            assert len(failures) == 1
            assert isinstance(failures[0], ValueError)
            assert "transcript cursor is stale" in str(failures[0])
            assert len(await store.load_transcript(fenced_session_id)) == 1
            assert len(await store.load_events(fenced_session_id)) == 1
            winning_contender = successes[0].receipt.intent["round"]
            assert await store.load_checkpoint(fenced_session_id) == {"winner": winning_contender}
            receipts = [
                await store.load_runtime_publication_receipt(
                    fenced_session_id,
                    f"contender:{contender}",
                )
                for contender in ("a", "b")
            ]
            assert sum(receipt is not None for receipt in receipts) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_references_and_namespace(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_namespace"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "enforce receipt namespace")],
                ),
                identity=_identity(),
            )
            missing_reference_request = RuntimePublicationRequest(
                publication_id="missing-reference",
                kind="model-step",
                intent={"round": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[Message.text("assistant", "not published")],
                events=[],
                referenced_events=[
                    runtime_publication_event_reference(
                        Event(
                            id="not-durable",
                            type=EventType.MODEL_STARTED,
                            session_id=session_id,
                        )
                    )
                ],
            )
            with pytest.raises(ValueError, match="not durable"):
                await store.publish_runtime_publication(
                    session_id,
                    request=missing_reference_request,
                )
            assert await store.load_transcript(session_id) == []
            assert await store.load_checkpoint(session_id) is None

            overlap_event = Event(
                id="overlap-event",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
            )
            with pytest.raises(ValueError, match="cannot overlap"):
                RuntimePublicationRequest(
                    publication_id="overlap-reference",
                    kind="model-step",
                    intent={"round": 1},
                    mutation=runtime_publication_checkpoint_mutation(
                        None,
                        {"phase": "published"},
                    ),
                    transcript_messages=[],
                    events=[overlap_event],
                    referenced_events=[runtime_publication_event_reference(overlap_event)],
                )
            assert await store.load_events(session_id) == []

            reserved_key = _RUNTIME_PUBLICATION_PREFIX + "caller-owned"
            with pytest.raises(ValueError, match="reserved runtime publication namespace"):
                await store.load_session_operation(session_id, reserved_key)
            with pytest.raises(ValueError, match="reserved runtime publication namespace"):
                await store.publish_session_operation(
                    session_id,
                    idempotency_key=reserved_key,
                    operation_transform=lambda _session, _checkpoint, _record: (
                        SessionOperationPublication(checkpoint={})
                    ),
                    events=[],
                )

            bypassed_publication = SessionOperationPublication.model_construct(
                checkpoint={"phase": "legacy"},
                operation_records={reserved_key: {"status": "completed"}},
            )
            with pytest.raises(ValueError, match="reserved runtime publication namespace"):
                await store.publish_session_operation(
                    session_id,
                    idempotency_key="ordinary-operation",
                    operation_transform=lambda _session, _checkpoint, _record: bypassed_publication,
                    events=[],
                )
            assert await store.load_checkpoint(session_id) is None
            assert await store.load_session_operation(session_id, "ordinary-operation") is None

            durable_reference = Event(
                id="durable-reference",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
            )
            await store.append_event(session_id, durable_reference)
            duplicate_event_request = RuntimePublicationRequest(
                publication_id="duplicate-appended-event",
                kind="model-step",
                intent={"round": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[],
                events=[durable_reference],
            )
            with pytest.raises(ValueError, match="Event already exists"):
                await store.publish_runtime_publication(
                    session_id,
                    request=duplicate_event_request,
                )
            assert await store.load_checkpoint(session_id) is None

            publication_id = "referenced-model-step"
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"round": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[Message.text("assistant", "referenced")],
                events=[],
                referenced_events=[runtime_publication_event_reference(durable_reference)],
            )
            result = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )
            assert result.receipt.referenced_events == (
                runtime_publication_event_reference(durable_reference),
            )
            assert result.receipt.appended_event_ids == ()
            assert await store.load_checkpoint(session_id) == {"phase": "published"}
            with pytest.raises(ValueError, match="reserved runtime publication namespace"):
                await store.load_session_operation(
                    session_id,
                    _runtime_publication_key(publication_id),
                )
            assert (
                await store.load_runtime_publication_receipt(session_id, publication_id)
                == result.receipt
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_fails_closed_on_receipt_drift(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_receipt_drift"
            publication_id = "receipt-drift"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "verify receipt")],
                ),
                identity=_identity(),
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[Message.text("assistant", "published")],
                events=[
                    Event(
                        id="receipt-drift-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
            )
            before_session = await store.load(session_id)
            before_checkpoint = await store.load_checkpoint(session_id)
            before_transcript = await store.load_transcript(session_id)
            before_events = await store.load_events(session_id)
            drifted_record = published.receipt.model_dump(mode="json")
            drifted_record["record_type"] = "caller-owned-record"
            await _replace_runtime_publication_record(
                session_store_case,
                store,
                session_id=session_id,
                publication_id=publication_id,
                record=drifted_record,
            )

            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="malformed or conflicts",
            ):
                await store.load_runtime_publication_receipt(session_id, publication_id)
            await _replace_runtime_publication_record(
                session_store_case,
                store,
                session_id=session_id,
                publication_id=publication_id,
                record="nested-json-is-not-a-receipt",
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="malformed or conflicts",
            ):
                await store.load_runtime_publication_receipt(session_id, publication_id)

            drifted_record = published.receipt.model_dump(mode="json")
            drifted_record["checkpoint_digest"] = "0" * 64
            await _replace_runtime_publication_record(
                session_store_case,
                store,
                session_id=session_id,
                publication_id=publication_id,
                record=drifted_record,
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="malformed or conflicts",
            ):
                await store.load_runtime_publication_receipt(session_id, publication_id)
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="malformed or conflicts",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                )
            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == before_checkpoint
            assert await store.load_transcript(session_id) == before_transcript
            assert await store.load_events(session_id) == before_events
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_checkpoint_patch_preserves_unrelated_keys(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_unrelated"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            source_checkpoint = {
                "owned": {"version": 1},
                "unrelated": {"version": 1},
            }
            target_checkpoint = {
                "owned": {"version": 2},
                "unrelated": {"version": 1},
            }
            await store.checkpoint(session_id, source_checkpoint)
            request = RuntimePublicationRequest(
                publication_id="checkpoint-unrelated",
                kind="model-step",
                intent={"round": "checkpoint-unrelated"},
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    target_checkpoint,
                ),
                transcript_messages=[Message.text("assistant", "publish owned state")],
                events=[
                    Event(
                        id="checkpoint-unrelated-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            assert tuple(operation.key for operation in request.mutation.operations) == ("owned",)

            await store.transform_checkpoint(
                session_id,
                lambda _session, checkpoint: {
                    **(checkpoint or {}),
                    "unrelated": {"version": 2},
                    "late_runtime_state": {"preserved": True},
                },
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {
                "owned": {"version": 2},
                "unrelated": {"version": 2},
                "late_runtime_state": {"preserved": True},
            }
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == list(request.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_checkpoint_patch_rejects_touched_drift(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_touched_drift"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            source_checkpoint = {
                "owned": {"version": 1},
                "unrelated": {"stable": True},
            }
            await store.checkpoint(session_id, source_checkpoint)
            request = RuntimePublicationRequest(
                publication_id="checkpoint-touched-drift",
                kind="model-step",
                intent={"round": "checkpoint-touched-drift"},
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {
                        "owned": {"version": 2},
                        "unrelated": {"stable": True},
                    },
                ),
                transcript_messages=[Message.text("assistant", "must not publish")],
                events=[
                    Event(
                        id="checkpoint-touched-drift-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            drifted_checkpoint = {
                "owned": {"version": 99},
                "unrelated": {"stable": True},
                "late_runtime_state": True,
            }
            await store.checkpoint(session_id, drifted_checkpoint)
            before_session = await store.load(session_id)

            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="checkpoint key changed",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_transcript_cursor=0,
                )

            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == drifted_checkpoint
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_checkpoint_patch_distinguishes_null(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_null"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            await store.checkpoint(session_id, {"nullable": None})

            absent_mutation = runtime_publication_checkpoint_mutation(
                {},
                {"nullable": "must-not-publish"},
            )
            assert absent_mutation.operations[0].expected_value_digest is None
            absent_request = RuntimePublicationRequest(
                publication_id="checkpoint-null-expected-absent",
                kind="model-step",
                intent={"round": "null-expected-absent"},
                mutation=absent_mutation,
                transcript_messages=[Message.text("assistant", "must not publish")],
                events=[
                    Event(
                        id="checkpoint-null-expected-absent-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="checkpoint key changed",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=absent_request,
                    expected_transcript_cursor=0,
                )
            assert await store.load_checkpoint(session_id) == {"nullable": None}
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []

            present_null_mutation = runtime_publication_checkpoint_mutation(
                {"nullable": None},
                {"nullable": "updated"},
            )
            assert present_null_mutation.operations[0].expected_value_digest is not None
            present_null_request = RuntimePublicationRequest(
                publication_id="checkpoint-null-expected-present",
                kind="model-step",
                intent={"round": "null-expected-present"},
                mutation=present_null_mutation,
                transcript_messages=[Message.text("assistant", "published")],
                events=[
                    Event(
                        id="checkpoint-null-expected-present-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=present_null_request,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {"nullable": "updated"}
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_checkpoint_operations_normalize_and_bind(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_order"
            publication_id = "checkpoint-operation-order"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            first = RuntimePublicationCheckpointOperation(
                key="first",
                expected_value_digest=None,
                action="set",
                value={"version": 1},
            )
            second = RuntimePublicationCheckpointOperation(
                key="second",
                expected_value_digest=None,
                action="set",
                value={"version": 2},
            )
            reverse_order = RuntimePublicationMutation(operations=(second, first))
            forward_order = RuntimePublicationMutation(operations=(first, second))
            assert reverse_order == forward_order
            assert tuple(operation.key for operation in reverse_order.operations) == (
                "first",
                "second",
            )
            event = Event(
                id="checkpoint-operation-order-event",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"round": "checkpoint-operation-order"},
                mutation=reverse_order,
                transcript_messages=[Message.text("assistant", "ordered mutation")],
                events=[event],
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {
                "first": {"version": 1},
                "second": {"version": 2},
            }

            equivalent_request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind=request.kind,
                intent=request.intent,
                mutation=forward_order,
                transcript_messages=request.transcript_messages,
                events=request.events,
            )
            replayed = await store.publish_runtime_publication(
                session_id,
                request=equivalent_request,
                expected_transcript_cursor=99,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt

            conflicting_request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind=request.kind,
                intent=request.intent,
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {
                        "first": {"version": 1},
                        "second": {"version": 3},
                    },
                ),
                transcript_messages=request.transcript_messages,
                events=request.events,
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="different request",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=conflicting_request,
                )
            assert await store.load_checkpoint(session_id) == {
                "first": {"version": 1},
                "second": {"version": 2},
            }
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == [event]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_empty_checkpoint_mutation_preserves_none(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_empty"
            publication_id = "checkpoint-empty-mutation"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"logical_step": "checkpoint-empty-mutation"},
                mutation=runtime_publication_checkpoint_mutation(None, None),
                transcript_messages=[Message.text("assistant", "no checkpoint state")],
                events=[
                    Event(
                        id="checkpoint-empty-mutation-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            assert request.mutation.operations == ()
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) is None

            store = await _reopen_store(session_store_case, store)
            assert await store.load_checkpoint(session_id) is None
            replayed = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=99,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt
            assert await store.load_checkpoint(session_id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_persisted_event_side_effect_recovery(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_side_effect_recovery",
                    messages=[Message.text("user", "persist")],
                ),
                identity=_identity(),
            )
            event = Event(type=EventType.MODEL_COMPLETED, session_id=session.id)
            await store.append_event(session.id, event)
            pending = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert pending is not None
            assert pending.status is PersistedEventSideEffectStatus.PENDING
            assert (
                await store.get_persisted_event_side_effect_delivery(
                    session_id=session.id,
                    event_id="missing-event",
                )
                is None
            )
            await store.append_event(
                session.id,
                Event(
                    type=EventType.RUNTIME_SINK_FAILED,
                    session_id=session.id,
                    payload={"event_id": event.id},
                ),
            )

            store = await _reopen_store(session_store_case, store)
            first_claim = await store.claim_persisted_event_side_effect()
            assert first_claim is not None
            assert first_claim.event.id == event.id
            assert first_claim.attempt == 1
            leased = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert leased is not None
            assert leased.status is PersistedEventSideEffectStatus.LEASED
            failed = await store.mark_persisted_event_side_effect_failed(
                first_claim,
                error="sink unavailable",
                max_attempts=2,
                retry_delay_seconds=0,
            )
            assert failed.status is PersistedEventSideEffectStatus.FAILED
            loaded_failed = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert loaded_failed == failed

            store = await _reopen_store(session_store_case, store)
            second_claim = await store.claim_persisted_event_side_effect()
            assert second_claim is not None
            assert second_claim.event.id == event.id
            assert second_claim.attempt == 2
            delivered = await store.mark_persisted_event_side_effect_delivered(second_claim)
            assert delivered.status is PersistedEventSideEffectStatus.DELIVERED
            loaded_delivered = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert loaded_delivered == delivered
            assert await store.claim_persisted_event_side_effect() is None
            states = await store.list_persisted_event_side_effect_deliveries()
            assert [(state.event_id, state.status, state.attempts) for state in states] == [
                (event.id, PersistedEventSideEffectStatus.DELIVERED, 2)
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_nonportable_side_effect_errors_atomically(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_side_effect_error_portability",
                    messages=[Message.text("user", "persist")],
                ),
                identity=_identity(),
            )
            event = Event(type=EventType.SESSION_STARTED, session_id=session.id)
            await store.append_event(session.id, event)
            claim = await store.claim_persisted_event_side_effect(
                session_id=session.id,
                event_id=event.id,
            )
            assert claim is not None
            leased = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert leased is not None

            invalid_errors = (
                "sink\x00secret",
                "sink \ud800 secret",
                "x" * (PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES + 1),
            )
            for error in invalid_errors:
                with pytest.raises(ValueError):
                    await store.mark_persisted_event_side_effect_failed(
                        claim,
                        error=error,
                        max_attempts=3,
                        retry_delay_seconds=0,
                    )
                assert (
                    await store.get_persisted_event_side_effect_delivery(
                        session_id=session.id,
                        event_id=event.id,
                    )
                    == leased
                )

            failed = await store.mark_persisted_event_side_effect_failed(
                claim,
                error="portable failure",
                max_attempts=3,
                retry_delay_seconds=0,
            )
            assert failed.status is PersistedEventSideEffectStatus.FAILED
            assert failed.last_error == "portable failure"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_legacy_unbounded_side_effect_error_remains_readable() -> None:
    delivery = PersistedEventSideEffectDelivery(
        session_id="sess_legacy_side_effect_error",
        event_id="event_legacy_side_effect_error",
        event_sequence=1,
        status=PersistedEventSideEffectStatus.FAILED,
        last_error="é" * (PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES + 1),
    )

    assert delivery.last_error is not None
    assert len(delivery.last_error.encode("utf-8")) <= (PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES)
    assert delivery.last_error.endswith("... [truncated]")


def test_session_store_conformance_persisted_event_side_effect_claim_fencing(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_side_effect_fencing",
                    messages=[Message.text("user", "persist")],
                ),
                identity=_identity(),
            )
            event = Event(type=EventType.SESSION_STARTED, session_id=session.id)
            await store.append_event(session.id, event)

            stale = await store.claim_persisted_event_side_effect(lease_seconds=0.05)
            assert stale is not None
            pending = Event(type="custom.pending", session_id=session.id)
            await store.append_event(session.id, pending)
            claimable = await store.list_persisted_event_side_effect_deliveries(
                claimable_only=True,
                limit=1,
            )
            assert [delivery.event_id for delivery in claimable] == [pending.id]
            await asyncio.sleep(0.06)
            replacement = await store.claim_persisted_event_side_effect()
            assert replacement is not None
            assert replacement.event.id == event.id
            assert replacement.attempt == 2
            with pytest.raises(PersistedEventSideEffectClaimLost, match="no longer active"):
                await store.mark_persisted_event_side_effect_delivered(stale)
            dead_lettered = await store.mark_persisted_event_side_effect_failed(
                replacement,
                error="still unavailable",
                max_attempts=2,
                retry_delay_seconds=0,
            )
            assert dead_lettered.status is PersistedEventSideEffectStatus.DEAD_LETTERED
            loaded_dead_lettered = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert loaded_dead_lettered == dead_lettered
            pending_claim = await store.claim_persisted_event_side_effect()
            assert pending_claim is not None
            assert pending_claim.event.id == pending.id
            await store.mark_persisted_event_side_effect_delivered(pending_claim)
            assert await store.claim_persisted_event_side_effect() is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_persisted_event_side_effect_retry_spacing_and_paging(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_side_effect_retry_spacing",
                    messages=[Message.text("user", "persist")],
                ),
                identity=_identity(),
            )
            events = [
                Event(type=f"custom.page.{index}", session_id=session.id) for index in range(3)
            ]
            await store.append_events(session.id, events)

            async def exercise_retry_clock():
                claim = await store.claim_persisted_event_side_effect(
                    session_id=session.id,
                    event_id=events[0].id,
                )
                assert claim is not None
                failed = await store.mark_persisted_event_side_effect_failed(
                    claim,
                    error="try later",
                    max_attempts=3,
                    retry_delay_seconds=60,
                )
                assert failed.next_attempt_at is not None
                assert failed.next_attempt_at > failed.updated_at
                assert (
                    await store.claim_persisted_event_side_effect(
                        session_id=session.id,
                        event_id=events[0].id,
                    )
                    is None
                )

            if session_store_case[0] == "postgres":

                class NodeClockMustNotBeRead:
                    @classmethod
                    def now(cls, *args, **kwargs):
                        raise AssertionError("Postgres handoff eligibility must use DB time")

                with monkeypatch.context() as context:
                    context.setattr("cayu.storage.postgres.datetime", NodeClockMustNotBeRead)
                    await exercise_retry_clock()
            else:
                await exercise_retry_clock()

            claimable = await store.list_persisted_event_side_effect_deliveries(
                claimable_only=True,
            )
            assert [state.event_id for state in claimable] == [events[1].id, events[2].id]

            first_page = await store.list_persisted_event_side_effect_deliveries(limit=2)
            second_page = await store.list_persisted_event_side_effect_deliveries(
                after_sequence=first_page[-1].event_sequence,
                limit=2,
            )
            assert [state.event_id for state in [*first_page, *second_page]] == [
                event.id for event in events
            ]
            assert second_page[0].event_sequence > first_page[-1].event_sequence
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fences_reclaimed_compaction_attempts(
    session_store_case,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store = await _open_store(session_store_case)
        try:
            compactor = _ConformanceOverlappingCompactor()

            def configured_app(*, now: datetime) -> CayuApp:
                app = CayuApp(
                    session_store=store,
                    request_footprint=RequestFootprintConfig(enabled=False),
                    enable_logging=False,
                    clock=lambda: now,
                )
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    context_policy=CheckpointCompactionContextPolicy(
                        compactor=compactor,
                        max_user_turns=1,
                    ),
                )
                return app

            first_app = configured_app(now=accepted_at)
            recovered_app = configured_app(now=accepted_at + timedelta(minutes=6))
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_claim_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
                Message.text("assistant", "current answer"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            first_request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-claim-conformance",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
                requested_by=ResolutionActor(subject="operator-a"),
            )
            recovered_request = first_request.model_copy(
                update={"requested_by": ResolutionActor(subject="operator-b")}
            )

            async def collect(app: CayuApp, request: CompactSessionRequest) -> list[Event]:
                return [event async for event in app.compact_session(request)]

            first_task = asyncio.create_task(collect(first_app, first_request))
            await compactor.started[0].wait()
            recovered_task = asyncio.create_task(collect(recovered_app, recovered_request))
            await compactor.started[1].wait()
            compactor.release[1].set()
            recovered_events = await recovered_task
            compactor.release[0].set()
            with pytest.raises(RuntimeError, match="superseded"):
                await first_task

            store = await _reopen_store(session_store_case, store)
            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            durable_records = await store.query_events(EventQuery(session_id=created.id, limit=100))
            durable_events = [record.event for record in durable_records]
            replay = [event async for event in replay_app.compact_session(recovered_request)]

            assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED
            assert [public_event_sequence(event.id) for event in replay] == [
                record.sequence for record in durable_records
            ]
            assert (
                sum(
                    event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in durable_events
                )
                == 1
            )
            assert (
                sum(event.type == EventType.SESSION_CHECKPOINTED for event in durable_events) == 1
            )
            assert sum(event.type == EventType.MODEL_COMPLETED for event in durable_events) == 2
            operation_events = [
                event
                for event in durable_events
                if event.type != EventType.REQUEST_FOOTPRINT_RECORDED
            ]
            assert len({event.payload["operation_id"] for event in operation_events}) == 1
            assert len({event.payload["attempt_id"] for event in operation_events}) == 2
            delivery_ids = {
                delivery.event_id
                for delivery in await store.list_persisted_event_side_effect_deliveries(limit=1000)
            }
            assert {event.id for event in durable_events} <= delivery_ids
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["summary"] == "summary from attempt 2"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_heartbeats_active_compaction_claim(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = await _open_store(session_store_case)
        compactor = _ConformanceBlockingCompactor()
        task: asyncio.Task[list[Event]] | None = None
        try:
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                clock=lambda: now["value"],
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_heartbeat_conformance",
                    messages=[],
                ),
                identity=_identity(),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-heartbeat-conformance",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            async def collect() -> list[Event]:
                return [event async for event in app.compact_session(request)]

            task = asyncio.create_task(collect())
            await asyncio.wait_for(compactor.started.wait(), timeout=5)
            now["value"] = accepted_at + timedelta(minutes=4)
            first_renewal_expiry = accepted_at + timedelta(minutes=9)
            async with asyncio.timeout(5):
                while True:
                    checkpoint = await store.load_checkpoint(created.id)
                    assert checkpoint is not None
                    record = checkpoint["session_operations"]["records"][request.idempotency_key]
                    if datetime.fromisoformat(record["claim_expires_at"]) >= first_renewal_expiry:
                        break
                    await asyncio.sleep(0)

            now["value"] = accepted_at + timedelta(minutes=6)
            expected_expiry = accepted_at + timedelta(minutes=11)
            async with asyncio.timeout(5):
                while True:
                    checkpoint = await store.load_checkpoint(created.id)
                    assert checkpoint is not None
                    record = checkpoint["session_operations"]["records"][request.idempotency_key]
                    if datetime.fromisoformat(record["claim_expires_at"]) >= expected_expiry:
                        break
                    await asyncio.sleep(0)

            with pytest.raises(RuntimeError, match="already running"):
                async for _event in app.compact_session(request):
                    pass
            assert compactor.calls == 1

            compactor.release.set()
            events = await task
            task = None
            assert events[-1].type == EventType.SESSION_CHECKPOINTED
            assert compactor.calls == 1

            store = await _reopen_store(session_store_case, store)
            operation = await store.load_session_operation(created.id, request.idempotency_key)
            assert operation is not None
            assert operation["status"] == "completed"
        finally:
            compactor.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_operation_commit_guard_is_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_operation_commit_guard_conformance",
                    messages=[],
                ),
                identity=_identity(),
            )
            event = Event(
                type=EventType.CONTEXT_COMPACTION_STARTED,
                session_id=created.id,
                agent_name="assistant",
                payload={"operation_id": "guarded-operation"},
            )

            def transform(_session, checkpoint, _persisted_record):
                updated = {} if checkpoint is None else dict(checkpoint)
                updated["guarded_operation"] = True
                return SessionOperationPublication(
                    checkpoint=updated,
                    operation_records={
                        "guarded-request": {
                            "operation_id": "guarded-operation",
                            "status": "completed",
                        }
                    },
                )

            def reject_commit() -> None:
                raise RuntimeError("operation commit guard rejected publication")

            with pytest.raises(RuntimeError, match="commit guard rejected"):
                await store.publish_session_operation_guarded(
                    created.id,
                    idempotency_key="guarded-request",
                    operation_transform=transform,
                    commit_guard=reject_commit,
                    events=[event],
                )

            assert await store.load_checkpoint(created.id) is None
            assert await store.load_session_operation(created.id, "guarded-request") is None
            assert await store.load_events(created.id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_provider_progress_event_and_cursor_are_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_provider_progress_conformance"
        interaction_id = "interaction-provider-progress-conformance"
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="provider-progress-conformance-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[],
            )
            await store.checkpoint(created.id, {})
            identity = ModelAttemptIdentity(
                model_step_id="mstep_" + "a" * 32,
                model_attempt_id="matt_" + "b" * 32,
            )
            stage_result = await store.prepare_model_completion_stage(
                created.id,
                request=ModelCompletionStageRequest(
                    stage_id=f"{identity.model_step_id}:dispatch:0",
                    logical_step_id=identity.model_step_id,
                    dispatch_ordinal=0,
                    intent={
                        **identity.payload(),
                        "provider_name": "conformance-provider",
                        "requested_model": "conformance-model",
                    },
                ),
                expected_statuses={created.status},
                expected_run_epoch=created.run_epoch,
                expected_transcript_cursor=0,
            )
            state = ProviderOperationState(
                operation_id="provider-operation-conformance",
                stream_protocol="conformance-v1",
                recovery_metadata={"cursor": 0, "opaque": {"page": "start"}},
            )
            await store.append_events(
                created.id,
                [
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            "provider": "conformance-provider",
                            "model": "conformance-model",
                            "step": 1,
                            "attempt": 1,
                            "max_attempts": 1,
                            **identity.payload(),
                        },
                    ),
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            "provider": "conformance-provider",
                            "model": "conformance-model",
                            "step": 1,
                            "attempt": 1,
                            "max_attempts": 1,
                            **identity.payload(),
                            "source_run_epoch": created.run_epoch,
                            "start_id": f"provider-operation:{identity.model_attempt_id}",
                            "state_version": state.version,
                            "operation_id": state.operation_id,
                            "stream_protocol": state.stream_protocol,
                            "status": ProviderOperationStatus.IN_PROGRESS.value,
                            "recovery_metadata": state.recovery_metadata.model_dump(mode="json"),
                        },
                    ),
                ],
            )

            async def commit(delta: str, cursor: int, current: ProviderOperationState):
                stream_event = ModelStreamEvent.text_delta(
                    delta,
                    recovery_metadata={
                        "cursor": cursor,
                        "opaque": {"page": f"after-{cursor}"},
                    },
                )
                event = Event(
                    id=provider_operation_progress_event_id(
                        stage_result.stage.stage_id,
                        cursor,
                    ),
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id=created.id,
                    interaction_id=interaction_id,
                    agent_name="assistant",
                    payload={
                        "delta": delta,
                        "step": 1,
                        "attempt": 1,
                        "max_attempts": 1,
                        **identity.payload(),
                        "provider_operation_progress": provider_operation_progress_payload(
                            current,
                            stream_event,
                        ),
                    },
                )
                return await commit_provider_operation_progress(
                    store,
                    stage=stage_result.stage,
                    model_attempt_identity=identity,
                    current_state=current,
                    stream_event=stream_event,
                    event=event,
                    expected_run_epoch=created.run_epoch,
                )

            first = await commit("hel", 1, state)
            replay = await commit("hel", 1, state)
            second = await commit("lo", 2, first.state)

            assert first.replayed is False
            assert replay.replayed is True
            assert second.replayed is False
            assert second.state.recovery_metadata.cursor == 2
            text_events = [
                event
                for event in await store.load_events(created.id)
                if event.type == EventType.MODEL_TEXT_DELTA
            ]
            assert [event.payload["delta"] for event in text_events] == ["hel", "lo"]
            recovered = await load_recoverable_provider_operation(
                store,
                stage_result.stage,
            )
            assert recovered is not None
            assert recovered.state.recovery_metadata.cursor == 2
            assert recovered.state.recovery_metadata.opaque == {"page": "after-2"}
            assert [event.delta for event in recovered.accepted_stream_events] == [
                "hel",
                "lo",
            ]
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_provider_resolution_after_restart(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_provider_resolution_conformance"
        interaction_id = "interaction-provider-resolution-conformance"
        try:
            assert store._supports_atomic_model_completion_stage_release_protocol()
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="provider-resolution-conformance-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[],
            )
            await store.checkpoint(created.id, {})
            identity = ModelAttemptIdentity(
                model_step_id="mstep_" + "c" * 32,
                model_attempt_id="matt_" + "d" * 32,
            )
            stage = (
                await store.prepare_model_completion_stage(
                    created.id,
                    request=ModelCompletionStageRequest(
                        stage_id=f"{identity.model_step_id}:dispatch:0",
                        logical_step_id=identity.model_step_id,
                        dispatch_ordinal=0,
                        intent={
                            **identity.payload(),
                            "provider_name": "conformance-provider",
                            "requested_model": "conformance-model",
                            "recovery_context": {
                                "execution_profile_fingerprint": "e" * 64,
                            },
                        },
                    ),
                    expected_statuses={created.status},
                    expected_run_epoch=created.run_epoch,
                    expected_transcript_cursor=0,
                )
            ).stage
            state = ProviderOperationState(
                operation_id="provider-resolution-operation",
                stream_protocol="conformance-v1",
            )
            common_payload = {
                "provider": "conformance-provider",
                "model": "conformance-model",
                "step": 1,
                "attempt": 1,
                "max_attempts": 1,
                **identity.payload(),
            }
            await store.append_events(
                created.id,
                [
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload=common_payload,
                    ),
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTING,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            **common_payload,
                            "source_run_epoch": created.run_epoch,
                            "start_id": f"provider-operation:{identity.model_attempt_id}",
                            "start_idempotency_support": "unsupported",
                        },
                    ),
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            **common_payload,
                            "source_run_epoch": created.run_epoch,
                            "start_id": f"provider-operation:{identity.model_attempt_id}",
                            "state_version": state.version,
                            "operation_id": state.operation_id,
                            "stream_protocol": state.stream_protocol,
                            "status": ProviderOperationStatus.UNAVAILABLE.value,
                            "recovery_metadata": {},
                        },
                    ),
                    event_with_runtime_payload_authority(
                        Event(
                            type=EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                            session_id=created.id,
                            interaction_id=interaction_id,
                            agent_name="assistant",
                            payload={
                                **common_payload,
                                "source_run_epoch": created.run_epoch,
                                "run_epoch": created.run_epoch,
                                "operation_id": state.operation_id,
                                "stream_protocol": state.stream_protocol,
                                "status": "unavailable",
                                "recovery_reason": "unavailable",
                            },
                        ),
                        "model_attempt_id",
                        "model_step_id",
                        "operation_id",
                        "stream_protocol",
                    ),
                ],
            )
            interrupted = await store.transition_status(
                created.id,
                from_statuses={created.status},
                to_status=SessionStatus.INTERRUPTED,
            )
            request = ProviderOperationResolutionRequest(
                session_id=created.id,
                stage_id=stage.stage_id,
                expected_run_epoch=interrupted.run_epoch,
                action=ProviderOperationResolutionAction.FAIL,
                reason="conformance operator failure",
            )
            resolved = await resolve_provider_operation_stage(
                store,
                request,
                redactor=SecretRedactor(),
            )
            assert resolved.replayed is False
            assert await store.load_active_model_completion_stage(created.id) is None
            pending = await load_pending_provider_operation_disposition(store, created.id)
            assert pending is not None
            assert pending[0].resolution_id == resolved.record.resolution_id

            store = await _reopen_store(session_store_case, store)
            reconstructed = await load_provider_operation_resolution(
                store,
                created.id,
                stage.stage_id,
            )
            assert reconstructed is not None
            assert reconstructed.replayed is True
            assert reconstructed.record.action is ProviderOperationResolutionAction.FAIL
            reconstructed_pending = await load_pending_provider_operation_disposition(
                store,
                created.id,
            )
            assert reconstructed_pending is not None
            assert reconstructed_pending[0] == pending[0]
            inspection = await inspect_provider_operation(store, created.id)
            assert (
                inspection.status
                is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
            )
            assert inspection.resolution_action is ProviderOperationResolutionAction.FAIL
            replay = await resolve_provider_operation_stage(
                store,
                request,
                redactor=SecretRedactor(),
            )
            assert replay.replayed is True
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_provider_resolution_loses_to_terminal_completion(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_provider_resolution_terminal_race"
        interaction_id = "interaction-provider-resolution-terminal-race"
        release_resolution = asyncio.Event()
        resolution_entered = asyncio.Event()
        resolution_task: asyncio.Task | None = None
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="provider-resolution-terminal-race-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[],
            )
            await store.checkpoint(created.id, {})
            identity = ModelAttemptIdentity(
                model_step_id="mstep_" + "4" * 32,
                model_attempt_id="matt_" + "5" * 32,
            )
            intent = {
                **identity.payload(),
                "provider_name": "conformance-provider",
                "requested_model": "conformance-model",
                "recovery_context": {
                    "execution_profile_fingerprint": "6" * 64,
                },
            }
            stage = (
                await store.prepare_model_completion_stage(
                    created.id,
                    request=ModelCompletionStageRequest(
                        stage_id=f"{identity.model_step_id}:dispatch:0",
                        logical_step_id=identity.model_step_id,
                        dispatch_ordinal=0,
                        intent=intent,
                    ),
                    expected_statuses={created.status},
                    expected_run_epoch=created.run_epoch,
                    expected_transcript_cursor=0,
                )
            ).stage
            common_payload = {
                "provider": "conformance-provider",
                "model": "conformance-model",
                "step": 1,
                "attempt": 1,
                "max_attempts": 1,
                **identity.payload(),
            }
            await store.append_events(
                created.id,
                [
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload=common_payload,
                    ),
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            **common_payload,
                            "source_run_epoch": created.run_epoch,
                            "start_id": f"provider-operation:{identity.model_attempt_id}",
                            "state_version": 1,
                            "operation_id": "provider-resolution-terminal-race-operation",
                            "stream_protocol": "conformance-v1",
                            "status": ProviderOperationStatus.UNAVAILABLE.value,
                            "recovery_metadata": {},
                        },
                    ),
                    event_with_runtime_payload_authority(
                        Event(
                            type=EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                            session_id=created.id,
                            interaction_id=interaction_id,
                            agent_name="assistant",
                            payload={
                                **common_payload,
                                "source_run_epoch": created.run_epoch,
                                "run_epoch": created.run_epoch,
                                "operation_id": "provider-resolution-terminal-race-operation",
                                "stream_protocol": "conformance-v1",
                                "status": "unavailable",
                                "recovery_reason": "unavailable",
                            },
                        ),
                        "model_attempt_id",
                        "model_step_id",
                        "operation_id",
                        "stream_protocol",
                    ),
                ],
            )
            interrupted = await store.transition_status(
                created.id,
                from_statuses={created.status},
                to_status=SessionStatus.INTERRUPTED,
            )
            request = ProviderOperationResolutionRequest(
                session_id=created.id,
                stage_id=stage.stage_id,
                expected_run_epoch=interrupted.run_epoch,
                action=ProviderOperationResolutionAction.FAIL,
                reason="terminal completion must win the resolution race",
            )
            original_publish_session_operation = store.publish_session_operation

            async def publish_after_terminal_completion(*args, **kwargs):
                resolution_entered.set()
                await release_resolution.wait()
                return await original_publish_session_operation(*args, **kwargs)

            monkeypatch.setattr(
                store,
                "publish_session_operation",
                publish_after_terminal_completion,
            )
            resolution_task = asyncio.create_task(
                resolve_provider_operation_stage(
                    store,
                    request,
                    redactor=SecretRedactor(),
                )
            )
            await asyncio.wait_for(resolution_entered.wait(), timeout=5)

            completion = await store.complete_model_completion_stage(
                created.id,
                stage_id=stage.stage_id,
                publication=_assistant_model_completion_publication(
                    session_id=created.id,
                    stage_id=stage.stage_id,
                    logical_step_id=stage.logical_step_id,
                    intent=intent,
                    completion_event_id="provider-resolution-terminal-race-completed",
                    source_transcript_cursor=0,
                    assistant_message=Message.text("assistant", "terminal completion won"),
                    event_payload=identity.payload(),
                ),
            )
            assert completion.stage.state == "completed"
            release_resolution.set()

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="became terminal before external disposition",
            ):
                await resolution_task
            assert (
                await load_provider_operation_resolution(
                    store,
                    created.id,
                    stage.stage_id,
                )
                is None
            )
            assert (
                await load_pending_provider_operation_disposition(
                    store,
                    created.id,
                )
                is None
            )
            active = await store.load_active_model_completion_stage(created.id)
            assert active is not None
            assert active.stage.state == "completed"
            assert not any(
                event.type is EventType.PROVIDER_OPERATION_RESOLVED
                for event in await store.load_events(created.id)
            )
        finally:
            release_resolution.set()
            if resolution_task is not None and not resolution_task.done():
                resolution_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await resolution_task
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_future_checkpoint_before_operation_load(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_future_operation_load",
                    messages=[],
                ),
                identity=_identity(),
            )

            def transform(_session, checkpoint, _persisted_record):
                return SessionOperationPublication(
                    checkpoint={} if checkpoint is None else dict(checkpoint),
                    operation_records={
                        "completed-request": {
                            "status": "completed",
                            "private_detail": "must-not-be-interpreted",
                        }
                    },
                )

            await store.publish_session_operation(
                created.id,
                idempotency_key="completed-request",
                operation_transform=transform,
                events=[],
            )
            await store.checkpoint(
                created.id,
                {
                    CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                    "private_checkpoint_detail": "must-not-be-reported",
                },
            )

            runtime_store = CayuApp(
                session_store=store,
                enable_logging=False,
            )._runtime_session_store
            with pytest.raises(CheckpointCompatibilityError) as caught:
                await runtime_store.load_session_operation(
                    created.id,
                    "completed-request",
                )

            assert caught.value.reason == "checkpoint_schema_version_too_new"
            assert "must-not-be-interpreted" not in str(caught.value)
            assert "must-not-be-reported" not in str(caught.value)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_guarded_operation_publication_requires_native_commit_boundary() -> None:
    class LegacyOverrideStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.publication_calls = 0

        async def publish_session_operation(
            self,
            session_id: str,
            *,
            idempotency_key: str,
            operation_transform,
            events: list[Event],
            expected_statuses: set[SessionStatus] | None = None,
            expected_run_epoch: int | None = None,
            expected_transcript_cursor: int | None = None,
        ) -> Session:
            self.publication_calls += 1
            return await super().publish_session_operation(
                session_id,
                idempotency_key=idempotency_key,
                operation_transform=operation_transform,
                events=events,
                expected_statuses=expected_statuses,
                expected_run_epoch=expected_run_epoch,
                expected_transcript_cursor=expected_transcript_cursor,
            )

    async def run() -> None:
        store = LegacyOverrideStore()
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_guarded_legacy_override",
                messages=[],
            ),
            identity=_identity(),
        )
        guard_calls = 0

        def transform(_session, checkpoint, _persisted_record):
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["legacy_guarded_operation"] = True
            return SessionOperationPublication(checkpoint=updated)

        def commit_guard() -> None:
            nonlocal guard_calls
            guard_calls += 1

        await store.publish_session_operation_guarded(
            created.id,
            idempotency_key="guarded-request",
            operation_transform=transform,
            commit_guard=commit_guard,
            events=[],
        )

        assert store.publication_calls == 0
        assert guard_calls == 1
        assert await store.load_checkpoint(created.id) == {"legacy_guarded_operation": True}

        with pytest.raises(NotImplementedError, match="atomic guarded operation publication"):
            await SessionStore.publish_session_operation_guarded(
                store,
                created.id,
                idempotency_key="unsupported-guarded-request",
                operation_transform=transform,
                commit_guard=commit_guard,
                events=[],
            )

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_during_explicit_compaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            compactor = _ConformanceOverlappingCompactor()
            app = CayuApp(
                session_store=store,
                request_footprint=RequestFootprintConfig(enabled=False),
                enable_logging=False,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_delete_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
                Message.text("assistant", "current answer"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            await store.append_event(
                created.id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=created.id,
                ),
            )
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-delete-conformance",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            async def collect() -> list[Event]:
                return [event async for event in app.compact_session(request)]

            task = asyncio.create_task(collect())
            await compactor.started[0].wait()
            with pytest.raises(ValueError, match="durable operation .* is active"):
                await store.delete_session(created.id)
            assert await store.load(created.id) is not None

            compactor.release[0].set()
            events = await task
            assert events[-1].type == EventType.SESSION_CHECKPOINTED
            await store.delete_session(created.id)
            assert await store.load(created.id) is None
            with pytest.raises(KeyError, match="Session not found"):
                await store.query_latest_interaction_events(created.id, limit=1)
            with pytest.raises(KeyError, match="Session not found"):
                await store.load_session_operation(created.id, request.idempotency_key)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_during_incomplete_recovery_claim(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_recovery_claim_delete_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            claimed_at = datetime.now(UTC)
            claim_id = "recovery-delete-conformance"
            await store.checkpoint(
                created.id,
                {
                    "incomplete_session_recovery_claim": {
                        "version": 1,
                        "claim_id": claim_id,
                        "claimed_at": claimed_at.isoformat(),
                        "claim_expires_at": (claimed_at + timedelta(minutes=5)).isoformat(),
                    }
                },
            )

            with pytest.raises(
                ValueError,
                match=f"incomplete-session recovery claim {claim_id} is active",
            ):
                await store.delete_session(created.id)
            assert await store.load(created.id) is not None

            await store.checkpoint(
                created.id,
                {
                    "incomplete_session_recovery_claim": {
                        "version": 1,
                        "claim_id": claim_id,
                        "claimed_at": (claimed_at - timedelta(minutes=10)).isoformat(),
                        "claim_expires_at": (claimed_at - timedelta(minutes=5)).isoformat(),
                    }
                },
            )
            await store.delete_session(created.id)
            assert await store.load(created.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_until_budget_settlement_is_delivered(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_pending_budget_settlement_delete",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            reservation_id = "bres_" + "1" * 32
            reserved = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=created.id,
                payload={"reservation_id": reservation_id},
            )
            await store.claim_budget_reservation_identity(
                reservation_id=reservation_id,
                publication_session_id=created.id,
                publication_id=reserved.id,
            )
            await store.append_event(created.id, reserved)
            reserved_claim = await store.claim_persisted_event_side_effect(
                session_id=created.id,
                event_id=reserved.id,
            )
            assert reserved_claim is not None
            await store.mark_persisted_event_side_effect_delivered(reserved_claim)

            with pytest.raises(
                ValueError,
                match="budget settlement audit event is pending",
            ):
                await store.delete_session(created.id)

            released = Event(
                type=EventType.BUDGET_RESERVATION_RELEASED,
                session_id=created.id,
                payload={"reservation_id": reservation_id},
            )
            await store.append_event(created.id, released)
            with pytest.raises(
                ValueError,
                match="budget settlement audit event is pending",
            ):
                await store.delete_session(created.id)

            released_claim = await store.claim_persisted_event_side_effect(
                session_id=created.id,
                event_id=released.id,
            )
            assert released_claim is not None
            await store.mark_persisted_event_side_effect_delivered(released_claim)
            await store.delete_session(created.id)
            assert await store.load(created.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_during_terminal_publication(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_terminal_publication_delete_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            operation_id = "terminal-publication-delete-conformance"
            running = await store.transition_status_and_checkpoint(
                created.id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda current_session, checkpoint: (
                    _checkpoint_with_session_run_operation(
                        checkpoint=checkpoint,
                        current_session=current_session,
                        operation_id=operation_id,
                    )
                ),
            )
            assert running.run_epoch == completed.run_epoch + 1
            await store.update_status(created.id, SessionStatus.INTERRUPTED)

            with pytest.raises(
                ValueError,
                match=f"terminal publication {operation_id} is incomplete",
            ):
                await store.delete_session(created.id)
            assert await store.load(created.id) is not None

            await store.append_event(
                created.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=created.id,
                    payload={"session_run_operation_id": operation_id},
                ),
            )
            await store.checkpoint(created.id, {})
            await store.delete_session(created.id)
            assert await store.load(created.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_until_markerless_terminal_evidence_is_safe(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            pending_interrupt = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_pending_interrupt_terminal_delete",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            await store.update_status(
                pending_interrupt.id,
                SessionStatus.INTERRUPTED,
            )
            await store.checkpoint(
                pending_interrupt.id,
                {
                    "pending_session_interrupt": {
                        "reason": "operator request",
                        "interruption_type": "operator_requested",
                        "interruption_request_id": "interrupt-delete-conformance",
                    }
                },
            )
            with pytest.raises(
                ValueError,
                match="pending interruption terminal publication is incomplete",
            ):
                await store.delete_session(pending_interrupt.id)
            await store.append_event(
                pending_interrupt.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=pending_interrupt.id,
                    payload={
                        "reason": "operator request",
                        "interruption_type": "operator_requested",
                        "interruption_request_id": "interrupt-delete-conformance",
                    },
                ),
            )
            with pytest.raises(
                ValueError,
                match="pending interruption terminal publication is incomplete",
            ):
                await store.delete_session(pending_interrupt.id)
            await store.checkpoint(pending_interrupt.id, {})
            await store.delete_session(pending_interrupt.id)
            assert await store.load(pending_interrupt.id) is None

            pending_action = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_pending_action_terminal_delete",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            await store.update_status(pending_action.id, SessionStatus.INTERRUPTED)
            await store.checkpoint(
                pending_action.id,
                {"pending_tool_round": {"recovery_marker": True}},
            )
            with pytest.raises(
                ValueError,
                match="terminal publication evidence is incomplete",
            ):
                await store.delete_session(pending_action.id)
            await store.append_event(
                pending_action.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=pending_action.id,
                ),
            )
            await store.delete_session(pending_action.id)
            assert await store.load(pending_action.id) is None

            markerless = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_markerless_current_terminal_delete",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            await store.update_status(markerless.id, SessionStatus.INTERRUPTED)
            await store.append_event(
                markerless.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=markerless.id,
                ),
            )
            await store.update_status(markerless.id, SessionStatus.RUNNING)
            await store.append_event(
                markerless.id,
                Event(
                    type=EventType.SESSION_RESUMED,
                    session_id=markerless.id,
                ),
            )
            await store.update_status(markerless.id, SessionStatus.INTERRUPTED)
            with pytest.raises(
                ValueError,
                match="terminal publication evidence is incomplete",
            ):
                await store.delete_session(markerless.id)
            await store.append_event(
                markerless.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=markerless.id,
                ),
            )
            await store.delete_session(markerless.id)
            assert await store.load(markerless.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_fork_and_delete_with_active_model_stage(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_active_model_stage_control_plane"
            source_message = Message.text("user", "preserve the staged provider boundary")
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[source_message],
                ),
                identity=_identity(),
            )
            await store.append_transcript_messages(created.id, [source_message])
            running = await store.transition_status(
                created.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            prepared = await store.prepare_model_completion_stage(
                created.id,
                request=ModelCompletionStageRequest(
                    stage_id="control-plane-stage",
                    logical_step_id="control-plane-step",
                    dispatch_ordinal=0,
                    intent={"request_fingerprint": "control-plane"},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=1,
            )
            interrupted = await store.update_status(
                created.id,
                SessionStatus.INTERRUPTED,
            )
            await store.append_event(
                created.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=created.id,
                ),
            )
            fork = Session(
                id="sess_active_model_stage_control_plane_fork",
                agent_name=interrupted.agent_name,
                provider_name=interrupted.provider_name,
                model=interrupted.model,
                parent_session_id=interrupted.id,
                causal_budget_id=interrupted.causal_budget_id,
                invocation=fork_session_invocation(interrupted),
                status=interrupted.status,
            )

            with pytest.raises(ValueError, match="model-completion stage is active"):
                await store.create_fork(
                    source_session_id=interrupted.id,
                    fork=fork,
                    source_statuses={SessionStatus.INTERRUPTED},
                    transcript_cursor=None,
                    checkpoint_transform=None,
                    expected_source_run_epoch=interrupted.run_epoch,
                )
            with pytest.raises(ValueError, match="model-completion stage is active"):
                await store.delete_session(interrupted.id)
            assert await store.load(interrupted.id) is not None
            assert await store.load_active_model_completion_stage(interrupted.id) is not None

            await store.abandon_model_completion_stage(
                interrupted.id,
                stage_id=prepared.stage.stage_id,
                preparation_digest=prepared.stage.preparation_digest,
                expected_run_epoch=interrupted.run_epoch,
            )
            created_fork = await store.create_fork(
                source_session_id=interrupted.id,
                fork=fork,
                source_statuses={SessionStatus.INTERRUPTED},
                transcript_cursor=None,
                checkpoint_transform=None,
                expected_source_run_epoch=interrupted.run_epoch,
            )
            assert created_fork.id == fork.id
            await store.delete_session(interrupted.id)
            assert await store.load(interrupted.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_atomically_fences_checkpoint_owner(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_checkpoint_fence_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            original_checkpoint = {"owner": "expired", "preserved": {"value": 1}}
            await store.checkpoint(created.id, original_checkpoint)

            def replace_owner(
                current: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                assert current.run_epoch == completed.run_epoch
                assert checkpoint == original_checkpoint
                assert checkpoint is not None
                updated = dict(checkpoint)
                updated["owner"] = "replacement"
                return updated

            fenced = await store.fence_run_and_transform_checkpoint(
                created.id,
                statuses={SessionStatus.COMPLETED},
                checkpoint_transform=replace_owner,
            )
            assert fenced.run_epoch == completed.run_epoch + 1
            persisted = await store.load(created.id)
            assert persisted is not None
            assert persisted.run_epoch == fenced.run_epoch
            assert await store.load_checkpoint(created.id) == {
                "owner": "replacement",
                "preserved": {"value": 1},
            }
            await store.release_run_fence(created.id)

            before_rejected_fence = await store.load(created.id)
            before_rejected_checkpoint = await store.load_checkpoint(created.id)
            assert before_rejected_fence is not None

            def reject_fence(
                _current: Session,
                _checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                raise RuntimeError("checkpoint owner changed")

            with pytest.raises(RuntimeError, match="checkpoint owner changed"):
                await store.fence_run_and_transform_checkpoint(
                    created.id,
                    statuses={SessionStatus.COMPLETED},
                    checkpoint_transform=reject_fence,
                )
            assert await store.load(created.id) == before_rejected_fence
            assert await store.load_checkpoint(created.id) == before_rejected_checkpoint

            def cancel_fence(
                _current: Session,
                _checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                raise asyncio.CancelledError("cancel atomic fence")

            with pytest.raises(asyncio.CancelledError, match="cancel atomic fence"):
                await store.fence_run_and_transform_checkpoint(
                    created.id,
                    statuses={SessionStatus.COMPLETED},
                    checkpoint_transform=cancel_fence,
                )
            assert await store.load(created.id) == before_rejected_fence
            assert await store.load_checkpoint(created.id) == before_rejected_checkpoint

            fenced_after_cancel = await store.fence_run_and_transform_checkpoint(
                created.id,
                statuses={SessionStatus.COMPLETED},
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
            )
            assert fenced_after_cancel.run_epoch == before_rejected_fence.run_epoch + 1
            await store.release_run_fence(created.id)
            before_rejected_fence = await store.load(created.id)
            before_rejected_checkpoint = await store.load_checkpoint(created.id)
            assert before_rejected_fence is not None

            def omit_replacement(
                _current: Session,
                _checkpoint: dict[str, Any] | None,
            ) -> None:
                return None

            with pytest.raises(
                ValueError,
                match="Fenced checkpoint transform must return a checkpoint",
            ):
                await store.fence_run_and_transform_checkpoint(
                    created.id,
                    statuses={SessionStatus.COMPLETED},
                    checkpoint_transform=omit_replacement,
                )
            assert await store.load(created.id) == before_rejected_fence
            assert await store.load_checkpoint(created.id) == before_rejected_checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_durable_session_message_queue(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            idle_request = EnqueueSessionMessageRequest(
                session_id=created.id,
                idempotency_key="queue-idle",
                content="idle",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
            next_one_request = EnqueueSessionMessageRequest(
                session_id=created.id,
                idempotency_key="queue-next-1",
                content="next one",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            next_two_request = EnqueueSessionMessageRequest(
                session_id=created.id,
                idempotency_key="queue-next-2",
                content="next two",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            idle = await store.enqueue_session_message(idle_request)
            next_one = await store.enqueue_session_message(next_one_request)
            next_two = await store.enqueue_session_message(next_two_request)
            replay = await store.enqueue_session_message(next_one_request)
            assert replay.replayed is True
            assert replay.message.queue_id == next_one.message.queue_id
            with pytest.raises(ValueError, match="different request"):
                await store.enqueue_session_message(
                    next_one_request.model_copy(update={"content": "changed"})
                )

            store = await _reopen_store(session_store_case, store)
            reconstructed = await store.enqueue_session_message(next_one_request)
            assert reconstructed.replayed is True
            assert reconstructed.message == next_one.message

            await store.transition_status(
                created.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.transition_status(
                created.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
            )
            with pytest.raises(
                SessionStatusConflict,
                match="delivered only while running",
            ):
                await store.deliver_queued_session_messages(
                    created.id,
                    include_on_idle=True,
                )
            await store.transition_status(
                created.id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
            )
            first = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=True,
                limit=1,
                interaction_id="interaction-queue-first",
                interaction_started_event=Event(
                    id="evt_interaction_queue_first",
                    type=EventType.INTERACTION_STARTED,
                    session_id=created.id,
                    interaction_id="interaction-queue-first",
                ),
            )
            assert [event.type for event in first.events] == [
                EventType.INTERACTION_STARTED,
                EventType.SESSION_MESSAGE_DELIVERED,
            ]
            assert [message.queue_id for message in first.messages] == [next_one.message.queue_id]
            assert first.has_more is True
            late = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=created.id,
                    idempotency_key="queue-late",
                    content="late next boundary",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            second = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=True,
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id="interaction-queue-first",
            )
            third = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=True,
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id="interaction-queue-first",
            )
            assert [message.queue_id for message in second.messages] == [next_two.message.queue_id]
            assert [message.queue_id for message in third.messages] == [idle.message.queue_id]
            assert third.has_more is False

            with pytest.raises(SessionQueuedMessagesPending):
                await store.transition_status_if_no_queued_messages(
                    created.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )
            late_batch = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=False,
                interaction_id="interaction-queue-late",
                interaction_started_event=Event(
                    id="evt_interaction_queue_late",
                    type=EventType.INTERACTION_STARTED,
                    session_id=created.id,
                    interaction_id="interaction-queue-late",
                ),
            )
            assert [message.queue_id for message in late_batch.messages] == [late.message.queue_id]
            completed = await store.transition_status_if_no_queued_messages(
                created.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )
            assert completed.status == SessionStatus.COMPLETED
            transcript = await store.load_transcript(created.id)
            assert [message.content[0].text for message in transcript] == [  # type: ignore[union-attr]
                "next one",
                "next two",
                "idle",
                "late next boundary",
            ]
            transcript_page = await store.query_transcript(
                TranscriptQuery(session_id=created.id, limit=10)
            )
            assert [record.interaction_id for record in transcript_page.records] == [
                "interaction-queue-first",
                "interaction-queue-first",
                "interaction-queue-first",
                "interaction-queue-late",
            ]
            lifecycle = await store.query_events(
                EventQuery(
                    session_id=created.id,
                    event_types=(EventType.INTERACTION_STARTED,),
                )
            )
            assert [record.event.id for record in lifecycle] == [
                "evt_interaction_queue_first",
                "evt_interaction_queue_late",
            ]
            queue_events = [
                event
                for event in await store.load_events(created.id)
                if event.type
                in {EventType.SESSION_MESSAGE_QUEUED, EventType.SESSION_MESSAGE_DELIVERED}
            ]
            assert len(queue_events) == 8
            assert all("content" not in event.payload for event in queue_events)
            deliveries = await store.list_persisted_event_side_effect_deliveries(limit=1000)
            assert {delivery.event_id for delivery in deliveries} == {
                event.id for event in [*queue_events, *(record.event for record in lifecycle)]
            }
            assert all(
                delivery.status is PersistedEventSideEffectStatus.PENDING for delivery in deliveries
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_enqueue_completion_race_is_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_completion_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            await store.transition_status(
                created.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            start = asyncio.Event()

            async def enqueue():
                await start.wait()
                return await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=created.id,
                        idempotency_key="completion-race",
                        content="race steering",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )

            async def complete():
                await start.wait()
                return await store.transition_status_if_no_queued_messages(
                    created.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )

            enqueue_task = asyncio.create_task(enqueue())
            completion_task = asyncio.create_task(complete())
            start.set()
            enqueue_result, completion_result = await asyncio.gather(
                enqueue_task,
                completion_task,
                return_exceptions=True,
            )

            if isinstance(enqueue_result, Exception):
                assert isinstance(enqueue_result, SessionStatusConflict)
                assert "pending or running" in str(enqueue_result)
                assert not isinstance(completion_result, Exception)
                assert completion_result.status is SessionStatus.COMPLETED
                events = await store.query_events(
                    EventQuery(
                        session_id=created.id,
                        event_type=EventType.SESSION_MESSAGE_QUEUED,
                    )
                )
                assert events == []
            else:
                assert enqueue_result.message.content == "race steering"
                assert isinstance(completion_result, SessionQueuedMessagesPending)
                delivered = await store.deliver_queued_session_messages(
                    created.id,
                    include_on_idle=False,
                )
                assert [message.queue_id for message in delivered.messages] == [
                    enqueue_result.message.queue_id
                ]
                completed = await store.transition_status_if_no_queued_messages(
                    created.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )
                assert completed.status is SessionStatus.COMPLETED
            with pytest.raises(SessionStatusConflict, match="pending or running"):
                await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=created.id,
                        idempotency_key="after-completion",
                        content="too late",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_queue_delivery_acknowledgement(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_queue_delivery_reconstruction"
            interaction_id = "interaction-queue-delivery-reconstruction"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_queue_delivery_initial_interaction",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id="interaction-queue-delivery-initial",
                ),
                interaction_source_messages=[Message.text("user", "initial")],
            )
            for index in range(2):
                await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=session_id,
                        idempotency_key=f"queue-delivery-reconstruction-{index}",
                        content=f"queued {index}",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )

            started = Event(
                id="evt_queue_delivery_reconstruction_started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            first = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=interaction_id,
                limit=1,
                interaction_id=interaction_id,
                interaction_started_event=started,
            )
            assert first.replayed is False
            assert first.delivery_id == interaction_id
            assert first.has_more is True

            store = await _reopen_store(session_store_case, store)
            replayed = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=interaction_id,
                limit=1,
                interaction_id=interaction_id,
                interaction_started_event=started,
            )
            assert replayed.replayed is True
            assert replayed == first.model_copy(update={"replayed": True})

            with pytest.raises(ValueError, match="different queue delivery"):
                await store.deliver_queued_session_messages(
                    session_id,
                    include_on_idle=True,
                    delivery_id=interaction_id,
                    limit=1,
                    interaction_id=interaction_id,
                    interaction_started_event=started,
                )

            second = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=f"{interaction_id}:batch:1",
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id=interaction_id,
            )
            assert second.replayed is False
            assert [message.content for message in second.messages] == ["queued 1"]
            empty_delivery_id = f"{interaction_id}:batch:2"
            empty = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=empty_delivery_id,
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id=interaction_id,
            )
            assert empty.messages == ()
            assert empty.interaction_id == interaction_id
            store = await _reopen_store(session_store_case, store)
            replayed_empty = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=empty_delivery_id,
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id=interaction_id,
            )
            assert replayed_empty == empty.model_copy(update={"replayed": True})
            transcript = await store.load_transcript(session_id)
            assert [
                message.content[0].text  # type: ignore[union-attr]
                for message in transcript
            ] == ["queued 0", "queued 1"]
            lifecycle = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_id=started.id,
                )
            )
            assert [record.event for record in lifecycle] == [started]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_queue_boundary_is_global_and_stable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            primary = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_global_boundary_primary",
                    messages=[Message.text("user", "primary")],
                ),
                identity=_identity(),
            )
            other = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_global_boundary_other",
                    messages=[Message.text("user", "other")],
                ),
                identity=_identity(),
            )
            for session in (primary, other):
                await store.transition_status(
                    session.id,
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                )

            primary_request = EnqueueSessionMessageRequest(
                session_id=primary.id,
                idempotency_key="primary-before-boundary",
                content="deliver before boundary",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            accepted = await store.enqueue_session_message(primary_request)
            other_message = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=other.id,
                    idempotency_key="other-before-boundary",
                    content="advance global boundary",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )

            first = await store.deliver_queued_session_messages(
                primary.id,
                include_on_idle=False,
            )
            assert [message.queue_id for message in first.messages] == [accepted.message.queue_id]
            assert first.eligible_through >= other_message.message.ordering_key

            replay = await store.enqueue_session_message(primary_request)
            assert replay.replayed is True
            assert replay.message.status is SessionMessageQueueStatus.DELIVERED

            late = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=primary.id,
                    idempotency_key="primary-after-boundary",
                    content="deliver after boundary",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            fenced = await store.deliver_queued_session_messages(
                primary.id,
                include_on_idle=False,
                eligible_through=first.eligible_through,
            )
            assert fenced.messages == ()

            current = await store.deliver_queued_session_messages(
                primary.id,
                include_on_idle=False,
            )
            assert [message.queue_id for message in current.messages] == [late.message.queue_id]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_atomically_transforms_checkpoint(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_checkpoint_transform",
                    messages=[Message.text("user", "hello")],
                ),
                identity=_identity(),
            )
            await store.checkpoint("sess_atomic_checkpoint_transform", {"original": True})

            def add_key(key: str):
                def transform(_session: Session, checkpoint: dict[str, Any] | None):
                    updated = {} if checkpoint is None else dict(checkpoint)
                    updated[key] = True
                    return updated

                return transform

            await asyncio.gather(
                store.transform_checkpoint(
                    "sess_atomic_checkpoint_transform",
                    add_key("first"),
                ),
                store.transform_checkpoint(
                    "sess_atomic_checkpoint_transform",
                    add_key("second"),
                ),
            )
            await asyncio.gather(
                store.transform_checkpoint(
                    "sess_atomic_checkpoint_transform",
                    add_key("third"),
                ),
                store.append_transcript_messages_and_transform_checkpoint(
                    "sess_atomic_checkpoint_transform",
                    [Message.text("assistant", "done")],
                    add_key("fourth"),
                ),
            )

            assert await store.load_checkpoint("sess_atomic_checkpoint_transform") == {
                "original": True,
                "first": True,
                "second": True,
                "third": True,
                "fourth": True,
            }
            assert [
                message.content[0].text
                for message in await store.load_transcript("sess_atomic_checkpoint_transform")
            ] == ["done"]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_lists_pending_interruption_cascades(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            for session_id in (
                "sess_cascade_index_a",
                "sess_cascade_index_b",
                "sess_cascade_index_none",
                "sess_cascade_index_running",
            ):
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", session_id)],
                    ),
                    identity=_identity(),
                )
            for session_id in (
                "sess_cascade_index_a",
                "sess_cascade_index_b",
                "sess_cascade_index_none",
            ):
                await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.update_status(
                "sess_cascade_index_running",
                SessionStatus.RUNNING,
            )
            for session_id in (
                "sess_cascade_index_a",
                "sess_cascade_index_b",
                "sess_cascade_index_running",
            ):
                await store.checkpoint(
                    session_id,
                    {
                        "pending_interruption_cascade": {
                            "attempt_id": session_id,
                            "interrupt_payload": {"interruption_type": "operator_requested"},
                        }
                    },
                )
            await store.checkpoint(
                "sess_cascade_index_none",
                {"unrelated_checkpoint": True},
            )

            first = await store.list_sessions_with_pending_interruption_cascade(
                SessionQuery(
                    status=SessionStatus.INTERRUPTED,
                    order_by=SessionOrder.CREATED_AT_ASC,
                    limit=1,
                    include_total_count=True,
                )
            )
            second = await store.list_sessions_with_pending_interruption_cascade(
                SessionQuery(
                    status=SessionStatus.INTERRUPTED,
                    order_by=SessionOrder.CREATED_AT_ASC,
                    limit=1,
                    cursor=first.next_cursor,
                )
            )
            running = await store.list_sessions_with_pending_interruption_cascade(
                SessionQuery(status=SessionStatus.RUNNING)
            )

            assert first.total_count == 2
            assert first.next_cursor is not None
            assert [session.id for session in first.sessions + second.sessions] == [
                "sess_cascade_index_a",
                "sess_cascade_index_b",
            ]
            assert second.next_cursor is None
            assert [session.id for session in running.sessions] == ["sess_cascade_index_running"]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_future_checkpoint_before_marker_projection(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_future_cascade_projection"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(
                session_id,
                {
                    CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                    "pending_interruption_cascade": {
                        "attempt_id": "must-not-be-interpreted",
                    },
                },
            )

            with pytest.raises(CheckpointCompatibilityError) as caught:
                await CayuApp(
                    session_store=store,
                    enable_logging=False,
                ).interruption_cascade_status(session_id)

            assert caught.value.reason == "checkpoint_schema_version_too_new"
            assert "must-not-be-interpreted" not in str(caught.value)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_applies_query_filters(session_store_case) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            await session_store.create(
                RunRequest(
                    agent_name="alpha",
                    session_id="sess_query_alpha",
                    causal_budget_id="budget_runtime",
                    environment_name="local",
                    labels={"team": "runtime"},
                    messages=[Message.text("user", "alpha")],
                ),
                identity=_identity(),
            )
            await session_store.create(
                RunRequest(
                    agent_name="beta",
                    session_id="sess_query_beta",
                    causal_budget_id="budget_runtime",
                    environment_name="remote",
                    labels={"team": "review"},
                    messages=[Message.text("user", "beta")],
                ),
                identity=_identity(),
            )
            await session_store.append_events(
                "sess_query_alpha",
                [
                    Event(
                        id="evt_query_alpha",
                        type=EventType.TOOL_CALL_COMPLETED,
                        session_id="sess_query_alpha",
                        agent_name="alpha",
                        environment_name="local",
                        tool_name="read_file",
                        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                    )
                ],
            )
            await session_store.append_events(
                "sess_query_beta",
                [
                    Event(
                        id="evt_query_beta",
                        type=EventType.TOOL_CALL_FAILED,
                        session_id="sess_query_beta",
                        agent_name="beta",
                        environment_name="remote",
                        tool_name="edit_file",
                        timestamp=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
                    )
                ],
            )

            sessions = await session_store.list_sessions(
                SessionQuery(q="ALPHA", labels={"team": "runtime"}, include_total_count=True)
            )
            assert [session.id for session in sessions.sessions] == ["sess_query_alpha"]
            assert sessions.total_count == 1

            records = await session_store.query_events(
                EventQuery(
                    causal_budget_id="budget_runtime",
                    event_types=(EventType.TOOL_CALL_COMPLETED,),
                    agent_name="alpha",
                    tool_name="read_file",
                )
            )
            assert [record.event.id for record in records] == ["evt_query_alpha"]
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_preserves_interaction_attribution(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interaction_attribution",
                    messages=[Message.text("user", "bootstrap")],
                ),
                identity=_identity(),
            )
            first_id = "interaction-first"
            second_id = "interaction-second"
            await store.append_events(
                session.id,
                [
                    Event(
                        id="evt_interaction_first",
                        type=EventType.MODEL_STARTED,
                        session_id=session.id,
                        interaction_id=first_id,
                    ),
                    Event(
                        id="evt_interaction_legacy",
                        type="custom.legacy",
                        session_id=session.id,
                    ),
                    Event(
                        id="evt_interaction_second",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session.id,
                        interaction_id=second_id,
                    ),
                ],
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("user", "first"), Message.text("assistant", "one")],
                interaction_id=first_id,
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("user", "legacy")],
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("user", "second"), Message.text("assistant", "two")],
                interaction_id=second_id,
            )

            store = await _reopen_store(session_store_case, store)
            first_events = await store.query_events(
                EventQuery(session_id=session.id, interaction_id=first_id)
            )
            assert [(record.event.id, record.event.interaction_id) for record in first_events] == [
                ("evt_interaction_first", first_id)
            ]
            all_events = await store.query_events(EventQuery(session_id=session.id))
            assert [record.event.interaction_id for record in all_events] == [
                first_id,
                None,
                second_id,
            ]

            first_transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, interaction_id=first_id)
            )
            assert [record.index for record in first_transcript.records] == [0, 1]
            assert [record.interaction_id for record in first_transcript.records] == [
                first_id,
                first_id,
            ]
            assert first_transcript.total_records == 2

            all_transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, limit=10)
            )
            assert [record.index for record in all_transcript.records] == [0, 1, 2, 3, 4]
            assert [record.interaction_id for record in all_transcript.records] == [
                first_id,
                first_id,
                None,
                second_id,
                second_id,
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_failed_transition_admission_is_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_failed_transition_admission_atomic"
        duplicate_event_id = "evt_duplicate_transition_admission"
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, {"state": "before"})
            existing = Event(
                id=duplicate_event_id,
                type=EventType.SESSION_STARTED,
                session_id=session_id,
            )
            await store.append_event(session_id, existing)
            original = await store.load(session_id)
            assert original is not None

            with pytest.raises(ValueError, match="Event already exists for session"):
                await store.transition_status_and_checkpoint(
                    session_id,
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                    checkpoint_transform=lambda _session, _checkpoint: {"state": "after"},
                    interaction_started_event=Event(
                        id=duplicate_event_id,
                        type=EventType.INTERACTION_STARTED,
                        session_id=session_id,
                        interaction_id="interaction-failed-transition-admission",
                    ),
                    interaction_source_messages=[Message.text("user", "must not persist")],
                )

            current = await store.load(session_id)
            assert current == original
            assert await store.load_checkpoint(session_id) == {"state": "before"}
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == [existing]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_create_atomically_claims_and_admits_first_interaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            source: list[Message] = []
            started = Event(
                id="evt_atomic_create_interaction",
                type=EventType.INTERACTION_STARTED,
                session_id="sess_atomic_create_interaction",
                interaction_id="interaction-atomic-create",
            )
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_create_interaction",
                    messages=source,
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=source,
            )

            assert session.status is SessionStatus.RUNNING
            assert session.run_epoch == 1
            store = await _reopen_store(session_store_case, store)
            assert [
                record.event.id
                for record in await store.query_events(EventQuery(session_id=session.id))
            ] == [started.id]
            assert (
                await store.query_transcript(TranscriptQuery(session_id=session.id))
            ).records == []
            deferred = await store.load_deferred_interaction_input(session.id)
            assert deferred is not None
            assert deferred.interaction_id == "interaction-atomic-create"
            assert deferred.source_messages == []
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
                "initial_transcript_pending": {
                    "version": 1,
                    "interaction_id": "interaction-atomic-create",
                },
            }
            assert await store.materialize_deferred_interaction_input(
                session.id,
                interaction_id="interaction-atomic-create",
            )
            assert await store.load_deferred_interaction_input(session.id) is None
            assert await store.load_checkpoint(session.id) == checkpoint
        finally:
            await store.release_run_fence("sess_atomic_create_interaction")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_create_atomically_stages_pending_checkpoint(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            assert store.supports_pending_session_initial_checkpoint is True
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_pending_checkpoint",
                    messages=[Message.text("user", "prepared work")],
                ),
                identity=_identity(),
                checkpoint_transform=lambda current, checkpoint: {
                    "prepared": current.id,
                    "prior": checkpoint,
                },
            )
            assert session.status is SessionStatus.PENDING
            assert session.run_epoch == 0
            store = await _reopen_store(session_store_case, store)
            assert await store.load_checkpoint(session.id) == {
                "prepared": session.id,
                "prior": None,
            }

            def reject_create(_current: Session, _checkpoint: dict[str, Any] | None):
                raise RuntimeError("reject staged checkpoint")

            with pytest.raises(RuntimeError, match="reject staged checkpoint"):
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_failed_atomic_pending_checkpoint",
                        messages=[],
                    ),
                    identity=_identity(),
                    checkpoint_transform=reject_create,
                )
            assert await store.load("sess_failed_atomic_pending_checkpoint") is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_transition_atomically_claims_and_admits_interaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_resume_interaction",
                    messages=[],
                ),
                identity=_identity(),
            )
            source = [Message.text("user", "resume source")]
            started = Event(
                id="evt_atomic_resume_interaction",
                type=EventType.INTERACTION_STARTED,
                session_id=session.id,
                interaction_id="interaction-atomic-resume",
            )
            transitioned = await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_started_event=started,
                interaction_source_messages=source,
            )

            assert transitioned.status is SessionStatus.RUNNING
            assert transitioned.run_epoch == 1
            store = await _reopen_store(session_store_case, store)
            events = await store.query_events(
                EventQuery(session_id=session.id, interaction_id="interaction-atomic-resume")
            )
            assert [record.event.id for record in events] == [started.id]
            transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, limit=10)
            )
            assert [record.message for record in transcript.records] == source
            assert [record.interaction_id for record in transcript.records] == [
                "interaction-atomic-resume"
            ]
            assert await store.load_deferred_interaction_input(session.id) is None
        finally:
            await store.release_run_fence("sess_atomic_resume_interaction")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_initial_transcript_publication_clears_authority_marker(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_initial_transcript_authority"
        interaction_id = "interaction-initial-authority"
        source = [Message.text("user", "start")]
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=source,
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_initial_transcript_authority",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=source,
            )
            assert await store.load_checkpoint(session.id) is not None

            final = [Message.text("system", "authoritative"), *source]
            await store.replace_initial_transcript_messages(
                session.id,
                source,
                final,
                interaction_id=interaction_id,
            )

            store = await _reopen_store(session_store_case, store)
            assert await store.load_checkpoint(session.id) == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
            }
            transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, limit=10)
            )
            assert [record.message for record in transcript.records] == final
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_future_checkpoint_before_initial_transcript_write(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_future_initial_transcript"
        interaction_id = "interaction-future-initial-transcript"
        source = [Message.text("user", "start")]
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=source,
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_future_initial_transcript",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=source,
            )
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint is not None
            checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
            checkpoint["private_checkpoint_detail"] = "must-not-be-reported"
            await store.checkpoint(session.id, checkpoint)

            runtime_store = CayuApp(
                session_store=store,
                enable_logging=False,
            )._runtime_session_store
            with pytest.raises(CheckpointCompatibilityError) as caught:
                await runtime_store.replace_initial_transcript_messages(
                    session.id,
                    source,
                    [Message.text("system", "authoritative"), *source],
                    interaction_id=interaction_id,
                )

            assert caught.value.reason == "checkpoint_schema_version_too_new"
            assert "must-not-be-reported" not in str(caught.value)
            assert (
                await store.query_transcript(TranscriptQuery(session_id=session.id))
            ).records == []
            deferred = await store.load_deferred_interaction_input(session.id)
            assert deferred is not None
            assert deferred.interaction_id == interaction_id
            assert await store.load_checkpoint(session.id) == checkpoint
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_transition_can_defer_continued_interaction_input(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_continue_interaction",
                    messages=[],
                ),
                identity=_identity(),
            )
            source = [Message.text("user", "continue after recovery")]
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_source_messages=source,
                continued_interaction_id="interaction-existing",
                defer_interaction_source=True,
            )

            store = await _reopen_store(session_store_case, store)
            assert await store.query_events(EventQuery(session_id=session.id)) == []
            assert (
                await store.query_transcript(TranscriptQuery(session_id=session.id))
            ).records == []
            deferred = await store.load_deferred_interaction_input(session.id)
            assert deferred is not None
            assert deferred.interaction_id == "interaction-existing"
            assert deferred.source_messages == source
            assert await store.materialize_deferred_interaction_input(
                session.id,
                interaction_id="interaction-existing",
            )
            transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, limit=10)
            )
            assert [record.interaction_id for record in transcript.records] == [
                "interaction-existing"
            ]
        finally:
            await store.release_run_fence("sess_atomic_continue_interaction")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_transition_extends_matching_deferred_interaction_input(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_extend_deferred_interaction",
                    messages=[],
                ),
                identity=_identity(),
            )
            first_source = [Message.text("user", "continue after recovery")]
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_source_messages=first_source,
                continued_interaction_id="interaction-existing",
                defer_interaction_source=True,
            )
            await store.update_status(session.id, SessionStatus.FAILED)
            await store.release_run_fence(session.id)
            store = await _reopen_store(session_store_case, store)

            combined_source = [
                *first_source,
                Message.text("user", "retry recovery"),
            ]
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.FAILED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_source_messages=combined_source,
                continued_interaction_id="interaction-existing",
                defer_interaction_source=True,
            )

            store = await _reopen_store(session_store_case, store)
            deferred = await store.load_deferred_interaction_input(session.id)
            assert deferred is not None
            assert deferred.interaction_id == "interaction-existing"
            assert deferred.source_messages == combined_source
            assert (
                await store.query_transcript(TranscriptQuery(session_id=session.id))
            ).records == []
        finally:
            await store.release_run_fence("sess_atomic_extend_deferred_interaction")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interaction_transition_is_atomic_and_reconstructable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_interaction_transition_conformance"
            interaction_id = "interaction-transition-conformance"
            started = Event(
                id="evt_interaction_transition_started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )
            failed = Event(
                id="evt_interaction_transition_failed",
                type=EventType.INTERACTION_FAILED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            published = await store.publish_interaction_transition(
                session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
            )
            failed_transition = InteractionTransitionSpec(
                event=failed,
                from_statuses=(SessionStatus.RUNNING,),
                to_status=SessionStatus.FAILED,
            )
            assert published.status_changed is True
            assert published.replayed is False
            assert published.session.status is SessionStatus.FAILED
            assert published.event == failed

            loaded_receipt = await store.load_interaction_transition_receipt(
                session_id,
                transition=failed_transition,
            )
            assert loaded_receipt is not None
            assert loaded_receipt.status_changed is True
            assert loaded_receipt.replayed is True
            assert loaded_receipt.session == published.session
            assert loaded_receipt.transition == failed_transition

            store = await _reopen_store(session_store_case, store)
            replayed = await store.publish_interaction_transition(
                session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
            )
            assert replayed.status_changed is True
            assert replayed.replayed is True
            assert replayed.session.status is SessionStatus.FAILED
            assert replayed.event == failed
            records = await store.query_events(
                EventQuery(session_id=session_id, event_id=failed.id)
            )
            assert [record.event for record in records] == [failed]

            with pytest.raises(ValueError, match="different data"):
                await store.publish_interaction_transition(
                    session_id,
                    event=failed.model_copy(update={"payload": {"changed": True}}),
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                )
            with pytest.raises(ValueError, match="different data"):
                await store.load_interaction_transition_receipt(
                    session_id,
                    transition=failed_transition.model_copy(
                        update={"event": failed.model_copy(update={"payload": {"changed": True}})}
                    ),
                )

            receipt_storage_key = (
                "__cayu_interaction_transition_v1__:"
                + sha256(failed.id.encode("utf-8")).hexdigest()
            )
            receipt = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=receipt_storage_key,
            )
            assert receipt is not None
            receipt["session"]["model"] = "drifted-model"
            await _set_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=receipt_storage_key,
                record=receipt,
            )
            with pytest.raises(
                RuntimeError,
                match="Stored interaction-transition receipt is invalid",
            ):
                await store.load_interaction_transition_receipt(
                    session_id,
                    transition=failed_transition,
                )
            with pytest.raises(
                RuntimeError,
                match="Stored interaction-transition receipt is invalid",
            ):
                await store.publish_interaction_transition(
                    session_id,
                    event=failed,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_receipt_lookup_rejects_non_event_transition_conflicts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)

        async def publish_and_reject_conflict(
            *,
            suffix: str,
            event_type: EventType,
            stored_from_statuses: set[SessionStatus],
            stored_to_status: SessionStatus,
            stored_queue_guard: bool,
            conflicting_from_statuses: tuple[SessionStatus, ...],
            conflicting_to_status: SessionStatus,
            conflicting_queue_guard: bool,
        ) -> None:
            session_id = f"sess_interaction_receipt_conflict_{suffix}"
            interaction_id = f"interaction-receipt-conflict-{suffix}"
            started = Event(
                id=f"evt_interaction_receipt_conflict_started_{suffix}",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )
            event = Event(
                id=f"evt_interaction_receipt_conflict_{suffix}",
                type=event_type,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.publish_interaction_transition(
                session_id,
                event=event,
                from_statuses=stored_from_statuses,
                to_status=stored_to_status,
                only_if_no_queued_messages=stored_queue_guard,
            )
            conflicting = InteractionTransitionSpec(
                event=event,
                from_statuses=conflicting_from_statuses,
                to_status=conflicting_to_status,
                only_if_no_queued_messages=conflicting_queue_guard,
            )
            with pytest.raises(ValueError, match="different data"):
                await store.load_interaction_transition_receipt(
                    session_id,
                    transition=conflicting,
                )

        try:
            await publish_and_reject_conflict(
                suffix="source_statuses",
                event_type=EventType.INTERACTION_FAILED,
                stored_from_statuses={SessionStatus.RUNNING},
                stored_to_status=SessionStatus.FAILED,
                stored_queue_guard=False,
                conflicting_from_statuses=(SessionStatus.INTERRUPTING,),
                conflicting_to_status=SessionStatus.FAILED,
                conflicting_queue_guard=False,
            )
            await publish_and_reject_conflict(
                suffix="target_status",
                event_type=EventType.INTERACTION_PAUSED,
                stored_from_statuses={SessionStatus.RUNNING},
                stored_to_status=SessionStatus.FAILED,
                stored_queue_guard=False,
                conflicting_from_statuses=(SessionStatus.RUNNING,),
                conflicting_to_status=SessionStatus.INTERRUPTED,
                conflicting_queue_guard=False,
            )
            await publish_and_reject_conflict(
                suffix="queue_guard",
                event_type=EventType.INTERACTION_COMPLETED,
                stored_from_statuses={SessionStatus.RUNNING},
                stored_to_status=SessionStatus.COMPLETED,
                stored_queue_guard=True,
                conflicting_from_statuses=(SessionStatus.RUNNING,),
                conflicting_to_status=SessionStatus.COMPLETED,
                conflicting_queue_guard=False,
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("lost_acknowledgements", [1, 2])
def test_runtime_conformance_replays_exact_interaction_transition_after_ack_loss(
    session_store_case,
    lost_acknowledgements: int,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = (
            f"interaction-transition-ack-loss-{lost_acknowledgements}-{session_store_case[0]}"
        )
        original_publish = store.publish_interaction_transition
        attempted_events: list[Event] = []
        remaining = lost_acknowledgements
        try:

            async def publish_then_lose_ack(session_id: str, **kwargs):
                nonlocal remaining
                event = kwargs["event"]
                if event.type == EventType.INTERACTION_COMPLETED:
                    attempted_events.append(event.model_copy(deep=True))
                result = await original_publish(session_id, **kwargs)
                if event.type == EventType.INTERACTION_COMPLETED and remaining > 0:
                    remaining -= 1
                    raise ConnectionError("interaction transition acknowledgement lost")
                return result

            store.publish_interaction_transition = publish_then_lose_ack  # type: ignore[method-assign]
            provider = _TransitionReplayConformanceProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))

            emitted = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "complete")],
                    )
                )
            ]

            assert emitted[-1].type == EventType.SESSION_COMPLETED
            assert remaining == 0
            assert len(attempted_events) == lost_acknowledgements + 1
            assert all(event == attempted_events[0] for event in attempted_events)
            assert len(provider.requests) == 1

            store.publish_interaction_transition = original_publish  # type: ignore[method-assign]
            store = await _reopen_store(session_store_case, store)
            session = await store.load(session_id)
            durable = await store.load_events(session_id)
            assert session is not None
            assert session.status is SessionStatus.COMPLETED
            terminal_indexes = [
                index
                for index, event in enumerate(durable)
                if event.type == EventType.INTERACTION_COMPLETED
            ]
            assert len(terminal_indexes) == 1
            assert all(event.interaction_id is None for event in durable[terminal_indexes[0] + 1 :])
            assert sum(event.type == EventType.SESSION_COMPLETED for event in durable) == 1
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interaction_completion_replays_queue_decision(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_interaction_completion_queue_conformance"
            interaction_id = "interaction-completion-queue-conformance"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_interaction_completion_queue_started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[Message.text("user", "start")],
            )
            await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="queued-before-interaction-completion",
                    content="continue",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            completed = Event(
                id="evt_interaction_completion_queue_completed",
                type=EventType.INTERACTION_COMPLETED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            publication = await store.publish_interaction_transition(
                session_id,
                event=completed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=True,
            )
            assert publication.status_changed is False
            assert publication.session.status is SessionStatus.RUNNING
            receipt = await store.load_interaction_transition_receipt(
                session_id,
                transition=InteractionTransitionSpec(
                    event=completed,
                    from_statuses=(SessionStatus.RUNNING,),
                    to_status=SessionStatus.COMPLETED,
                    only_if_no_queued_messages=True,
                ),
            )
            assert receipt is not None
            assert receipt.status_changed is False
            assert receipt.session.status is SessionStatus.RUNNING

            await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                interaction_id="interaction-after-queued-completion",
                interaction_started_event=Event(
                    id="evt_interaction_after_queued_completion_started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id="interaction-after-queued-completion",
                ),
            )
            later_completed = Event(
                id="evt_interaction_after_queued_completion_completed",
                type=EventType.INTERACTION_COMPLETED,
                session_id=session_id,
                interaction_id="interaction-after-queued-completion",
            )
            later_publication = await store.publish_interaction_transition(
                session_id,
                event=later_completed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=True,
            )
            assert later_publication.status_changed is True
            assert later_publication.session.status is SessionStatus.COMPLETED

            replayed = await store.publish_interaction_transition(
                session_id,
                event=completed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=True,
            )
            assert replayed.replayed is True
            assert replayed.status_changed is False
            assert replayed.session == publication.session
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interaction_receipts_do_not_bypass_run_fencing(
    session_store_case,
) -> None:
    async def fence_in_child(
        store: SessionStore,
        session_id: str,
        status: SessionStatus,
    ) -> None:
        fenced = await store.fence_stalled_run(
            session_id,
            statuses={status},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
        assert fenced is not None

    async def run() -> None:
        store = await _open_store(session_store_case)
        transition_session_id = "sess_stale_interaction_transition_replay"
        delivery_session_id = "sess_stale_interaction_delivery_replay"
        try:
            transition_interaction_id = "interaction-stale-transition"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=transition_session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_stale_transition_started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=transition_session_id,
                    interaction_id=transition_interaction_id,
                ),
                interaction_source_messages=[Message.text("user", "start")],
            )
            failed = Event(
                id="evt_stale_transition_failed",
                type=EventType.INTERACTION_FAILED,
                session_id=transition_session_id,
                interaction_id=transition_interaction_id,
            )
            await store.publish_interaction_transition(
                transition_session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
            )
            await asyncio.create_task(
                fence_in_child(store, transition_session_id, SessionStatus.FAILED)
            )
            with pytest.raises(SessionRunFenced):
                await store.publish_interaction_transition(
                    transition_session_id,
                    event=failed,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                )

            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=delivery_session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.transition_status(
                delivery_session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            queued = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=delivery_session_id,
                    idempotency_key="stale-delivery",
                    content="continue",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            delivery_id = "delivery-stale-replay"
            delivered = await store.deliver_queued_session_messages(
                delivery_session_id,
                include_on_idle=False,
                delivery_id=delivery_id,
                eligible_through=queued.message.ordering_key,
                interaction_id="interaction-stale-delivery",
                interaction_started_event=Event(
                    id="evt_stale_delivery_started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=delivery_session_id,
                    interaction_id="interaction-stale-delivery",
                ),
            )
            assert delivered.messages
            await asyncio.create_task(
                fence_in_child(store, delivery_session_id, SessionStatus.RUNNING)
            )
            with pytest.raises(SessionRunFenced):
                await store.deliver_queued_session_messages(
                    delivery_session_id,
                    include_on_idle=False,
                    delivery_id=delivery_id,
                    eligible_through=queued.message.ordering_key,
                    interaction_id="interaction-stale-delivery",
                    interaction_started_event=Event(
                        id="evt_stale_delivery_started",
                        type=EventType.INTERACTION_STARTED,
                        session_id=delivery_session_id,
                        interaction_id="interaction-stale-delivery",
                    ),
                )
        finally:
            await store.release_run_fence(transition_session_id)
            await store.release_run_fence(delivery_session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_invalid_queue_boundary_before_mutation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_invalid_queue_boundary_conformance"
            request = EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="invalid-queue-boundary-message",
                content="must remain queued",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            accepted = await store.enqueue_session_message(request)
            await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            started = Event(
                id="evt_invalid_queue_boundary_started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id="interaction-invalid-queue-boundary",
            )

            invalid_boundaries: tuple[object, ...] = (
                True,
                1.5,
                MAX_DURABLE_JSON_INTEGER + 1,
            )
            for index, invalid_boundary in enumerate(invalid_boundaries):
                with pytest.raises(ValueError, match="eligible_through must be an integer"):
                    await store.deliver_queued_session_messages(
                        session_id,
                        include_on_idle=False,
                        eligible_through=invalid_boundary,  # type: ignore[arg-type]
                        delivery_id=f"invalid-queue-boundary-{index}",
                        interaction_id="interaction-invalid-queue-boundary",
                        interaction_started_event=started,
                    )

            replayed = await store.enqueue_session_message(request)
            assert replayed.replayed is True
            assert replayed.message.queue_id == accepted.message.queue_id
            assert replayed.message.status is SessionMessageQueueStatus.QUEUED
            assert await store.load_transcript(session_id) == []
            lifecycle = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(EventType.INTERACTION_STARTED,),
                )
            )
            assert lifecycle == []
        finally:
            await store.release_run_fence("sess_invalid_queue_boundary_conformance")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_materializes_deferred_interaction_input_fallback(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_deferred_interaction_fallback",
                    messages=[Message.text("user", "bootstrap")],
                ),
                identity=_identity(),
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("assistant", "previous response")],
                interaction_id=None,
            )
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_started_event=Event(
                    id="evt_deferred_interaction_fallback",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session.id,
                    interaction_id="interaction-deferred-fallback",
                ),
                interaction_source_messages=[Message.text("user", "deferred source")],
                defer_interaction_source=True,
            )
            store = await _reopen_store(session_store_case, store)

            with pytest.raises(
                RuntimeError,
                match="Deferred interaction input belongs to another interaction",
            ):
                await store.materialize_deferred_interaction_input(
                    session.id,
                    interaction_id="interaction-wrong",
                )
            before = await store.query_transcript(TranscriptQuery(session_id=session.id, limit=10))
            assert [record.interaction_id for record in before.records] == [None]

            assert await store.materialize_deferred_interaction_input(
                session.id,
                interaction_id="interaction-deferred-fallback",
            )
            store = await _reopen_store(session_store_case, store)
            after = await store.query_transcript(TranscriptQuery(session_id=session.id, limit=10))
            assert [record.interaction_id for record in after.records] == [
                None,
                "interaction-deferred-fallback",
            ]
            assert [record.message for record in after.records] == [
                Message.text("assistant", "previous response"),
                Message.text("user", "deferred source"),
            ]
            assert not await store.materialize_deferred_interaction_input(
                session.id,
                interaction_id="interaction-deferred-fallback",
            )
        finally:
            await store.release_run_fence("sess_deferred_interaction_fallback")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_paginates_latest_interaction_states(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interaction_pagination",
                    messages=[Message.text("user", "bootstrap")],
                ),
                identity=_identity(),
            )
            await store.append_events(
                session.id,
                [
                    Event(
                        id="interaction_1_started",
                        type=EventType.INTERACTION_STARTED,
                        session_id=session.id,
                        interaction_id="interaction-1",
                    ),
                    Event(
                        id="interaction_2_started",
                        type=EventType.INTERACTION_STARTED,
                        session_id=session.id,
                        interaction_id="interaction-2",
                    ),
                    Event(
                        id="interaction_1_paused",
                        type=EventType.INTERACTION_PAUSED,
                        session_id=session.id,
                        interaction_id="interaction-1",
                    ),
                    Event(
                        id="interaction_3_started",
                        type=EventType.INTERACTION_STARTED,
                        session_id=session.id,
                        interaction_id="interaction-3",
                    ),
                    Event(
                        id="interaction_2_completed",
                        type=EventType.INTERACTION_COMPLETED,
                        session_id=session.id,
                        interaction_id="interaction-2",
                    ),
                ],
            )

            first_page = await store.query_latest_interaction_events(session.id, limit=2)
            assert [record.event.id for record in first_page] == [
                "interaction_2_completed",
                "interaction_3_started",
            ]
            second_page = await store.query_latest_interaction_events(
                session.id,
                before_sequence=first_page[-1].sequence,
                limit=2,
            )
            assert [record.event.id for record in second_page] == ["interaction_1_paused"]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_pages_hundreds_of_interactions(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interaction_scale",
                    messages=[Message.text("user", "bootstrap")],
                ),
                identity=_identity(),
            )
            interaction_count = 250
            await store.append_events(
                session.id,
                [
                    Event(
                        id=f"interaction_{index}_completed",
                        type=EventType.INTERACTION_COMPLETED,
                        session_id=session.id,
                        interaction_id=f"interaction-{index}",
                    )
                    for index in range(interaction_count)
                ],
            )

            observed: list[str] = []
            before_sequence = None
            while True:
                page = await store.query_latest_interaction_events(
                    session.id,
                    before_sequence=before_sequence,
                    limit=37,
                )
                if not page:
                    break
                observed.extend(record.event.interaction_id or "" for record in page)
                before_sequence = page[-1].sequence

            assert observed == [
                f"interaction-{index}" for index in reversed(range(interaction_count))
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_distinguishes_inherited_and_explicit_null_transcript_ids(
    session_store_case,
) -> None:
    from cayu.runtime.sessions import (
        _activate_session_interaction,
        _deactivate_session_interaction,
    )

    async def run() -> None:
        store = await _open_store(session_store_case)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_transcript_attribution_modes",
                messages=[Message.text("user", "bootstrap")],
            ),
            identity=_identity(),
        )
        try:
            _activate_session_interaction(session.id, "interaction-active")
            await store.append_transcript_messages(
                session.id,
                [Message.text("user", "inherit")],
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("assistant", "unassociated")],
                interaction_id=None,
            )
            page = await store.query_transcript(TranscriptQuery(session_id=session.id))
            assert [record.interaction_id for record in page.records] == [
                "interaction-active",
                None,
            ]
        finally:
            _deactivate_session_interaction(session.id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_validates_event_batch_preamble(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_event_preamble",
                    messages=[Message.text("user", "events")],
                ),
                identity=_identity(),
            )
            append_events: Any = session_store.append_events

            with pytest.raises(TypeError, match="Session events must be a list."):
                await append_events("sess_event_preamble", ())
            with pytest.raises(TypeError, match="Session events must be Event instances."):
                await append_events("sess_event_preamble", ["not-an-event"])
            with pytest.raises(ValueError, match="Event session_id does not match target session."):
                await session_store.append_events(
                    "sess_event_preamble",
                    [
                        Event(
                            id="evt_wrong_session",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_other",
                        )
                    ],
                )
            with pytest.raises(ValueError, match="Event already exists for session"):
                await session_store.append_events(
                    "sess_event_preamble",
                    [
                        Event(
                            id="evt_duplicate",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_event_preamble",
                        ),
                        Event(
                            id="evt_duplicate",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_event_preamble",
                        ),
                    ],
                )
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_validates_fork_request_preamble(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_fork_source",
                    messages=[Message.text("user", "fork")],
                ),
                identity=_identity(),
            )

            with pytest.raises(ValueError, match="Fork parent_session_id must match"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_parent",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id="sess_other",
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="transcript_cursor must be greater"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_cursor",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=-1,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Source session status is not forkable"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_status_source",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.COMPLETED},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Fork status must match source session status"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_status_fork",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.COMPLETED,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Fork provider_name must match"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_provider",
                        agent_name="assistant",
                        provider_name="other",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        invocation=fork_session_invocation(source),
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    expected_source_run_epoch=source.run_epoch,
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_validates_exact_fork_transcript_atomically(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_validated_fork_source",
                    messages=[Message.text("user", "fork")],
                ),
                identity=_identity(),
            )
            await session_store.append_transcript_messages(
                source.id,
                [Message.text("user", "copied-prefix")],
                interaction_id="source-interaction-copied",
            )
            await session_store.append_transcript_messages(
                source.id,
                [Message.text("user", "excluded-suffix")],
                interaction_id="source-interaction-excluded",
            )
            source = await session_store.update_status(
                source.id,
                SessionStatus.COMPLETED,
            )
            observed: list[tuple[Message, ...]] = []

            fork = await session_store.create_fork_with_transcript_validation(
                source_session_id=source.id,
                fork=Session(
                    id="sess_validated_fork_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=1,
                checkpoint_transform=None,
                transcript_validator=lambda messages: not observed.append(messages),
            )
            assert fork.id == "sess_validated_fork_child"
            assert len(observed) == 1
            assert [message.content[0].text for message in observed[0]] == ["copied-prefix"]
            assert [
                message.content[0].text for message in await session_store.load_transcript(fork.id)
            ] == ["copied-prefix"]
            fork_transcript = await session_store.query_transcript(
                TranscriptQuery(session_id=fork.id, limit=10)
            )
            assert [record.interaction_id for record in fork_transcript.records] == [
                "source-interaction-copied"
            ]

            mutation_source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_validated_fork_mutation_source",
                    messages=[],
                ),
                identity=_identity(),
            )
            await session_store.append_transcript_messages(
                mutation_source.id,
                [
                    Message.tool_call(
                        tool_call_id="call_validator_mutation",
                        tool_name="validator_mutation",
                        arguments={"nested": {"value": "original"}},
                    )
                ],
            )
            mutation_source = await session_store.update_status(
                mutation_source.id,
                SessionStatus.COMPLETED,
            )

            def mutate_validation_projection(messages: tuple[Message, ...]) -> bool:
                part = messages[0].content[0]
                assert isinstance(part, ToolCallPart)
                part.arguments["nested"]["value"] = "mutated"
                return True

            await session_store.create_fork_with_transcript_validation(
                source_session_id=mutation_source.id,
                fork=Session(
                    id="sess_validated_fork_mutation_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=mutation_source.id,
                    invocation=fork_session_invocation(mutation_source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=mutation_source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
                transcript_validator=mutate_validation_projection,
            )
            for session_id in (
                mutation_source.id,
                "sess_validated_fork_mutation_child",
            ):
                transcript = await session_store.load_transcript(session_id)
                part = transcript[0].content[0]
                assert isinstance(part, ToolCallPart)
                assert part.arguments == {"nested": {"value": "original"}}

            for child_id, invalid_result in (
                ("sess_rejected_fork_false", False),
                ("sess_rejected_fork_ambiguous", 1),
            ):
                with pytest.raises(ValueError, match="atomic copy validation"):
                    await session_store.create_fork_with_transcript_validation(
                        source_session_id=source.id,
                        fork=Session(
                            id=child_id,
                            agent_name="assistant",
                            provider_name="fake",
                            model="fake-model",
                            parent_session_id=source.id,
                            invocation=fork_session_invocation(source),
                            status=SessionStatus.COMPLETED,
                        ),
                        source_statuses={SessionStatus.COMPLETED},
                        expected_source_run_epoch=source.run_epoch,
                        transcript_cursor=None,
                        checkpoint_transform=None,
                        transcript_validator=lambda _messages, result=invalid_result: result,
                    )
                assert await session_store.load(child_id) is None
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_replaces_fork_system_prompt_atomically(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_prompt_replacement_source",
                    messages=[],
                ),
                identity=_identity(),
            )
            await session_store.append_transcript_messages(
                source.id,
                [Message.text("system", "source prompt")],
            )
            await session_store.append_transcript_messages(
                source.id,
                [Message.text("user", "inherited question")],
                interaction_id="source-prompt-interaction",
            )
            source = await session_store.update_status(source.id, SessionStatus.COMPLETED)
            observed: list[tuple[Message, ...]] = []

            child = await session_store.create_fork_with_transcript_validation(
                source_session_id=source.id,
                fork=Session(
                    id="sess_prompt_replacement_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
                system_prompt_replacement=ForkSystemPromptReplacement(
                    Message.text("system", "child prompt")
                ),
                transcript_validator=lambda messages: not observed.append(messages),
            )

            source_transcript = await session_store.load_transcript(source.id)
            child_transcript = await session_store.load_transcript(child.id)
            child_page = await session_store.query_transcript(
                TranscriptQuery(session_id=child.id, limit=10)
            )
            assert [message.content[0].text for message in source_transcript] == [
                "source prompt",
                "inherited question",
            ]
            assert [message.content[0].text for message in child_transcript] == [
                "child prompt",
                "inherited question",
            ]
            assert observed == [tuple(child_transcript)]
            assert [record.interaction_id for record in child_page.records] == [
                None,
                "source-prompt-interaction",
            ]
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_rejects_unsafe_derived_fork_before_mutation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"derived_fork_source_{store_kind}"
        child_id = f"derived_fork_child_{store_kind}"
        secret = "derived-store-fork-secret"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="source-agent",
                    session_id=source_id,
                    messages=[Message.text("user", "fork")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="source-model",
                ),
            )
            await store.append_transcript_messages(
                source.id,
                [Message.text("user", "copied transcript")],
            )
            await store.checkpoint(source.id, {"safe": "checkpoint"})
            await store.update_status(source.id, SessionStatus.COMPLETED)
            source_before = await store.load(source.id)
            transcript_before = await store.load_transcript(source.id)
            checkpoint_before = await store.load_checkpoint(source.id)
            events_before = await store.load_events(source.id)

            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="source-agent", model="source-model"))
            app.register_agent(AgentSpec(name="target-agent", model=secret))

            with pytest.raises(ValueError, match="model"):
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=child_id,
                            agent_name="target-agent",
                            execution_profile_selection=(
                                ForkExecutionProfileSelection.CURRENT_CHILD
                            ),
                            profile_adoption=ExecutionProfileAdoptionIntent(
                                idempotency_key=f"derived-fork-{store_kind}",
                                reason="Exercise derived fork authority validation.",
                                requested_by=ResolutionActor(
                                    subject="test-caller",
                                    source=ResolutionActorSource.REQUEST,
                                ),
                            ),
                        )
                    )
                ]

            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_transcript(child_id)
            assert await store.load_checkpoint(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert await store.load(source.id) == source_before
            assert await store.load_transcript(source.id) == transcript_before
            assert await store.load_checkpoint(source.id) == checkpoint_before
            assert await store.load_events(source.id) == events_before
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_same_model_cross_provider_preflight_is_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"portable_fork_source_{store_kind}_{uuid4().hex}"
        child_id = f"portable_fork_child_{store_kind}_{uuid4().hex}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="source-agent",
                    session_id=source_id,
                    messages=[Message.text("user", "fork source")],
                ),
                identity=profiled_session_identity(
                    provider_name="source-provider",
                    model="shared-model",
                ),
            )
            await store.append_transcript_messages(
                source.id,
                [Message.text("user", "portable history")],
            )
            source = await store.update_status(source.id, SessionStatus.COMPLETED)
            source_before = await store.load(source.id)
            transcript_before = await store.load_transcript(source.id)
            events_before = await store.load_events(source.id)

            target_provider = _RejectingPortableForkProvider("target-provider")
            app = CayuApp(
                session_store=store,
                execution_profile_policy=_AuthorizeForkProfilePolicy(),
                enable_logging=False,
            )
            app.register_provider(_UnusedNamedForkProvider("source-provider"), default=True)
            app.register_provider(target_provider)
            app.register_agent(
                AgentSpec(
                    name="source-agent",
                    provider_name="source-provider",
                    model="shared-model",
                )
            )
            app.register_agent(
                AgentSpec(
                    name="target-agent",
                    provider_name="target-provider",
                    model="shared-model",
                )
            )

            with pytest.raises(ValueError, match="target provider rejected source history"):
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=child_id,
                            agent_name="target-agent",
                            execution_profile_selection=(
                                ForkExecutionProfileSelection.CURRENT_CHILD
                            ),
                            profile_adoption=ExecutionProfileAdoptionIntent(
                                idempotency_key=f"portable-fork-{store_kind}",
                                reason="Authorize the target provider.",
                                requested_by=ResolutionActor(
                                    subject="test-caller",
                                    source=ResolutionActorSource.REQUEST,
                                ),
                            ),
                        )
                    )
                ]

            assert len(target_provider.preflight_calls) == 1
            model, messages, tools = target_provider.preflight_calls[0]
            assert model == "shared-model"
            assert list(messages) == transcript_before
            assert tools == ()
            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert await store.load(source.id) == source_before
            assert await store.load_transcript(source.id) == transcript_before
            assert await store.load_events(source.id) == events_before
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_equal_current_child_authorization_is_replayable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"equal_profile_fork_source_{store_kind}_{uuid4().hex}"
        child_id = f"equal_profile_fork_child_{store_kind}_{uuid4().hex}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "fork source")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            source = await store.update_status(source.id, SessionStatus.COMPLETED)
            app = CayuApp(
                session_store=store,
                execution_profile_policy=_AuthorizeForkProfilePolicy(),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            request = ForkSessionRequest(
                source_session_id=source.id,
                session_id=child_id,
                execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key=f"equal-profile-fork-{store_kind}",
                    reason="Authorize decision-bearing current registrations.",
                    requested_by=ResolutionActor(
                        subject="test-caller",
                        source=ResolutionActorSource.REQUEST,
                    ),
                ),
            )

            async def collect_fork() -> list[Event]:
                return [event async for event in app.fork_session(request)]

            first, replay = await asyncio.gather(collect_fork(), collect_fork())

            assert [event.type for event in first] == [EventType.SESSION_FORKED]
            assert [event.id for event in replay] == [event.id for event in first]
            child = await store.load(child_id)
            assert child is not None
            relationship = sessions_module.session_fork_profile_relationship(child)
            assert relationship is not None
            assert relationship.selected_profile == relationship.source_profile
            assert relationship.decision is not None
            assert relationship.decision.kind is ExecutionProfileDecisionKind.ADOPTED
            assert (
                relationship.decision.authority_decision
                is ExecutionProfileAuthorityDecision.AUTHORIZED
            )
            records = await store.query_events(EventQuery(session_id=child_id, limit=10))
            assert [record.event.type for record in records] == [
                EventType.SESSION_EXECUTION_PROFILE_DECIDED,
                EventType.SESSION_FORKED,
            ]
            assert records[-1].event.id == relationship.fork_event_id
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("profile_mode", ["inherit", "current-model", "current-provider"])
def test_session_store_conformance_profiled_fork_is_atomic_and_exactly_replayable(
    session_store_case,
    profile_mode: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"profiled_fork_source_{profile_mode}_{store_kind}_{uuid4().hex}"
        child_id = f"profiled_fork_child_{profile_mode}_{store_kind}_{uuid4().hex}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "fork source")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            source = await store.update_status(source.id, SessionStatus.COMPLETED)
            app = CayuApp(
                session_store=store,
                execution_profile_policy=_AuthorizeForkProfilePolicy(),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            if profile_mode == "current-provider":
                app.register_provider(_UnusedNamedForkProvider("child-provider"))
                app.register_agent(
                    AgentSpec(
                        name="child-assistant",
                        provider_name="child-provider",
                        model="child-model",
                    )
                )
            request = (
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id=child_id,
                )
                if profile_mode == "inherit"
                else ForkSessionRequest(
                    source_session_id=source.id,
                    session_id=child_id,
                    agent_name=("child-assistant" if profile_mode == "current-provider" else None),
                    model=("child-model" if profile_mode == "current-model" else None),
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key=f"profiled-fork-{profile_mode}-{store_kind}",
                        reason="Select the current child model profile.",
                        requested_by=ResolutionActor(
                            subject="test-caller",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )

            async def collect_fork() -> list[Event]:
                return [event async for event in app.fork_session(request)]

            first, replay = await asyncio.gather(
                collect_fork(),
                collect_fork(),
            )

            assert [event.type for event in first] == [EventType.SESSION_FORKED]
            assert [event.id for event in replay] == [event.id for event in first]
            child = await store.load(child_id)
            assert child is not None
            relationship = sessions_module.session_fork_profile_relationship(child)
            assert relationship is not None
            assert relationship.source_session_id == source.id
            if profile_mode == "inherit":
                assert relationship.selected_profile == relationship.source_profile
                expected_event_types = [EventType.SESSION_FORKED]
            else:
                assert relationship.selected_profile != relationship.source_profile
                assert relationship.decision is not None
                if profile_mode == "current-provider":
                    assert child.provider_name == "child-provider"
                expected_event_types = [
                    EventType.SESSION_EXECUTION_PROFILE_DECIDED,
                    EventType.SESSION_FORKED,
                ]
            records = await store.query_events(EventQuery(session_id=child_id, limit=10))
            assert [record.event.type for record in records] == expected_event_types
            assert records[-1].event.id == relationship.fork_event_id
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_generated_profiled_fork_is_idempotent(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        source_id = f"generated_profiled_fork_source_{store_kind}_{uuid4().hex}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "fork source")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            source = await store.update_status(source.id, SessionStatus.COMPLETED)
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            request = ForkSessionRequest(source_session_id=source.id)

            async def collect_fork() -> list[Event]:
                return [event async for event in app.fork_session(request)]

            first, replay = await asyncio.gather(collect_fork(), collect_fork())

            assert [event.type for event in first] == [EventType.SESSION_FORKED]
            assert [event.id for event in replay] == [event.id for event in first]
            children = (
                await store.list_sessions(SessionQuery(parent_session_id=source.id, limit=10))
            ).sessions
            assert len(children) == 1
            relationship = sessions_module.session_fork_profile_relationship(children[0])
            assert relationship is not None
            assert relationship.source_session_id == source.id
            records = await store.query_events(EventQuery(session_id=children[0].id, limit=10))
            assert [record.event.type for record in records] == [EventType.SESSION_FORKED]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_hook_fork_preserves_generated_provenance(
    session_store_case,
) -> None:
    class ForkCompletedSessionHook(RuntimeHook):
        def __init__(self) -> None:
            self.calls = 0
            self.fork_events: list[Event] = []

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls += 1
            self.fork_events = await context.fork_session(
                ForkSessionRequest(source_session_id=context.session.id)
            )

    async def run() -> None:
        store = await _open_store(session_store_case)
        hook = ForkCompletedSessionHook()
        try:
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor("-"),
                runtime_hooks=[hook],
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[Message.text("user", "start")],
                    )
                )
            ]

            assert hook.calls == 1
            assert [event.type for event in hook.fork_events] == [EventType.SESSION_FORKED]
            assert [event.type for event in events].count(EventType.SESSION_COMPLETED) == 1
            assert [event.type for event in events].count(EventType.HOOK_STARTED) == 1
            assert [event.type for event in events].count(EventType.HOOK_COMPLETED) == 1
            sessions = (await store.list_sessions(SessionQuery(limit=10))).sessions
            assert len(sessions) == 2
            source = next(session for session in sessions if session.parent_session_id is None)
            child = next(session for session in sessions if session.parent_session_id == source.id)
            assert "-" in source.id
            assert "-" in child.id
            source_events = await store.load_events(source.id)
            assert [event.type for event in source_events].count(EventType.SESSION_COMPLETED) == 1
            assert [event.type for event in source_events].count(EventType.HOOK_STARTED) == 1
            assert [event.type for event in source_events].count(EventType.HOOK_COMPLETED) == 1
            child_events = await store.load_events(child.id)
            assert [event.type for event in child_events] == [EventType.SESSION_FORKED]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fork_source_provenance_distinguishes_callers(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        secret = "private-fork-source"
        source_id = f"legacy-{secret}-session-{store_kind}"
        raw_child_id = f"raw-fork-child-{store_kind}"
        alias_child_id = f"alias-fork-child-{store_kind}"
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "source")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fakemodel",
                ),
            )
            await store.update_status(source.id, SessionStatus.COMPLETED)
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

            with pytest.raises(ValueError, match="source_session_id"):
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=raw_child_id,
                        )
                    )
                ]
            assert await store.load(raw_child_id) is None

            public_source_id = app.project_session_id_for_exposure(source.id)
            await store.register_public_authority_alias(
                public_source_id,
                field_name="session_id",
                private_value=source.id,
            )
            alias_events = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=public_source_id,
                        session_id=alias_child_id,
                    )
                )
            ]

            assert [event.type for event in alias_events] == [EventType.SESSION_FORKED]
            child = await store.load(alias_child_id)
            assert child is not None
            assert child.parent_session_id == source.id
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("source_kind", ("generated", "legacy-secret"))
def test_session_store_conformance_stale_fork_alias_omits_private_authority(
    session_store_case,
    source_kind: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        secret = "stale-fork-private-secret" if source_kind == "legacy-secret" else "-"
        requested_source_id = (
            f"legacy-{secret}-session-{store_kind}" if source_kind == "legacy-secret" else None
        )
        child_id = f"stale_fork_child_{source_kind.replace('-', '_')}_{store_kind}"
        sink = InMemoryEventSink()
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=requested_source_id,
                    messages=[Message.text("user", "source")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fakemodel"),
            )
            await store.append_transcript_messages(
                source.id,
                [Message.text("user", "copied transcript")],
            )
            await store.checkpoint(source.id, {"safe": "checkpoint"})
            await store.update_status(source.id, SessionStatus.COMPLETED)
            await store.append_event(
                source.id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=source.id,
                ),
            )

            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
            public_source_id = app.project_session_id_for_exposure(source.id)
            await store.register_public_authority_alias(
                public_source_id,
                field_name="session_id",
                private_value=source.id,
            )

            await store.delete_session(source.id)
            assert await store.load(source.id) is None
            assert (
                await store.resolve_public_authority_alias(
                    public_source_id,
                    field_name="session_id",
                )
                == source.id
            )

            with pytest.raises(KeyError) as raised:
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=public_source_id,
                            session_id=child_id,
                        )
                    )
                ]

            assert raised.value.args == ("Fork source session was not found.",)
            private_values = (source.id, secret) if len(secret) > 1 else (source.id,)
            _assert_exception_omits_private_values(raised.value, *private_values)
            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_transcript(child_id)
            assert await store.load_checkpoint(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert sink.events == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fork_source_deleted_after_initial_load_omits_private_authority(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        secret = "fork-race-private-secret"
        source_id = f"legacy-{secret}-session-{store_kind}"
        child_id = f"fork_race_child_{store_kind}"
        sink = InMemoryEventSink()
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "source")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fakemodel",
                ),
            )
            await store.update_status(source.id, SessionStatus.COMPLETED)
            await store.append_event(
                source.id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=source.id,
                ),
            )

            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
            public_source_id = app.project_session_id_for_exposure(source.id)
            await store.register_public_authority_alias(
                public_source_id,
                field_name="session_id",
                private_value=source.id,
            )

            original_create_fork = store.create_profiled_fork
            deleted_after_initial_load = False

            async def delete_source_then_create_fork(**kwargs):
                nonlocal deleted_after_initial_load
                deleted_after_initial_load = True
                await store.delete_session(source.id)
                return await original_create_fork(**kwargs)

            monkeypatch.setattr(
                store,
                "create_profiled_fork",
                delete_source_then_create_fork,
            )

            with pytest.raises(KeyError) as raised:
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=public_source_id,
                            session_id=child_id,
                        )
                    )
                ]

            assert deleted_after_initial_load is True
            assert raised.value.args == ("Fork source session was not found.",)
            _assert_exception_omits_private_values(raised.value, source.id, secret)
            assert await store.load(source.id) is None
            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_transcript(child_id)
            assert await store.load_checkpoint(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert sink.events == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_active_stage_fork_omits_private_source_authority(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        secret = "active-stage-fork-private-secret"
        source_id = f"legacy-{secret}-session-{store_kind}"
        child_id = f"active_stage_fork_child_{store_kind}"
        sink = InMemoryEventSink()
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "source")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fakemodel",
                ),
            )
            await store.append_transcript_messages(
                source.id,
                [Message.text("user", "source")],
            )
            running = await store.transition_status(
                source.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                source.id,
                request=ModelCompletionStageRequest(
                    stage_id="active-stage-fork-stage",
                    logical_step_id="active-stage-fork-step",
                    dispatch_ordinal=0,
                    intent={"request_fingerprint": "active-stage-fork"},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=1,
            )
            source = await store.update_status(source.id, SessionStatus.INTERRUPTED)
            await store.append_event(
                source.id,
                Event(type=EventType.SESSION_INTERRUPTED, session_id=source.id),
            )

            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
            public_source_id = app.project_session_id_for_exposure(source.id)
            await store.register_public_authority_alias(
                public_source_id,
                field_name="session_id",
                private_value=source.id,
            )

            with pytest.raises(ValueError) as raised:
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=public_source_id,
                            session_id=child_id,
                        )
                    )
                ]

            assert raised.value.args == (
                "Fork source session has an active model-completion stage.",
            )
            assert raised.value.__cause__ is None
            assert raised.value.__context__ is None
            _assert_exception_omits_private_values(raised.value, source.id, secret)
            assert await store.load(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_transcript(child_id)
            assert await store.load_checkpoint(child_id) is None
            with pytest.raises(KeyError, match=child_id):
                await store.load_events(child_id)
            assert sink.events == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("source_status", "copy_checkpoint"),
    (
        pytest.param(SessionStatus.INTERRUPTED, True, id="copy-checkpoint"),
        pytest.param(SessionStatus.FAILED, False, id="discard-checkpoint"),
    ),
)
def test_session_store_conformance_rejects_fork_with_pending_tool_round(
    session_store_case,
    source_status: SessionStatus,
    copy_checkpoint: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        mode = "copy" if copy_checkpoint else "discard"
        source_id = f"sess_pending_round_fork_source_{store_kind}_{mode}"
        child_id = f"sess_pending_round_fork_child_{store_kind}_{mode}"
        identity = {
            "model_step_id": f"mstep_{'1' * 32}",
            "model_attempt_id": f"matt_{'2' * 32}",
            "tool_round_id": f"tround_{'3' * 32}",
        }
        checkpoint = {
            "pending_tool_round": {
                **identity,
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        "tool_call_id": "call_external_effect",
                        "tool_name": "external_effect",
                        "arguments": {},
                    }
                ],
            }
        }
        try:
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "perform the effect")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            await store.append_event(
                source.id,
                Event(
                    id=f"evt_pending_round_started_{store_kind}",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=source.id,
                    tool_name="external_effect",
                    payload={
                        **identity,
                        "tool_call_id": "call_external_effect",
                    },
                ),
            )
            await store.checkpoint(source.id, checkpoint)
            await store.update_status(source.id, source_status)
            source_events = await store.load_events(source.id)

            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))

            with pytest.raises(RuntimeError, match="pending tool round cannot be forked"):
                [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=source.id,
                            session_id=child_id,
                            copy_checkpoint=copy_checkpoint,
                        )
                    )
                ]

            assert await store.load(child_id) is None
            assert await store.load_checkpoint(source.id) == checkpoint
            assert await store.load_events(source.id) == source_events
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_atomically_fences_mcp_manifest_baselines(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            sessions = []
            for index in range(2):
                sessions.append(
                    await store.create(
                        RunRequest(
                            agent_name="assistant",
                            session_id=f"mcp_manifest_cas_{index}",
                            messages=[Message.text("user", "check manifest")],
                        ),
                        identity=_identity(),
                    )
                )

            history_key = "sha256:" + "1" * 64
            publications = []
            for index, session in enumerate(sessions):
                event = Event(
                    id=f"evt_mcp_manifest_cas_{index}",
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    payload={
                        "history_key": history_key,
                        "manifest_identity": "sha256:" + "2" * 64,
                        "manifest_hash": _mcp_test_manifest_hash(
                            source_manifest_hash="sha256:" + "6" * 64,
                            server_hash=f"sha256:{index + 5}" + "0" * 63,
                        ),
                        "source_manifest_hash": "sha256:" + "6" * 64,
                        "server_hash": f"sha256:{index + 5}" + "0" * 63,
                        "status": "first_seen",
                        "outcome": "accepted",
                    },
                )
                baseline = McpManifestBaseline(
                    history_key=history_key,
                    generation=1,
                    manifest_identity=event.payload["manifest_identity"],
                    manifest_hash=event.payload["manifest_hash"],
                    source_manifest_hash=event.payload["source_manifest_hash"],
                    server_hash=event.payload["server_hash"],
                    tools=(),
                    exposed_tools=(),
                    accepted_session_ref=_mcp_manifest_session_ref(session.id),
                    accepted_event_id=event.id,
                    accepted_at=event.timestamp,
                )
                publications.append(
                    store.compare_and_publish_mcp_manifest_checks(
                        session.id,
                        expected_generations={history_key: None},
                        baseline_updates={history_key: baseline},
                        events=[event],
                    )
                )

            results = await asyncio.gather(*publications)
            assert [result.published for result in results].count(True) == 1
            assert [result.published for result in results].count(False) == 1

            loaded = await store.load_mcp_manifest_baselines((history_key,))
            baselines = loaded.baselines
            assert baselines[history_key].generation == 1
            accepted_events = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert len(accepted_events) == 1
            assert baselines[history_key].accepted_event_id == accepted_events[0].event.id
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rolls_back_mcp_baseline_when_event_insert_fails(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_manifest_atomic_failure",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            duplicate_event_id = "evt_mcp_manifest_duplicate"
            await store.append_event(
                session.id,
                Event(
                    id=duplicate_event_id,
                    type=EventType.SESSION_STARTED,
                    session_id=session.id,
                ),
            )
            history_key = "sha256:" + "6" * 64
            candidate = Event(
                id=duplicate_event_id,
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "7" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash="sha256:" + "a" * 64,
                        server_hash="sha256:" + "9" * 64,
                    ),
                    "source_manifest_hash": "sha256:" + "a" * 64,
                    "server_hash": "sha256:" + "9" * 64,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=candidate.payload["manifest_identity"],
                manifest_hash=candidate.payload["manifest_hash"],
                source_manifest_hash=candidate.payload["source_manifest_hash"],
                server_hash=candidate.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=candidate.id,
                accepted_at=candidate.timestamp,
            )

            with pytest.raises(ValueError, match="Event already exists"):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: baseline},
                    events=[candidate],
                )

            loaded = await store.load_mcp_manifest_baselines((history_key,))
            assert loaded.baselines == {}
            manifest_events = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert manifest_events == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_revalidates_constructed_mcp_baselines(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_manifest_constructed_baseline",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "a" * 64
            event = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "b" * 64,
                    "manifest_hash": "sha256:" + "c" * 64,
                    "source_manifest_hash": "sha256:" + "e" * 64,
                    "server_hash": "sha256:" + "d" * 64,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            invalid = McpManifestBaseline.model_construct(
                history_key=history_key,
                generation=1,
                manifest_identity="not-a-hash",
                manifest_hash=event.payload["manifest_hash"],
                source_manifest_hash=event.payload["source_manifest_hash"],
                server_hash=event.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=event.id,
                accepted_at=event.timestamp,
            )

            with pytest.raises(ValueError, match="SHA-256"):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: invalid},
                    events=[event],
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("status", ["first_seen", "changed"])
def test_session_store_conformance_rejects_accepted_mcp_transition_without_baseline(
    session_store_case,
    status: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"mcp_manifest_missing_baseline_{status}",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "1" * 64
            event = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "2" * 64,
                    "manifest_hash": "sha256:" + "3" * 64,
                    "source_manifest_hash": "sha256:" + "5" * 64,
                    "server_hash": "sha256:" + "4" * 64,
                    "status": status,
                    "outcome": "accepted",
                },
            )

            with pytest.raises(
                ValueError,
                match="Every accepted MCP manifest transition",
            ):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={},
                    events=[event],
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_partially_updated_mcp_transition_batch(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_manifest_partial_baseline_batch",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_keys = ("sha256:" + "5" * 64, "sha256:" + "6" * 64)
            events = [
                Event(
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    payload={
                        "history_key": history_key,
                        "manifest_identity": "sha256:" + ("7" if index == 0 else "8") * 64,
                        "manifest_hash": _mcp_test_manifest_hash(
                            source_manifest_hash="sha256:" + "c" * 64,
                            server_hash="sha256:" + "b" * 64,
                        ),
                        "source_manifest_hash": "sha256:" + "c" * 64,
                        "server_hash": "sha256:" + "b" * 64,
                        "status": "first_seen",
                        "outcome": "accepted",
                    },
                )
                for index, history_key in enumerate(history_keys)
            ]
            first = events[0]
            baseline = McpManifestBaseline(
                history_key=history_keys[0],
                generation=1,
                manifest_identity=first.payload["manifest_identity"],
                manifest_hash=first.payload["manifest_hash"],
                source_manifest_hash=first.payload["source_manifest_hash"],
                server_hash=first.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=first.id,
                accepted_at=first.timestamp,
            )

            with pytest.raises(
                ValueError,
                match="Every accepted MCP manifest transition",
            ):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={key: None for key in history_keys},
                    baseline_updates={history_keys[0]: baseline},
                    events=events,
                )

            assert (await store.load_mcp_manifest_baselines(history_keys)).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_duplicate_accepted_mcp_transitions(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_manifest_duplicate_accepted_transitions",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "c" * 64
            events = [
                Event(
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    payload={
                        "history_key": history_key,
                        "manifest_identity": "sha256:" + "d" * 64,
                        "manifest_hash": _mcp_test_manifest_hash(
                            source_manifest_hash="sha256:" + "2" * 64,
                            server_hash="sha256:" + "1" * 64,
                        ),
                        "source_manifest_hash": "sha256:" + "2" * 64,
                        "server_hash": "sha256:" + "1" * 64,
                        "status": "first_seen",
                        "outcome": "accepted",
                    },
                )
                for index in range(2)
            ]
            first = events[0]
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=first.payload["manifest_identity"],
                manifest_hash=first.payload["manifest_hash"],
                source_manifest_hash=first.payload["source_manifest_hash"],
                server_hash=first.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=first.id,
                accepted_at=first.timestamp,
            )

            with pytest.raises(
                ValueError,
                match="Every accepted MCP manifest transition",
            ):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: baseline},
                    events=events,
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_accepts_unchanged_mcp_event_without_update(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            sessions = [
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=f"mcp_manifest_unchanged_publication_{index}",
                        messages=[Message.text("user", "check manifest")],
                    ),
                    identity=_identity(),
                )
                for index in range(2)
            ]
            history_key = "sha256:" + "2" * 64
            accepted = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=sessions[0].id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "3" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash="sha256:" + "6" * 64,
                        server_hash="sha256:" + "5" * 64,
                    ),
                    "source_manifest_hash": "sha256:" + "6" * 64,
                    "server_hash": "sha256:" + "5" * 64,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=accepted.payload["manifest_identity"],
                manifest_hash=accepted.payload["manifest_hash"],
                source_manifest_hash=accepted.payload["source_manifest_hash"],
                server_hash=accepted.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(sessions[0].id),
                accepted_event_id=accepted.id,
                accepted_at=accepted.timestamp,
            )
            first_result = await store.compare_and_publish_mcp_manifest_checks(
                sessions[0].id,
                expected_generations={history_key: None},
                baseline_updates={history_key: baseline},
                events=[accepted],
            )
            assert first_result.published is True

            unchanged = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=sessions[1].id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": baseline.manifest_identity,
                    "manifest_hash": baseline.manifest_hash,
                    "source_manifest_hash": baseline.source_manifest_hash,
                    "server_hash": baseline.server_hash,
                    "status": "unchanged",
                    "outcome": "accepted",
                },
            )
            second_result = await store.compare_and_publish_mcp_manifest_checks(
                sessions[1].id,
                expected_generations={history_key: 1},
                baseline_updates={},
                events=[unchanged],
            )

            assert second_result.published is True
            assert second_result.baselines == {history_key: baseline}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert [record.event.id for record in records] == [accepted.id, unchanged.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mismatch",
    [
        "manifest_identity",
        "manifest_hash",
        "source_manifest_hash",
        "server_hash",
    ],
)
def test_session_store_conformance_rejects_false_unchanged_mcp_evidence(
    session_store_case,
    mismatch: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            sessions = [
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=f"mcp_false_unchanged_{mismatch}_{index}",
                        messages=[Message.text("user", "check manifest")],
                    ),
                    identity=_identity(),
                )
                for index in range(2)
            ]
            history_key = "sha256:" + "2" * 64
            source_hash = "sha256:" + "6" * 64
            server_hash = "sha256:" + "5" * 64
            accepted = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=sessions[0].id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "3" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=source_hash,
                        server_hash=server_hash,
                    ),
                    "source_manifest_hash": source_hash,
                    "server_hash": server_hash,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=accepted.payload["manifest_identity"],
                manifest_hash=accepted.payload["manifest_hash"],
                source_manifest_hash=source_hash,
                server_hash=server_hash,
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(sessions[0].id),
                accepted_event_id=accepted.id,
                accepted_at=accepted.timestamp,
            )
            first_result = await store.compare_and_publish_mcp_manifest_checks(
                sessions[0].id,
                expected_generations={history_key: None},
                baseline_updates={history_key: baseline},
                events=[accepted],
            )
            assert first_result.published is True

            false_payload = {
                "history_key": history_key,
                "manifest_identity": baseline.manifest_identity,
                "manifest_hash": baseline.manifest_hash,
                "source_manifest_hash": baseline.source_manifest_hash,
                "server_hash": baseline.server_hash,
                "status": "unchanged",
                "outcome": "accepted",
            }
            false_payload[mismatch] = "sha256:" + "f" * 64
            false_unchanged = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=sessions[1].id,
                payload=false_payload,
            )
            with pytest.raises(ValueError, match="unchanged|identity"):
                await store.compare_and_publish_mcp_manifest_checks(
                    sessions[1].id,
                    expected_generations={history_key: 1},
                    baseline_updates={},
                    events=[false_unchanged],
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {
                history_key: baseline
            }
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert [record.event.id for record in records] == [accepted.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("status", ["first_seen", "changed"])
def test_session_store_conformance_rejects_mcp_status_incompatible_with_current_state(
    session_store_case,
    status: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"mcp_invalid_{status}_state",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "7" * 64
            source_hash = "sha256:" + "8" * 64
            server_hash = "sha256:" + "9" * 64
            expected_generation: int | None = None
            generation = 1
            if status == "first_seen":
                seed_event = Event(
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    payload={
                        "history_key": history_key,
                        "manifest_identity": "sha256:" + "a" * 64,
                        "manifest_hash": _mcp_test_manifest_hash(
                            source_manifest_hash=source_hash,
                            server_hash=server_hash,
                        ),
                        "source_manifest_hash": source_hash,
                        "server_hash": server_hash,
                        "status": "first_seen",
                        "outcome": "accepted",
                    },
                )
                seed_baseline = McpManifestBaseline(
                    history_key=history_key,
                    generation=1,
                    manifest_identity=seed_event.payload["manifest_identity"],
                    manifest_hash=seed_event.payload["manifest_hash"],
                    source_manifest_hash=source_hash,
                    server_hash=server_hash,
                    tools=(),
                    exposed_tools=(),
                    accepted_session_ref=_mcp_manifest_session_ref(session.id),
                    accepted_event_id=seed_event.id,
                    accepted_at=seed_event.timestamp,
                )
                seeded = await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: seed_baseline},
                    events=[seed_event],
                )
                assert seeded.published is True
                expected_generation = 1
                generation = 2

            event = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "a" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=source_hash,
                        server_hash=server_hash,
                    ),
                    "source_manifest_hash": source_hash,
                    "server_hash": server_hash,
                    "status": status,
                    "outcome": "accepted",
                },
            )
            update = McpManifestBaseline(
                history_key=history_key,
                generation=generation,
                manifest_identity=event.payload["manifest_identity"],
                manifest_hash=event.payload["manifest_hash"],
                source_manifest_hash=source_hash,
                server_hash=server_hash,
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=event.id,
                accepted_at=event.timestamp,
            )
            with pytest.raises(ValueError, match="first_seen|changed"):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: expected_generation},
                    baseline_updates={history_key: update},
                    events=[event],
                )

            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert len(records) == (1 if status == "first_seen" else 0)
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("event_type", "outcome"),
    [
        (EventType.MCP_MANIFEST_BLOCKED, "blocked"),
        (EventType.MCP_MANIFEST_CHECKED, "batch_blocked"),
    ],
)
def test_session_store_conformance_accepts_mcp_block_without_baseline_update(
    session_store_case,
    event_type: EventType,
    outcome: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"mcp_{outcome}_publication",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "b" * 64
            source_hash = "sha256:" + "c" * 64
            server_hash = "sha256:" + "d" * 64
            event = Event(
                type=event_type,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "e" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=source_hash,
                        server_hash=server_hash,
                    ),
                    "source_manifest_hash": source_hash,
                    "server_hash": server_hash,
                    "status": "first_seen",
                    "outcome": outcome,
                },
            )
            result = await store.compare_and_publish_mcp_manifest_checks(
                session.id,
                expected_generations={history_key: None},
                baseline_updates={},
                events=[event],
            )

            assert result.published is True
            assert result.baselines == {}
            records = await store.query_events(EventQuery(event_type=event_type, limit=10))
            assert [record.event.id for record in records] == [event.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_missing_mcp_history_outcome(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_missing_history_outcome",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_keys = ("sha256:" + "1" * 64, "sha256:" + "2" * 64)
            source_hash = "sha256:" + "3" * 64
            server_hash = "sha256:" + "4" * 64
            event = Event(
                type=EventType.MCP_MANIFEST_BLOCKED,
                session_id=session.id,
                payload={
                    "history_key": history_keys[0],
                    "manifest_identity": "sha256:" + "5" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=source_hash,
                        server_hash=server_hash,
                    ),
                    "source_manifest_hash": source_hash,
                    "server_hash": server_hash,
                    "status": "first_seen",
                    "outcome": "blocked",
                },
            )
            with pytest.raises(ValueError, match="exactly one outcome"):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={key: None for key in history_keys},
                    baseline_updates={},
                    events=[event],
                )

            assert (await store.load_mcp_manifest_baselines(history_keys)).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_BLOCKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mismatch",
    ["server_hash", "source_manifest_hash", "accepted_event_id", "blocked_event"],
)
def test_session_store_conformance_rejects_mcp_baselines_without_matching_accepted_events(
    session_store_case,
    mismatch: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"mcp_manifest_event_mismatch_{mismatch}",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "e" * 64
            event_source_hash = "sha256:" + "4" * 64
            event_server_hash = "sha256:" + "3" * 64
            event = Event(
                type=(
                    EventType.MCP_MANIFEST_BLOCKED
                    if mismatch == "blocked_event"
                    else EventType.MCP_MANIFEST_CHECKED
                ),
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "1" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=event_source_hash,
                        server_hash=event_server_hash,
                    ),
                    "source_manifest_hash": event_source_hash,
                    "server_hash": event_server_hash,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            baseline_source_hash = (
                "sha256:" + "5" * 64 if mismatch == "source_manifest_hash" else event_source_hash
            )
            baseline_server_hash = (
                "sha256:" + "6" * 64 if mismatch == "server_hash" else event_server_hash
            )
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=event.payload["manifest_identity"],
                manifest_hash=_mcp_test_manifest_hash(
                    source_manifest_hash=baseline_source_hash,
                    server_hash=baseline_server_hash,
                ),
                source_manifest_hash=baseline_source_hash,
                server_hash=baseline_server_hash,
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=(
                    "evt_missing_manifest_acceptance"
                    if mismatch == "accepted_event_id"
                    else event.id
                ),
                accepted_at=event.timestamp,
            )

            expected_error = (
                "matching baseline update"
                if mismatch == "accepted_event_id"
                else "must match exactly one accepted"
            )
            with pytest.raises(ValueError, match=expected_error):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: baseline},
                    events=[event],
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {}
            records = await store.query_events(
                EventQuery(
                    event_types=(
                        EventType.MCP_MANIFEST_CHECKED,
                        EventType.MCP_MANIFEST_BLOCKED,
                    ),
                    limit=10,
                )
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())
