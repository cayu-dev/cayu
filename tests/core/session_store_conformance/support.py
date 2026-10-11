"""Shared store fixtures, durable-evidence builders and behavioral recovery fakes."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast

import pytest
from pydantic import SecretStr

from cayu._validation import (
    extract_durable_value_error,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    PendingToolApprovalEventView,
    PendingToolCallApproval,
)
from cayu.budgets.base import (
    MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY,
    BudgetReconciliation,
    budget_reconciliation_payload,
    budget_settlement_id,
)
from cayu.context.base import (
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    ModelCompactor,
)
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)
from cayu.environments.base import EnvironmentSpec
from cayu.environments.bindings import (
    BoundWorkspace,
    WorkspaceBinding,
    WorkspaceSnapshot,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.memory.base import AutomaticRecallPolicy
from cayu.memory.context import AutomaticRecallContextPolicy, AutomaticRecallSourceConfig
from cayu.memory.evidence import (
    ContextExposure,
    ContextExposureEvidenceKind,
    ContextExposureState,
    ContextExposureTransition,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptItem,
    RecallSourceCoverage,
)
from cayu.memory.recall import KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL
from cayu.memory.retrieval import (
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    WeightedReciprocalRankFusionConfig,
)
from cayu.messages import (
    Message,
)
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._event_projection import (
    public_event_sequence,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions._checkpoint_mutations import (
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.sessions._operation_records import (
    RuntimePublicationOperationRecordMutation,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationRequest,
    runtime_publication_event_reference,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.mcp_manifest_history import (
    _mcp_authoritative_manifest_hash,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionIdentity
from cayu.sessions.store import SessionStore
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import (
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.tools.exposure import (
    ToolExposureDecision,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
)
from cayu.tools.grants import (
    TargetedToolGrantRecord,
    TargetedToolUseRequest,
)
from cayu.tools.policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import (
    WorkspaceObservationLifecycle,
    _admit_workspace_observation_intent,
)


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
    "cayu_targeted_tool_grant_uses",
    "cayu_targeted_tool_grants",
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_public_authority_alias_config",
    "cayu_task_terminalization_receipts",
    "cayu_completion_decision_application_receipts",
    "cayu_completion_decisions",
    "cayu_completion_verification_claims",
    "cayu_completion_evaluation_runs",
    "cayu_completion_verifier_dispatches",
    "cayu_completion_verifier_profiles",
    "cayu_completion_proposals",
    "cayu_work_attempt_execution_claims",
    "cayu_work_attempt_lifecycle_receipts",
    "cayu_work_attempt_preparation_holds",
    "cayu_work_attempt_admissions",
    "cayu_work_attempts",
    "cayu_task_session_execution_authority",
    "cayu_work_contracts",
    "cayu_recall_item_exposures",
    "cayu_context_exposures",
    "cayu_recall_receipts",
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
    "cayu_child_session_lifecycle_candidates",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_search_configuration",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_local_execution_attempts",
    "cayu_tasks",
    "cayu_participant_session_bindings",
    "cayu_sessions",
    "cayu_eval_baseline_mutations",
    "cayu_eval_baselines",
    "cayu_eval_result_records",
    "cayu_eval_results",
    "cayu_eval_run_trial_checkpoints",
    "cayu_eval_runs",
    "cayu_eval_cases",
    "cayu_eval_suites",
    "cayu_eval_corpora",
    "cayu_storage_retention_entries",
    "cayu_storage_retention_runs",
    "cayu_schema_migrations",
)
_POSTGRES_DATA_TABLES = (
    "cayu_public_authority_alias_config",
    *(
        table
        for table in _POSTGRES_TABLES
        if table
        not in {
            "cayu_public_authority_alias_config",
            "cayu_schema_migrations",
            "cayu_transcript_search_configuration",
        }
    ),
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _memory_evidence_digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _memory_evidence_fingerprint(label: str, domain: str) -> dict[str, str]:
    return {
        "algorithm": "hmac-sha256",
        "domain": domain,
        "key_id": "memory-evidence-test",
        "digest": _memory_evidence_digest(label),
    }


def _recall_receipt(
    label: str,
    *,
    session_id: str = "memory-evidence-session",
    interaction_id: str = "memory-evidence-interaction",
    model_step_id: str = f"mstep_{'1' * 32}",
    created_at: datetime | None = None,
) -> RecallReceipt:
    item = RecallReceiptItem(
        ordinal=0,
        identity={
            "record_type": "knowledge_entry",
            "record_id": f"knowledge-{label}",
            "revision": "1",
        },
        representation_id="entry_text",
        content_sha256=_memory_evidence_digest(f"content:{label}"),
        locator={
            "kind": "knowledge_entry",
            "entry_id": f"knowledge-{label}",
            "entry_revision": 1,
        },
        admission="admitted",
        selection_reason="calibrated_strong_match",
        fused_rank=1,
        match_channels=("knowledge.lexical",),
    )
    return RecallReceipt(
        receipt_id=f"receipt-{label}",
        session_id=session_id,
        interaction_id=interaction_id,
        model_step_id=model_step_id,
        created_at=created_at or datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
        situation_fingerprint=_memory_evidence_fingerprint(
            f"situation:{label}", "recall_situation"
        ),
        engine_version="cayu.memory.recall.v1",
        source_configuration_fingerprint=_memory_evidence_fingerprint(
            f"sources:{label}", "recall_source_configuration"
        ),
        admission_policy_fingerprint=_memory_evidence_fingerprint(
            f"policy:{label}", "recall_admission_policy"
        ),
        access_scope_fingerprint=_memory_evidence_fingerprint(
            f"scope:{label}", "recall_access_scope"
        ),
        frontier_fingerprint=_memory_evidence_fingerprint(f"frontier:{label}", "recall_frontier"),
        sources=(
            RecallSourceCoverage(
                source="knowledge",
                required=True,
                channels=("knowledge.lexical",),
                state="complete",
                inspected_count=1,
                candidate_limit=10,
            ),
        ),
        inspected_count=1,
        eligible_count=1,
        admitted_count=1,
        offered_count=0,
        silent_count=0,
        omitted_count=0,
        truncated=False,
        items=(item,),
    )


def _large_recall_receipt(label: str) -> RecallReceipt:
    """Build a valid receipt large enough to exercise the page byte fence."""

    channels = tuple(f"channel-{index}-{'x' * 180}" for index in range(32))
    items = tuple(
        RecallReceiptItem(
            ordinal=index,
            identity={
                "record_type": "custom_memory_record",
                "record_id": f"large-{label}-{index}",
                "revision": "1",
            },
            representation_id=f"representation-{index}-{'y' * 220}",
            content_sha256=_memory_evidence_digest(f"large-content:{label}:{index}"),
            locator={
                "kind": "opaque",
                "fingerprint": _memory_evidence_fingerprint(
                    f"large-locator:{label}:{index}",
                    "recall_source_locator",
                ),
            },
            admission="admitted",
            selection_reason="calibrated_strong_match",
            fused_rank=index + 1,
            match_channels=channels,
        )
        for index in range(32)
    )
    receipt = _recall_receipt(
        label,
        interaction_id="memory-evidence-byte-page",
    )
    return RecallReceipt.model_validate(
        {
            **receipt.model_dump(mode="python"),
            "sources": (
                RecallSourceCoverage(
                    source="custom",
                    required=True,
                    channels=channels,
                    state="complete",
                    inspected_count=len(items),
                    candidate_limit=len(items),
                ),
            ),
            "inspected_count": len(items),
            "eligible_count": len(items),
            "admitted_count": len(items),
            "items": items,
        }
    )


def _planned_context_exposure(
    label: str,
    receipt: RecallReceipt,
) -> tuple[ContextExposure, tuple[RecallItemExposure, ...]]:
    planned = ContextExposureTransition(
        transition_id=f"transition-{label}-planned",
        revision=0,
        state=ContextExposureState.PLANNED,
        occurred_at=receipt.created_at,
        evidence_kind=ContextExposureEvidenceKind.COMPOSITION_PLANNED,
        evidence_ref=f"plan-{label}",
    )
    exposure = ContextExposure(
        exposure_id=f"exposure-{label}",
        session_id=receipt.session_id,
        interaction_id=receipt.interaction_id,
        model_step_id=receipt.model_step_id,
        model_attempt_id=f"matt_{_memory_evidence_digest(f'model:{label}')[:32]}",
        provider_attempt_id=f"patt_{_memory_evidence_digest(f'provider:{label}')[:32]}",
        provider_name="scripted",
        model_name="scripted-model",
        composition_fingerprint=_memory_evidence_fingerprint(
            f"composition:{label}", "context_composition"
        ),
        execution_profile_fingerprint=_memory_evidence_fingerprint(
            f"profile:{label}", "execution_profile"
        ),
        context_policy_fingerprint=_memory_evidence_fingerprint(
            f"context:{label}", "context_policy"
        ),
        tool_exposure_fingerprint=_memory_evidence_fingerprint(f"tools:{label}", "tool_exposure"),
        request_contract_fingerprint=_memory_evidence_fingerprint(
            f"request:{label}", "provider_request_contract"
        ),
        receipt_ids=(receipt.receipt_id,),
        contributor_ids=("memory_focus",),
        created_at=planned.occurred_at,
        updated_at=planned.occurred_at,
        state=planned.state,
        state_revision=planned.revision,
        transitions=(planned,),
    )
    receipt_item = receipt.items[0]
    item = RecallItemExposure(
        exposure_id=exposure.exposure_id,
        receipt_id=receipt.receipt_id,
        ordinal=0,
        receipt_item_ordinal=0,
        identity=receipt_item.identity,
        representation_id=receipt_item.representation_id,
        content_sha256=receipt_item.content_sha256,
        locator=receipt_item.locator,
        admission=receipt_item.admission,
        selection_reason=receipt_item.selection_reason,
    )
    return exposure, (item,)


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
    pending_round = pending_rounds.PendingToolRound(
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
            pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY: (
                pending_round.model_dump(mode="json")
            ),
            pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: (
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
    store_type: type[Any] | None = None,
) -> SessionStore:
    from cayu.storage.postgres import PostgresSessionStore

    selected_store_type = PostgresSessionStore if store_type is None else store_type
    return selected_store_type(
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


async def _collect_events(stream: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in stream]


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
    checkpoint_approval = pending_approval_reader.pending_approval_from_checkpoint(
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

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session_store_conformance:compactor",
            behavior_version="1",
            implementation_version="1",
        )

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

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session_store_conformance:overlapping_compactor",
            behavior_version="1",
            implementation_version="1",
        )

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

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session_store_conformance:partial_compactor",
            behavior_version="1",
            implementation_version="1",
        )

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

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session_store_conformance:partial_cancellation_compactor",
            behavior_version="1",
            implementation_version="1",
        )

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _ConformancePartialOverlapCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.calls = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session_store_conformance:partial_overlap_compactor",
            behavior_version="1",
            implementation_version="1",
        )

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

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session_store_conformance:unused_fork_provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _UnusedNamedForkProvider(ModelProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name=f"tests:session-store-conformance:{type(self).__name__}",
            behavior_version="1",
            implementation_version="1",
        )

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

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:transition-replay-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("completed")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _WorkspaceObservationRecoveryProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:workspace-recovery-provider",
            behavior_version="1",
            implementation_version="1",
        )

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

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name=f"tests:session-store-conformance:{type(self).__name__}",
            behavior_version="1",
            implementation_version="1",
        )

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


class _UserInputRecoveryProvider(ModelProvider):
    name = "fake"

    def __init__(self, *, complete_without_tools: bool = False) -> None:
        self._complete_without_tools = complete_without_tools
        self.requests: list[ModelRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:user-input-recovery-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if self._complete_without_tools:
            yield ModelStreamEvent.text_delta("answer accepted")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(
            id="call_user_input",
            name="ask_user",
            arguments={"question": "Continue?"},
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


class _AutomaticRecallRecoveryKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self) -> None:
        super().__init__(access_scope=KnowledgeAccessScope.privileged())
        self.search_count = 0

    async def search(self, query, *, access_scope=None):
        self.search_count += 1
        return await super().search(query, access_scope=access_scope)


def _automatic_recall_recovery_policy() -> AutomaticRecallContextPolicy:
    configuration_version = "session-store-recovery-automatic-recall-v1"
    return AutomaticRecallContextPolicy(
        admission_policy=AutomaticRecallPolicy(
            calibration_version="session-store-recovery-calibration-v1",
            fusion_strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
            fusion_configuration_version=configuration_version,
            minimum_inject_score=0.01,
            minimum_offer_score=0.005,
        ),
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version=configuration_version,
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            },
            max_candidates_per_channel=20,
            fused_head_limit=20,
        ),
        sources=AutomaticRecallSourceConfig(
            include_transcript=False,
            transcript_required=False,
        ),
    )


class _CeilingConformanceTool(Tool):
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name=name,
            effect=ToolEffect.NONE,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name=f"tests:session-store-conformance:ceiling-tool:{name}",
                behavior_version="1",
                implementation_version="1",
            ),
        )
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        return ToolResult(content=self.name)


def _targeted_use_request(
    record: TargetedToolGrantRecord,
    *,
    run_epoch: int,
    identity: str,
    arguments: str | None = None,
) -> TargetedToolUseRequest:
    argument_evidence = identity if arguments is None else arguments
    return TargetedToolUseRequest(
        tool_ref=record.tool_ref,
        session_id=record.session_id,
        interaction_id=record.interaction_id,
        generation_id=record.generation_id,
        agent_name=record.agent_name,
        task_id=record.task_id,
        environment_name=record.environment_name,
        principal=record.principal,
        tenant=record.tenant,
        catalogue_revision=record.catalogue_revision,
        descriptor_version=record.descriptor_version,
        schema_fingerprint=record.schema_fingerprint,
        tool_id=record.tool_id,
        tool_name=record.tool_name,
        model_step_id=f"model-step-{identity}",
        outer_tool_call_id=f"outer-call-{identity}",
        arguments_sha256=f"sha256:{sha256(argument_evidence.encode()).hexdigest()}",
        invocation_id=f"invocation-{identity}",
        expected_run_epoch=run_epoch,
    )


class _PreviousProfileConformancePolicy(ToolExposurePolicy):
    def __init__(self) -> None:
        self.requests: list[ToolExposurePolicyRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:session-store-conformance:previous-exposure-policy",
            behavior_version="1",
            implementation_version="1",
        )

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        self.requests.append(request)
        if request.previous_profile_id is None:
            return ToolExposureDecision(
                profile_id="initial-phase",
                tool_names=("inspect",),
            )
        if request.previous_profile_id != "initial-phase":
            raise AssertionError(
                f"unexpected previous exposure profile: {request.previous_profile_id}"
            )
        return ToolExposureDecision(
            profile_id="resumed-phase",
            tool_names=("inspect",),
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


class _WorkspaceObservationProcessLossStoreMixin:
    """Class-declared fault injection that preserves lifecycle capability attestation."""

    invocation_lifecycle_command_version = 1
    _workspace_observation_crash_phase: str | None = None
    _workspace_observation_process_lost = False

    def arm_workspace_observation_process_loss(self, crash_phase: str) -> None:
        self._workspace_observation_crash_phase = crash_phase
        self._workspace_observation_process_lost = False

    def disarm_workspace_observation_process_loss(self) -> bool:
        process_lost = self._workspace_observation_process_lost
        self._workspace_observation_crash_phase = None
        return process_lost

    async def publish_runtime_publication(self, session_id: str, **kwargs):
        result = await cast("Any", super()).publish_runtime_publication(
            session_id,
            **kwargs,
        )
        request = kwargs["request"]
        if (
            not self._workspace_observation_process_lost
            and request.kind == "workspace-observation"
            and request.intent.get("phase") == self._workspace_observation_crash_phase
        ):
            self._workspace_observation_process_lost = True
            raise _SimulatedProcessLoss(self._workspace_observation_crash_phase)
        return result

    async def transform_checkpoint(self, session_id: str, checkpoint_transform):
        result = await cast("Any", super()).transform_checkpoint(
            session_id,
            checkpoint_transform,
        )
        await self._fail_after_workspace_terminal_stage(session_id)
        return result

    async def publish_session_operation(self, session_id: str, **kwargs):
        result = await cast("Any", super()).publish_session_operation(session_id, **kwargs)
        # External-effect selection and its staged terminal share this atomic
        # operation; they no longer pass through transform_checkpoint.
        await self._fail_after_workspace_terminal_stage(session_id)
        return result

    async def _fail_after_workspace_terminal_stage(self, session_id: str) -> None:
        if (
            self._workspace_observation_crash_phase == "terminal-stage"
            and not self._workspace_observation_process_lost
        ):
            checkpoint = await cast("Any", self).load_checkpoint(session_id)
            observations = None if checkpoint is None else checkpoint.get("workspace_observations")
            pending_round = None if checkpoint is None else checkpoint.get("pending_tool_round")
            if (
                type(observations) is dict
                and len(observations) == 1
                and next(iter(observations.values())).get("phase") == "before_captured"
                and type(pending_round) is dict
                and pending_round.get("staged_terminals")
            ):
                self._workspace_observation_process_lost = True
                raise _SimulatedProcessLoss(self._workspace_observation_crash_phase)


class _WorkspaceObservationProcessLossInMemorySessionStore(
    _WorkspaceObservationProcessLossStoreMixin,
    InMemorySessionStore,
):
    invocation_lifecycle_command_version = 1


class _WorkspaceObservationProcessLossSQLiteSessionStore(
    _WorkspaceObservationProcessLossStoreMixin,
    SQLiteSessionStore,
):
    invocation_lifecycle_command_version = 1


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


async def _expire_compaction_claim_for_test(
    store: SessionStore, session_id: str, idempotency_key: str
) -> None:
    def expire(_session, checkpoint, store_now):
        assert checkpoint is not None
        copied = copy.deepcopy(checkpoint)
        record = copied["session_operations"]["records"][idempotency_key]
        assert record["status"] == "running"
        record["claim_expires_at"] = (store_now - timedelta(seconds=1)).isoformat()
        return copied

    await store.transform_checkpoint_with_store_time(session_id, expire)


async def _open_store(
    case,
    *,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
    workspace_observation_crash_phase: str | None = None,
) -> SessionStore:
    if public_authority_alias_codec is None:
        public_authority_alias_codec = _public_authority_alias_codec()
    store_kind, tmp_path, postgres_dsn = case
    if store_kind == "memory":
        store_type = (
            InMemorySessionStore
            if workspace_observation_crash_phase is None
            else _WorkspaceObservationProcessLossInMemorySessionStore
        )
        store = store_type(
            public_authority_alias_codec=public_authority_alias_codec,
        )
    elif store_kind == "sqlite":
        store_type = (
            SQLiteSessionStore
            if workspace_observation_crash_phase is None
            else _WorkspaceObservationProcessLossSQLiteSessionStore
        )
        store = store_type(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=public_authority_alias_codec,
        )
    else:
        await _reset_postgres_data(postgres_dsn)
        postgres_store_type = None
        if workspace_observation_crash_phase is not None:
            from cayu.storage.postgres import PostgresSessionStore

            class _WorkspaceObservationProcessLossPostgresSessionStore(
                _WorkspaceObservationProcessLossStoreMixin,
                PostgresSessionStore,
            ):
                invocation_lifecycle_command_version = 1

            postgres_store_type = _WorkspaceObservationProcessLossPostgresSessionStore
        store = _new_postgres_store(
            postgres_dsn,
            public_authority_alias_codec=public_authority_alias_codec,
            store_type=postgres_store_type,
        )
    if workspace_observation_crash_phase is not None:
        assert isinstance(store, _WorkspaceObservationProcessLossStoreMixin)
        store.arm_workspace_observation_process_loss(workspace_observation_crash_phase)
    return store


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


async def _open_peer_store(case, store: SessionStore) -> SessionStore:
    store_kind, tmp_path, postgres_dsn = case
    if store_kind == "memory":
        return store
    if store_kind == "sqlite":
        return SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=store.public_authority_alias_codec,
        )
    return _new_postgres_store(
        postgres_dsn,
        public_authority_alias_codec=store.public_authority_alias_codec,
    )


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
    operation_record_mutations: tuple[RuntimePublicationOperationRecordMutation, ...] = (),
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
        operation_record_mutations=operation_record_mutations,
    )


def _model_completion_publication_checkpoint(
    publication: RuntimePublicationRequest,
) -> dict[str, Any]:
    (operation,) = publication.mutation.operations
    assert operation.action == "set"
    assert operation.key == LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY
    return {operation.key: operation.value}


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
