from __future__ import annotations

import asyncio
import contextvars
import hashlib
import io
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast

import pytest
from pydantic import ValidationError

import cayu.runtime._model_step_executor as model_step_executor_module
import cayu.runtime._session_engine as session_engine_module
from cayu import SQLiteSessionStore
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    extract_durable_value_error,
)
from cayu.core import AgentSpec, Event, EventType, Message, ToolCallPart
from cayu.core.billing import BillingIdentity
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
    bedrock_billing_identity,
    completed_bedrock_billing_identity,
)
from cayu.runtime import (
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    CompactSessionRequest,
    ContextCompactor,
    EnqueueSessionMessageRequest,
    EventQuery,
    ForkSessionRequest,
    InMemorySessionStore,
    McpManifestBaseline,
    ModelCompactor,
    ModelCompletionStageRequest,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectStatus,
    ResolutionActor,
    RunRequest,
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    RuntimePublicationRequest,
    Session,
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
    ToolRoundIdentity,
    UsageRollupQuery,
    runtime_publication_checkpoint_mutation,
    runtime_publication_checkpoint_value_digest,
    runtime_publication_event_reference,
)
from cayu.runtime._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.runtime.aggregates import estimate_usage_rollup_cost
from cayu.runtime.budgets import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.sessions import (
    PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES,
    BudgetReservationIdentityConflict,
    PersistedEventSideEffectDelivery,
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
)
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)
from cayu.runtime.usage import UsageMetrics
from cayu.storage.jsonl_export import export_sessions, import_sessions

_POSTGRES_TABLES = (
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _publication_tool_round_identity(label: str) -> dict[str, str]:
    digest = sha256(label.encode("utf-8")).hexdigest()
    return {
        "model_step_id": f"mstep_{digest[:32]}",
        "model_attempt_id": f"matt_{digest[16:48]}",
        "tool_round_id": f"tround_{digest[32:]}",
    }


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


async def _truncate_postgres(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _POSTGRES_TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def _new_postgres_store(dsn: str) -> SessionStore:
    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    return PostgresSessionStore(dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE)


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


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

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
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
    async def compact(self, request: CompactionRequest) -> CompactionResult:
        return CompactionResult(
            summary=_summary_with_existing(request, "partial coverage"),
            covered_message_count=min(1, len(request.messages)),
            represented_existing_summary_sha256=(_represented_existing_summary_sha256(request)),
        )


class _ConformancePartialCancellationCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _ConformancePartialOverlapCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.calls = 0

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


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def session_store_case(request, tmp_path):
    if request.param == "memory":
        return request.param, tmp_path, None
    if request.param == "sqlite":
        return request.param, tmp_path, None
    return request.param, tmp_path, request.getfixturevalue("postgres_dsn")


async def _open_store(case) -> SessionStore:
    store_kind, tmp_path, postgres_dsn = case
    if store_kind == "memory":
        return InMemorySessionStore()
    if store_kind == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    await _truncate_postgres(postgres_dsn)
    return _new_postgres_store(postgres_dsn)


async def _reopen_store(case, store: SessionStore) -> SessionStore:
    store_kind, tmp_path, postgres_dsn = case
    if store_kind == "memory":
        return store
    await _close_store(store)
    if store_kind == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    return _new_postgres_store(postgres_dsn)


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
            await store.append_event(
                first.id,
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id=first.id,
                    payload={"reservation_id": reservation_id},
                ),
            )
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

            with pytest.raises(DurableValueError) as invalid_model:
                await store.update_model(session_id, invalid_text)
            assert invalid_model.value.code == code
            assert "workload-secret-value" not in str(invalid_model.value)
            loaded = await store.load(session_id)
            assert loaded is not None
            assert loaded.model == "fake-model"

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
    postgres_dsn,
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

        await _truncate_postgres(postgres_dsn)
        postgres_store = _new_postgres_store(postgres_dsn)
        try:
            source = imported.session
            await postgres_store.create(
                RunRequest(
                    agent_name=source.agent_name,
                    session_id=source.id,
                    parent_session_id=source.parent_session_id,
                    causal_budget_id=source.causal_budget_id,
                    provider_name=source.provider_name,
                    model=source.model,
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
                ("postgres", tmp_path, postgres_dsn),
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
) -> RuntimePublicationRequest:
    if classification is None:
        assert assistant_message is not None
        classification = {"type": "final"}
    transcript_end_cursor = source_transcript_cursor + int(assistant_message is not None)
    completion_payload = {} if event_payload is None else dict(event_payload)
    completion_payload.update(
        step_classification=classification,
        transcript_cursor=transcript_end_cursor,
    )
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
            if corruption == "transcript":
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
            if corruption == "transcript":
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
            compactor = _ConformanceCompactor()
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
                    session_id="sess_compaction_conformance",
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
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-conformance-1",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            first = [event async for event in app.compact_session(request)]
            store = await _reopen_store(session_store_case, store)
            app = CayuApp(session_store=store, enable_logging=False)
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
            assert await store.load(session_id) == before_replay_session
            assert await store.load_checkpoint(session_id) == before_replay_checkpoint
            assert await store.load_transcript(session_id) == before_replay_transcript
            assert await store.load_events(session_id) == before_replay_events
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
            "valid structured-output event conflicts with its grouped tool result",
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
            "deterministic_validity",
            "validity conflicts with the durable tool calls",
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
                    publication=RuntimePublicationRequest(
                        publication_id=stage_id,
                        kind="model-step",
                        intent={"logical_step": 1},
                        mutation=runtime_publication_checkpoint_mutation(None, None),
                        transcript_messages=[],
                        events=[
                            Event(
                                id="unclassified-contentless-completion",
                                type=EventType.MODEL_COMPLETED,
                                session_id=session_id,
                            )
                        ],
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
    ["transcript", "event", "reference", "reference-content"],
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
            )
            await store.append_event(session_id, referenced_event)
            appended_event = Event(
                id=f"material-completion-{corruption}",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                payload={"usage": {"input_tokens": 2, "output_tokens": 1}},
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
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
                app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now)
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
            durable_events = [
                record.event
                for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
            ]
            replay = [event async for event in replay_app.compact_session(recovered_request)]

            assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED
            assert [event.id for event in replay] == [event.id for event in durable_events]
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
            assert len({event.payload["operation_id"] for event in durable_events}) == 1
            assert len({event.payload["attempt_id"] for event in durable_events}) == 2
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
            fork = Session(
                id="sess_active_model_stage_control_plane_fork",
                agent_name=interrupted.agent_name,
                provider_name=interrupted.provider_name,
                model=interrupted.model,
                parent_session_id=interrupted.id,
                causal_budget_id=interrupted.causal_budget_id,
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
            )
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
            )
            third = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=True,
                eligible_through=first.eligible_through,
                limit=1,
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
                event.id for event in queue_events
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
                [
                    Message.text("user", "copied-prefix"),
                    Message.text("user", "excluded-suffix"),
                ],
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
                with pytest.raises(ValueError, match="workload secret"):
                    await session_store.create_fork_with_transcript_validation(
                        source_session_id=source.id,
                        fork=Session(
                            id=child_id,
                            agent_name="assistant",
                            provider_name="fake",
                            model="fake-model",
                            parent_session_id=source.id,
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
                identity=_identity(),
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
