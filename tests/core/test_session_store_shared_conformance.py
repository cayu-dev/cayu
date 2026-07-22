from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from cayu import SQLiteSessionStore
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.billing import BillingIdentity
from cayu.providers import bedrock_billing_identity, completed_bedrock_billing_identity
from cayu.runtime import (
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    CompactSessionRequest,
    ContextCompactor,
    EnqueueSessionMessageRequest,
    EventQuery,
    InMemorySessionStore,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectStatus,
    ResolutionActor,
    RunRequest,
    Session,
    SessionIdentity,
    SessionMessageDeliveryMode,
    SessionMessageQueueStatus,
    SessionOrder,
    SessionQuery,
    SessionQueuedMessagesPending,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    UsageRollupQuery,
)
from cayu.runtime.usage import UsageMetrics

_POSTGRES_TABLES = (
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_persisted_event_side_effects",
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


class _ConformanceCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0
        self.fail_next = False

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("conformance compactor failed")
        return CompactionResult(summary=f"summary-{self.calls}")


class _ConformanceOverlappingCompactor(ContextCompactor):
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
            summary=f"summary from attempt {call + 1}",
            model_completed_payloads=[
                {
                    "provider_name": "overlap-compactor",
                    "model": "summary-model",
                    "usage": {"input_tokens": call + 1, "output_tokens": 1},
                }
            ],
        )


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
