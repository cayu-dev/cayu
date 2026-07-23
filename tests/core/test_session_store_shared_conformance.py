from __future__ import annotations

import asyncio
import hashlib
import io
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

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
    InMemorySessionStore,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectStatus,
    ResolutionActor,
    RunRequest,
    Session,
    SessionIdentity,
    SessionMessageDeliveryMode,
    SessionMessageQueueStatus,
    SessionOperationPublication,
    SessionOrder,
    SessionQuery,
    SessionQueuedMessage,
    SessionQueuedMessagesPending,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    UsageRollupQuery,
)
from cayu.runtime.budgets import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.sessions import (
    PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES,
    PersistedEventSideEffectDelivery,
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
        self.started = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.calls = 0

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        call = self.calls
        self.calls += 1
        self.started[call].set()
        await self.release[call].wait()
        return CompactionResult(
            summary=_summary_with_existing(request, f"summary from attempt {call + 1}"),
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(_represented_existing_summary_sha256(request)),
            model_completed_payloads=[
                {
                    "provider_name": "overlap-compactor",
                    "model": "summary-model",
                    "usage": {"input_tokens": call + 1, "output_tokens": 1},
                }
            ],
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
