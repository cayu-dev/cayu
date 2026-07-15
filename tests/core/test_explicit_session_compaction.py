from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    CompactSessionRequest,
    ContextCompactor,
    EventQuery,
    ForkSessionRequest,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    ModelCompactor,
    ModelPrice,
    PriceBook,
    PromptCacheCompactor,
    ResolutionActor,
    ResumeRequest,
    RunLimits,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.storage import SQLiteSessionStore


class RecordingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.requests: list[CompactionRequest] = []

    def provider_budget_identity(self, _session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request)
        return CompactionResult(
            summary="durable compact summary",
            metadata={"compactor": type(self).__name__, "mode": "deterministic"},
        )


class NoFullHistoryReplayStore(InMemorySessionStore):
    async def load_events(self, session_id: str):
        raise AssertionError("compaction replay must use indexed event lookup")


class FailingCompletionPublishStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.publications = 0

    async def publish_session_operation(self, session_id: str, **kwargs):
        self.publications += 1
        if self.publications == 2:
            raise RuntimeError("simulated completion publication failure")
        return await super().publish_session_operation(session_id, **kwargs)


class BlockingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.started.set()
        await self.release.wait()
        return CompactionResult(summary="summary")


class OverlappingCompactor(ContextCompactor):
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


class CompletingProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest):
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({})


class UsageCompactionProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self, *, summary: str = "provider summary") -> None:
        self.calls = 0
        self.summary = summary

    async def stream(self, request: ModelRequest):
        self.calls += 1
        yield ModelStreamEvent.text_delta(self.summary)
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})


class FinalRenewalFailureBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.heartbeat_calls = 0

    async def heartbeat(self, *, reservation_id: str) -> bool:
        self.heartbeat_calls += 1
        if self.heartbeat_calls == 2:
            return False
        return await super().heartbeat(reservation_id=reservation_id)


class InitialRenewalFailureBudgetLedger(InMemoryBudgetLedger):
    async def heartbeat(self, *, reservation_id: str) -> bool:
        return False


class HeartbeatCancellationBudgetLedger(FinalRenewalFailureBudgetLedger):
    @property
    def reservation_ttl_seconds(self) -> int:
        return 0


class FailingSecondReservationBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.reserve_calls = 0
        self.release_calls = 0

    async def reserve(self, **kwargs):
        self.reserve_calls += 1
        if self.reserve_calls == 2:
            raise RuntimeError("simulated reservation store failure")
        return await super().reserve(**kwargs)

    async def release(self, **kwargs):
        self.release_calls += 1
        return await super().release(**kwargs)


class UndeclaredProviderCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        return CompactionResult(
            summary="undeclared provider summary",
            model_completed_payloads=[
                {
                    "provider_name": "compaction-provider",
                    "model": "summary-model",
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                }
            ],
        )


class UnsafeProviderIdentityCompactor(ContextCompactor):
    def __init__(self, identity: tuple[str, str]) -> None:
        self.identity = identity
        self.calls = 0

    def provider_budget_identity(self, _session) -> tuple[str, str]:
        return self.identity

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        return CompactionResult(summary="must not execute")


class CancellationCompletingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0

    def provider_budget_identity(self, _session) -> tuple[str, str]:
        return "compaction-provider", "summary-model"

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return CompactionResult(
                summary="completed while cancellation was handled",
                model_completed_payloads=[
                    {
                        "provider_name": "compaction-provider",
                        "model": "summary-model",
                        "usage": {"input_tokens": 8, "output_tokens": 2},
                    }
                ],
            )
        raise AssertionError("blocking compactor unexpectedly resumed")


class FailOnceCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("provider prompt echoed: secret instructions")
        return CompactionResult(summary="retry summary")


class AdversarialTelemetryCompactor(ContextCompactor):
    async def compact(self, request: CompactionRequest) -> CompactionResult:
        return CompactionResult(
            summary="private summary text",
            metadata={
                "summary": "metadata summary leak",
                "instructions": "metadata instruction leak",
                "huge": "m" * 100_000,
            },
            model_completed_payloads=[
                {
                    "provider_name": "fake-compactor",
                    "requested_model": "summary-model",
                    "model": "summary-model-v1",
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                    "summary": "model event summary leak",
                    "instructions": "model event instruction leak",
                    "provider_state": {"secret": "opaque continuation leak"},
                    "arbitrary": "p" * 100_000,
                }
            ],
        )


def test_compact_session_request_requires_idempotency_and_source_fences() -> None:
    with pytest.raises(ValidationError):
        CompactSessionRequest(  # type: ignore[call-arg]
            session_id="sess_compact_contract",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
        )

        with pytest.raises(ValidationError):
            CompactSessionRequest(
                session_id="sess_compact_contract",
                idempotency_key="compact-1",
                expected_run_epoch=0,
                expected_transcript_cursor=3,
                instructions="x" * 4097,
            )

    with pytest.raises(ValidationError):
        CompactSessionRequest(
            session_id="sess_compact_contract",
            idempotency_key="compact-1",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
            instructions="invalid surrogate: \ud800",
        )

    for field_name, value in (
        ("session_id", "invalid-surrogate-\ud800"),
        ("session_id", "invalid-nul-\x00"),
        ("idempotency_key", "invalid-nul-\x00"),
        ("instructions", "invalid-nul-\x00"),
    ):
        values = {
            "session_id": "sess_compact_contract",
            "idempotency_key": "compact-1",
            "expected_run_epoch": 0,
            "expected_transcript_cursor": 3,
            field_name: value,
        }
        with pytest.raises(ValidationError):
            CompactSessionRequest(**values)

    invalid_price = ModelPrice.fixed(
        provider_name="invalid-provider-\ud800",
        model="summary-model",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("1"),
    )
    with pytest.raises(ValidationError):
        CompactSessionRequest(
            session_id="sess_compact_contract",
            idempotency_key="compact-1",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
            budget_limits=(
                BudgetLimit(
                    max_estimated_cost=Decimal("1"),
                    pricing=PriceBook(prices=(invalid_price,)),
                ),
            ),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"summary": "invalid-\x00summary"},
        {"summary": "invalid-\ud800-summary"},
        {"summary": "summary", "metadata": {"nested": "invalid-\x00metadata"}},
        {
            "summary": "summary",
            "model_completed_payloads": [{"model": "invalid-\ud800-model"}],
        },
    ],
)
def test_compaction_result_rejects_text_that_cannot_be_durably_persisted(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CompactionResult.model_validate(values)


def test_compact_session_preserves_transcript_and_replays_original_outcome() -> None:
    async def run() -> None:
        store = NoFullHistoryReplayStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_compact",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("system", "Be precise."),
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-explicit-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
            instructions="Keep the decisions and file names.",
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert [event.id for event in durable_events] == [event.id for event in first]
        assert len(compactor.requests) == 1
        assert compactor.requests[0].instructions == "Keep the decisions and file names."
        assert await store.load_transcript(created.id) == transcript
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"] == {
            "version": 1,
            "summary": "durable compact summary",
            "compacted_transcript_cursor": 3,
            "metadata": {"compactor": "RecordingCompactor", "mode": "deterministic"},
        }
        for event in first:
            assert "summary" not in event.payload
            assert "instructions" not in event.payload
            assert event.payload["operation_id"]
            assert event.payload["reason"] == "application_requested"

    asyncio.run(run())


def test_compact_session_replays_original_outcome_after_session_advances() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_advanced_replay",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
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
            idempotency_key="compact-before-resume",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        resume_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=created.id,
                    messages=[Message.text("user", "advance the durable session")],
                )
            )
        ]
        advanced = await store.load(created.id)
        assert advanced is not None
        assert advanced.run_epoch > request.expected_run_epoch
        assert len(await store.load_transcript(created.id)) > request.expected_transcript_cursor
        assert EventType.SESSION_COMPLETED in [event.type for event in resume_events]

        replay = [event async for event in app.compact_session(request)]
        operation_id = first[0].payload["operation_id"]
        durable_operation_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
            if record.event.payload.get("operation_id") == operation_id
        ]

        assert [event.id for event in replay] == [event.id for event in first]
        assert [event.id for event in durable_operation_events] == [event.id for event in first]
        assert len(compactor.requests) == 1

    asyncio.run(run())


def test_compact_session_events_allowlist_adversarial_compactor_telemetry() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=AdversarialTelemetryCompactor(),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_private_events",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-private-events",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    instructions="private caller instructions",
                )
            )
        ]

        serialized_events = json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
        )
        for secret in (
            "private summary text",
            "metadata summary leak",
            "metadata instruction leak",
            "model event summary leak",
            "model event instruction leak",
            "opaque continuation leak",
            "private caller instructions",
            "m" * 1_000,
            "p" * 1_000,
        ):
            assert secret not in serialized_events
        assert len(serialized_events.encode("utf-8")) < 10_000
        assert all(event.payload["mode"] == "bounded" for event in events)

        usage_event = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        assert usage_event.payload["purpose"] == "context_compaction"
        assert usage_event.payload["provider_name"] == "fake-compactor"
        assert usage_event.payload["usage_metrics"]["total_tokens"] == 10
        assert "usage" not in usage_event.payload
        assert "provider_state" not in usage_event.payload
        completed_event = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
        )
        assert "metadata" not in completed_event.payload

    asyncio.run(run())


def test_compact_session_claim_blocks_concurrent_resume() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = BlockingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_resume_race",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        compact_events = []

        async def compact() -> None:
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-race-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                compact_events.append(event)

        compact_task = asyncio.create_task(compact())
        await compactor.started.wait()
        try:
            with pytest.raises(RuntimeError, match="active durable operation"):
                async for _event in app.resume(
                    ResumeRequest(
                        session_id=created.id,
                        messages=[Message.text("user", "race")],
                    )
                ):
                    pass
            with pytest.raises(RuntimeError, match="active durable operation"):
                async for _event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=created.id,
                        session_id="sess_compact_race_fork",
                    )
                ):
                    pass
        finally:
            compactor.release.set()
            await compact_task

        assert [event.type for event in compact_events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]

    asyncio.run(run())


def test_compact_session_rejects_an_equivalent_unexpired_concurrent_claim() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = BlockingCompactor()
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
                session_id="sess_compact_same_key_race",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
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
            idempotency_key="compact-same-key-race",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first_events = []

        async def compact() -> None:
            async for event in app.compact_session(request):
                first_events.append(event)

        first_task = asyncio.create_task(compact())
        await compactor.started.wait()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                async for _event in app.compact_session(request):
                    pass
        finally:
            compactor.release.set()
            await first_task

        assert [event.type for event in first_events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]

    asyncio.run(run())


def test_compact_session_no_context_failure_is_replayable_without_provider_spend() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = RecordingCompactor()
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
                session_id="sess_compact_no_context",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [Message.text("user", "only current request")]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-no-context",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first = []

        with pytest.raises(ValueError, match="no complete older context"):
            async for event in app.compact_session(request):
                first.append(event)
        replay = [event async for event in app.compact_session(request)]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert compactor.requests == []
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_compact_session_recovers_an_abandoned_accepted_operation() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store = InMemorySessionStore()
        first_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_recovery",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
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
            idempotency_key="compact-recover-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        abandoned_stream = first_app.compact_session(request)
        abandoned_started = await anext(abandoned_stream)
        await abandoned_stream.aclose()

        recovering_compactor = RecordingCompactor()
        recovered_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        recovered_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=recovering_compactor,
                max_user_turns=1,
            ),
        )
        recovered = [event async for event in recovered_app.compact_session(request)]
        replay = [event async for event in recovered_app.compact_session(request)]

        assert [event.type for event in recovered] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]
        assert [event.id for event in replay] == [
            abandoned_started.id,
            *[event.id for event in recovered],
        ]
        assert len(recovering_compactor.requests) == 1

    asyncio.run(run())


def test_compact_session_recovery_fences_a_late_attempt_and_preserves_its_usage() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store = InMemorySessionStore()
        compactor = OverlappingCompactor()

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
                session_id="sess_compact_overlap",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
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
            idempotency_key="compact-overlap-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
            requested_by=ResolutionActor(subject="operator-a"),
        )
        recovered_request = first_request.model_copy(
            update={"requested_by": ResolutionActor(subject="operator-b")}
        )
        first_events = []
        recovered_events = []

        async def collect_first() -> None:
            async for event in first_app.compact_session(first_request):
                first_events.append(event)

        async def collect_recovered() -> None:
            async for event in recovered_app.compact_session(recovered_request):
                recovered_events.append(event)

        first_task = asyncio.create_task(collect_first())
        await compactor.started[0].wait()
        recovered_task = asyncio.create_task(collect_recovered())
        await compactor.started[1].wait()
        compactor.release[1].set()
        await recovered_task
        compactor.release[0].set()
        with pytest.raises(RuntimeError, match="superseded"):
            await first_task

        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        replay = [event async for event in recovered_app.compact_session(recovered_request)]

        assert (
            sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in durable_events)
            == 1
        )
        assert sum(event.type == EventType.SESSION_CHECKPOINTED for event in durable_events) == 1
        assert sum(event.type == EventType.MODEL_COMPLETED for event in durable_events) == 2
        assert first_events[-1].type == EventType.CONTEXT_COMPACTION_FAILED
        assert first_events[-1].payload["error_type"] == "SessionCompactionAttemptSuperseded"
        assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED
        assert [event.id for event in replay] == [event.id for event in durable_events]
        assert len({event.payload["operation_id"] for event in durable_events}) == 1
        assert len({event.payload["attempt_id"] for event in durable_events}) == 2
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["summary"] == "summary from attempt 2"

    asyncio.run(run())


def test_compact_session_completion_publication_failure_releases_the_claim() -> None:
    async def run() -> None:
        store = FailingCompletionPublishStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_completion_failure",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
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
            idempotency_key="compact-completion-failure",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first = []

        with pytest.raises(RuntimeError, match="simulated completion publication failure"):
            async for event in app.compact_session(request):
                first.append(event)
        replay = [event async for event in app.compact_session(request)]
        checkpoint = await store.load_checkpoint(created.id)

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_expired_compaction_claim_does_not_block_resume() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store = InMemorySessionStore()
        first_app = CayuApp(session_store=store, enable_logging=False, clock=lambda: accepted_at)
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_compact_resume",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        abandoned = first_app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-expired-resume",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )
        await anext(abandoned)
        await abandoned.aclose()

        resumed_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        resumed_app.register_provider(CompletingProvider())
        resumed_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        events = [
            event
            async for event in resumed_app.resume(
                ResumeRequest(
                    session_id=created.id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

        assert EventType.SESSION_COMPLETED in [event.type for event in events]
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        operations = checkpoint["session_operations"]
        assert operations["active_operation_id"] is None
        assert operations["records"]["compact-expired-resume"]["status"] == "abandoned"

    asyncio.run(run())


def test_expired_compaction_claim_does_not_block_fork_or_a_new_key() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store = InMemorySessionStore()
        first_app = CayuApp(session_store=store, enable_logging=False, clock=lambda: accepted_at)
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_compact_fork",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        abandoned = first_app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-expired-fork",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )
        await anext(abandoned)
        await abandoned.aclose()

        recovered_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        recovered_app.register_provider(CompletingProvider())
        recovered_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        forked = [
            event
            async for event in recovered_app.fork_session(
                ForkSessionRequest(
                    source_session_id=created.id,
                    session_id="sess_after_expired_compact",
                )
            )
        ]
        compacted = [
            event
            async for event in recovered_app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-after-expired-key",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert forked[-1].type == EventType.SESSION_FORKED
        assert compacted[-1].type == EventType.SESSION_CHECKPOINTED
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        abandoned_operation = await store.load_session_operation(
            created.id,
            "compact-expired-fork",
        )
        completed_operation = await store.load_session_operation(
            created.id,
            "compact-after-expired-key",
        )
        assert abandoned_operation is not None
        assert abandoned_operation["status"] == "abandoned"
        assert completed_operation is not None
        assert completed_operation["status"] == "completed"

    asyncio.run(run())


def test_fork_inherits_compacted_context_without_source_operation_records() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_fork_source",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        source = await store.update_status(created.id, SessionStatus.COMPLETED)
        idempotency_key = "compact-fork-independent-key"
        source_events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=source.id,
                    idempotency_key=idempotency_key,
                    expected_run_epoch=source.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        assert source_events[-1].type == EventType.SESSION_CHECKPOINTED

        fork_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="sess_compact_fork_child",
                )
            )
        ]
        assert fork_events[-1].type == EventType.SESSION_FORKED
        child = await store.load("sess_compact_fork_child")
        assert child is not None
        child_checkpoint = await store.load_checkpoint(child.id)
        assert child_checkpoint is not None
        assert child_checkpoint["context_compaction"]["summary"] == "durable compact summary"
        assert "session_operations" not in child_checkpoint

        child_messages = [
            Message.text("user", "child request"),
            Message.text("assistant", "child answer"),
        ]
        await store.append_transcript_messages(child.id, child_messages)
        child_events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=child.id,
                    idempotency_key=idempotency_key,
                    expected_run_epoch=child.run_epoch,
                    expected_transcript_cursor=len(transcript) + len(child_messages),
                )
            )
        ]

        assert child_events[-1].type == EventType.SESSION_CHECKPOINTED
        child_checkpoint = await store.load_checkpoint(child.id)
        assert child_checkpoint is not None
        assert "session_operations" not in child_checkpoint
        child_operation = await store.load_session_operation(child.id, idempotency_key)
        assert child_operation is not None
        assert child_operation["status"] == "completed"

    asyncio.run(run())


def test_compact_session_failure_replays_without_spend_and_new_key_retries() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = FailOnceCompactor()
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
                session_id="sess_compact_failure",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        failed_request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-failed-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
            instructions="secret instructions",
        )
        failed_events = []
        with pytest.raises(RuntimeError, match="provider prompt echoed"):
            async for event in app.compact_session(failed_request):
                failed_events.append(event)

        replay = [event async for event in app.compact_session(failed_request)]
        assert [event.id for event in replay] == [event.id for event in failed_events]
        assert [event.type for event in replay] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert compactor.calls == 1
        assert "error" not in replay[-1].payload

        retry = [
            event
            async for event in app.compact_session(
                failed_request.model_copy(update={"idempotency_key": "compact-retry-1"})
            )
        ]
        assert retry[-1].type == EventType.SESSION_CHECKPOINTED
        assert compactor.calls == 2

        with pytest.raises(ValueError, match="different request"):
            async for _event in app.compact_session(
                failed_request.model_copy(update={"instructions": "different"})
            ):
                pass
        assert compactor.calls == 2

    asyncio.run(run())


def test_sqlite_compaction_outcome_reconstructs_after_reopen(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "compact-reconstruction.sqlite"
        store = SQLiteSessionStore(path)
        compactor = RecordingCompactor()
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
                session_id="sess_compact_sqlite_reopen",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
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
            idempotency_key="compact-sqlite-reopen-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first = [event async for event in app.compact_session(request)]
        await store.close()

        reopened = SQLiteSessionStore(path)
        replay_compactor = RecordingCompactor()
        replay_app = CayuApp(session_store=reopened, enable_logging=False)
        replay_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=replay_compactor,
                max_user_turns=1,
            ),
        )
        try:
            replay = [event async for event in replay_app.compact_session(request)]
            assert [event.id for event in replay] == [event.id for event in first]
            assert replay_compactor.requests == []
            assert await reopened.load_transcript(created.id) == transcript
        finally:
            await reopened.close()

    asyncio.run(run())


def test_compact_session_attributes_provider_usage_and_honors_run_limits() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=UsageCompactionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_usage",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        first = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-usage-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        usage_event = next(event for event in first if event.type == EventType.MODEL_COMPLETED)
        assert usage_event.payload["operation_id"]
        assert usage_event.payload["reason"] == "application_requested"
        assert usage_event.payload["usage_metrics"]["total_tokens"] == 10

        tail = [
            Message.text("user", "later request"),
            Message.text("assistant", "later answer"),
        ]
        await store.append_transcript_messages(created.id, tail)
        checkpoint_before_limit = await store.load_checkpoint(created.id)
        limited_events = []
        with pytest.raises(RuntimeError, match="limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-usage-limited",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript) + len(tail),
                    limits=RunLimits(max_total_tokens=5),
                )
            ):
                limited_events.append(event)

        assert [event.type for event in limited_events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        checkpoint_after_limit = await store.load_checkpoint(created.id)
        assert (
            checkpoint_after_limit["context_compaction"]
            == checkpoint_before_limit["context_compaction"]
        )

    asyncio.run(run())


def test_compact_session_preserves_usage_for_durably_invalid_summary() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider(summary="invalid\x00summary")
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_invalid_summary",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
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
            idempotency_key="compact-invalid-summary",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first = []

        with pytest.raises(RuntimeError, match="must not contain NUL characters"):
            async for event in app.compact_session(request):
                first.append(event)
        replay = [event async for event in app.compact_session(request)]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert provider.calls == 1
        usage_event = first[3]
        assert usage_event.payload["compaction_outcome"] == "invalid_summary"
        assert usage_event.payload["usage_metrics"]["total_tokens"] == 10
        assert "compaction_attempt_id" not in usage_event.payload
        assert first[4].payload["actual_amount"] == "0.000012"
        assert first[5].payload["error_type"] == "ContextBuildError"
        usage = await app.get_session_usage(created.id)
        assert usage.model_steps == 1
        assert usage.usage.total_tokens == 10
        cost = await app.get_session_cost(created.id, pricing)
        assert cost.model_steps == 1
        assert cost.total_cost == Decimal("0.000012")
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"
        assert operation["event_ids"] == [event.id for event in first]

    asyncio.run(run())


def test_compact_session_usage_counts_against_supplied_cost_budget() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=UsageCompactionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_budget",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        events = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-budget-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    budget_limits=(
                        BudgetLimit(
                            scope="run",
                            max_estimated_cost=Decimal("0.00001"),
                            pricing=pricing,
                        ),
                    ),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        budget_event = events[-2]
        assert budget_event.payload["actual"] == "0.000012"
        assert budget_event.payload["operation_id"]
        assert await store.load_transcript(created.id) == transcript
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_applies_app_policy_and_reserves_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.00001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=8,
                            max_output_tokens=2,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_policy_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="reservation failed"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-policy-reservation",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVATION_FAILED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert provider.calls == 0
        assert all(event.payload["operation_id"] for event in events[1:])

    asyncio.run(run())


def test_compact_session_stops_on_an_exhausted_app_budget_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.00001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_exhausted_policy",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        await store.append_events(
            created.id,
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=created.id,
                    agent_name="assistant",
                    payload={
                        "provider_name": "compaction-provider",
                        "model": "summary-model",
                        "usage_metrics": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        },
                    },
                )
            ],
        )
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-exhausted-policy",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_releases_partial_reservations_when_later_acquisition_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=10)
        provider = UsageCompactionProvider()
        ledger = FailingSecondReservationBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_partial_reservation_failure",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="simulated reservation store failure"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-partial-reservation-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    budget_limits=(
                        BudgetLimit(
                            scope="agent",
                            key="assistant",
                            max_estimated_cost=Decimal("0.001"),
                            pricing=pricing,
                            reservation=reservation,
                        ),
                    ),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        reservation_id = events[2].payload["reservation_id"]
        assert ledger.reserve_calls == 2
        assert ledger.release_calls == 1
        assert not await ledger.heartbeat(reservation_id=reservation_id)
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_reconciles_reservations_and_replays_budget_events() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_policy_reconcile",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
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
            idempotency_key="compact-policy-reconcile",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.BUDGET_CHECKED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert provider.calls == 1
        reconciled = first[4]
        assert reconciled.payload["actual_amount"] == "0.000012"
        assert reconciled.payload["operation_id"]

    asyncio.run(run())


def test_compact_session_reserves_prompt_cache_compactor_with_session_model() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="summary-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=PromptCacheCompactor(provider=provider),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_prompt_cache_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(
                provider_name="compaction-provider",
                model="summary-model",
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-prompt-cache-reservation",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert EventType.BUDGET_RESERVED in [event.type for event in events]
        assert EventType.BUDGET_RECONCILED in [event.type for event in events]
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_preserves_usage_when_final_reservation_renewal_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        ledger = FinalRenewalFailureBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_final_renewal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="lease was lost"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-final-renewal",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[4].payload["actual_amount"] == "0.000012"
        assert provider.calls == 1
        assert ledger.heartbeat_calls == 2

    asyncio.run(run())


def test_compact_session_releases_reservation_when_initial_renewal_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=InitialRenewalFailureBudgetLedger(),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_initial_renewal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="before provider dispatch"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-initial-renewal",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[3].payload["actual_amount"] is None
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_preserves_usage_returned_while_heartbeat_cancels() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        compactor = CancellationCompletingCompactor()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=HeartbeatCancellationBudgetLedger(),
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
                session_id="sess_compact_heartbeat_cancellation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="lease was lost"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-heartbeat-cancellation",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[4].payload["actual_amount"] == "0.000012"
        assert compactor.calls == 1

    asyncio.run(run())


def test_compact_session_skips_provider_reservations_for_deterministic_compaction() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_deterministic_compaction_budget",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-deterministic-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert not any(str(event.type).startswith("budget.") for event in events)

    asyncio.run(run())


def test_compact_session_rejects_unknown_compactor_budget_identity_before_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        compactor = UndeclaredProviderCompactor()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                    ),
                )
            ),
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
                session_id="sess_unknown_compactor_budget_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="declare provider_budget_identity"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-unknown-budget-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.calls == 0
        assert await store.load_checkpoint(created.id) is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "identity",
    [
        ("compaction-provider\x00suffix", "summary-model"),
        ("compaction-provider", "summary\nmodel"),
        ("compaction-provider", "summary-\ud800-model"),
        ("compaction-provider", "m" * 513),
    ],
)
def test_compact_session_rejects_unsafe_compactor_budget_identity_before_work(
    identity: tuple[str, str],
) -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        compactor = UnsafeProviderIdentityCompactor(identity)
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                    ),
                )
            ),
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
                session_id="sess_unsafe_compactor_budget_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="without control characters"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-unsafe-budget-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.calls == 0
        assert await store.load_checkpoint(created.id) is None

    asyncio.run(run())


def test_compact_session_persists_reservation_lifecycle_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_persisted_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        stream = app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-persisted-reservation",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )

        started = await anext(stream)
        checked = await anext(stream)
        reserved = await anext(stream)
        assert [started.type, checked.type, reserved.type] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
        ]
        assert provider.calls == 0
        durable_before_close = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert [event.id for event in durable_before_close] == [
            started.id,
            checked.id,
            reserved.id,
        ]

        await stream.aclose()

        durable_after_close = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert [event.type for event in durable_after_close] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_RELEASED,
        ]
        assert durable_after_close[-1].payload["operation_id"] == started.payload["operation_id"]
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        record = checkpoint["session_operations"]["records"]["compact-persisted-reservation"]
        assert record["event_ids"] == [event.id for event in durable_after_close]

    asyncio.run(run())
