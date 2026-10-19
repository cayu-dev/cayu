"""Explicit compaction usage evidence and billing identity through failures."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from tests.core._bounded_waits import wait_for_caller_settlement, wait_for_fake_step
from tests.core._explicit_compaction_fixtures import (
    ContextualBedrockCompactionProvider,
    FinalRenewalFailureBudgetLedger,
    UsageCompactionProvider,
    _create_profiled_session,
    _public_ids_for_private_event_ids,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactor,
    ModelCompactor,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent, UsageDialect
from cayu.providers.bedrock import bedrock_billing_identity, completed_bedrock_billing_identity
from cayu.runtime._compaction import explicit as _session_compaction
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_sequence
from cayu.runtime._run_limits import BudgetReservationLeaseLost
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import CompactSessionRequest, RunRequest


def test_compact_session_invalid_usage_fails_closed_on_next_strict_budget_check() -> None:
    class InvalidUsageProvider(ModelProvider):
        name = "invalid-usage-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage_metrics": {
                        "input_tokens": 2**63,
                        "output_tokens": 1,
                        "total_tokens": 2**63,
                    },
                }
            )

    async def run() -> None:
        store = InMemorySessionStore()
        provider = InvalidUsageProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        allow_unpriced=False,
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
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_usage_strict_budget",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ContextBuildError, match="integer_out_of_range"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="invalid-usage-first",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass
        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="invalid-usage-second",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert provider.calls == 1
        completions = [
            event
            for event in await store.load_events(created.id)
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert "usage_metrics" not in completions[0].payload
        assert completions[0].payload["usage_unavailable_reason"] == (
            "invalid compaction usage telemetry"
        )

    asyncio.run(run())


class HeartbeatCancellationBudgetLedger(FinalRenewalFailureBudgetLedger):
    def __init__(self, provider_started: asyncio.Event) -> None:
        super().__init__()
        self.provider_started = provider_started

    async def heartbeat(self, *, reservation_id: str) -> bool:
        if self.heartbeat_calls == 1:
            await self.provider_started.wait()
        return await super().heartbeat(reservation_id=reservation_id)

    @property
    def reservation_ttl_seconds(self) -> int:
        return 0


class CancellationCompletingProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()

    async def stream(self, request: ModelRequest):
        self.calls += 1
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The in-flight read returns terminal evidence directly. A text
            # delta would require a new read after cancellation to reach usage.
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})
            return
        raise AssertionError("blocking compactor unexpectedly resumed")


class AdversarialTelemetryCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.billing_identity = BillingIdentity(
            provider_name="fake-compactor",
            resource_id="summary-model-v1",
        )

    def provider_budget_identity(self, _session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        return CompactionResult(
            summary="private summary text",
            covered_message_count=len(request.messages),
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
                    "usage_metrics": {
                        "provider_name": "fake-compactor",
                        "requested_model": "summary-model",
                        "model": "summary-model-v1",
                        "billing_identity": self.billing_identity.model_dump(mode="json"),
                        "input_tokens": 8,
                        "output_tokens": 2,
                        "total_tokens": 10,
                    },
                    "summary": "model event summary leak",
                    "instructions": "model event instruction leak",
                    "provider_state": {"secret": "opaque continuation leak"},
                    "arbitrary": "p" * 100_000,
                }
            ],
        )


def test_compact_session_rejects_unobserved_adversarial_completion_telemetry() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        compactor = AdversarialTelemetryCompactor()
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
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

        with pytest.raises(
            RuntimeError,
            match="Compaction completion was observed outside its provider dispatch",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-private-events",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    instructions="private caller instructions",
                )
            ):
                pass

        events = await store.load_events(created.id)

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
        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[0].payload["model_step_id"] == events[1].payload["model_step_id"]
        assert "model_attempt_id" not in events[1].payload

    asyncio.run(run())


def test_compact_session_rejects_second_completion_for_one_provider_dispatch() -> None:
    class DuplicatingWrapperCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            result = await ModelCompactor(
                provider=self.provider,
                model="summary-model",
            ).compact(request)
            forged = dict(result.model_completed_payloads[0])
            forged.pop("compaction_attempt_id", None)
            return result.model_copy(
                update={
                    "model_completed_payloads": [
                        *result.model_completed_payloads,
                        forged,
                    ]
                }
            )

    async def run() -> None:
        store = InMemorySessionStore()
        provider = UsageCompactionProvider()
        app = CayuApp(
            session_store=store,
            request_footprint=RequestFootprintConfig(enabled=False),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=DuplicatingWrapperCompactor(provider),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_duplicate_dispatch_completion",
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

        with pytest.raises(
            ValueError,
            match="Compaction provider dispatch produced conflicting completion identities",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-duplicate-dispatch-completion",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        events = await store.load_events(created.id)
        assert provider.calls == 1
        assert sum(event.type == EventType.MODEL_COMPLETED for event in events) == 1
        assert events[-1].type == EventType.CONTEXT_COMPACTION_FAILED

    asyncio.run(run())


def test_compact_session_generator_exit_preserves_completed_hierarchy_usage() -> None:
    class AbandonSecondHierarchyCallProvider(ModelProvider):
        name = "abandon-hierarchy"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 2:
                raise GeneratorExit("abandon hierarchy after first completion")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_explicit_hierarchy_generator_exit"
        store = InMemorySessionStore()
        provider = AbandonSecondHierarchyCallProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=session_id,
            idempotency_key="explicit-generator-exit",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(GeneratorExit, match="abandon hierarchy after first completion"):
            async for _event in app.compact_session(request):
                pass

        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=session_id, limit=100))
        ]
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 2
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert completions[1].payload["compaction_outcome"] == "unfinished_stream"
        assert completions[1].payload["usage_unavailable_reason"]
        failures = [event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["error_type"] == "GeneratorExit"
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = await store.load_session_operation(session_id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_compact_session_generator_exit_propagates_abandonment_accounting_failure() -> None:
    class FailingAbandonmentPublishStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            events = kwargs.get("events", [])
            if any(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events):
                raise RuntimeError("simulated abandonment accounting failure")
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class AbandonSecondHierarchyCallProvider(ModelProvider):
        name = "abandon-hierarchy-accounting-failure"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 2:
                raise GeneratorExit("abandon hierarchy before accounting failure")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_explicit_hierarchy_abandonment_accounting_failure"
        store = FailingAbandonmentPublishStore()
        provider = AbandonSecondHierarchyCallProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=session_id,
            idempotency_key="explicit-generator-exit-accounting-failure",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(
            RuntimeError,
            match="simulated abandonment accounting failure",
        ) as exc_info:
            async for _event in app.compact_session(request):
                pass

        assert isinstance(exc_info.value.__cause__, GeneratorExit)
        assert exc_info.value.__cause__.args == ("abandon hierarchy before accounting failure",)
        assert any(
            "accounting failure is authoritative" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )
        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=session_id, limit=100))
        ]
        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
        ]
        assert events[3].payload["model"] == "summary-model"
        assert events[3].payload["usage"] == {"input_tokens": 5, "output_tokens": 2}
        assert events[3].payload["purpose"] == "context_compaction"
        assert events[6].payload["compaction_outcome"] == "unfinished_stream"
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = await store.load_session_operation(session_id, request.idempotency_key)
        assert operation is None

    asyncio.run(run())


def test_compact_session_generator_exit_keeps_accounting_failure_authoritative_when_heartbeat_fails(
    monkeypatch,
) -> None:
    class FailingHeartbeatAndAbandonmentStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_heartbeat_reconciliation = False
            self.heartbeat_failed = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if self.fail_heartbeat_reconciliation and kwargs.get("events") == []:
                self.fail_heartbeat_reconciliation = False
                self.heartbeat_failed.set()
                raise _session_compaction.SessionCompactionAttemptSuperseded(
                    "Session compaction operation ownership changed during renewal."
                )
            if any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in kwargs.get("events", [])
            ):
                self.fail_heartbeat_reconciliation = True
                await asyncio.wait_for(self.heartbeat_failed.wait(), timeout=5)
                raise RuntimeError("simulated abandonment accounting failure")
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class AbandonAfterHeartbeatFailureProvider(ModelProvider):
        name = "abandon-after-heartbeat-failure"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 2:
                raise GeneratorExit("abandon after heartbeat failure")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        session_id = "sess_explicit_abandonment_accounting_and_heartbeat_failure"
        store = FailingHeartbeatAndAbandonmentStore()
        provider = AbandonAfterHeartbeatFailureProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)

        with pytest.raises(
            RuntimeError,
            match="simulated abandonment accounting failure",
        ) as exc_info:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="explicit-generator-exit-accounting-and-heartbeat-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert any(
            "accounting failure is authoritative" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )
        assert isinstance(exc_info.value.__cause__, BaseExceptionGroup)
        heartbeat_failures = exc_info.value.__cause__.exceptions
        assert len(heartbeat_failures) == 1
        assert isinstance(heartbeat_failures[0], RuntimeError)
        assert "ownership changed" in str(heartbeat_failures[0])
        assert isinstance(exc_info.value.__cause__.__cause__, GeneratorExit)
        assert exc_info.value.__cause__.__cause__.args == ("abandon after heartbeat failure",)

    asyncio.run(run())


def test_compact_session_cancellation_preserves_completed_hierarchy_usage() -> None:
    class CancellationBoundaryStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failure_publish_started = asyncio.Event()
            self.allow_failure_publish = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            events = kwargs.get("events", [])
            if any(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events):
                self.failure_publish_started.set()
                await self.allow_failure_publish.wait()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class CancelSecondHierarchyCallProvider(ModelProvider):
        name = "cancel-explicit-hierarchy"

        def __init__(self) -> None:
            self.calls = 0
            self.second_started = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 2:
                self.second_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_explicit_hierarchy_cancelled"
        store = CancellationBoundaryStore()
        provider = CancelSecondHierarchyCallProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=session_id,
            idempotency_key="explicit-cancelled-hierarchy",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def consume() -> None:
            async for _event in app.compact_session(request):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(provider.second_started.wait(), timeout=10)
        task.cancel("cancel explicit hierarchy after first completion")
        await asyncio.wait_for(store.failure_publish_started.wait(), timeout=10)
        task.cancel("later cancellation during failure publication")
        store.allow_failure_publish.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel explicit hierarchy after first completion",)
        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=session_id, limit=100))
        ]
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 2
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert completions[1].payload["compaction_outcome"] == "cancelled"
        assert completions[1].payload["usage_unavailable_reason"]
        failures = [event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["error_type"] == "CancelledError"
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = await store.load_session_operation(session_id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

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
        created = await _create_profiled_session(
            app,
            store,
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
        terminal_context_events = [
            event
            for event in first
            if event.type
            in {
                EventType.CONTEXT_COMPACTION_COMPLETED,
                EventType.SESSION_CHECKPOINTED,
            }
        ]
        assert usage_event.payload["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
        usage_sequence = public_event_sequence(usage_event.id)
        assert usage_sequence is not None
        private_usage_record = next(
            record
            for record in await store.query_events(
                EventQuery(
                    session_id=created.id,
                    after_sequence=usage_sequence - 1,
                    limit=1,
                )
            )
            if record.sequence == usage_sequence
        )
        assert private_usage_record.event.payload["model_attempt_id"].startswith("matt_")
        assert all(
            event.payload["model_step_id"] == usage_event.payload["model_step_id"]
            for event in terminal_context_events
        )
        assert all("model_attempt_id" not in event.payload for event in terminal_context_events)

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
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        checkpoint_after_limit = await store.load_checkpoint(created.id)
        assert (
            checkpoint_after_limit["context_compaction"]
            == checkpoint_before_limit["context_compaction"]
        )

    asyncio.run(run())


@pytest.mark.parametrize("invalid_text", ["100\x00", "\ud800"], ids=["nul", "surrogate"])
@pytest.mark.parametrize(
    "invalid_primary_counter",
    [True, False],
    ids=["invalid-primary", "invalid-extra-field"],
)
def test_explicit_compaction_persists_undurable_usage_as_unpriced_evidence(
    invalid_text: str,
    invalid_primary_counter: bool,
) -> None:
    class UndurableUsageProvider(ModelProvider):
        name = "renamed-openai"
        usage_dialect = UsageDialect.OPENAI

        async def stream(self, request: ModelRequest):
            yield ModelStreamEvent.text_delta("provider summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
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
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="renamed-openai",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
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
                compactor=ModelCompactor(
                    provider=UndurableUsageProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=(
                    "explicit-undurable-compaction-"
                    f"{invalid_primary_counter}-{ord(invalid_text[-1])}"
                ),
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed_session = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []
        with pytest.raises(ContextBuildError):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-undurable-compaction",
                    expected_run_epoch=completed_session.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)
        completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
        reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
        cost = await app.get_session_cost(created.id, pricing)

        assert "usage" not in completed.payload
        assert "usage_metrics" not in completed.payload
        assert completed.payload["usage_normalization_failed"] is True
        assert completed.payload["usage_unavailable_reason"] == (
            "invalid compaction usage telemetry"
        )
        assert cost.model_steps == 1
        assert cost.priced_model_steps == 0
        assert cost.unpriced_model_steps == 1
        assert reserved.payload["actual"] == "1"
        assert reconciled.payload["actual_amount"] == "1"
        assert reconciled.payload["released_amount"] == "0"
        assert reconciled.payload["reason"] == (
            "compaction completed without priced usage; charged reserved amount"
        )
        ledger_record = next(iter(ledger._records.values()))
        assert ledger_record.reason == (
            "compaction completed without priced usage; charged reserved amount"
        )
        assert events[-1].type == EventType.CONTEXT_COMPACTION_FAILED

    asyncio.run(run())


def test_compact_session_deduplicates_persisted_usage_for_session_run_limit() -> None:
    async def run() -> None:
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_session_limit_deduplication",
                messages=[],
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
                    idempotency_key="compact-session-limit-deduplication",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    limits=RunLimits(max_total_tokens=15, scope="session"),
                )
            )
        ]

        assert provider.calls == 1
        completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events) == 1
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2

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
        created = await _create_profiled_session(
            app,
            store,
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
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert provider.calls == 1
        usage_event = first[5]
        assert usage_event.payload["compaction_outcome"] == "invalid_summary"
        assert usage_event.payload["usage_metrics"]["total_tokens"] == 10
        assert "compaction_attempt_id" not in usage_event.payload
        assert first[6].payload["actual_amount"] == "0.000012"
        assert first[7].payload["error_type"] == "ContextBuildError"
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
        assert [event.id for event in first] == await _public_ids_for_private_event_ids(
            store,
            created.id,
            operation["event_ids"],
        )

    asyncio.run(run())


def test_compact_session_cancellation_reconciles_observed_completion_usage() -> None:
    class CompletedThenBlockingProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(self) -> None:
            self.completed = asyncio.Event()
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("provider summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})
            self.completed.set()
            await asyncio.Event().wait()

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
        provider = CompletedThenBlockingProvider()
        ledger = InMemoryBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
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
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_cancel_after_compaction_completion",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="cancel-after-compaction-completion",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await wait_for_fake_step(provider.completed, task)
        task.cancel("cancel after completed")
        await wait_for_caller_settlement(task)
        with pytest.raises(asyncio.CancelledError, match="cancel after completed"):
            await task

        durable = await store.load_events(created.id)
        completion = next(event for event in durable if event.type == EventType.MODEL_COMPLETED)
        reconciliation = next(
            event for event in durable if event.type == EventType.BUDGET_RECONCILED
        )
        assert completion.payload["usage_metrics"]["total_tokens"] == 10
        assert reconciliation.payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.actual_amount == Decimal("0.000012")
        assert record.status == "reconciled"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("terminal_signal", "stream_delay_seconds"),
    [
        pytest.param("caller_cancellation", 0.0, id="caller_cancellation"),
        pytest.param("lease_loss", 0.0, id="lease_loss"),
        # A slow runner reaches the provider after the renewal loop's first tick
        # (TTL / 3). The lease must still be lost only after completion.
        pytest.param("lease_loss", 0.5, id="lease_loss_after_slow_stream_start"),
    ],
)
def test_compact_session_preserves_observed_usage_when_terminal_signal_replaces_child_failure(
    terminal_signal: str,
    stream_delay_seconds: float,
) -> None:
    class CompletedToolCallThenBlockingProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(self) -> None:
            self.completed = asyncio.Event()
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if stream_delay_seconds:
                await asyncio.sleep(stream_delay_seconds)
            yield ModelStreamEvent.text_delta("unusable provider summary")
            yield ModelStreamEvent.tool_call(name="unexpected", arguments={})
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})
            self.completed.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                # The provider's own failure turns the already-observed tool call into
                # an ordinary protocol failure while the outer signal stays primary.
                raise RuntimeError("provider failed during stream cleanup") from exc

    class LoseLeaseOnSecondHeartbeatLedger(InMemoryBudgetLedger):
        """Lose the lease on the first timed renewal, after the provider completes.

        The one-second TTL only makes the renewal loop tick quickly. The lease loss
        waits for the provider's completion so a slow runner cannot lose the lease
        before the tool call and usage are observed, and the frozen clock keeps
        the reservation from expiring by wall time before dispatch.
        """

        def __init__(self, provider_completed: asyncio.Event) -> None:
            frozen_now = datetime.now(UTC)
            super().__init__(clock=lambda: frozen_now, reservation_ttl_seconds=1)
            self.heartbeat_calls = 0
            self.provider_completed = provider_completed

        async def heartbeat(self, *, reservation_id: str) -> bool:
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 2:
                await self.provider_completed.wait()
                return False
            return await super().heartbeat(reservation_id=reservation_id)

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
        provider = CompletedToolCallThenBlockingProvider()
        ledger = (
            LoseLeaseOnSecondHeartbeatLedger(provider.completed)
            if terminal_signal == "lease_loss"
            else InMemoryBudgetLedger(reservation_ttl_seconds=30)
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
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
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_child_failure_{terminal_signal}",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key=f"child-failure-{terminal_signal}",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await wait_for_fake_step(provider.completed, task)
        if terminal_signal == "caller_cancellation":
            task.cancel("cancel after completed tool call")
            await wait_for_caller_settlement(task)
            with pytest.raises(
                asyncio.CancelledError,
                match="cancel after completed tool call",
            ):
                await task
            assert task.cancelling() == 0
            assert task.cancelled()
        else:
            await wait_for_caller_settlement(task)
            with pytest.raises(ContextBuildError, match="lease was lost") as raised:
                await task
            assert isinstance(raised.value.__cause__, BudgetReservationLeaseLost)
            assert raised.value.__cause__.__dict__["completed_metadata"]["usage"] == {
                "input_tokens": 8,
                "output_tokens": 2,
            }

        durable = await store.load_events(created.id)
        completion_events = [event for event in durable if event.type == EventType.MODEL_COMPLETED]
        assert len(completion_events) == 1
        assert completion_events[0].payload["usage_metrics"]["total_tokens"] == 10
        assert completion_events[0].payload["compaction_outcome"] == (
            "cancelled_after_completion"
            if terminal_signal == "caller_cancellation"
            else "provider_error_after_completion"
        )
        reconciliations = [event for event in durable if event.type == EventType.BUDGET_RECONCILED]
        assert len(reconciliations) == 1
        assert reconciliations[0].payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.actual_amount == Decimal("0.000012")
        assert record.status == "reconciled"
        assert provider.calls == 1
        if isinstance(ledger, LoseLeaseOnSecondHeartbeatLedger):
            assert ledger.heartbeat_calls == 2

    asyncio.run(run())


def test_compact_session_cancellation_during_final_renewal_reconciles_actual_usage() -> None:
    class BlockingFinalRenewalLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=30)
            self.heartbeat_calls = 0
            self.final_renewal_started = asyncio.Event()

        async def heartbeat(self, *, reservation_id: str) -> bool:
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 2:
                self.final_renewal_started.set()
                await asyncio.Event().wait()
            return await super().heartbeat(reservation_id=reservation_id)

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
        ledger = BlockingFinalRenewalLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
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
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_cancel_during_final_compaction_renewal",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="cancel-during-final-compaction-renewal",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(ledger.final_renewal_started.wait(), timeout=10)
        task.cancel("cancel during final reservation renewal")
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel during final reservation renewal",
        ):
            await task

        durable = await store.load_events(created.id)
        completion = next(event for event in durable if event.type == EventType.MODEL_COMPLETED)
        reconciliation = next(
            event for event in durable if event.type == EventType.BUDGET_RECONCILED
        )
        assert provider.calls == 1
        assert completion.payload["usage_metrics"]["total_tokens"] == 10
        assert reconciliation.payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.actual_amount == Decimal("0.000012")
        assert record.status == "reconciled"

    asyncio.run(run())


def test_compact_session_failure_after_completion_reconciles_actual_usage() -> None:
    class EventAfterCompletionProvider(ModelProvider):
        name = "compaction-provider"

        async def stream(self, request: ModelRequest):
            yield ModelStreamEvent.text_delta("provider summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})
            yield ModelStreamEvent.text_delta("invalid trailing event")

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
        ledger = InMemoryBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
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
                compactor=ModelCompactor(
                    provider=EventAfterCompletionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_event_after_completion",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(
            RuntimeError,
            match="Compaction provider emitted event after completed: text_delta",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="event-after-compaction-completion",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        durable = await store.load_events(created.id)
        completion = next(event for event in durable if event.type == EventType.MODEL_COMPLETED)
        reconciliation = next(
            event for event in durable if event.type == EventType.BUDGET_RECONCILED
        )
        assert completion.payload["usage_metrics"]["total_tokens"] == 10
        assert reconciliation.payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.actual_amount == Decimal("0.000012")
        assert record.status == "reconciled"

    asyncio.run(run())


def test_compact_session_uses_frozen_canonical_billing_provider_without_identity() -> None:
    class GatewayProvider(UsageCompactionProvider):
        name = "gateway"
        billing_provider_name = "billco"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            assert request.model == "summary-model"
            self.name = "poisoned\x00provider"
            self.billing_provider_name = "poisoned\ud800billing"
            return None

    async def run() -> None:
        provider = GatewayProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="billco",
                    model="summary-model",
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
                        max_estimated_cost=Decimal("1"),
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
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_gateway_compaction_billing",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="gateway-compaction-billing",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        completion = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
        reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
        assert provider.calls == 1
        assert completion.payload["provider_name"] == "billco"
        assert completion.payload["usage_metrics"]["provider_name"] == "billco"
        assert reserved.payload["provider_name"] == "billco"
        assert reconciled.payload["actual_amount"] == "0.000012"
        assert provider.name == "poisoned\x00provider"
        assert provider.billing_provider_name == "poisoned\ud800billing"

    asyncio.run(run())


def test_compact_session_resolves_bedrock_identity_before_reservation_and_replays() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
        )
        provider = ContextualBedrockCompactionProvider(identity)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model=identity.resource_id,
                    match="exact",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                    pricing_context={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    },
                ),
            )
        )
        ledger = InMemoryBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("10"),
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
                compactor=ModelCompactor(provider=provider, model=identity.resource_id),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_contextual_explicit_compaction",
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
            idempotency_key="contextual-explicit-compaction",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]

        assert provider.calls == 1
        assert [event.id for event in replay] == [event.id for event in first]
        reserved = next(event for event in first if event.type == EventType.BUDGET_RESERVED)
        reconciled = next(event for event in first if event.type == EventType.BUDGET_RECONCILED)
        completion = next(event for event in first if event.type == EventType.MODEL_COMPLETED)
        completed_identity = BillingIdentity.model_validate(completion.payload["billing_identity"])
        assert reserved.payload["billing_identity"] == identity.model_dump(mode="json")
        assert reconciled.payload["billing_identity"] == completion.payload["billing_identity"]
        assert reconciled.payload["actual_amount"] == "0.000054"
        assert next(iter(ledger._records.values())).billing_identity == completed_identity
        assert len([event for event in first if event.type == EventType.BUDGET_CHECKED]) == 1

    asyncio.run(run())


def test_compact_session_preserves_bedrock_identity_without_usage_metrics() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
        )
        completed_identity = completed_bedrock_billing_identity(
            identity,
            effective_service_tier="default",
        )
        provider = ContextualBedrockCompactionProvider(identity, report_usage=False)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model=identity.resource_id,
                    match="exact",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                    pricing_context={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    },
                ),
            )
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model=identity.resource_id),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_contextual_compaction_without_usage",
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
            idempotency_key="contextual-compaction-without-usage",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]

        assert provider.calls == 1
        assert [event.id for event in replay] == [event.id for event in first]
        completion = next(event for event in first if event.type == EventType.MODEL_COMPLETED)
        assert completion.payload["billing_identity"] == completed_identity.model_dump(mode="json")
        assert "usage_metrics" not in completion.payload
        completion_sequence = public_event_sequence(completion.id)
        assert completion_sequence is not None
        stored_completion = next(
            record.event
            for record in await store.query_events(
                EventQuery(
                    session_id=created.id,
                    after_sequence=completion_sequence - 1,
                    limit=1,
                )
            )
            if record.sequence == completion_sequence
        )
        assert (
            stored_completion.payload["billing_identity"] == completion.payload["billing_identity"]
        )
        cost = await app.get_session_cost(created.id, pricing)
        assert cost.model_steps == 1
        assert cost.unpriced_model_steps == 1
        assert cost.line_items[0].billing_identity == completed_identity
        assert (
            cost.line_items[0].missing_pricing_reason
            == "model.completed event has no token usage metrics"
        )

    asyncio.run(run())


def test_compact_session_rejects_completion_billing_identity_rewrite() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
        )

        class RewritingCompactionProvider(ContextualBedrockCompactionProvider):
            def billing_identity_for_completion(
                self,
                requested: BillingIdentity | None,
                payload: dict,
            ) -> BillingIdentity | None:
                assert requested == identity
                return identity.model_copy(
                    update={
                        "request_evidence": {
                            **identity.request_evidence,
                            "source_region": "eu-west-1",
                        }
                    }
                )

        provider = RewritingCompactionProvider(identity)
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model=identity.resource_id),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_rewritten_compaction_billing_identity",
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

        with pytest.raises(
            ContextBuildError,
            match="Model provider billing identity resolution failed",
        ) as exc_info:
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="rewritten-compaction-billing-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 1
        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        completion = events[3]
        assert completion.payload["billing_identity"] == identity.model_dump(mode="json")
        assert completion.payload["usage_metrics"]["input_tokens"] == 8
        assert completion.payload["usage_metrics"]["output_tokens"] == 2
        assert completion.payload["usage_metrics"]["total_tokens"] == 10
        assert completion.payload["compaction_outcome"] == "completion_observation_failed"
        assert "compaction_attempt_id" not in completion.payload
        failed = events[-1]
        assert failed.payload["error_type"] == type(exc_info.value).__name__
        assert "error" not in failed.payload
        assert exc_info.value.cause is not None
        assert "Model provider billing identity resolution failed" in str(exc_info.value)
        assert "Model provider billing identity resolution failed" in str(exc_info.value.cause)
        assert await store.load_transcript(created.id) == transcript
        completion_sequence = public_event_sequence(completion.id)
        assert completion_sequence is not None
        stored_completion = await store.query_events(
            EventQuery(
                session_id=created.id,
                after_sequence=completion_sequence - 1,
                limit=1,
            )
        )
        assert [record.sequence for record in stored_completion] == [completion_sequence]
        usage = await app.get_session_usage(created.id)
        assert usage.model_steps == 1
        assert usage.usage.input_tokens == 8
        assert usage.usage.output_tokens == 2
        assert usage.usage.total_tokens == 10

    asyncio.run(run())


def test_compact_session_rejects_unpriced_bedrock_identity_before_dispatch() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="eu-west-1",
            resource_type="inference_profile",
            profile_scope="global",
        )
        provider = ContextualBedrockCompactionProvider(identity)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model=identity.resource_id,
                    match="exact",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                    pricing_context={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    },
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
                        max_estimated_cost=Decimal("10"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model=identity.resource_id),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unpriced_contextual_explicit_compaction",
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
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="unpriced-contextual-explicit-compaction",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 0
        assert EventType.BUDGET_RESERVED not in [event.type for event in events]
        assert EventType.BUDGET_LIMIT_REACHED in [event.type for event in events]
        assert events[-1].type == EventType.CONTEXT_COMPACTION_FAILED

    asyncio.run(run())


def test_compact_session_does_not_claim_unobserved_usage_returned_after_heartbeat_cancels() -> None:
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
        provider = CancellationCompletingProvider()
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
            budget_ledger=HeartbeatCancellationBudgetLedger(provider.started),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
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
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[5].payload["compaction_outcome"] == "unfinished_stream"
        assert events[5].payload["usage_unavailable_reason"] == (
            "compaction provider dispatch ended without completion usage"
        )
        assert "usage_metrics" not in events[5].payload
        assert events[6].payload["actual_amount"] == "0.00003"
        assert provider.calls == 1

    asyncio.run(run())
