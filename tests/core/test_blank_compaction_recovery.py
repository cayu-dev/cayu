from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from cayu import AgentSpec, Message
from cayu.providers import ModelCompletion, ModelProvider, ModelProviderError, ModelStreamEvent
from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    InMemorySessionStore,
    ModelCompactor,
    ModelPrice,
    PriceBook,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.storage import SQLiteSessionStore


class SummaryProvider(ModelProvider):
    name = "test-summary"

    def __init__(self, answers, *, finish="stop", status="completed"):
        self.answers = answers
        self.finish = finish
        self.status = status
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        answer = self.answers[len(self.requests) - 1]
        if isinstance(answer, BaseException):
            raise answer
        if answer:
            yield ModelStreamEvent.text_delta(answer)
        yield ModelStreamEvent(
            type="completed",
            payload={"status": self.status, "usage": {"input_tokens": 10, "output_tokens": 2}},
            completion=ModelCompletion(finish_reason=self.finish, status=self.status),
        )


def compactor(provider, *, enabled=True, attempts=2, policy=None):
    return ModelCompactor(
        provider=provider,
        model="summary",
        retry_empty_summaries=enabled,
        retry_policy=policy or RetryPolicy(max_attempts=attempts, initial_delay_s=0.0),
    )


async def compact(provider, **kwargs):
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name=provider.name, model="summary"),
    )
    return await compactor(provider, **kwargs).compact(
        CompactionRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="summary"),
            messages=[Message.text("user", "Preserve my pending request.")],
        )
    )


@pytest.mark.parametrize("blank", ["", " \n\t"])
def test_completed_blank_summary_retries_and_accounts_for_both_calls(blank):
    provider = SummaryProvider([blank, "Preserved request."])
    result = asyncio.run(compact(provider))
    assert result.summary == "Preserved request."
    assert result.covered_message_count == 1
    assert len(provider.requests) == 2
    assert len(result.model_completed_payloads) == 2
    assert result.model_completed_payloads[0]["compaction_outcome"] == "empty_summary"
    assert sum(p["usage_metrics"]["input_tokens"] for p in result.model_completed_payloads) == 20


def test_output_recovery_does_not_enable_network_retries():
    policy = RetryPolicy(
        max_attempts=3,
        max_unknown_attempts=1,
        initial_delay_s=0.0,
        jitter_s=0.0,
        retry_on_connection_error=False,
        retry_on_timeout=False,
        retry_on_rate_limit=False,
        retry_on_status_codes=(),
    )
    provider = SummaryProvider(["", "Recovered"])
    result = asyncio.run(compact(provider, policy=policy))
    assert result.summary == "Recovered"
    assert len(provider.requests) == 2
    assert len(result.model_completed_payloads) == 2

    provider = SummaryProvider(
        [
            "",
            ModelProviderError("Connection reset", provider="test-summary", retryable=True),
            "Must not be reached",
        ]
    )
    with pytest.raises(ModelProviderError, match="Connection reset"):
        asyncio.run(compact(provider, policy=policy))
    assert len(provider.requests) == 2


def test_empty_summary_attempts_are_bounded():
    provider = SummaryProvider(["", "", "Must not be reached"])
    with pytest.raises(ModelProviderError, match="nonblank summary") as raised:
        asyncio.run(compact(provider))
    assert raised.value.error_code == "compaction_empty_summary"
    assert len(provider.requests) == 2


def test_recovery_is_opt_in():
    provider = SummaryProvider(["", "Must not be reached"])
    with pytest.raises(ValueError, match="cannot be blank"):
        asyncio.run(compact(provider, enabled=False))
    assert len(provider.requests) == 1


def test_one_attempt_policy_still_prevents_retry_when_enabled():
    provider = SummaryProvider(["", "Must not be reached"])
    with pytest.raises(ModelProviderError, match="nonblank summary"):
        asyncio.run(compact(provider, attempts=1))
    assert len(provider.requests) == 1


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_retry_option_requires_a_boolean(invalid):
    with pytest.raises(TypeError, match="boolean"):
        ModelCompactor(provider=SummaryProvider([]), model="summary", retry_empty_summaries=invalid)


def test_invalid_completion_metadata_keeps_completed_usage_without_retry():
    class MalformedCompletionProvider(SummaryProvider):
        async def stream(self, request):
            async for event in super().stream(request):
                if event.type == "completed":
                    event.completion = ModelCompletion.model_construct(
                        finish_reason="not-a-finish-reason",
                        raw_finish_reason=None,
                        status="completed",
                        end_turn=None,
                    )
                yield event

    async def run():
        provider = MalformedCompletionProvider(["", "Must not be reached"])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="summary"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor(provider), max_user_turns=1, compact_after_messages=2
            ),
        )
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "answer"),
                        Message.text("user", "new"),
                    ],
                )
            )
        ]
        assert len(provider.requests) == 1
        completions = [e for e in events if e.type == "model.completed"]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 12
        assert any(e.type == "session.failed" for e in events)
        assert not any(e.type == "context.compaction.completed" for e in events)

    asyncio.run(run())


def test_cancellation_after_completed_blank_stream_is_not_retried():
    class CancelAfterCompletion(SummaryProvider):
        async def stream(self, request):
            async for event in super().stream(request):
                yield event
            raise asyncio.CancelledError()

    provider = CancelAfterCompletion(["", "Must not be reached"])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(compact(provider))
    assert len(provider.requests) == 1


def test_tool_call_completion_is_not_retried_as_empty():
    class ToolCallingProvider(SummaryProvider):
        async def stream(self, request):
            yield ModelStreamEvent.tool_call(id="call-1", name="forbidden", arguments={})
            async for event in super().stream(request):
                yield event

    provider = ToolCallingProvider(["", "Must not be reached"])
    with pytest.raises(RuntimeError, match="must not call tools"):
        asyncio.run(compact(provider))
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "finish,status",
    [("content_filter", "completed"), ("length", "incomplete"), ("unknown", "completed")],
)
def test_other_terminal_outcomes_are_not_empty_summary_retries(finish, status):
    provider = SummaryProvider(["", "Must not be reached"], finish=finish, status=status)
    with pytest.raises(ValueError, match="cannot be blank"):
        asyncio.run(compact(provider))
    assert len(provider.requests) == 1


def test_cancellation_does_not_trigger_replacement():
    provider = SummaryProvider([asyncio.CancelledError(), "Must not be reached"])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(compact(provider))
    assert len(provider.requests) == 1


def test_recovery_changes_profile_without_changing_default_material():
    from cayu.runtime._execution_profile_admission import _cayu_compactor_material

    provider = SummaryProvider([])
    disabled = _cayu_compactor_material(
        compactor(provider, enabled=False), behavior_identities={}, process_identity="test"
    )
    enabled = _cayu_compactor_material(
        compactor(provider), behavior_identities={}, process_identity="test"
    )
    assert "retry_empty_summaries" not in disabled
    assert enabled == {**disabled, "retry_empty_summaries": True}


def test_retry_cannot_bypass_budget_admission():
    async def run():
        provider = SummaryProvider(["", "Must not be dispatched"])
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.000025"),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name=provider.name,
                                    model="summary",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                ),
                            )
                        ),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=10),
                    ),
                )
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="summary"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor(provider), max_user_turns=1, compact_after_messages=2
            ),
        )
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="budgeted-blank",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "answer"),
                        Message.text("user", "new"),
                    ],
                )
            )
        ]
        assert len(provider.requests) == 1
        assert any(e.type == "budget.limit_reached" for e in events)
        completions = [e for e in events if e.type == "model.completed"]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 12
        assert "context_compaction" not in await store.load_checkpoint("budgeted-blank")

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("recovers", [True, False])
def test_runtime_records_each_completion_once_before_advancing_coverage(
    tmp_path, backend, recovers
):
    async def run():
        summaries = SummaryProvider(["", "Preserved old conversation." if recovers else ""])
        actor = SummaryProvider(["Done"])
        path = tmp_path / "sessions.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(actor, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="actor"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor(summaries),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="blank-recovery",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "new"),
                    ],
                )
            )
        ]
        completed = [
            e
            for e in events
            if e.type == "model.completed" and e.payload.get("purpose") == "context_compaction"
        ]
        assert len(completed) == 2
        assert len({e.id for e in completed}) == 2
        assert sum(e.payload["usage_metrics"]["input_tokens"] for e in completed) == 20
        assert any(e.type == "context.compaction.completed" for e in events) == recovers
        assert any(e.type == "session.failed" for e in events) != recovers
        assert len(actor.requests) == int(recovers)
        checkpoint = await store.load_checkpoint("blank-recovery")
        if recovers:
            assert checkpoint["context_compaction"]["summary"] == "Preserved old conversation."
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] > 0
        else:
            assert "context_compaction" not in checkpoint
        if backend == "sqlite":
            await store.close()
            store = SQLiteSessionStore(path)
            assert await store.load_checkpoint("blank-recovery") == checkpoint
        durable = await store.load_events("blank-recovery")
        recorded = [
            e
            for e in durable
            if e.type == "model.completed" and e.payload.get("purpose") == "context_compaction"
        ]
        assert len(recorded) == 2
        assert len({e.payload["model_attempt_id"] for e in recorded}) == 2
        if backend == "sqlite":
            await store.close()

    asyncio.run(run())


def test_failed_replacement_preserves_previous_summary_and_cursor(tmp_path):
    async def run():
        path = tmp_path / "previous.sqlite"
        store = SQLiteSessionStore(path)
        summaries = SummaryProvider(["Original valid summary.", "", ""])
        actor = SummaryProvider(["First response"])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(actor, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="actor"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor(summaries),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )
        first = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="previous",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                )
            )
        ]
        assert not any(e.type == "session.failed" for e in first)
        before = (await store.load_checkpoint("previous"))["context_compaction"]
        transcript_before = await store.load_transcript("previous")
        second = [
            e
            async for e in app.resume(
                ResumeRequest(
                    session_id="previous", messages=[Message.text("user", "Next request")]
                )
            )
        ]
        assert any(e.type == "session.failed" for e in second)
        assert len(summaries.requests) == 3
        assert len(actor.requests) == 1
        await store.close()
        store = SQLiteSessionStore(path)
        after = (await store.load_checkpoint("previous"))["context_compaction"]
        assert after["summary"] == before["summary"]
        assert after["compacted_transcript_cursor"] == before["compacted_transcript_cursor"]
        transcript_after = await store.load_transcript("previous")
        assert transcript_after[: len(transcript_before)] == transcript_before
        await store.close()

    asyncio.run(run())
