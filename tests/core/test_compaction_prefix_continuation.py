from __future__ import annotations

import asyncio
import base64
import hashlib
from decimal import Decimal

import pytest
from pydantic import SecretStr

from cayu import AgentSpec, Message
from cayu.core.events import EventType
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime import (
    BudgetLimit,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionResult,
    ContextCompactor,
    ContextRequest,
    InMemorySessionStore,
    ModelCompactor,
    ModelPrice,
    PriceBook,
    ResumeRequest,
    RunLimits,
    RunRequest,
    SessionIdentity,
)
from cayu.runtime.context import ContextBuildError, _estimate_model_facing_context_pressure
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.storage import SQLiteSessionStore
from cayu.vaults import SecretRedactor


class PrefixSummarizer(ModelProvider):
    name = "prefix-summarizer"

    def __init__(self, outcome="success"):
        self.requests = []
        self.outcome = outcome
        self.second_started = asyncio.Event()

    async def stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 2:
            if self.outcome == "failure":
                raise ValueError("scripted second-prefix failure")
            if self.outcome == "cancelled":
                raise asyncio.CancelledError("scripted second-prefix cancellation")
            if self.outcome == "caller_cancelled":
                self.second_started.set()
                await asyncio.Event().wait()
        yield ModelStreamEvent.text_delta("The fictional read results were examined.")
        yield ModelStreamEvent.completed(
            {"finish_reason": "stop", "usage": {"input_tokens": 100, "output_tokens": 10}}
        )


async def context_request():
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name="prefix-summarizer", model="synthetic"),
    )
    messages = [
        Message.text("system", "Preserve the original request."),
        Message.text("user", "ORIGINAL_FICTIONAL_REQUEST"),
    ]
    for index in range(6):
        messages.extend(
            [
                Message.tool_call(tool_call_id=f"read-{index}", tool_name="read", arguments={}),
                Message.tool_result(
                    tool_call_id=f"read-{index}", tool_name="read", content="x" * 7_000
                ),
            ]
        )
    return ContextRequest(
        session=session,
        agent=AgentSpec(name="assistant", model="synthetic"),
        messages=messages,
        step=7,
    )


def policy_for(provider, **overrides):
    return CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=provider, model="synthetic", max_input_chars=10_000),
        compact_after_estimated_context_tokens=1_000,
        max_recent_context_tokens=500,
        reserved_output_tokens=100,
        reserved_summary_tokens=50,
        **overrides,
    )


def test_bounded_prefixes_fit_in_one_automatic_context_build():
    async def exercise():
        provider = PrefixSummarizer()
        request = await context_request()
        original = request.model_dump(mode="json")
        result = await policy_for(provider).build_with_checkpoint(request, checkpoint=None)
        assert len(provider.requests) == 6
        assert all(len(r.messages[-1].content[0].text) <= 10_000 for r in provider.requests)
        assert request.model_dump(mode="json") == original
        assert result.messages[:2] == request.messages[:2]
        assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 14
        pressure = _estimate_model_facing_context_pressure(
            request=request, messages=result.messages, reserved_output_tokens=100
        )
        assert pressure.estimated_context_input_tokens <= 500
        assert pressure.estimated_context_window_tokens < 1_000
        completions = [
            t for t in result.compaction_telemetry if t.event_type == EventType.MODEL_COMPLETED
        ]
        assert len(completions) == 6
        covered = [
            t.payload["represented_source_end"]
            for t in result.compaction_telemetry
            if t.event_type == EventType.CONTEXT_COMPACTION_COMPLETED
        ]
        assert covered == [4, 6, 8, 10, 12, 14]

    asyncio.run(exercise())


@pytest.mark.parametrize("limit", [1, 3])
def test_pass_limit_keeps_progress_and_resume_does_not_repeat_source(limit):
    async def exercise():
        provider = PrefixSummarizer()
        request = await context_request()
        with pytest.raises(ContextBuildError) as failed:
            await policy_for(provider, max_compaction_passes=limit).build_with_checkpoint(
                request, checkpoint={"application_state": {"keep": True}}
            )
        assert len(provider.requests) == limit
        checkpoint = failed.value.checkpoint
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2 + 2 * limit
        assert checkpoint["application_state"] == {"keep": True}
        assert "pass limit reached" in str(failed.value.cause.__notes__)
        assert f"represented_cursor={2 + 2 * limit}, requested_cursor=14" in str(failed.value)
        assert "estimated_input_tokens=" in str(failed.value)
        assert "stop=pass_limit" in str(failed.value)
        result = await policy_for(provider).build_with_checkpoint(request, checkpoint=checkpoint)
        assert len(provider.requests) == 6
        assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 14
        next_prompt = provider.requests[limit].messages[-1].content[0].text
        assert f"read-{limit}" in next_prompt
        assert f"read-{limit - 1}" not in next_prompt

    asyncio.run(exercise())


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_pass_limit_validation(value):
    with pytest.raises((TypeError, ValueError), match="max_compaction_passes"):
        CheckpointCompactionContextPolicy(max_compaction_passes=value)


@pytest.mark.parametrize(
    "mode", ["wrong_binding", "split_round", "no_progress", "oversized", "exhausted"]
)
def test_unsafe_continuation_fails_with_last_valid_prefix(mode):
    class PrefixCompactor(ContextCompactor):
        def __init__(self):
            self.requests = []

        def _progress_key(self):
            return "synthetic-prefix-terminal"

        async def compact(self, request):
            self.requests.append(request)
            later = len(self.requests) == 2
            return CompactionResult(
                summary="s" * 10_000 if mode == "oversized" else "summary",
                covered_message_count=(
                    0
                    if later and mode in {"no_progress", "exhausted"}
                    else 1
                    if later and mode == "split_round"
                    else 2
                ),
                represented_existing_summary_sha256=(
                    hashlib.sha256(request.existing_summary.encode()).hexdigest()
                    if request.existing_summary is not None
                    and mode not in {"wrong_binding", "exhausted"}
                    else None
                ),
                progress_exhausted=later and mode == "exhausted",
                progress_key=self._progress_key() if later and mode == "exhausted" else None,
            )

    async def exercise():
        compactor = PrefixCompactor()
        policy = policy_for(PrefixSummarizer())
        policy.compactor = compactor
        with pytest.raises(ContextBuildError) as failed:
            await policy.build_with_checkpoint(await context_request(), checkpoint=None)
        assert len(compactor.requests) == (1 if mode == "oversized" else 2)
        assert failed.value.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4
        if mode == "exhausted":
            with pytest.raises(ContextBuildError):
                await policy.build_with_checkpoint(
                    await context_request(), checkpoint=failed.value.checkpoint
                )
            assert len(compactor.requests) == 2

    asyncio.run(exercise())


def test_explicit_request_does_not_implicitly_continue():
    async def exercise():
        provider = PrefixSummarizer()
        request = (await context_request()).model_copy(update={"force_compaction": True})
        with pytest.raises(ContextBuildError):
            await policy_for(provider).build_with_checkpoint(request, checkpoint=None)
        assert len(provider.requests) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "outcome", ["success", "failure", "cancelled", "caller_cancelled", "limit", "budget"]
)
def test_runtime_prefix_progress_and_accounting_are_durable(backend, outcome, tmp_path):
    async def exercise():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "prefixes.sqlite")
        )
        provider = PrefixSummarizer(outcome)
        actor = PrefixSummarizer()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(actor, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="synthetic"), context_policy=policy_for(provider)
        )
        request = RunRequest(
            agent_name="assistant",
            session_id="prefix-test",
            messages=(await context_request()).messages,
            limits=RunLimits(max_total_tokens=110) if outcome == "limit" else RunLimits(),
            budget_limits=(
                BudgetLimit(
                    scope="run",
                    max_estimated_cost=Decimal("0.000110"),
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name=provider.name,
                                model="synthetic",
                                input_per_million=Decimal("1"),
                                output_per_million=Decimal("1"),
                            ),
                        )
                    ),
                ),
            )
            if outcome == "budget"
            else (),
        )
        try:

            async def run():
                return [event async for event in app.run(request)]

            if outcome == "caller_cancelled":
                task = asyncio.create_task(run())
                await asyncio.wait_for(provider.second_started.wait(), timeout=5)
                task.cancel("caller stopped prefix continuation")
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif outcome == "cancelled":
                with pytest.raises(asyncio.CancelledError, match="second-prefix cancellation"):
                    await run()
            else:
                await run()
            events = await store.load_events("prefix-test")
            checkpoint = await store.load_checkpoint("prefix-test")
            expected_cursor = 14 if outcome == "success" else 4
            assert (
                checkpoint["context_compaction"]["compacted_transcript_cursor"] == expected_cursor
            )
            assert len(provider.requests) == (
                6 if outcome == "success" else 1 if outcome in {"limit", "budget"} else 2
            )
            assert len(actor.requests) == (1 if outcome == "success" else 0)
            completions = [
                e
                for e in events
                if e.type == EventType.MODEL_COMPLETED
                and e.payload.get("purpose") == "context_compaction"
            ]
            assert len(completions) == len(provider.requests)
            assert len({e.payload["model_attempt_id"] for e in completions}) == len(completions)
            starts = [e for e in events if e.type == EventType.CONTEXT_COMPACTION_STARTED]
            assert len(starts) == 1
            completed = [e for e in events if e.type == EventType.CONTEXT_COMPACTION_COMPLETED]
            assert [e.payload["represented_source_end"] for e in completed] == (
                [4, 6, 8, 10, 12, 14] if outcome == "success" else [4]
            )
            checkpoint_events = [
                e
                for e in events
                if e.type == EventType.SESSION_CHECKPOINTED
                and e.payload.get("checkpoint") == "context_compaction"
            ]
            assert len(checkpoint_events) == 1
            assert (
                checkpoint_events[0].payload["newly_compacted_message_count"] == expected_cursor - 2
            )
            assert "compaction_model_calls_unrepresented" not in checkpoint_events[0].payload
            if outcome == "failure":
                resumed = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="prefix-test",
                            messages=[Message.text("user", "Continue.")],
                        )
                    )
                ]
                assert resumed[-1].type == EventType.SESSION_COMPLETED
                assert len(provider.requests) == 7  # Six covered rounds plus one failed call.
                resumed_prompt = provider.requests[2].messages[-1].content[0].text
                assert "read-1" in resumed_prompt
                assert "read-0" not in resumed_prompt
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(exercise())


def test_continuation_does_not_reselect_below_trigger_or_consume_retained_round():
    async def exercise():
        provider = PrefixSummarizer()
        request = await context_request()
        request.messages[-1] = Message.tool_result(
            tool_call_id="read-5", tool_name="read", content="z" * 2_000
        )
        retained = [
            Message.tool_call(tool_call_id="latest", tool_name="read", arguments={}),
            Message.tool_result(tool_call_id="latest", tool_name="read", content="latest evidence"),
            Message.text("user", "Keep working on my original request."),
        ]
        request.messages.extend(retained)
        policy = policy_for(provider)
        policy.compactor.max_input_chars = 8_500
        result = await policy.build_with_checkpoint(request, checkpoint=None)
        assert len(provider.requests) == 6
        assert result.messages[-3:] == retained
        assert all(
            "latest evidence" not in r.messages[-1].content[0].text for r in provider.requests
        )
        assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 14
        assert (
            _estimate_model_facing_context_pressure(
                request=request, messages=result.messages
            ).estimated_context_input_tokens
            <= 500
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("outcome", ["success", "cancelled"])
@pytest.mark.parametrize("committed", [False, True])
def test_prefix_checkpoint_and_completion_evidence_remain_atomic(outcome, committed):
    class AckLossStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.injected = False

        async def publish_checkpoint_and_events(self, session_id, **kwargs):
            if self.injected or not any(
                e.type == EventType.SESSION_CHECKPOINTED
                and e.payload.get("checkpoint") == "context_compaction"
                for e in kwargs["events"]
            ):
                return await super().publish_checkpoint_and_events(session_id, **kwargs)
            self.injected = True
            if committed:
                await super().publish_checkpoint_and_events(session_id, **kwargs)
            raise ConnectionError("scripted checkpoint acknowledgement failure")

    async def exercise():
        store = AckLossStore()
        summarizer = PrefixSummarizer(outcome)
        actor = PrefixSummarizer()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(actor, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="synthetic"), context_policy=policy_for(summarizer)
        )

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="prefix-ack",
                        messages=(await context_request()).messages,
                    )
                )
            ]

        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await run()
        else:
            await run()
        assert store.injected
        checkpoint = await store.load_checkpoint("prefix-ack")
        events = await store.load_events("prefix-ack")
        prefix_events = [e for e in events if e.type == EventType.CONTEXT_COMPACTION_COMPLETED]
        checkpoint_events = [
            e
            for e in events
            if e.type == EventType.SESSION_CHECKPOINTED
            and e.payload.get("checkpoint") == "context_compaction"
        ]
        assert len(checkpoint_events) == int(committed)
        assert len(prefix_events) == (6 if outcome == "success" else 1) * int(committed)
        assert len(actor.requests) == int(committed and outcome == "success")
        if committed:
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == (
                14 if outcome == "success" else 4
            )
        else:
            assert checkpoint is None or "context_compaction" not in checkpoint
        completions = [
            e
            for e in events
            if e.type == EventType.MODEL_COMPLETED
            and e.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == len(summarizer.requests)

    asyncio.run(exercise())


@pytest.mark.parametrize("unsafe_pass", [1, 2])
def test_intermediate_summary_secret_is_rejected_before_next_dispatch(unsafe_pass):
    canary = "fictional-prefix-secret-boundary-canary-1710"

    class UnsafeSummarizer(PrefixSummarizer):
        async def stream(self, request):
            self.requests.append(request)
            yield ModelStreamEvent.text_delta(
                canary if len(self.requests) == unsafe_pass else "Safe fictional summary."
            )
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop", "usage": {"input_tokens": 100, "output_tokens": 10}}
            )

    async def exercise():
        store = InMemorySessionStore()
        summarizer = UnsafeSummarizer()
        actor = PrefixSummarizer()
        app = CayuApp(
            session_store=store, secret_redactor=SecretRedactor(canary), enable_logging=False
        )
        app.register_provider(actor, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="synthetic"), context_policy=policy_for(summarizer)
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="prefix-secret",
                    messages=(await context_request()).messages,
                )
            )
        ]
        assert len(summarizer.requests) == unsafe_pass
        assert not actor.requests
        assert events[-1].type == EventType.SESSION_FAILED
        assert all(canary not in str(request.model_dump()) for request in summarizer.requests)
        checkpoint = await store.load_checkpoint("prefix-secret")
        assert canary not in str(checkpoint)
        if unsafe_pass == 1:
            assert checkpoint is None or "context_compaction" not in checkpoint
        else:
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4
        completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == unsafe_pass

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["failure", "cancelled"])
def test_partial_checkpoint_does_not_acknowledge_unrepresented_hierarchy_calls(
    backend, outcome, tmp_path
):
    class InterruptedHierarchy(PrefixSummarizer):
        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 3:
                if outcome == "cancelled":
                    raise asyncio.CancelledError("scripted hierarchy cancellation")
                raise ValueError("scripted hierarchy failure")
            yield ModelStreamEvent.text_delta("Short fictional summary.")
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop", "usage": {"input_tokens": 100, "output_tokens": 10}}
            )

    async def exercise():
        alias_codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={
                    "test": SecretStr(
                        base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                    )
                },
            )
        )
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(
                tmp_path / "hierarchy.sqlite", public_authority_alias_codec=alias_codec
            )
        )
        summarizer = InterruptedHierarchy()
        # A registered secret colliding with the schema key must not erase the
        # runtime's recovery fence during event projection or durable reload.
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor("compaction_model_calls_unrepresented"),
            enable_logging=False,
        )
        app.register_provider(PrefixSummarizer(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="synthetic"), context_policy=policy_for(summarizer)
        )
        request = await context_request()
        request.messages[5] = Message.tool_result(
            tool_call_id="read-1", tool_name="read", content="z" * 40_000
        )
        try:

            async def run():
                return [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id="hierarchy-prefix",
                            messages=request.messages,
                        )
                    )
                ]

            if outcome == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await run()
            else:
                await run()
            assert len(summarizer.requests) == 3
            checkpoint = await store.load_checkpoint("hierarchy-prefix")
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4
            checkpoint_events = [
                event
                for event in await store.load_events("hierarchy-prefix")
                if event.type == EventType.SESSION_CHECKPOINTED
                and event.payload.get("checkpoint") == "context_compaction"
            ]
            assert len(checkpoint_events) == 1
            assert checkpoint_events[0].payload["compaction_model_calls_unrepresented"] is True
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="hierarchy-prefix",
                        messages=[Message.text("user", "Continue.")],
                    )
                )
            ]
            assert len(summarizer.requests) == 3
            assert resumed[-1].type == EventType.SESSION_FAILED
            assert "no later durable context checkpoint" in resumed[-1].payload["error"]
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("committed", [False, True])
def test_cancellation_cleanup_timeout_cancels_actual_checkpoint_writer(monkeypatch, committed):
    import cayu.runtime._model_step_executor as executor

    monkeypatch.setattr(executor, "_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S", 0.02)

    class StalledStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.finished = asyncio.Event()
            self.writer = None

        async def publish_checkpoint_and_events(self, session_id, **kwargs):
            if not any(
                event.type == EventType.SESSION_CHECKPOINTED
                and event.payload.get("checkpoint") == "context_compaction"
                for event in kwargs["events"]
            ):
                return await super().publish_checkpoint_and_events(session_id, **kwargs)
            self.writer = asyncio.current_task()
            try:
                if committed:
                    await super().publish_checkpoint_and_events(session_id, **kwargs)
                await self.release.wait()
                if not committed:
                    await super().publish_checkpoint_and_events(session_id, **kwargs)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            finally:
                self.finished.set()

    async def exercise():
        store = StalledStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(PrefixSummarizer(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="synthetic"),
            context_policy=policy_for(PrefixSummarizer("cancelled")),
        )
        try:
            with pytest.raises(asyncio.CancelledError, match="second-prefix cancellation"):
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="prefix-writer",
                        messages=(await context_request()).messages,
                    )
                ):
                    pass
            await asyncio.wait_for(store.cancelled.wait(), 0.2)
            assert store.writer.done()
            checkpoint = await store.load_checkpoint("prefix-writer")
            events = await store.load_events("prefix-writer")
            checkpoint_events = [
                event
                for event in events
                if event.type == EventType.SESSION_CHECKPOINTED
                and event.payload.get("checkpoint") == "context_compaction"
            ]
            assert len(checkpoint_events) == int(committed)
            if committed:
                assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4
            else:
                assert checkpoint is None or "context_compaction" not in checkpoint
        finally:
            store.release.set()
            await asyncio.wait_for(store.finished.wait(), 5)

    asyncio.run(exercise())
