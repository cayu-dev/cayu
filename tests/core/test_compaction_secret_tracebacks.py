from __future__ import annotations

import asyncio
import traceback

import pytest
from tests.core.test_compaction_prefix_continuation import PrefixSummarizer, context_request
from tests.core.test_explicit_compaction_transcript_redaction import (
    _CapturingCompactor,
    _create_completed_session,
)
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import AgentSpec, Message
from cayu.core.events import EventType
from cayu.runtime import (
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactSessionRequest,
    InMemorySessionStore,
    ModelCompactor,
    RunRequest,
)
from cayu.runtime.context import ContextBuildError
from cayu.runtime.request_footprints import RequestFootprintConfig
from cayu.vaults import SecretRedactor

_CANARY = "fictional-rejected-compaction-result-secret-1710"


def _with_secret(result, location):
    update = (
        {"summary": _CANARY}
        if location == "summary"
        else {"metadata": {"note": _CANARY}}
        if location == "metadata_value"
        else {"metadata": {_CANARY: "fictional value"}}
    )
    return result.model_copy(update=update, deep=True)


def _assert_safe_exception_graph(error):
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert _CANARY not in str(current)
        assert _CANARY not in repr(vars(current))
        captured = traceback.TracebackException.from_exception(current, capture_locals=True)
        leaking = [
            (frame.name, name)
            for frame in captured.stack
            if is_cayu_source_filename(frame.filename)
            for name, value in (frame.locals or {}).items()
            if _CANARY in value
        ]
        assert not leaking, leaking
        pending.extend(
            linked
            for linked in (current.__cause__, current.__context__, getattr(current, "cause", None))
            if isinstance(linked, BaseException)
        )
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


class _RecordingPolicy(CheckpointCompactionContextPolicy):
    failure = None

    async def build_with_checkpoint(self, request, *, checkpoint):
        try:
            return await super().build_with_checkpoint(request, checkpoint=checkpoint)
        except ContextBuildError as error:
            self.failure = error
            raise


@pytest.mark.parametrize("location", ["summary", "metadata_value", "metadata_key"])
@pytest.mark.parametrize("unsafe_pass", [1, 2])
def test_automatic_compaction_rejection_drops_secret_traceback_locals(location, unsafe_pass):
    class SecretResultCompactor(ModelCompactor):
        returned_result = None

        async def compact(self, request):
            result = await super().compact(request)
            if len(self.provider.requests) == unsafe_pass:
                self.returned_result = _with_secret(result, location)
                return self.returned_result
            return result

    async def exercise():
        store = InMemorySessionStore()
        summarizer = PrefixSummarizer()
        compactor = SecretResultCompactor(
            provider=summarizer, model="synthetic", max_input_chars=10_000
        )
        policy = _RecordingPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=1_000,
            max_recent_context_tokens=500,
            reserved_output_tokens=100,
            reserved_summary_tokens=50,
        )
        actor = PrefixSummarizer()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(_CANARY),
            # This test wrapper intentionally alters the built-in result. Its
            # real provider calls remain accounted for, but opaque wrappers are
            # not admitted with request footprints enabled.
            request_footprint=RequestFootprintConfig(enabled=False),
            enable_logging=False,
        )
        app.register_provider(actor, default=True)
        app.register_agent(AgentSpec(name="assistant", model="synthetic"), context_policy=policy)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="secret-traceback",
                    messages=(await context_request()).messages,
                )
            )
        ]
        assert events[-1].type == EventType.SESSION_FAILED
        assert len(summarizer.requests) == unsafe_pass, events[-1].payload
        assert not actor.requests
        assert isinstance(policy.failure, ContextBuildError)
        _assert_safe_exception_graph(policy.failure)
        # Only runtime-owned copies may be consumed by rejection cleanup.
        assert _CANARY in repr(compactor.returned_result)
        checkpoint = await store.load_checkpoint("secret-traceback")
        assert _CANARY not in repr(checkpoint)
        assert _CANARY not in repr(await store.load_events("secret-traceback"))
        if unsafe_pass == 1:
            assert checkpoint is None or "context_compaction" not in checkpoint
        else:
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4
        completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == unsafe_pass
        assert sum(event.payload["usage_metrics"]["total_tokens"] for event in completions) == (
            unsafe_pass * 110
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("location", ["summary", "metadata_value", "metadata_key"])
def test_explicit_compaction_rejection_drops_secret_traceback_locals(location):
    class SecretResultCompactor(_CapturingCompactor):
        returned_result = None

        async def compact(self, request):
            self.returned_result = _with_secret(await super().compact(request), location)
            return self.returned_result

    async def exercise():
        store = InMemorySessionStore()
        compactor = SecretResultCompactor()
        app = CayuApp(
            session_store=store, secret_redactor=SecretRedactor(_CANARY), enable_logging=False
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(compactor=compactor, max_user_turns=1),
        )
        messages = [
            Message.text("user", "Old fictional request."),
            Message.text("assistant", "Old fictional answer."),
            Message.text("user", "Current fictional request."),
        ]
        session = await _create_completed_session(
            app, store, session_id="explicit-secret-traceback", transcript=messages
        )
        with pytest.raises(ContextBuildError, match="contains a workload secret") as raised:
            async for _ in app.compact_session(
                CompactSessionRequest(
                    session_id=session.id,
                    idempotency_key="secret-traceback",
                    expected_run_epoch=session.run_epoch,
                    expected_transcript_cursor=len(messages),
                )
            ):
                pass
        _assert_safe_exception_graph(raised.value)
        assert len(compactor.requests) == 1
        assert _CANARY in repr(compactor.returned_result)
        assert _CANARY not in repr(await store.load_checkpoint(session.id))
        assert _CANARY not in repr(await store.load_events(session.id))
        assert await store.load_transcript(session.id) == messages

    asyncio.run(exercise())
