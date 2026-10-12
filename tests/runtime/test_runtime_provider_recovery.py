"""Provider recovery contracts."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.runtime.runtime_provider_support import FakeProvider, StageRecordingFakeProvider
from tests.runtime.runtime_session_support import (
    _private_events_for_public_events,
    _without_interaction_lifecycle,
    _without_request_evidence,
    collect_events,
)
from tests.runtime.runtime_tool_support import EchoTool, SideEffectTool

import cayu
import cayu.providers.deadlines as provider_deadlines_module
import cayu.sessions.base as sessions_module
from cayu._exception_groups import iter_exception_tree
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.configuration import CayuConfig, RunDefaults
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.providers.chat_completions import ChatCompletionsProvider
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.providers.openai import OpenAIProvider
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions._model_stage import ModelCompletionManualRecoveryRequest
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore


def test_cayu_app_retries_retryable_model_error_before_tool_side_effects():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_failed_attempt",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.error("OpenAI API request failed with HTTP 429: rate limit"),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_successful_attempt",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_before_tool",
                messages=[Message.text("user", "use the tool")],
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_retry_before_tool"))
    interaction = asyncio.run(
        store.query_latest_interaction_events("sess_retry_before_tool", limit=1)
    )[0].event

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 3
    assert len(tool.calls) == 1
    assert events[3].payload["reason"] == "http_status"
    assert events[3].payload["status_code"] == 429
    assert events[3].payload["attempt"] == 1
    assert events[3].payload["next_attempt"] == 2
    assert events[4].payload["attempt"] == 1
    assert events[4].payload["next_attempt"] == 2
    assert events[4].payload["status_code"] == 429
    assert events[7].payload["tool_call_id"] == f"{events[7].id}:tool_call_id"
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert transcript[1].content[0].type == "tool_call"
    assert transcript[1].content[0].tool_call_id == "call_successful_attempt"
    from cayu.sessions.interactions import InteractionSummaryEvidence

    evidence = InteractionSummaryEvidence.model_validate(interaction.payload)
    assert evidence.model_step_count == 3
    assert evidence.tool_call_count == 1
    assert evidence.provider_names == ["fake"]
    assert evidence.models == ["fake-model"]


def test_cayu_app_retries_transient_provider_failure_by_default():
    typed_error = ModelProviderError(
        "provider rate limited",
        provider="fake",
        status_code=429,
        retryable=True,
        retry_after_s=0.0,
    )
    provider = FakeProvider(
        [
            [ModelStreamEvent.error("provider rate limited", cause=typed_error)],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_default",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    retry = next(event for event in events if event.type is EventType.MODEL_RETRY)
    assert retry.payload["attempt"] == 1
    assert retry.payload["next_attempt"] == 2
    assert retry.payload["max_attempts"] == 5
    assert retry.payload["delay_seconds"] == 0.0


def test_cayu_app_does_not_retry_non_retryable_model_error():
    provider = FakeProvider(
        [
            ModelStreamEvent.error("OpenAI API request failed with HTTP 400: bad request"),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_non_retryable",
                messages=[Message.text("user", "hi")],
                retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]


def test_cayu_app_retries_provider_exception_and_keeps_transcript_clean():
    class TimeoutThenSuccessProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise TimeoutError("stream idle timeout")
            yield ModelStreamEvent.text_delta("ok")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    store = InMemorySessionStore()
    provider = TimeoutThenSuccessProvider()
    app = CayuApp(
        session_store=store,
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_exception",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_retry_exception"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert events[3].payload["reason"] == "timeout"
    assert events[4].payload["reason"] == "timeout"
    assert events[4].payload["attempt"] == 1
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[1].content[0].text == "ok"


def test_cayu_app_does_not_redispatch_after_established_openai_sse_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResetAfterIdentityBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                b'data: {"type":"response.created","sequence_number":0,'
                b'"response":{"id":"resp-reset","model":"fake-model",'
                b'"status":"in_progress","output":[]}}\n\n'
            )
            request = httpx.Request("POST", "https://example.test/v1/responses")
            raise httpx.RemoteProtocolError(
                "peer reset the established response stream",
                request=request,
            )

        async def aclose(self) -> None:
            return None

    class StreamContext:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self) -> httpx.Response:
            return self.response

        async def __aexit__(self, *args: Any) -> None:
            await self.response.aclose()

    class ResetHttpClient:
        calls = 0

        def __init__(self, **kwargs: Any) -> None:
            self.is_closed = False

        def stream(self, *args: Any, **kwargs: Any) -> StreamContext:
            type(self).calls += 1
            request = httpx.Request("POST", "https://example.test/v1/responses")
            return StreamContext(
                httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=ResetAfterIdentityBody(),
                    request=request,
                )
            )

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", ResetHttpClient)
    provider = OpenAIProvider(
        api_key="test-key",
        base_url="https://example.test",
    )
    app = CayuApp(
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess-established-openai-reset",
                messages=[Message.text("user", "do not duplicate this request")],
            ),
        )
    )

    assert ResetHttpClient.calls == 1
    assert EventType.MODEL_ERROR in {event.type for event in events}
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    model_error = next(event for event in events if event.type is EventType.MODEL_ERROR)
    assert model_error.payload["provider_error_type"] == "RemoteProtocolError"
    assert model_error.payload["retryable"] is False


_SEMANTIC_DEADLINE_PROCESS_EXIT_CODE = 87


_SEMANTIC_DEADLINE_PROCESS_SESSION_ID = "semantic-deadline-process-loss"


class ProcessLossSemanticDeadlineProvider(ModelProvider):
    name = "process-loss-semantic-deadline"

    def __init__(self, marker: str) -> None:
        self._marker = Path(marker)

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test.process-loss-semantic-deadline-provider",
            behavior_version="1",
            implementation_version="1",
        )

    @property
    def stream_deadlines(self) -> ProviderStreamDeadlines:
        return ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.02,
            absolute_stream_timeout_s=1,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        prior = int(self._marker.read_text(encoding="utf-8")) if self._marker.exists() else 0
        self._marker.write_text(str(prior + 1), encoding="utf-8")
        while True:
            yield ModelStreamEvent.text_delta(" \n")
            await asyncio.sleep(0.001)


def _run_until_semantic_deadline_process_exit(database: str, marker: str) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(database)
        provider = ProcessLossSemanticDeadlineProvider(marker)
        app = CayuApp(
            session_store=store,
            config=CayuConfig(
                run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=_SEMANTIC_DEADLINE_PROCESS_SESSION_ID,
                    messages=[Message.text("user", "wait until the semantic deadline")],
                )
            ):
                pass
        except ModelStreamDeadlineError as failure:
            if failure.deadline_evidence.deadline_kind.value != "semantic_idle":
                raise
            active = await store.load_active_model_completion_stage(
                _SEMANTIC_DEADLINE_PROCESS_SESSION_ID
            )
            durable_events = await store.load_events(_SEMANTIC_DEADLINE_PROCESS_SESSION_ID)
            deadline_events = [
                event for event in durable_events if event.type is EventType.MODEL_ERROR
            ]
            if (
                active is None
                or active.stage.state != "in_flight"
                or len(deadline_events) != 1
                or deadline_events[0].payload.get("provider_recovery_disposition")
                != "manual_settlement_required"
            ):
                raise AssertionError(
                    "semantic deadline evidence was not durable before exit"
                ) from None
            os._exit(_SEMANTIC_DEADLINE_PROCESS_EXIT_CODE)
        raise AssertionError("the child provider did not reach its semantic deadline")

    asyncio.run(run())


def test_semantic_deadline_process_loss_cannot_redispatch_before_settlement(
    tmp_path: Path,
) -> None:
    database = tmp_path / "semantic-deadline-process-loss.sqlite"
    marker = tmp_path / "semantic-deadline-provider-count"
    repository_root = Path(__file__).resolve().parents[2]
    child_script = (
        "from tests.runtime.test_runtime_provider_recovery import "
        "_run_until_semantic_deadline_process_exit as run; "
        f"run({str(database)!r}, {str(marker)!r})"
    )
    child_environment = os.environ.copy()
    existing_python_path = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (str(Path(cayu.__file__).resolve().parent.parent), existing_python_path)
        if path
    )
    child = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=repository_root,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert child.returncode == _SEMANTIC_DEADLINE_PROCESS_EXIT_CODE, child.stderr
    assert marker.read_text(encoding="utf-8") == "1"

    store = SQLiteSessionStore(database)
    active = asyncio.run(
        store.load_active_model_completion_stage(_SEMANTIC_DEADLINE_PROCESS_SESSION_ID)
    )
    durable_events = asyncio.run(store.load_events(_SEMANTIC_DEADLINE_PROCESS_SESSION_ID))
    deadline_events = [event for event in durable_events if event.type is EventType.MODEL_ERROR]
    assert active is not None
    assert active.stage.state == "in_flight"
    assert len(deadline_events) == 1
    deadline = deadline_events[0]
    assert deadline.payload["provider_deadline_kind"] == "semantic_idle"
    assert deadline.payload["provider_whitespace_since_progress"] is True
    assert deadline.payload["provider_recovery_disposition"] == "manual_settlement_required"
    assert deadline.payload["model_step_id"] == active.stage.logical_step_id
    assert deadline.payload["model_attempt_id"] == active.stage.intent["model_attempt_id"]
    assert EventType.MODEL_RETRY not in {event.type for event in durable_events}

    restarted_provider = ProcessLossSemanticDeadlineProvider(str(marker))
    restarted = CayuApp(session_store=store, enable_logging=False)
    restarted.register_provider(restarted_provider, default=True)
    restarted.register_agent(AgentSpec(name="assistant", model="fake-model"))
    with pytest.raises(ModelCompletionManualRecoveryRequired):
        asyncio.run(
            restarted.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=_SEMANTIC_DEADLINE_PROCESS_SESSION_ID,
                    inactive_for_seconds=0,
                )
            )
        )

    assert marker.read_text(encoding="utf-8") == "1"
    after_recovery = asyncio.run(
        store.load_active_model_completion_stage(_SEMANTIC_DEADLINE_PROCESS_SESSION_ID)
    )
    assert after_recovery is not None
    assert after_recovery.stage.stage_id == active.stage.stage_id
    assert (
        after_recovery.stage.intent["model_attempt_id"] == active.stage.intent["model_attempt_id"]
    )
    asyncio.run(store.close())


def test_cayu_app_fences_typed_semantic_stream_deadline_until_manual_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class StreamingBody(httpx.AsyncByteStream):
        def __init__(self, *, idle: bool) -> None:
            self.idle = idle

        async def __aiter__(self):
            if self.idle:
                yield (
                    b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                    b'"model":"fake-model","choices":[{"index":0,'
                    b'"delta":{"content":"partial answer"},"finish_reason":null}]}'
                    b"\n\n"
                )
                # Block until the real idle deadline cancels this pending read.
                # A fixed sleep made the successful retry race a 1 ms deadline
                # under slower event-loop schedulers (notably CI on Python 3.14).
                await asyncio.Event().wait()
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{"content":"ok"},"finish_reason":null}]}'
                b"\n\n"
            )
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{},"finish_reason":"stop"}]}'
                b"\n\n"
            )
            yield b"data: [DONE]\n\n"

    class StreamContext:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self) -> httpx.Response:
            return self.response

        async def __aexit__(self, *args: Any) -> None:
            await self.response.aclose()

    class FlakyHttpClient:
        calls = 0

        def __init__(self, **kwargs: Any) -> None:
            self.is_closed = False

        def stream(self, *args: Any, **kwargs: Any) -> StreamContext:
            type(self).calls += 1
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            response = httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=StreamingBody(idle=type(self).calls == 1),
                request=request,
            )
            return StreamContext(response)

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", FlakyHttpClient)
    database = tmp_path / "semantic-deadline-recovery.sqlite"
    store = SQLiteSessionStore(database)
    provider = ChatCompletionsProvider(
        api_key="test-key",
        base_url="https://example.test/v1",
        stream_deadlines=ProviderStreamDeadlines(
            transport_idle_timeout_s=1.0,
            semantic_progress_timeout_s=0.05,
            protocol_idle_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
        ),
    )
    app = CayuApp(
        session_store=store,
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def collect_until_deadline() -> tuple[list[Event], ModelStreamDeadlineError]:
        observed: list[Event] = []
        with pytest.raises(ModelStreamDeadlineError) as caught:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_retry_sse_idle",
                    messages=[Message.text("user", "hi")],
                )
            ):
                observed.append(event)
        return (
            _without_request_evidence(_without_interaction_lifecycle(observed)),
            caught.value,
        )

    events, failure = asyncio.run(collect_until_deadline())
    transcript = asyncio.run(store.load_transcript("sess_retry_sse_idle"))
    private_events = asyncio.run(
        _private_events_for_public_events(store, "sess_retry_sse_idle", events)
    )

    assert FlakyHttpClient.calls == 1
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_ERROR,
    ]
    assert failure.deadline_evidence.deadline_kind.value == "semantic_idle"
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    assert EventType.MODEL_ATTEMPT_DISCARDED not in {event.type for event in events}
    assert events[2].payload == {
        "delta": "partial answer",
        "step": 1,
        "attempt": 1,
        "max_attempts": 2,
        "model_step_id": events[1].payload["model_step_id"],
        "model_attempt_id": events[1].payload["model_attempt_id"],
        "execution_profile_fingerprint": events[1].payload["execution_profile_fingerprint"],
    }
    deadline = events[3]
    assert deadline.payload["error_type"] == "ModelStreamDeadlineError"
    assert deadline.payload["provider_error_type"] == "ModelStreamDeadlineError"
    assert deadline.payload["provider_error_code"] == "provider_stream_semantic_idle_timeout"
    assert deadline.payload["provider_deadline_kind"] == "semantic_idle"
    assert deadline.payload["provider_deadline_timeout_s"] == 0.05
    assert deadline.payload["provider_stream_elapsed_s"] >= 0.05
    assert deadline.payload["provider_last_progress_kind"] == "content"
    assert (
        0
        <= deadline.payload["provider_last_progress_elapsed_s"]
        <= deadline.payload["provider_stream_elapsed_s"]
    )
    assert (
        datetime.fromisoformat(deadline.payload["provider_last_progress_at"]).utcoffset()
        is not None
    )
    assert deadline.payload["provider_effect_outcome"] == "unknown"
    assert deadline.payload["provider_recovery_disposition"] == "manual_settlement_required"
    assert deadline.payload["retryable"] is False
    assert deadline.payload["effective_max_attempts"] == 1
    assert deadline.payload["retry_disposition"] == "suppressed"
    assert deadline.payload["retry_suppression"] == "deadline"
    assert deadline.payload["model_step_id"] == events[1].payload["model_step_id"]
    assert deadline.payload["model_attempt_id"] == events[1].payload["model_attempt_id"]
    assert "partial answer" not in repr(deadline.payload)

    private_started = private_events[1]
    private_deadline = private_events[3]
    assert private_deadline.payload["model_step_id"] == private_started.payload["model_step_id"]
    assert (
        private_deadline.payload["model_attempt_id"] == private_started.payload["model_attempt_id"]
    )
    assert [message.role for message in transcript] == ["user"]

    active = asyncio.run(store.load_active_model_completion_stage("sess_retry_sse_idle"))
    session = asyncio.run(store.load("sess_retry_sse_idle"))
    assert active is not None
    assert active.stage.state == "in_flight"
    assert active.stage.logical_step_id == private_started.payload["model_step_id"]
    assert active.stage.intent["model_attempt_id"] == private_started.payload["model_attempt_id"]
    assert session is not None and session.status is SessionStatus.RUNNING
    assert (
        asyncio.run(
            store.load_model_completion_stage_dispatch(
                "sess_retry_sse_idle",
                active.stage.stage_id,
            )
        )
        is not None
    )

    asyncio.run(store.close())
    reopened_store = SQLiteSessionStore(database)
    restarted_provider = ChatCompletionsProvider(
        api_key="test-key",
        base_url="https://example.test/v1",
        stream_deadlines=ProviderStreamDeadlines(
            transport_idle_timeout_s=1.0,
            semantic_progress_timeout_s=0.05,
            protocol_idle_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
        ),
    )
    restarted = CayuApp(session_store=reopened_store, enable_logging=False)
    restarted.register_provider(restarted_provider, default=True)
    restarted.register_agent(AgentSpec(name="assistant", model="fake-model"))
    with pytest.raises(ModelCompletionManualRecoveryRequired):
        asyncio.run(
            restarted.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="sess_retry_sse_idle",
                    inactive_for_seconds=0,
                )
            )
        )
    assert FlakyHttpClient.calls == 1
    recovered_owner = asyncio.run(reopened_store.load("sess_retry_sse_idle"))
    assert recovered_owner is not None
    assert recovered_owner.run_epoch > session.run_epoch

    settlement = asyncio.run(
        restarted.recover_model_completion_stage(
            ModelCompletionManualRecoveryRequest(
                session_id="sess_retry_sse_idle",
                stage_id=active.stage.stage_id,
                expected_run_epoch=recovered_owner.run_epoch,
                terminal_status=SessionStatus.FAILED,
            )
        )
    )
    assert settlement.settlement.disposition is (
        sessions_module.ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
    )
    assert settlement.settlement.reason_code == "operator_outcome_unknown"
    assert (
        asyncio.run(reopened_store.load_active_model_completion_stage("sess_retry_sse_idle"))
        is None
    )
    assert FlakyHttpClient.calls == 1
    asyncio.run(reopened_store.close())


def test_cayu_app_preserves_stream_deadline_when_model_error_acknowledgement_is_lost() -> None:
    class ModelErrorAcknowledgementLosingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.publication_failure = RuntimeError("model.error acknowledgement lost")
            self.failed = False

        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if not self.failed and event.type is EventType.MODEL_ERROR:
                self.failed = True
                raise self.publication_failure

    class BlockingDeadlineProvider(ModelProvider):
        name = "blocking-deadline"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        @property
        def stream_deadlines(self) -> ProviderStreamDeadlines:
            return ProviderStreamDeadlines(
                transport_idle_timeout_s=1,
                protocol_idle_timeout_s=1,
                semantic_progress_timeout_s=0.01,
                absolute_stream_timeout_s=1,
            )

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            await asyncio.Event().wait()
            yield ModelStreamEvent.completed({})  # pragma: no cover

    async def run() -> None:
        session_id = "sess_deadline_model_error_acknowledgement_lost"
        store = ModelErrorAcknowledgementLosingStore()
        provider = BlockingDeadlineProvider()
        app = CayuApp(
            session_store=store,
            config=CayuConfig(
                run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ExceptionGroup) as caught:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "wait")],
                )
            ):
                pass

        leaves = [
            failure
            for failure in iter_exception_tree(caught.value)
            if not isinstance(failure, BaseExceptionGroup)
        ]
        deadlines = [failure for failure in leaves if isinstance(failure, ModelStreamDeadlineError)]
        assert len(leaves) == 2
        assert len(deadlines) == 1
        assert sum(failure is store.publication_failure for failure in leaves) == 1
        assert len(provider.requests) == 1
        events = await store.load_events(session_id)
        assert [event.type for event in events].count(EventType.MODEL_ERROR) == 1
        active = await store.load_active_model_completion_stage(session_id)
        session = await store.load(session_id)
        assert active is not None and active.stage.state == "in_flight"
        assert session is not None and session.status is SessionStatus.RUNNING

    asyncio.run(run())


def test_cayu_app_tags_failed_attempt_stream_events_and_keeps_transcript_clean():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("partial answer"),
                ModelStreamEvent.error("OpenAI API request failed with HTTP 500: unavailable"),
            ],
            [
                ModelStreamEvent.text_delta("final answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_stream_attempt_metadata",
                messages=[Message.text("user", "hi")],
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_retry_stream_attempt_metadata"))
    private_events = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_retry_stream_attempt_metadata",
            events,
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[2].payload == {
        "delta": "partial answer",
        "step": 1,
        "attempt": 1,
        "max_attempts": 2,
        "model_step_id": events[1].payload["model_step_id"],
        "model_attempt_id": events[1].payload["model_attempt_id"],
        "execution_profile_fingerprint": events[1].payload["execution_profile_fingerprint"],
    }
    assert events[3].payload["attempt"] == 1
    assert events[3].payload["max_attempts"] == 2
    assert events[4].payload["attempt"] == 1
    assert events[4].payload["next_attempt"] == 2
    assert events[4].payload["status_code"] == 500
    assert events[7].payload == {
        "delta": "final answer",
        "step": 1,
        "attempt": 2,
        "max_attempts": 2,
        "model_step_id": events[6].payload["model_step_id"],
        "model_attempt_id": events[6].payload["model_attempt_id"],
        "execution_profile_fingerprint": events[6].payload["execution_profile_fingerprint"],
    }
    assert private_events[1].payload["model_step_id"] == private_events[6].payload["model_step_id"]
    assert (
        private_events[1].payload["model_attempt_id"]
        != private_events[6].payload["model_attempt_id"]
    )
    assert events[8].payload["attempt"] == 2
    assert events[8].payload["max_attempts"] == 2
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[1].content[0].text == "final answer"


def test_cayu_app_emits_model_error_for_final_failed_exception_attempt():
    class AlwaysTimeoutProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise TimeoutError("stream idle timeout")
            yield

    provider = AlwaysTimeoutProvider()
    app = CayuApp(
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_final_exception_error",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    private_events = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_retry_final_exception_error",
            events,
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert len(provider.requests) == 2
    assert events[2].payload == {
        "error": "stream idle timeout",
        "error_type": "TimeoutError",
        "provider_name": "fake",
        "requested_model": "fake-model",
        "step": 1,
        "attempt": 1,
        "max_attempts": 2,
        "effective_max_attempts": 2,
        "reason": "timeout",
        "retry": True,
        "retry_disposition": "retry_scheduled",
        "retry_suppression": None,
        "provider_retryable": None,
        "model_step_id": events[1].payload["model_step_id"],
        "model_attempt_id": events[1].payload["model_attempt_id"],
        "execution_profile_fingerprint": events[1].payload["execution_profile_fingerprint"],
    }
    assert events[4].payload["attempt"] == 1
    assert events[4].payload["reason"] == "timeout"
    assert events[6].payload == {
        "error": "stream idle timeout",
        "error_type": "TimeoutError",
        "provider_name": "fake",
        "requested_model": "fake-model",
        "step": 1,
        "attempt": 2,
        "max_attempts": 2,
        "effective_max_attempts": 2,
        "reason": "timeout",
        "retry": False,
        "retry_disposition": "configured_attempt_exhaustion",
        "retry_suppression": None,
        "provider_retryable": None,
        "model_step_id": events[5].payload["model_step_id"],
        "model_attempt_id": events[5].payload["model_attempt_id"],
        "execution_profile_fingerprint": events[5].payload["execution_profile_fingerprint"],
    }
    assert private_events[1].payload["model_step_id"] == private_events[5].payload["model_step_id"]
    assert (
        private_events[1].payload["model_attempt_id"]
        != private_events[5].payload["model_attempt_id"]
    )


def test_cayu_app_retries_using_typed_provider_status_code_without_http_text():
    # The error message carries no HTTP token; retry classification must come
    # from the typed provider status code preserved on the error-event payload.
    typed_error = ModelProviderError(
        "provider overloaded",
        provider="fake",
        status_code=529,
    )
    session_id = "sess_typed_status_retry"
    store = InMemorySessionStore()
    provider = StageRecordingFakeProvider(
        [
            [ModelStreamEvent.error("provider overloaded", cause=typed_error)],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ],
        store=store,
        session_id=session_id,
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hi")],
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
        )
    )

    types = [event.type for event in events]
    assert EventType.MODEL_RETRY in types
    discarded = next(e for e in events if e.type == EventType.MODEL_ATTEMPT_DISCARDED)
    assert discarded.payload["status_code"] == 529
    assert discarded.payload["attempt"] == 1
    assert discarded.payload["next_attempt"] == 2
    retry_event = next(e for e in events if e.type == EventType.MODEL_RETRY)
    assert retry_event.payload["status_code"] == 529
    assert retry_event.payload["reason"] == "http_status"
    assert len(provider.stage_ids) == 2
    superseded = asyncio.run(
        store.load_model_completion_stage_settlement(session_id, provider.stage_ids[0])
    )
    assert superseded is not None
    assert superseded.disposition is sessions_module.ModelCompletionStageDisposition.SUPERSEDED
    assert superseded.superseding_stage_id == provider.stage_ids[1]
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None


def test_cayu_app_does_not_retry_when_provider_marks_error_non_retryable():
    # Message text looks like a transient timeout, but the provider's typed
    # verdict (retryable=False) forces a terminal decision.
    typed_error = ModelProviderError(
        "request timed out",
        provider="fake",
        retryable=False,
    )
    provider = FakeProvider(
        [
            [ModelStreamEvent.error("request timed out", cause=typed_error)],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_typed_non_retryable",
                messages=[Message.text("user", "hi")],
                retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
            ),
        )
    )

    types = [event.type for event in events]
    assert EventType.MODEL_RETRY not in types
    assert EventType.MODEL_ATTEMPT_DISCARDED not in types
    assert len(provider.requests) == 1


def test_cayu_app_explains_non_retryable_contract_failure():
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "echo", "arguments": "not-an-object"},
            )
        ]
    )
    app = CayuApp(
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_non_retryable_contract_failure",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]

    assert events[2].payload["retry_disposition"] == "classification_unavailable"
    assert events[2].payload["retry"] is False


def test_cayu_app_deadline_capacity_exhaustion_precedes_provider_dispatch() -> None:
    async def run() -> None:
        existing_owners = set(provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS)
        capacity = len(existing_owners) + 1
        cancellation_seen = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()

        class NonsettlingEvents:
            def __aiter__(self) -> NonsettlingEvents:
                return self

            async def __anext__(self) -> ModelStreamEvent:
                try:
                    await asyncio.Event().wait()
                    raise AssertionError("the retained provider read unexpectedly resumed")
                except asyncio.CancelledError:
                    cancellation_seen.set()
                    try:
                        await release.wait()
                        return ModelStreamEvent.text_delta("late provider value")
                    finally:
                        settled.set()

            async def aclose(self) -> None:
                return None

        async def completed_events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

        class CapacityProvider(ModelProvider):
            name = "runtime-deadline-capacity"

            def __init__(self, source: AsyncIterator[ModelStreamEvent]) -> None:
                self.source = source
                self.entered = 0

            @property
            def stream_deadlines(self) -> ProviderStreamDeadlines:
                return ProviderStreamDeadlines(
                    transport_idle_timeout_s=1,
                    protocol_idle_timeout_s=1,
                    semantic_progress_timeout_s=0.01,
                    absolute_stream_timeout_s=1,
                    max_concurrent_streams=capacity,
                )

            def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
                del request
                self.entered += 1
                return self.source

        class RejectedBackgroundAdapter(ProviderOperationAdapter):
            def __init__(self) -> None:
                self.start_calls = 0

            async def start(
                self,
                request: ProviderOperationStartRequest,
            ) -> ProviderOperationConnection:
                del request
                self.start_calls += 1
                raise AssertionError("background provider start must not be entered")

            async def retrieve(
                self,
                state: ProviderOperationState,
            ) -> ProviderOperationSnapshot:
                del state
                raise AssertionError("rejected background operation cannot be retrieved")

            async def reconnect(
                self,
                state: ProviderOperationState,
            ) -> ProviderOperationConnection:
                del state
                raise AssertionError("rejected background operation cannot reconnect")

        class BackgroundCapacityProvider(CapacityProvider):
            def __init__(self) -> None:
                super().__init__(completed_events())
                self.adapter = RejectedBackgroundAdapter()

            @property
            def provider_operation_mode(self) -> ProviderOperationMode:
                return ProviderOperationMode.BACKGROUND

            @property
            def provider_operations(self) -> ProviderOperationAdapter:
                return self.adapter

        occupying_provider = CapacityProvider(NonsettlingEvents())
        occupying_stream = occupying_provider.runtime_stream(
            ModelRequest(
                model="occupying-model",
                messages=[Message.text("user", "occupy the bounded deadline slot")],
            )
        )
        with pytest.raises(ModelStreamDeadlineError):
            await asyncio.wait_for(anext(occupying_stream), timeout=0.5)
        assert cancellation_seen.is_set()
        assert not settled.is_set()

        store = InMemorySessionStore()
        rejected_provider = CapacityProvider(completed_events())
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(rejected_provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        rejected_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_deadline_capacity_rejected",
                messages=[Message.text("user", "do not enter the provider")],
            ),
        )

        assert rejected_provider.entered == 0
        assert rejected_events[-1].type is EventType.SESSION_FAILED
        assert (
            await store.load_active_model_completion_stage("sess_deadline_capacity_rejected")
            is None
        )

        background_store = InMemorySessionStore()
        background_provider = BackgroundCapacityProvider()
        background_app = CayuApp(
            session_store=background_store,
            enable_logging=False,
        )
        background_app.register_provider(background_provider, default=True)
        background_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        background_events = await collect_events(
            background_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_background_deadline_capacity_rejected",
                messages=[Message.text("user", "do not start a background operation")],
            ),
        )
        assert background_provider.adapter.start_calls == 0
        assert background_provider.entered == 0
        assert background_events[-1].type is EventType.SESSION_FAILED
        assert (
            await background_store.load_active_model_completion_stage(
                "sess_background_deadline_capacity_rejected"
            )
            is None
        )

        release.set()
        await asyncio.wait_for(settled.wait(), timeout=0.5)
        async with asyncio.timeout(5):
            while len(provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS) != len(
                existing_owners
            ):
                await asyncio.sleep(0)
        assert len(provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS) == len(
            existing_owners
        )

        admitted_provider = CapacityProvider(completed_events())
        admitted_app = CayuApp(enable_logging=False)
        admitted_app.register_provider(admitted_provider, default=True)
        admitted_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        admitted_events = await collect_events(
            admitted_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_deadline_capacity_recovered",
                messages=[Message.text("user", "capacity is available again")],
            ),
        )
        assert admitted_provider.entered == 1
        assert admitted_events[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(run())
