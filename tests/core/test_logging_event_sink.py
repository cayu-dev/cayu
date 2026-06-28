from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.observability import TRACE_LEVEL, LoggingEventSink
from cayu.observability.logging import _level_for, _register_trace_level
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import CayuApp, EventSink, RunRequest
from cayu.vaults import REDACTED_SECRET, SecretRedactor

if TYPE_CHECKING:
    import pytest


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[ModelStreamEvent]) -> None:
        self.events = list(events)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        for event in self.events:
            yield event


class RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


def test_logging_event_sink_summarizes_lifecycle_events(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("cayu.test.lifecycle")
    sink = LoggingEventSink(logger=logger)
    event = Event(
        type=EventType.MODEL_STARTED,
        session_id="sess_logs",
        agent_name="assistant",
        environment_name="local",
        payload={"provider": "openai", "model": "gpt-5.5", "step": 2},
    )

    caplog.set_level(logging.INFO, logger=logger.name)
    asyncio.run(sink.emit(event))

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    assert "model.started" in record.message
    assert "sess_logs agent=assistant env=local" in record.message
    assert "provider=openai" in record.message
    assert "model=gpt-5.5" in record.message
    assert "step=2" in record.message


def test_logging_event_sink_routes_token_deltas_to_trace(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("cayu.test.delta")
    sink = LoggingEventSink(logger=logger)
    event = Event(
        type=EventType.MODEL_TEXT_DELTA,
        session_id="sess_delta",
        payload={"delta": "streamed token chunk"},
    )

    # DEBUG is now clean — token deltas live below it, at TRACE.
    caplog.set_level(logging.DEBUG, logger=logger.name)
    asyncio.run(sink.emit(event))
    assert caplog.records == []

    # They surface only when the logger opts into TRACE.
    caplog.clear()
    caplog.set_level(TRACE_LEVEL, logger=logger.name)
    asyncio.run(sink.emit(event))
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == TRACE_LEVEL
    assert "model.text.delta" in caplog.records[0].message


def test_level_for_routes_events_to_expected_levels() -> None:
    assert _level_for(EventType.MODEL_TEXT_DELTA) == TRACE_LEVEL
    assert _level_for(EventType.HOOK_STARTED) == logging.DEBUG
    assert _level_for(EventType.MODEL_STARTED) == logging.INFO
    assert _level_for(EventType.SESSION_INTERRUPTED) == logging.WARNING
    assert _level_for(EventType.SESSION_FAILED) == logging.ERROR


def test_trace_level_name_is_registered() -> None:
    assert TRACE_LEVEL == 5
    assert logging.getLevelName(TRACE_LEVEL) == "TRACE"


def test_register_trace_level_skips_when_name_already_taken(monkeypatch) -> None:
    # Another library already owns the "TRACE" name at a different level — don't clobber it.
    added: list[tuple[int, str]] = []
    monkeypatch.setattr(logging, "getLevelNamesMapping", lambda: {"TRACE": 15, "DEBUG": 10})
    monkeypatch.setattr(logging, "addLevelName", lambda level, name: added.append((level, name)))
    _register_trace_level()
    assert added == []


def test_register_trace_level_skips_when_level_number_already_taken(monkeypatch) -> None:
    # Another library already named level 5 — don't overwrite that name.
    added: list[tuple[int, str]] = []
    monkeypatch.setattr(logging, "getLevelNamesMapping", lambda: {"VERBOSE": TRACE_LEVEL})
    monkeypatch.setattr(logging, "addLevelName", lambda level, name: added.append((level, name)))
    _register_trace_level()
    assert added == []


def test_register_trace_level_registers_when_free(monkeypatch) -> None:
    added: list[tuple[int, str]] = []
    monkeypatch.setattr(logging, "getLevelNamesMapping", lambda: {"DEBUG": 10, "INFO": 20})
    monkeypatch.setattr(logging, "addLevelName", lambda level, name: added.append((level, name)))
    _register_trace_level()
    assert added == [(TRACE_LEVEL, "TRACE")]


def test_logging_event_sink_summarizes_normalized_usage_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("cayu.test.usage")
    sink = LoggingEventSink(logger=logger)
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="sess_usage",
        payload={
            "usage_metrics": {
                "provider_name": "openai",
                "model": "gpt-5.5",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "reasoning_output_tokens": 5,
                "cache": {
                    "read_tokens": 60,
                    "write_tokens": 0,
                    "cached_input_tokens": 60,
                    "uncached_input_tokens": 40,
                },
            }
        },
    )

    caplog.set_level(logging.INFO, logger=logger.name)
    asyncio.run(sink.emit(event))

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "input_tokens=100" in message
    assert "output_tokens=20" in message
    assert "reasoning_output_tokens=5" in message
    assert "cache_read_tokens=60" in message
    assert "cache_write_tokens=0" in message
    assert "cached_input_tokens=60" in message


def test_logging_event_sink_does_not_dump_raw_tool_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("cayu.test.tool")
    sink = LoggingEventSink(logger=logger)
    event = Event(
        type=EventType.TOOL_CALL_STARTED,
        session_id="sess_tool",
        tool_name="exec_command",
        payload={
            "tool_call_id": "call_1",
            "arguments": {
                "shell": "echo should_not_be_logged",
                "env": {"API_KEY": "sk-should-not-be-logged"},
            },
        },
    )

    caplog.set_level(logging.INFO, logger=logger.name)
    asyncio.run(sink.emit(event))

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "tool.call.started" in message
    assert "tool=exec_command" in message
    assert "call=call_1" in message
    assert "should_not_be_logged" not in message
    assert "API_KEY" not in message


def test_logging_event_sink_truncates_errors(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("cayu.test.error")
    sink = LoggingEventSink(logger=logger, error_summary_limit=12)
    event = Event(
        type=EventType.SESSION_FAILED,
        session_id="sess_failed",
        payload={"error_type": "RuntimeError", "error": "abcdefghijklmnopqrstuvwxyz"},
    )

    caplog.set_level(logging.ERROR, logger=logger.name)
    asyncio.run(sink.emit(event))

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "error_type=RuntimeError" in caplog.records[0].message
    assert "error=abcdefghijkl..." in caplog.records[0].message
    assert "mnopqrstuvwxyz" not in caplog.records[0].message


def test_logging_event_sink_escapes_control_characters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("cayu.test.escape")
    sink = LoggingEventSink(logger=logger)
    event = Event(
        type=EventType.MODEL_STARTED,
        session_id="sess_log\\ninjection",
        agent_name="assistant\\tname",
        environment_name="local",
        payload={"provider": "openai", "model": "gpt\\n5.5", "step": 1},
    )

    caplog.set_level(logging.INFO, logger=logger.name)
    asyncio.run(sink.emit(event))

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "\n" not in message
    assert "\t" not in message
    assert "sess_log\\\\ninjection" in message
    assert "agent=assistant\\\\tname" in message
    assert "model=gpt\\\\n5.5" in message


def test_logging_event_sink_redacts_configured_secret_from_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("cayu.test.redact")
    sink = LoggingEventSink(
        logger=logger,
        redactor=SecretRedactor("sk-secret-value"),
    )
    event = Event(
        type=EventType.MODEL_ERROR,
        session_id="sess_redact",
        payload={
            "error_type": "RuntimeError",
            "error": "provider failed with key sk-secret-value",
        },
    )

    caplog.set_level(logging.WARNING, logger=logger.name)
    asyncio.run(sink.emit(event))

    assert len(caplog.records) == 1
    assert "sk-secret-value" not in caplog.records[0].message
    assert REDACTED_SECRET in caplog.records[0].message


def test_logging_event_sink_redacts_before_truncating_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-" + ("x" * 80)
    logger = logging.getLogger("cayu.test.redact_before_truncate")
    sink = LoggingEventSink(
        logger=logger,
        error_summary_limit=30,
        redactor=SecretRedactor(secret),
    )
    event = Event(
        type=EventType.SESSION_FAILED,
        session_id="sess_redact_truncate",
        payload={
            "error_type": "RuntimeError",
            "error": f"provider failed with key {secret}",
        },
    )

    caplog.set_level(logging.ERROR, logger=logger.name)
    asyncio.run(sink.emit(event))

    assert len(caplog.records) == 1
    assert secret not in caplog.records[0].message
    assert "sk-xxxxxxxx" not in caplog.records[0].message
    assert "[REDA" in caplog.records[0].message


def test_logging_event_sink_rejects_invalid_redactor() -> None:
    try:
        LoggingEventSink(redactor="not-a-redactor")  # type: ignore[arg-type]
    except TypeError as exc:
        assert "redactor" in str(exc)
    else:
        raise AssertionError("Expected TypeError.")


def test_cayu_app_registers_logging_sink_by_default() -> None:
    app = CayuApp()

    assert any(isinstance(sink, LoggingEventSink) for sink in app._event_sinks)


def test_cayu_app_can_disable_default_logging_sink() -> None:
    app = CayuApp(enable_logging=False)

    assert not any(isinstance(sink, LoggingEventSink) for sink in app._event_sinks)


def test_default_logging_sink_does_not_replace_custom_sinks() -> None:
    recorder = RecordingSink()
    app = CayuApp(event_sinks=[recorder])
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_custom_sink",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert len(recorder.events) == len(events)


async def _collect_run(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]
