"""The default log line for a tool failure must carry the failure reason."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cayu import Event, EventType
from cayu.observability import LoggingEventSink

if TYPE_CHECKING:
    import pytest


def test_tool_call_failed_logs_the_result_reason(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("cayu.test.toolfail")
    sink = LoggingEventSink(logger=logger)
    event = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="sess_logs",
        tool_name="exec_command",
        payload={
            "tool_call_id": "call_1",
            "result": {"content": "No runner configured for this tool call.", "is_error": True},
        },
    )

    caplog.set_level(logging.INFO, logger=logger.name)
    asyncio.run(sink.emit(event))

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "tool.call.failed" in message
    assert "tool=exec_command" in message
    # The reason used to be dropped (it lives under payload["result"]["content"]).
    assert "reason=No runner configured" in message
