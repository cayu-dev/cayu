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


def test_tool_projection_logs_only_bounded_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("cayu.test.toolprojection")
    sink = LoggingEventSink(logger=logger)
    event = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="sess_logs",
        tool_name="search",
        payload={
            "tool_call_id": "call_1",
            "result": {
                "content": "private oversized preview content",
                "is_error": True,
            },
            "tool_result_projection": {
                "status": "externalized",
                "policy_id": "cayu.artifact_externalizing_tool_result.v1",
                "original_bytes": 40_000,
                "projected_bytes": 512,
                "original_token_estimate": 10_000,
                "projected_token_estimate": 128,
                "token_estimation_method": "unicode_codepoints_divided_by_4_ceiling_v1",
                "artifact_id": "art_11111111111111111111111111111111",
                "artifact_sha256": "a" * 64,
            },
        },
    )

    caplog.set_level(logging.INFO, logger=logger.name)
    asyncio.run(sink.emit(event))

    message = caplog.records[0].message
    assert "projection_status=externalized" in message
    assert "projection_original_bytes=40000" in message
    assert "projection_projected_bytes=512" in message
    assert "projection_artifact_id=art_11111111111111111111111111111111" in message
    assert "private oversized preview content" not in message


def test_unchanged_tool_projection_keeps_the_failure_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("cayu.test.toolprojection.unchanged")
    sink = LoggingEventSink(logger=logger)
    event = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="sess_logs",
        tool_name="exec_command",
        payload={
            "tool_call_id": "call_unchanged",
            "result": {"content": "ordinary tool failure", "is_error": True},
            "tool_result_projection": {
                "status": "unchanged",
                "policy_id": "cayu.artifact_externalizing_tool_result.v1",
                "original_bytes": 21,
                "projected_bytes": 21,
                "original_token_estimate": 6,
                "projected_token_estimate": 6,
                "token_estimation_method": "unicode_codepoints_divided_by_4_ceiling_v1",
            },
        },
    )

    caplog.set_level(logging.INFO, logger=logger.name)
    asyncio.run(sink.emit(event))

    message = caplog.records[0].message
    assert "projection_status=unchanged" in message
    assert "reason=ordinary tool failure" in message
