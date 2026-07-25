from __future__ import annotations

from cayu.cli.session import _tool_call_rows, _tool_inspection_record
from cayu.core import Event, EventType, ToolResult
from cayu.runtime import EventRecord


def _identity() -> dict[str, str]:
    return {
        "model_step_id": f"mstep_{'1' * 32}",
        "model_attempt_id": f"matt_{'2' * 32}",
        "tool_round_id": f"tround_{'3' * 32}",
    }


def _record(
    sequence: int,
    event_type: EventType,
    *,
    result: ToolResult | None = None,
) -> EventRecord:
    payload: dict[str, object] = {
        **_identity(),
        "tool_call_id": "call-1",
    }
    if event_type == EventType.TOOL_CALL_STARTED:
        payload["arguments"] = {}
    if result is not None:
        payload["result"] = result.model_dump(mode="json")
    return _tool_inspection_record(
        EventRecord(
            sequence=sequence,
            event=Event(
                type=event_type,
                session_id="session-1",
                tool_name="side_effect",
                payload=payload,
            ),
        )
    )


def test_tool_inspection_marks_conflicting_terminal_outcomes_unavailable() -> None:
    rows = _tool_call_rows(
        [
            _record(1, EventType.TOOL_CALL_STARTED),
            _record(
                2,
                EventType.TOOL_CALL_COMPLETED,
                result=ToolResult(content="completed"),
            ),
            _record(
                3,
                EventType.TOOL_CALL_FAILED,
                result=ToolResult(content="failed", is_error=True),
            ),
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "unavailable"
    assert row["completed_at"] is None
    assert row["duration_ms"] is None
    assert row["rendered_content_bytes"] is None
    assert row["structured_result_bytes"] is None
    assert row["artifact_bytes"] is None


def test_tool_inspection_marks_terminal_before_start_unavailable() -> None:
    rows = _tool_call_rows(
        [
            _record(
                1,
                EventType.TOOL_CALL_COMPLETED,
                result=ToolResult(content="completed"),
            ),
            _record(2, EventType.TOOL_CALL_STARTED),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"


def test_tool_inspection_marks_conflicting_approval_decisions_unavailable() -> None:
    rows = _tool_call_rows(
        [
            _record(1, EventType.TOOL_CALL_APPROVED),
            _record(
                2,
                EventType.TOOL_CALL_APPROVAL_DENIED,
                result=ToolResult(content="denied", is_error=True),
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["approval_state"] == "unavailable"


def test_tool_inspection_marks_conflicting_round_parent_identities_unavailable() -> None:
    started = _record(1, EventType.TOOL_CALL_STARTED)
    terminal = _record(
        2,
        EventType.TOOL_CALL_COMPLETED,
        result=ToolResult(content="completed"),
    )
    terminal.event.payload["model_step_id"] = f"mstep_{'9' * 32}"

    rows = _tool_call_rows([started, terminal])

    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"unavailable"}
    assert all(row["completed_at"] is None for row in rows)


def test_tool_inspection_marks_attempt_with_multiple_rounds_unavailable() -> None:
    first = _record(1, EventType.TOOL_CALL_STARTED)
    second = _record(2, EventType.TOOL_CALL_STARTED)
    second.event.payload["tool_round_id"] = f"tround_{'8' * 32}"

    rows = _tool_call_rows([first, second])

    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"unavailable"}


def test_tool_inspection_marks_malformed_identity_evidence_unavailable() -> None:
    terminal = _record(
        1,
        EventType.TOOL_CALL_COMPLETED,
        result=ToolResult(content="completed"),
    )
    terminal.event.payload["tool_round_id"] = "malformed-round-id"

    rows = _tool_call_rows([terminal])

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["completed_at"] is None
