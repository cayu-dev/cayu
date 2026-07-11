from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolApprovalRecoveryOutcome,
    ToolContext,
    ToolResult,
    ToolRoundRecoveryRequest,
    ToolSpec,
)
from cayu.runtime import SessionStatus
from cayu.storage import SQLiteSessionStore


class FileSideEffectTool(Tool):
    spec = ToolSpec(
        name="write_external_effect",
        description="Append one externally visible effect.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self, path: Path, store_path: Path) -> None:
        super().__init__()
        self.path = path
        self.store_path = store_path

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("effect\n")
        with sqlite3.connect(self.store_path) as connection:
            connection.execute("UPDATE cayu_fault_control SET armed = 1")
        return ToolResult(content="external effect recorded")


def _install_terminal_write_fault(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE cayu_fault_control (armed INTEGER NOT NULL);
            INSERT INTO cayu_fault_control(armed) VALUES (0);
            CREATE TABLE cayu_fault_probe (marker TEXT NOT NULL);
            CREATE TRIGGER cayu_audit_activity_after_effect
            AFTER UPDATE OF last_activity_at ON cayu_sessions
            WHEN (SELECT armed FROM cayu_fault_control) = 1
            BEGIN
                INSERT INTO cayu_fault_probe(marker)
                VALUES ('activity-after-effect:' || NEW.status);
            END;
            CREATE TRIGGER cayu_fail_tool_completed
            BEFORE INSERT ON cayu_events
            WHEN NEW.event_type = 'tool.call.completed'
            BEGIN
                INSERT INTO cayu_fault_probe(marker) VALUES ('terminal-trigger-entered');
                SELECT RAISE(ABORT, 'forced terminal tool event write failure');
            END;
            """
        )


def _remove_terminal_write_fault(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER cayu_fail_tool_completed")
        connection.execute("DROP TRIGGER cayu_audit_activity_after_effect")


def _effect_count(path: Path) -> int:
    if not path.exists():
        return 0
    return path.read_text(encoding="utf-8").splitlines().count("effect")


@pytest.mark.anyio
async def test_sqlite_terminal_write_failure_requires_manual_recovery_without_reexecution(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "sessions.sqlite"
    effect_path = tmp_path / "external-effects.log"
    store = SQLiteSessionStore(store_path)
    _install_terminal_write_fault(store_path)
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_external_effect",
                name="write_external_effect",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        name="sqlite-fault-provider",
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="sqlite-fault-assistant",
            model="scripted-model",
            provider_name="sqlite-fault-provider",
        ),
        tools=[FileSideEffectTool(effect_path, store_path)],
    )

    initial_events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="sqlite-fault-assistant",
                session_id="sqlite-terminal-write-failure",
                messages=[Message.text("user", "Record the external effect once.")],
            )
        )
    ]

    assert _effect_count(effect_path) == 1
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert "forced terminal tool event write failure" in initial_events[-1].payload["error"]
    assert EventType.TOOL_CALL_STARTED in [event.type for event in initial_events]
    assert EventType.TOOL_CALL_COMPLETED not in [event.type for event in initial_events]

    checkpoint = await store.load_checkpoint("sqlite-terminal-write-failure")
    assert checkpoint is not None
    pending_round = checkpoint["pending_tool_round"]
    assert pending_round["tool_calls"][0]["tool_call_id"] == "call_external_effect"
    await store.close()

    with sqlite3.connect(store_path) as connection:
        activity_markers = {
            row[0] for row in connection.execute("SELECT marker FROM cayu_fault_probe").fetchall()
        }
        persisted_types = [
            row[0]
            for row in connection.execute(
                "SELECT event_type FROM cayu_events ORDER BY sequence"
            ).fetchall()
        ]
    # The failed append touched last_activity_at while the session was still running.
    # Only later, successful failure-reporting writes may leave audit markers behind.
    assert activity_markers == {"activity-after-effect:failed"}
    assert persisted_types.count("tool.call.started") == 1
    assert "tool.call.completed" not in persisted_types

    _remove_terminal_write_fault(store_path)
    reopened = SQLiteSessionStore(store_path)
    recovery_provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("recovered after operator reconciliation"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="sqlite-fault-provider",
    )
    recovery_app = CayuApp(session_store=reopened, enable_logging=False)
    recovery_app.register_provider(recovery_provider, default=True)
    recovery_app.register_agent(
        AgentSpec(
            name="sqlite-fault-assistant",
            model="scripted-model",
            provider_name="sqlite-fault-provider",
        ),
        tools=[FileSideEffectTool(effect_path, store_path)],
    )
    try:
        failed_session = await reopened.load("sqlite-terminal-write-failure")
        failed_events = await reopened.load_events("sqlite-terminal-write-failure")
        assert failed_session is not None
        assert failed_session.status == SessionStatus.FAILED
        assert failed_events[-1].type == EventType.SESSION_FAILED
        assert _effect_count(effect_path) == 1

        recovery_events = [
            event
            async for event in recovery_app.recover_tool_round(
                ToolRoundRecoveryRequest(
                    session_id="sqlite-terminal-write-failure",
                    round_id=pending_round["round_id"],
                    tool_call_id="call_external_effect",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="operator verified the external effect",
                    structured={"verified": True},
                    reason="terminal event transaction failed after execution",
                )
            )
        ]
        session = await reopened.load("sqlite-terminal-write-failure")
        durable_events = await reopened.load_events("sqlite-terminal-write-failure")
        transcript = await reopened.load_transcript("sqlite-terminal-write-failure")
    finally:
        await reopened.close()

    assert _effect_count(effect_path) == 1
    manual_terminal = next(
        event
        for event in recovery_events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("manual_recovery") is True
    )
    assert manual_terminal.payload["tool_call_id"] == "call_external_effect"
    assert manual_terminal.payload["result"]["structured"] == {"verified": True}
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert durable_events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in transcript] == ["user", "assistant", "tool", "assistant"]
    with sqlite3.connect(store_path) as connection:
        durable_sequences = [
            row[0]
            for row in connection.execute(
                "SELECT sequence FROM cayu_events ORDER BY sequence"
            ).fetchall()
        ]
    assert durable_sequences == list(range(1, len(durable_events) + 1))
