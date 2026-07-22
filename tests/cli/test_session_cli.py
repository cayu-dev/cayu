from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cayu import SQLiteSessionStore
from cayu.cli import main
from cayu.core import (
    Event,
    EventType,
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from cayu.runtime import RunRequest, SessionIdentity, SessionStatus


def _write_project(root: Path, database: str = "data/cayu.db") -> Path:
    (root / "pyproject.toml").write_text(
        """
[tool.cayu]
factory = "app:build_app"

[tool.cayu.session_store]
backend = "sqlite"
path = """
        + json.dumps(database)
        + "\n",
        encoding="utf-8",
    )
    return root / database


def _seed_sessions(path: Path) -> None:
    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            first = await store.create(
                RunRequest(
                    agent_name="writer",
                    session_id="sess_older",
                    messages=[Message.text("user", "older")],
                    environment_name="local",
                    labels={"team": "docs"},
                ),
                identity=SessionIdentity(provider_name="fake", model="model-a"),
            )
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id="sess_newer",
                    messages=[Message.text("user", "newer")],
                    environment_name="sandbox",
                    labels={"team": "runtime"},
                ),
                identity=SessionIdentity(provider_name="fake", model="model-b"),
            )
            await store.update_status(first.id, SessionStatus.COMPLETED)
        finally:
            await store.close()

    asyncio.run(seed())


def test_session_list_uses_project_target_and_emits_stable_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    _seed_sessions(database)
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert (
        main(
            [
                "session",
                "list",
                "--status",
                "completed",
                "--label",
                "team=docs",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "has_more": False,
        "next_cursor": None,
        "schema_version": "1",
        "sessions": [
            {
                "agent": "writer",
                "created_at": payload["sessions"][0]["created_at"],
                "environment": "local",
                "id": "sess_older",
                "last_activity_at": payload["sessions"][0]["last_activity_at"],
                "model": "model-a",
                "provider": "fake",
                "run_epoch": 0,
                "status": "completed",
                "updated_at": payload["sessions"][0]["updated_at"],
            }
        ],
        "total_count": 1,
    }


def test_session_list_empty_result_is_successful_json(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    _seed_sessions(database)

    assert (
        main(
            [
                "session",
                "list",
                "--sqlite",
                str(database),
                "--agent",
                "missing",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions"] == []
    assert payload["total_count"] == 0
    assert payload["has_more"] is False


def test_session_output_names_a_destination_and_format_is_independent(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    _seed_sessions(database)
    destination = tmp_path / "sessions.json"

    assert (
        main(
            [
                "session",
                "list",
                "--sqlite",
                str(database),
                "--output",
                str(destination),
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == ""
    assert json.loads(destination.read_text(encoding="utf-8"))["sessions"]


def test_session_list_defaults_to_last_activity_order_and_pages_by_cursor(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    _seed_sessions(database)

    assert (
        main(
            [
                "session",
                "list",
                "--sqlite",
                str(database),
                "--limit",
                "1",
                "--json",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert [session["id"] for session in first["sessions"]] == ["sess_older"]
    assert first["has_more"] is True
    assert type(first["next_cursor"]) is str

    assert (
        main(
            [
                "session",
                "list",
                "--sqlite",
                str(database),
                "--limit",
                "1",
                "--cursor",
                first["next_cursor"],
                "--json",
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)
    assert [session["id"] for session in second["sessions"]] == ["sess_newer"]
    assert second["has_more"] is False


def test_session_list_filters_completed_failed_and_running_states(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            for status in (
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.RUNNING,
            ):
                session_id = f"sess_{status.value}"
                await store.create(
                    RunRequest(
                        agent_name="operator",
                        session_id=session_id,
                        messages=[Message.text("user", "inspect")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="model"),
                )
                await store.update_status(session_id, status)
        finally:
            await store.close()

    asyncio.run(seed())

    for status in ("completed", "failed", "running"):
        assert (
            main(
                [
                    "session",
                    "list",
                    "--sqlite",
                    str(database),
                    "--status",
                    status,
                    "--json",
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert [session["id"] for session in payload["sessions"]] == [f"sess_{status}"]


def test_session_commands_render_successful_explicit_tables(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_table",
                    messages=[Message.text("user", "inspect")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_transcript_messages(
                "sess_table",
                [Message.text("assistant", "done")],
            )
            for event in (
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_table",
                    payload={"usage_metrics": {"input_tokens": 1, "output_tokens": 1}},
                ),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_table",
                    tool_name="read_file",
                    payload={"tool_call_id": "call-1", "arguments": {"path": "README.md"}},
                ),
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="sess_table",
                    tool_name="read_file",
                    payload={
                        "tool_call_id": "call-1",
                        "result": {
                            "content": "done",
                            "structured": None,
                            "artifacts": [],
                            "is_error": False,
                        },
                    },
                ),
            ):
                await store.append_event("sess_table", event)
        finally:
            await store.close()

    asyncio.run(seed())

    commands = (
        (["list"], "id"),
        (["show", "sess_table"], "field"),
        (["usage", "sess_table"], "sequence"),
        (["tools", "sess_table"], "tool_call_id"),
        (["events", "sess_table"], "timestamp"),
        (["transcript", "sess_table"], "content_kinds"),
    )
    for command, header in commands:
        assert main(["session", *command, "--sqlite", str(database), "--table"]) == 0
        first_line = capsys.readouterr().out.splitlines()[0]
        assert header in first_line


def test_session_show_summarizes_oversized_state_without_printing_content(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    oversized_result = "x" * 500_000
    secret = "do-not-print-this-secret"

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_large",
                    messages=[Message.text("user", "inspect this")],
                    environment_name="local",
                    labels={"case": "oversized"},
                    metadata={
                        "subagent": {"mode": "background"},
                        "customer": {"id": "not-runtime-metadata"},
                    },
                ),
                identity=SessionIdentity(
                    provider_name="fake",
                    model="model-large",
                    runtime_version="1.2.3",
                ),
            )
            await store.append_transcript_messages(
                session.id,
                [
                    Message.text("user", "inspect this"),
                    Message.tool_result(
                        tool_call_id="call_large",
                        tool_name="read_file",
                        content=oversized_result,
                    ),
                ],
            )
            await store.append_event(
                session.id,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session.id,
                    payload={
                        "transcript_cursor": 2,
                        "usage_metrics": {
                            "provider_name": "fake",
                            "requested_model": "model-large",
                            "model": "model-large-2026",
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                            "cache": {
                                "cached_input_tokens": 80,
                                "uncached_input_tokens": 20,
                            },
                        },
                    },
                ),
            )
            await store.append_event(
                session.id,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session.id,
                    payload={
                        "usage_metrics": {
                            "input_tokens": 1,
                            "custom_counter": 1,
                        }
                    },
                ),
            )
            await store.append_event(
                session.id,
                Event(
                    type="custom.diagnostic",
                    session_id=session.id,
                    payload={"api_key": secret, "blob": "y" * 20_000},
                ),
            )
        finally:
            await store.close()

    asyncio.run(seed())

    assert (
        main(
            [
                "session",
                "show",
                "sess_large",
                "--sqlite",
                str(database),
                "--json",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert len(output.encode("utf-8")) < 20_000
    assert oversized_result[:1000] not in output
    assert secret not in output
    assert payload["session"]["id"] == "sess_large"
    assert payload["session"]["labels"] == {"case": "oversized"}
    assert payload["session"]["label_count"] == 1
    assert payload["session"]["labels_truncated"] is False
    assert "not-runtime-metadata" not in output
    assert payload["transcript"]["message_count"] == 2
    assert payload["transcript"]["largest_message_bytes"] >= 500_000
    assert payload["events"]["event_count"] == 3
    assert payload["events"]["largest_payload_bytes"] >= 20_000
    assert payload["activity"]["model_calls"] == 2
    assert payload["activity"]["model_calls_with_usage"] == 1
    assert payload["usage"]["input_tokens"] == 100
    assert payload["usage"]["cached_input_tokens"] == 80
    assert payload["budget"]["cost_state"] == "unknown"
    assert payload["budget"]["amount"] is None


def test_session_show_distinguishes_partial_and_mixed_currency_ledgers(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            for session_id in (
                "sess_partial_cost",
                "sess_mixed_currency",
                "sess_outstanding_cost",
                "sess_released_cost",
                "sess_parallel_limits",
                "sess_unpriced_cost",
                "sess_failed_reservation_cost",
            ):
                await store.create(
                    RunRequest(
                        agent_name="operator",
                        session_id=session_id,
                        messages=[Message.text("user", "inspect cost")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="model"),
                )

            for session_id, currencies, priced in (
                ("sess_partial_cost", ("USD", "USD"), (True, False)),
                ("sess_mixed_currency", ("USD", "EUR"), (True, True)),
            ):
                for index, (currency, has_pricing) in enumerate(
                    zip(currencies, priced, strict=True),
                    start=1,
                ):
                    reservation_id = f"{session_id}-reservation-{index}"
                    await store.append_event(
                        session_id,
                        Event(
                            type=EventType.BUDGET_RESERVED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "scope": "session",
                                "key": session_id,
                                "window": "lifetime",
                                "currency": currency,
                                "maximum": "1.00",
                                "action": "interrupt",
                                "requested": "1.00",
                            },
                        ),
                    )
                    await store.append_event(
                        session_id,
                        Event(
                            type=EventType.BUDGET_RECONCILED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "actual_amount": "0.25",
                                **(
                                    {"pricing": {"provider_name": "fake", "model": "model"}}
                                    if has_pricing
                                    else {}
                                ),
                            },
                        ),
                    )

            for session_id, second_terminal in (
                ("sess_outstanding_cost", None),
                ("sess_released_cost", EventType.BUDGET_RESERVATION_RELEASED),
            ):
                for index in (1, 2):
                    reservation_id = f"{session_id}-reservation-{index}"
                    await store.append_event(
                        session_id,
                        Event(
                            type=EventType.BUDGET_RESERVED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "scope": "session",
                                "key": session_id,
                                "window": "lifetime",
                                "currency": "USD",
                                "maximum": "1.00",
                                "action": "interrupt",
                                "requested": "1.00",
                            },
                        ),
                    )
                    if index == 1:
                        await store.append_event(
                            session_id,
                            Event(
                                type=EventType.BUDGET_RECONCILED,
                                session_id=session_id,
                                payload={
                                    "reservation_id": reservation_id,
                                    "actual_amount": "0.25",
                                    "pricing": {
                                        "provider_name": "fake",
                                        "model": "model",
                                    },
                                },
                            ),
                        )
                    elif second_terminal is not None:
                        await store.append_event(
                            session_id,
                            Event(
                                type=second_terminal,
                                session_id=session_id,
                                payload={"reservation_id": reservation_id},
                            ),
                        )

            for scope in ("app", "agent"):
                reservation_id = f"sess_parallel_limits-{scope}"
                await store.append_event(
                    "sess_parallel_limits",
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id="sess_parallel_limits",
                        payload={
                            "reservation_id": reservation_id,
                            "scope": scope,
                            "key": "runtime" if scope == "app" else "operator",
                            "window": "lifetime",
                            "currency": "USD",
                            "maximum": "1.00",
                            "action": "interrupt",
                            "requested": "1.00",
                        },
                    ),
                )
                await store.append_event(
                    "sess_parallel_limits",
                    Event(
                        type=EventType.BUDGET_RECONCILED,
                        session_id="sess_parallel_limits",
                        payload={
                            "reservation_id": reservation_id,
                            "actual_amount": "0.25",
                            "pricing": {"provider_name": "fake", "model": "model"},
                        },
                    ),
                )

            await store.append_event(
                "sess_unpriced_cost",
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id="sess_unpriced_cost",
                    payload={
                        "reservation_id": "sess_unpriced_cost-reservation",
                        "scope": "session",
                        "key": "sess_unpriced_cost",
                        "window": "lifetime",
                        "currency": "USD",
                        "maximum": "1.00",
                        "action": "interrupt",
                        "requested": "1.00",
                    },
                ),
            )
            await store.append_event(
                "sess_failed_reservation_cost",
                Event(
                    type=EventType.BUDGET_RESERVATION_FAILED,
                    session_id="sess_failed_reservation_cost",
                    payload={
                        "accepted": False,
                        "scope": "session",
                        "key": "sess_failed_reservation_cost",
                        "window": "lifetime",
                        "currency": "USD",
                        "requested": "1.00",
                    },
                ),
            )
            await store.append_event(
                "sess_unpriced_cost",
                Event(
                    type=EventType.BUDGET_RECONCILED,
                    session_id="sess_unpriced_cost",
                    payload={
                        "reservation_id": "sess_unpriced_cost-reservation",
                        "actual_amount": "0.25",
                    },
                ),
            )
        finally:
            await store.close()

    asyncio.run(seed())

    for session_id, expected_state, expected_amount, expected_currency in (
        ("sess_partial_cost", "partial", None, None),
        ("sess_mixed_currency", "mixed_currency", None, None),
        ("sess_outstanding_cost", "partial", None, None),
        ("sess_released_cost", "priced", "0.25", "USD"),
        ("sess_parallel_limits", "priced", "0.25", "USD"),
        ("sess_unpriced_cost", "unpriced", None, None),
        ("sess_failed_reservation_cost", "priced", "0", "USD"),
    ):
        assert (
            main(
                [
                    "session",
                    "show",
                    session_id,
                    "--sqlite",
                    str(database),
                    "--json",
                ]
            )
            == 0
        )
        budget = json.loads(capsys.readouterr().out)["budget"]
        assert budget["cost_state"] == expected_state
        assert budget["amount"] == expected_amount
        assert budget["currency"] == expected_currency


def test_session_show_reports_approval_and_user_input_pending_actions(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)

    def pending_call(call_id: str, tool_name: str) -> dict[str, object]:
        return {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "arguments": {},
            "policy_decision": None,
            "reason": None,
            "metadata": {},
            "active_taint_labels": [],
        }

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            for session_id in ("sess_approval", "sess_input"):
                await store.create(
                    RunRequest(
                        agent_name="operator",
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="model"),
                )

            await store.append_event(
                "sess_approval",
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="sess_approval",
                    tool_name="deploy",
                    payload={
                        "approval": {
                            "approval_id": "approval-1",
                            "tool_call_id": "call-approval",
                            "tool_name": "deploy",
                            "arguments": {},
                            "agent_name": "operator",
                            "tool_calls": [pending_call("call-approval", "deploy")],
                        }
                    },
                ),
            )
            await store.checkpoint(
                "sess_approval",
                {
                    "pending_tool_approval": {
                        "approval_id": "approval-1",
                        "tool_call_id": "call-approval",
                        "tool_name": "deploy",
                        "arguments": {},
                        "agent_name": "operator",
                        "tool_calls": [pending_call("call-approval", "deploy")],
                    }
                },
            )
            await store.update_status("sess_approval", SessionStatus.INTERRUPTED)

            await store.append_event(
                "sess_input",
                Event(
                    type=EventType.SESSION_AWAITING_USER_INPUT,
                    session_id="sess_input",
                    tool_name="ask_user",
                    payload={
                        "input_id": "input-1",
                        "tool_call_id": "call-input",
                        "question": "Deploy?",
                        "options": ["yes", "no"],
                    },
                ),
            )
            await store.checkpoint(
                "sess_input",
                {
                    "pending_user_input": {
                        "input_id": "input-1",
                        "tool_call_id": "call-input",
                        "tool_name": "ask_user",
                        "question": "Deploy?",
                        "options": ["yes", "no"],
                        "arguments": {},
                        "agent_name": "operator",
                        "tool_calls": [pending_call("call-input", "ask_user")],
                    }
                },
            )
            await store.update_status("sess_input", SessionStatus.INTERRUPTED)
        finally:
            await store.close()

    asyncio.run(seed())

    for session_id, expected_kind in (
        ("sess_approval", "tool_approval"),
        ("sess_input", "user_input"),
    ):
        assert (
            main(
                [
                    "session",
                    "show",
                    session_id,
                    "--sqlite",
                    str(database),
                    "--json",
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["session"]["status"] == "interrupted"
        assert payload["pending_action"] == {
            "count": 1,
            "issue_count": 0,
            "kinds": [expected_kind],
        }


def test_session_detail_commands_report_missing_session_concisely(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    _seed_sessions(database)

    for command in ("show", "usage", "tools", "events", "transcript"):
        assert main(["session", command, "missing", "--sqlite", str(database)]) == 1

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["error"] == {
            "code": "SESSION_INSPECTION_FAILED",
            "message": "Session not found: missing",
        }
        assert captured.err == ""


def test_session_show_missing_sqlite_target_does_not_create_it(
    tmp_path: Path,
    capsys,
) -> None:
    missing = tmp_path / "missing" / "data" / "cayu.db"

    assert main(["session", "show", "missing", "--sqlite", str(missing)]) == 1

    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "SESSION_INSPECTION_FAILED"
    assert captured.err == ""
    assert not missing.exists()
    assert not missing.parent.exists()


def test_session_cli_redacts_postgres_dsn_from_connection_errors(
    monkeypatch,
    capsys,
) -> None:
    from cayu.cli import session as session_cli

    secret = "database-password"
    dsn = f"postgresql://operator:{secret}@db.example/cayu"

    def fail_to_open(target) -> None:
        raise RuntimeError(f"connection rejected for {target.postgres_dsn}")

    monkeypatch.setattr(session_cli, "_open_read_only_store", fail_to_open)

    assert main(["session", "list", "--postgres", dsn]) == 1
    captured = capsys.readouterr()
    error = json.loads(captured.out)["error"]
    assert secret not in error["message"]
    assert "postgresql://db.example/cayu" in error["message"]
    assert captured.err == ""


def test_session_usage_reports_per_call_cache_and_honest_pricing_state(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    started_at = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="analyst",
                    session_id="sess_usage",
                    messages=[Message.text("user", "analyze")],
                ),
                identity=SessionIdentity(provider_name="fake", model="requested-model"),
            )
            events = [
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id="sess_usage",
                    timestamp=started_at,
                    payload={
                        "accepted": True,
                        "reservation_id": "reservation-1",
                        "currency": "USD",
                        "requested": "0.25",
                    },
                ),
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_usage",
                    timestamp=started_at + timedelta(seconds=1),
                    payload={
                        "transcript_cursor": 3,
                        "usage_metrics": {
                            "provider_name": "fake",
                            "requested_model": "requested-model",
                            "model": "resolved-model",
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "total_tokens": 12,
                            "reasoning_output_tokens": 1,
                            "cache": {
                                "read_tokens": 7,
                                "write_tokens": 0,
                                "cached_input_tokens": 7,
                                "uncached_input_tokens": 3,
                            },
                        },
                    },
                ),
                Event(
                    type=EventType.BUDGET_RECONCILED,
                    session_id="sess_usage",
                    timestamp=started_at + timedelta(seconds=2),
                    payload={
                        "reservation_id": "reservation-1",
                        "status": "reconciled",
                        "reserved_amount": "0.25",
                        "actual_amount": "0.01",
                        "released_amount": "0.24",
                        "pricing": {"provider_name": "fake", "model": "resolved-model"},
                    },
                ),
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_usage",
                    timestamp=started_at + timedelta(seconds=3),
                    payload={
                        "transcript_cursor": 5,
                        "usage_metrics": {"input_tokens": 1, "custom_counter": 1},
                    },
                ),
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id="sess_usage",
                    timestamp=started_at + timedelta(seconds=4),
                    payload={
                        "accepted": True,
                        "reservation_id": "reservation-unmatched",
                        "currency": "USD",
                        "requested": "0.50",
                    },
                ),
                Event(
                    type=EventType.BUDGET_RESERVATION_RELEASED,
                    session_id="sess_usage",
                    timestamp=started_at + timedelta(seconds=5),
                    payload={"reservation_id": "reservation-unmatched"},
                ),
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id="sess_usage",
                    timestamp=started_at + timedelta(seconds=6),
                    payload={
                        "accepted": True,
                        "reservation_id": "reservation-open",
                        "currency": "USD",
                        "requested": "0.75",
                    },
                ),
            ]
            for event in events:
                await store.append_event("sess_usage", event)
        finally:
            await store.close()

    asyncio.run(seed())

    assert (
        main(
            [
                "session",
                "usage",
                "sess_usage",
                "--sqlite",
                str(database),
                "--limit",
                "1",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["total_calls"] == 2
    assert payload["has_more"] is True
    assert payload["next_offset"] == 1
    assert payload["aggregate"] == {
        "cache_read_tokens": 7,
        "cache_write_tokens": 0,
        "cached_input_tokens": 7,
        "input_tokens": 10,
        "model_calls": 2,
        "model_calls_with_usage": 1,
        "output_tokens": 2,
        "reasoning_tokens": 1,
        "tool_calls": 0,
        "total_tokens": 12,
        "uncached_input_tokens": 3,
    }
    first = payload["calls"][0]
    assert first["provider"] == "fake"
    assert first["requested_model"] == "requested-model"
    assert first["resolved_model"] == "resolved-model"
    assert first["transcript_cursor"] == 3
    assert first["pricing_state"] == "unknown"
    assert first["ledger"] == []
    assert payload["unmatched_ledger"] == [
        {
            "actual_amount": "0.01",
            "currency": "USD",
            "outcome": "reconciled",
            "pricing_state": "priced",
            "reservation_id": "reservation-1",
            "reserved_amount": "0.25",
        },
    ]
    assert payload["unmatched_ledger_total"] == 3
    assert payload["unmatched_ledger_has_more"] is True
    assert payload["unmatched_ledger_next_offset"] == 1

    assert (
        main(
            [
                "session",
                "usage",
                "sess_usage",
                "--sqlite",
                str(database),
                "--offset",
                "1",
                "--json",
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)["calls"][0]
    assert second["input_tokens"] is None
    assert second["pricing_state"] == "unknown"
    assert second["ledger"] == []

    assert (
        main(
            [
                "session",
                "usage",
                "sess_usage",
                "--sqlite",
                str(database),
                "--table",
            ]
        )
        == 0
    )
    table_output = capsys.readouterr().out
    assert "unmatched ledger" in table_output.casefold()
    assert "aggregate usage" in table_output.casefold()
    usage_header = table_output.splitlines()[0]
    for field in (
        "transcript_cursor",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    ):
        assert field in usage_header
    assert "model_calls" in table_output
    assert "reservation-open" in table_output

    assert (
        main(
            [
                "session",
                "usage",
                "sess_usage",
                "--sqlite",
                str(database),
                "--jsonl",
            ]
        )
        == 0
    )
    jsonl_rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {row["record_type"] for row in jsonl_rows} == {
        "aggregate",
        "model_call",
        "unmatched_ledger",
    }
    aggregate_row = next(row for row in jsonl_rows if row["record_type"] == "aggregate")
    assert aggregate_row["total_tokens"] == 12


def test_session_aggregate_commands_bound_retained_projections_not_raw_payloads(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from cayu.cli import session as session_cli
    from cayu.runtime import sessions as session_runtime

    database = _write_project(tmp_path)

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_bounded",
                    messages=[Message.text("user", "inspect")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_event(
                "sess_bounded",
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_bounded",
                    payload={"blob": "x" * 4096},
                ),
            )
        finally:
            await store.close()

    asyncio.run(seed())

    monkeypatch.setattr(session_cli, "_MAX_COLLECTED_EVENT_BYTES", 512)
    assert (
        main(
            [
                "session",
                "usage",
                "sess_bounded",
                "--sqlite",
                str(database),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["total_calls"] == 1

    monkeypatch.setattr(session_runtime, "_SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES", 512)
    assert (
        main(
            [
                "session",
                "show",
                "sess_bounded",
                "--sqlite",
                str(database),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["events"]["largest_payload_bytes"] >= 4096

    monkeypatch.setattr(session_runtime, "_SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES", 32)
    assert (
        main(
            [
                "session",
                "show",
                "sess_bounded",
                "--sqlite",
                str(database),
                "--json",
            ]
        )
        == 1
    )
    assert "retained-event safety limit" in json.loads(capsys.readouterr().out)["error"]["message"]

    monkeypatch.setattr(session_cli, "_MAX_COLLECTED_EVENT_RECORDS", 0)
    assert (
        main(
            [
                "session",
                "usage",
                "sess_bounded",
                "--sqlite",
                str(database),
                "--json",
            ]
        )
        == 1
    )
    assert "0-event safety limit" in json.loads(capsys.readouterr().out)["error"]["message"]

    monkeypatch.setattr(session_runtime, "_SESSION_INSPECTION_MAX_RECORDS", 0)
    assert (
        main(
            [
                "session",
                "show",
                "sess_bounded",
                "--sqlite",
                str(database),
                "--json",
            ]
        )
        == 1
    )
    assert "0-event safety limit" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_session_tools_pairs_parallel_calls_and_omits_results(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    started_at = datetime(2026, 7, 20, 13, 0, tzinfo=UTC)
    secret = "tool-argument-secret"
    large_result = "sensitive result content" * 5000

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="builder",
                    session_id="sess_tools",
                    messages=[Message.text("user", "run tools")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            events = [
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_tools",
                    tool_name="read_file",
                    timestamp=started_at,
                    payload={
                        "tool_call_id": "call-1",
                        "tool_round_id": "round-1",
                        "arguments": {"path": "one.txt", "api_key": secret},
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_tools",
                    tool_name="search",
                    timestamp=started_at,
                    payload={
                        "tool_call_id": "call-2",
                        "tool_round_id": "round-1",
                        "arguments": {"query": "Cayu"},
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="sess_tools",
                    tool_name="search",
                    timestamp=started_at + timedelta(milliseconds=250),
                    payload={
                        "tool_call_id": "call-2",
                        "tool_round_id": "round-1",
                        "result": {
                            "content": large_result,
                            "structured": {"returned": 2, "truncated": True},
                            "artifacts": [
                                {"size_bytes": 10},
                                {"path": "report.txt"},
                            ],
                            "is_error": False,
                        },
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="sess_tools",
                    tool_name="deploy",
                    timestamp=started_at + timedelta(seconds=1),
                    payload={
                        "approval": {
                            "approval_id": "approval-3",
                            "tool_call_id": "call-3",
                            "tool_name": "deploy",
                            "arguments": {},
                            "agent_name": "builder",
                            "tool_calls": [],
                        }
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_APPROVED,
                    session_id="sess_tools",
                    tool_name="deploy",
                    timestamp=started_at + timedelta(seconds=2),
                    payload={"tool_call_id": "call-3", "tool_round_id": "round-2"},
                ),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_tools",
                    tool_name="deploy",
                    timestamp=started_at + timedelta(seconds=3),
                    payload={
                        "tool_call_id": "call-3",
                        "tool_round_id": "round-2",
                        "arguments": {},
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_BLOCKED,
                    session_id="sess_tools",
                    tool_name="deploy",
                    timestamp=started_at + timedelta(seconds=4),
                    payload={
                        "tool_call_id": "call-3",
                        "tool_round_id": "round-2",
                        "result": {
                            "content": "blocked",
                            "structured": None,
                            "artifacts": [],
                            "is_error": True,
                        },
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="sess_tools",
                    tool_name="publish",
                    timestamp=started_at + timedelta(seconds=5),
                    payload={
                        "approval": {
                            "approval_id": "approval-4",
                            "tool_call_id": "call-4",
                            "tool_name": "publish",
                            "arguments": {"environment": "production"},
                            "agent_name": "builder",
                            "tool_calls": [
                                {
                                    "tool_call_id": "call-4",
                                    "tool_name": "publish",
                                    "arguments": {"environment": "production"},
                                },
                                {
                                    "tool_call_id": "call-5",
                                    "tool_name": "delete_release",
                                    "arguments": {"release": "v1"},
                                    "policy_decision": "deny",
                                },
                            ],
                        }
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_BLOCKED,
                    session_id="sess_tools",
                    tool_name="delete_release",
                    timestamp=started_at + timedelta(seconds=6),
                    payload={
                        "approval_id": "approval-4",
                        "tool_call_id": "call-5",
                        "result": {
                            "content": "blocked",
                            "structured": None,
                            "artifacts": [],
                            "is_error": True,
                        },
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_FAILED,
                    session_id="sess_tools",
                    tool_name="cloud_login",
                    timestamp=started_at + timedelta(seconds=7),
                    payload={
                        "tool_call_id": "call-6",
                        "result": {
                            "content": "failed",
                            "structured": {
                                "returned": {"private_key": secret},
                                "truncated": "yes",
                            },
                            "artifacts": [],
                            "is_error": True,
                        },
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="sess_tools",
                    tool_name="read_file",
                    timestamp=started_at + timedelta(seconds=8),
                    payload={
                        "approval": {
                            "approval_id": "approval-7",
                            "tool_call_id": "call-7",
                            "tool_name": "read_file",
                            "arguments": {"path": "one"},
                            "agent_name": "builder",
                            "tool_calls": [
                                {
                                    "tool_call_id": "call-7",
                                    "tool_name": "read_file",
                                    "arguments": {"path": "one"},
                                },
                                {
                                    "tool_call_id": "call-8",
                                    "tool_name": "read_file",
                                    "arguments": {"path": "two"},
                                },
                            ],
                        }
                    },
                ),
                *[
                    Event(
                        type=event_type,
                        session_id="sess_tools",
                        tool_name="read_file",
                        timestamp=started_at + timedelta(seconds=offset),
                        payload={
                            "approval_id": "approval-7",
                            "tool_call_id": call_id,
                            **(
                                {"arguments": {"path": call_id}}
                                if event_type == EventType.TOOL_CALL_STARTED
                                else {
                                    "result": {
                                        "content": "done",
                                        "structured": None,
                                        "artifacts": [],
                                        "is_error": False,
                                    }
                                }
                            ),
                        },
                    )
                    for call_id, offset in (("call-7", 9), ("call-8", 10))
                    for event_type in (
                        EventType.TOOL_CALL_APPROVED,
                        EventType.TOOL_CALL_STARTED,
                        EventType.TOOL_CALL_COMPLETED,
                    )
                ],
                Event(
                    type=EventType.SESSION_AWAITING_USER_INPUT,
                    session_id="sess_tools",
                    tool_name="ask_user",
                    timestamp=started_at + timedelta(seconds=11),
                    payload={
                        "input_id": "input-9",
                        "tool_call_id": "call-9",
                        "question": "Continue?",
                        "options": ["yes", "no"],
                        "tool_calls": [
                            {
                                "tool_call_id": "call-9",
                                "tool_name": "ask_user",
                                "arguments": {"question": "Continue?"},
                            },
                            {
                                "tool_call_id": "call-10",
                                "tool_name": "read_file",
                                "arguments": {"path": "after-input"},
                            },
                        ],
                    },
                ),
            ]
            for event in events:
                await store.append_event("sess_tools", event)
        finally:
            await store.close()

    asyncio.run(seed())

    from cayu.cli import session as session_cli

    monkeypatch.setattr(session_cli, "_MAX_COLLECTED_EVENT_BYTES", 16 * 1024)

    assert (
        main(
            [
                "session",
                "tools",
                "sess_tools",
                "--sqlite",
                str(database),
                "--json",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert len(payload["calls"]) == 10
    assert secret not in output

    assert large_result[:1000] not in output
    by_id = {call["tool_call_id"]: call for call in payload["calls"]}
    assert by_id["call-1"]["status"] == "running"
    assert by_id["call-1"]["parallel_round_width"] == 2
    assert by_id["call-2"]["status"] == "success"
    assert by_id["call-2"]["parallel_round_width"] == 2
    assert by_id["call-2"]["duration_ms"] == 250
    assert by_id["call-2"]["rendered_content_bytes"] == len(large_result.encode("utf-8"))
    assert by_id["call-2"]["structured_result_bytes"] > 0
    assert by_id["call-2"]["artifact_bytes"] == len(
        json.dumps(
            [{"size_bytes": 10}, {"path": "report.txt"}],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert by_id["call-2"]["returned"] == 2
    assert by_id["call-2"]["truncated"] is True
    assert by_id["call-3"]["status"] == "blocked"
    assert by_id["call-3"]["approval_state"] == "approved"
    assert by_id["call-4"]["status"] == "approval_pending"
    assert by_id["call-4"]["approval_state"] == "requested"
    assert by_id["call-4"]["started_at"] is None
    assert by_id["call-5"]["status"] == "blocked"
    assert by_id["call-5"]["tool"] == "delete_release"
    assert by_id["call-6"]["status"] == "error"
    assert by_id["call-6"]["returned"] is None
    assert by_id["call-6"]["truncated"] is None
    assert by_id["call-7"]["tool_round_id"] == "approval:approval-7"
    assert by_id["call-7"]["parallel_round_width"] == 2
    assert by_id["call-8"]["tool_round_id"] == "approval:approval-7"
    assert by_id["call-8"]["parallel_round_width"] == 2
    assert by_id["call-9"]["status"] == "awaiting_input"
    assert by_id["call-9"]["tool_round_id"] == "input:input-9"
    assert by_id["call-9"]["parallel_round_width"] == 2
    assert by_id["call-10"]["status"] == "awaiting_input"

    assert (
        main(
            [
                "session",
                "tools",
                "sess_tools",
                "--sqlite",
                str(database),
                "--table",
            ]
        )
        == 0
    )
    table_header = capsys.readouterr().out.splitlines()[0]
    for field in (
        "argument_summary",
        "started_at",
        "completed_at",
        "returned",
        "truncated",
    ):
        assert field in table_header


def test_session_events_filters_paginates_and_bounds_explicit_payloads(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    secret = "event-payload-secret"

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_events",
                    environment_name="sandbox",
                    messages=[Message.text("user", "inspect events")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            for event in (
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_events",
                    agent_name="operator",
                    environment_name="sandbox",
                    tool_name="search",
                    payload={"tool_call_id": "call-1", "arguments": {"q": "cayu"}},
                ),
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="sess_events",
                    agent_name="operator",
                    environment_name="sandbox",
                    tool_name="search",
                    payload={
                        "tool_call_id": "call-1",
                        "api_key": secret,
                        "result": "z" * 500,
                    },
                ),
                Event(
                    type="custom.audit",
                    session_id="sess_events",
                    agent_name="operator",
                    environment_name="sandbox",
                    payload={"ok": True},
                ),
            ):
                await store.append_event("sess_events", event)
        finally:
            await store.close()

    asyncio.run(seed())

    from cayu.cli import session as session_cli

    monkeypatch.setattr(session_cli, "_MAX_COLLECTED_EVENT_BYTES", 64)

    assert (
        main(
            [
                "session",
                "events",
                "sess_events",
                "--sqlite",
                str(database),
                "--type",
                "tool.call.completed",
                "--tool",
                "search",
                "--include-payload",
                "64",
                "--jsonl",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    rows = [json.loads(line) for line in output.splitlines()]
    assert len(rows) == 1
    assert rows[0]["type"] == "tool.call.completed"
    assert rows[0]["payload_bytes"] > 500
    assert rows[0]["payload_truncated"] is True
    assert len(rows[0]["payload_preview"].encode("utf-8")) <= 64
    assert secret not in output

    assert (
        main(
            [
                "session",
                "events",
                "sess_events",
                "--sqlite",
                str(database),
                "--type",
                "tool.call.completed",
                "--include-payload",
                "64",
                "--table",
            ]
        )
        == 0
    )
    table_output = capsys.readouterr().out
    table_header = table_output.splitlines()[0]
    for field in ("agent", "environment", "payload_preview", "payload_truncated"):
        assert field in table_header
    assert "[REDACTED]" in table_output
    assert secret not in table_output

    assert (
        main(
            [
                "session",
                "events",
                "sess_events",
                "--sqlite",
                str(database),
                "--limit",
                "1",
                "--json",
            ]
        )
        == 0
    )
    page = json.loads(capsys.readouterr().out)
    assert page["has_more"] is True
    assert page["next_sequence"] == page["events"][0]["sequence"]
    assert "payload_preview" not in page["events"][0]


def test_session_transcript_sizes_find_large_records_without_dumping_content(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    oversized_result = "x" * 500_000
    secret = "transcript-api-key"

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_transcript",
                    messages=[Message.text("user", "inspect transcript")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_transcript_messages(
                "sess_transcript",
                [
                    Message.text("user", "inspect transcript"),
                    Message.tool_call(
                        tool_call_id="call-1",
                        tool_name="fetch",
                        arguments={"url": "https://example.test", "api_key": secret},
                    ),
                    Message.tool_result(
                        tool_call_id="call-1",
                        tool_name="fetch",
                        content=oversized_result,
                    ),
                ],
            )
        finally:
            await store.close()

    asyncio.run(seed())

    from cayu.cli import session as session_cli

    monkeypatch.setattr(session_cli, "_MAX_COLLECTED_EVENT_BYTES", 64)

    assert (
        main(
            [
                "session",
                "transcript",
                "sess_transcript",
                "--sqlite",
                str(database),
                "--offset",
                "1",
                "--limit",
                "2",
                "--sizes",
                "--json",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert len(output.encode("utf-8")) < 10_000
    assert secret not in output
    assert payload["total_messages"] == 3
    assert payload["has_more"] is False
    assert [item["index"] for item in payload["messages"]] == [1, 2]
    assert payload["messages"][0]["content_kinds"] == ["tool_call"]
    assert payload["messages"][0]["part_sizes"][0]["kind"] == "tool_call"
    assert payload["messages"][1]["message_bytes"] >= 500_000
    assert payload["messages"][1]["largest_part_bytes"] >= 500_000
    assert "content_json" not in payload["messages"][1]
    assert oversized_result[:1000] not in output

    assert (
        main(
            [
                "session",
                "transcript",
                "sess_transcript",
                "--sqlite",
                str(database),
                "--offset",
                "1",
                "--limit",
                "1",
                "--include-content",
                "64",
                "--jsonl",
            ]
        )
        == 0
    )
    included = json.loads(capsys.readouterr().out)
    assert included["content_truncated"] is True
    assert len(included["content_json"].encode("utf-8")) <= 64
    assert secret not in included["content_json"]

    assert (
        main(
            [
                "session",
                "transcript",
                "sess_transcript",
                "--sqlite",
                str(database),
                "--offset",
                "1",
                "--limit",
                "1",
                "--include-content",
                "64",
                "--table",
            ]
        )
        == 0
    )
    table_output = capsys.readouterr().out
    table_header = table_output.splitlines()[0]
    assert "content_json" in table_header
    assert "content_truncated" in table_header
    assert "[REDACTED]" in table_output
    assert secret not in table_output


def test_session_transcript_enforces_total_included_content_limit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    database = _write_project(tmp_path)

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_content_limit",
                    messages=[Message.text("user", "seed")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_transcript_messages(
                "sess_content_limit",
                [Message.text("user", "x" * 100) for _ in range(3)],
            )
        finally:
            await store.close()

    asyncio.run(seed())

    from cayu.cli import session as session_cli

    monkeypatch.setattr(session_cli, "_MAX_TRANSCRIPT_CONTENT_BYTES", 80)
    assert (
        main(
            [
                "session",
                "transcript",
                "sess_content_limit",
                "--sqlite",
                str(database),
                "--limit",
                "3",
                "--include-content",
                "64",
                "--json",
            ]
        )
        == 0
    )
    messages = json.loads(capsys.readouterr().out)["messages"]
    assert sum(len(row["content_json"].encode("utf-8")) for row in messages) <= 80
    assert messages[-1]["content_json"] == ""
    assert messages[-1]["content_truncated"] is True


def test_session_transcript_bounds_nested_part_metadata(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from cayu.cli import session as session_cli

    database = _write_project(tmp_path)

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_multipart",
                    messages=[Message.text("user", "inspect")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_transcript_messages(
                "sess_multipart",
                [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=tuple(TextPart(text=f"part-{index}") for index in range(5)),
                    )
                ],
            )
        finally:
            await store.close()

    asyncio.run(seed())
    monkeypatch.setattr(session_cli, "_MAX_TRANSCRIPT_SUMMARY_PARTS", 2)

    assert (
        main(
            [
                "session",
                "transcript",
                "sess_multipart",
                "--sqlite",
                str(database),
                "--sizes",
                "--json",
            ]
        )
        == 0
    )
    message = json.loads(capsys.readouterr().out)["messages"][0]
    assert message["content_part_count"] == 5
    assert message["content_parts_truncated"] is True
    assert message["content_kinds"] == ["text", "text"]
    assert len(message["part_sizes"]) == 2
    assert "part-4" not in message["preview"]


def test_session_content_views_strip_opaque_provider_state(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    anthropic_secret = "anthropic-redacted-thinking-data"
    openai_secret = "openai-encrypted-reasoning"
    aws_secret = "aws-secret-access-key-value"
    session_token = "aws-session-token-value"
    pem_secret = "-----BEGIN PRIVATE KEY-----\nprivate-key-material\n-----END PRIVATE KEY-----"
    bearer_secret = "bearer-secret-value"
    api_token = "sk-secret-token-value"
    assignment_secret = "assigned-secret-value"
    postgres_secret = "postgres-password-value"
    github_token_secret = "github-environment-token-value"

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_provider_state",
                    messages=[Message.text("user", "inspect provider state")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_transcript_messages(
                "sess_provider_state",
                [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=(
                            ThinkingPart(
                                text="bounded summary",
                                provider_state={
                                    "type": "redacted_thinking",
                                    "data": anthropic_secret,
                                    "signature": "anthropic-signature",
                                },
                            ),
                            ProviderStatePart(
                                provider="openai",
                                state={"blob": openai_secret},
                            ),
                        ),
                    )
                ],
            )
            await store.append_event(
                "sess_provider_state",
                Event(
                    type="custom.provider_state",
                    session_id="sess_provider_state",
                    payload={
                        "provider_state": [
                            {"type": "redacted_thinking", "data": anthropic_secret},
                            {"encrypted_content": openai_secret},
                        ],
                        "private_key": pem_secret,
                        "aws_secret_access_key": aws_secret,
                        "session_token": session_token,
                        "apikey": assignment_secret,
                        "secret_key": assignment_secret,
                        "signing_key": assignment_secret,
                        "id_token": assignment_secret,
                        "github_token": github_token_secret,
                        "message": (
                            f"Authorization: Bearer {bearer_secret}; "
                            f"token={api_token}; api_key={assignment_secret}; "
                            f"postgresql://operator:{postgres_secret}@db.example/cayu; "
                            f"pem={pem_secret}"
                        ),
                    },
                ),
            )
        finally:
            await store.close()

    asyncio.run(seed())

    assert (
        main(
            [
                "session",
                "transcript",
                "sess_provider_state",
                "--sqlite",
                str(database),
                "--include-content",
                "4096",
                "--json",
            ]
        )
        == 0
    )
    transcript_output = capsys.readouterr().out
    assert anthropic_secret not in transcript_output
    assert openai_secret not in transcript_output
    assert "anthropic-signature" not in transcript_output
    assert "[REDACTED]" in transcript_output

    assert (
        main(
            [
                "session",
                "events",
                "sess_provider_state",
                "--sqlite",
                str(database),
                "--type",
                "custom.provider_state",
                "--include-payload",
                "4096",
                "--json",
            ]
        )
        == 0
    )
    event_output = capsys.readouterr().out
    assert anthropic_secret not in event_output
    assert openai_secret not in event_output
    assert aws_secret not in event_output
    assert session_token not in event_output
    assert "private-key-material" not in event_output
    assert bearer_secret not in event_output
    assert api_token not in event_output
    assert assignment_secret not in event_output
    assert postgres_secret not in event_output
    assert github_token_secret not in event_output
    assert "[REDACTED]" in event_output


def test_session_default_views_redact_common_credential_field_names(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    credentials = {
        "credential": "credential-secret-value",
        "auth": "auth-secret-value",
        "aws_access_key_id": "AKIAEXAMPLESECRET",
        "accessToken": "access-token-secret-value",
        "refreshToken": "refresh-token-secret-value",
        "clientSecret": "client-secret-value",
        "privateKey": "private-key-secret-value",
        "awsAccessKeyId": "camel-aws-access-key-secret-value",
        "awsSecretAccessKey": "camel-aws-secret-access-key-value",
    }

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="operator",
                    session_id="sess_common_credentials",
                    messages=[Message.text("user", "inspect credentials")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_transcript_messages(
                "sess_common_credentials",
                [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=(
                            ToolCallPart(
                                tool_call_id="call-credential-preview",
                                tool_name="cloud_login",
                                arguments=credentials,
                            ),
                        ),
                    )
                ],
            )
            await store.append_event(
                "sess_common_credentials",
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_common_credentials",
                    tool_name="cloud_login",
                    payload={
                        "tool_call_id": "call-credential-summary",
                        "arguments": credentials,
                    },
                ),
            )
        finally:
            await store.close()

    asyncio.run(seed())

    for command in ("tools", "transcript"):
        assert (
            main(
                [
                    "session",
                    command,
                    "sess_common_credentials",
                    "--sqlite",
                    str(database),
                    "--json",
                ]
            )
            == 0
        )
        output = capsys.readouterr().out
        assert "[REDACTED]" in output
        for secret in credentials.values():
            assert secret not in output


def test_session_cli_rejects_malformed_filters_and_unsupported_schema(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)
    _seed_sessions(database)

    assert main(["session", "list", "--sqlite", ""]) == 1
    captured = capsys.readouterr()
    assert "--sqlite must be a non-empty path" in json.loads(captured.out)["error"]["message"]
    assert captured.err == ""

    assert (
        main(
            [
                "session",
                "list",
                "--sqlite",
                str(database),
                "--label",
                "not-an-assignment",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "KEY=VALUE" in json.loads(captured.out)["error"]["message"]
    assert captured.err == ""

    incompatible = tmp_path / "future.db"
    with sqlite3.connect(incompatible) as connection:
        connection.execute(
            "CREATE TABLE cayu_schema_migrations ("
            "revision INTEGER PRIMARY KEY, kind TEXT NOT NULL, "
            "compatible_from INTEGER NOT NULL, checksum TEXT, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO cayu_schema_migrations VALUES "
            "(999, 'breaking', 999, NULL, '2026-01-01T00:00:00+00:00')"
        )

    assert main(["session", "list", "--sqlite", str(incompatible)]) == 1
    captured = capsys.readouterr()
    assert (
        "requires an app that understands revision >= 999"
        in json.loads(captured.out)["error"]["message"]
    )
    assert captured.err == ""
    with sqlite3.connect(incompatible) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == {"cayu_schema_migrations"}


def test_session_event_and_transcript_pagination_stays_stable_at_scale(
    tmp_path: Path,
    capsys,
) -> None:
    database = _write_project(tmp_path)

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="scale",
                    session_id="sess_scale",
                    messages=[Message.text("user", "scale")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_transcript_messages(
                "sess_scale",
                [Message.text("user", f"message-{index}") for index in range(2001)],
            )
            await store.append_events(
                "sess_scale",
                [
                    Event(
                        type="custom.scale",
                        session_id="sess_scale",
                        payload={"index": index},
                    )
                    for index in range(2001)
                ],
            )
        finally:
            await store.close()

    asyncio.run(seed())

    assert (
        main(
            [
                "session",
                "events",
                "sess_scale",
                "--sqlite",
                str(database),
                "--after-sequence",
                "1999",
                "--limit",
                "2",
                "--json",
            ]
        )
        == 0
    )
    events = json.loads(capsys.readouterr().out)
    assert [item["sequence"] for item in events["events"]] == [2000, 2001]
    assert events["has_more"] is False

    assert (
        main(
            [
                "session",
                "transcript",
                "sess_scale",
                "--sqlite",
                str(database),
                "--offset",
                "1999",
                "--limit",
                "2",
                "--json",
            ]
        )
        == 0
    )
    transcript = json.loads(capsys.readouterr().out)
    assert transcript["total_messages"] == 2001
    assert [item["index"] for item in transcript["messages"]] == [1999, 2000]
    assert transcript["has_more"] is False
