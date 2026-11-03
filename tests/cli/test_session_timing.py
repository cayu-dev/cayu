"""`cayu session show`'s time breakdown, `session usage` call timing and rollup teardown."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from cayu.cli import main
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions._rollup import session_facts, session_timing
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage import SQLiteSessionStore

T0 = datetime(2026, 11, 2, 21, 0, tzinfo=UTC)
_ATTEMPT = {"model_step_id": "mstep_" + "1" * 32, "model_attempt_id": "matt_" + "1" * 32}
_SECOND_ATTEMPT = {"model_step_id": "mstep_" + "2" * 32, "model_attempt_id": "matt_" + "2" * 32}
_BINDING = {"binding_generation_id": "wbind_" + "a" * 32}


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _phase(phase: str, started: float, seconds: float, **extra: Any) -> dict[str, Any]:
    return {
        "phase": phase,
        "started_at": _at(started).isoformat(timespec="milliseconds"),
        "duration_ms": round(seconds * 1000),
        "outcome": "completed",
        "depth": 0,
        **extra,
    }


# One turn from 0 s to 100 s, then teardown until 130 s:
# setup 0-10, model 12-20 and 50-60, tools 25-40, a wait for approval 41-45, teardown 101-130.
_EVENTS: list[tuple[EventType, float, str | None, dict[str, Any]]] = [
    (EventType.ENVIRONMENT_BINDING_STARTED, 0, None, dict(_BINDING)),
    (
        EventType.ENVIRONMENT_BINDING_COMPLETED,
        10,
        None,
        {
            **_BINDING,
            "phase_timings": [
                _phase("transfer", 1, 8, items=3, bytes=4096),
                {**_phase("archive_preparation", 1, 2), "depth": 1},
                _phase("git_baseline", 9, 1),
            ],
        },
    ),
    (EventType.SESSION_STARTED, 11, None, {}),
    (EventType.MODEL_STARTED, 12, None, dict(_ATTEMPT)),
    (EventType.MODEL_TEXT_DELTA, 15, None, {**_ATTEMPT, "delta": "Hi"}),
    (EventType.MODEL_COMPLETED, 20, None, {**_ATTEMPT, "usage_metrics": {"input_tokens": 5}}),
    (EventType.TOOL_CALL_STARTED, 25, "shell", {"tool_round_id": "r", "tool_call_id": "a"}),
    (EventType.TOOL_CALL_COMPLETED, 40, "shell", {"tool_round_id": "r", "tool_call_id": "a"}),
    (EventType.TOOL_CALL_APPROVAL_REQUESTED, 41, "shell", {"approval_id": "ap"}),
    (EventType.TOOL_CALL_APPROVED, 45, "shell", {"approval_id": "ap"}),
    (EventType.MODEL_STARTED, 50, None, dict(_SECOND_ATTEMPT)),
    (EventType.MODEL_COMPLETED, 60, None, {**_SECOND_ATTEMPT, "usage_metrics": {}}),
    (
        EventType.INTERACTION_COMPLETED,
        100,
        None,
        {"active_duration_ms": 96_000, "wall_duration_ms": 100_000},
    ),
    (EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED, 101, None, dict(_BINDING)),
    (
        EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
        130,
        None,
        {**_BINDING, "phase_timings": [_phase("copy_back_publication", 102, 20, items=2)]},
    ),
]


async def _write(path: Path, events=_EVENTS, session_id: str = "timed") -> None:
    store = SQLiteSessionStore(path)
    try:
        await store.create(
            RunRequest(agent_name="a", session_id=session_id, messages=[Message.text("user", "x")]),
            identity=SessionIdentity(provider_name="fake", model="m"),
        )
        for event_type, seconds, tool, payload in events:
            await store.append_event(
                session_id,
                Event(
                    type=event_type,
                    session_id=session_id,
                    timestamp=_at(seconds),
                    tool_name=tool,
                    environment_name=(
                        "sandbox" if str(event_type).startswith("environment.") else None
                    ),
                    payload=payload,
                ),
            )
    finally:
        await store.close()


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "timing.db"
    asyncio.run(_write(path))
    return path


async def _read(path: Path, read):
    store = SQLiteSessionStore(path)
    try:
        return await read(store)
    finally:
        await store.close()


def test_the_parts_add_up_to_the_elapsed_time(database: Path) -> None:
    # Hand-written events: a binding, two model calls, a tool call, a wait and teardown.
    timing = asyncio.run(_read(database, lambda store: session_timing(store, "timed")))

    assert timing.elapsed_ms == 130_000
    assert (timing.setup_ms, timing.model_ms, timing.tool_ms) == (10_000, 18_000, 15_000)
    assert (timing.waiting_ms, timing.teardown_ms) == (4_000, 29_000)
    # The turn's 100 s less the 47 s of setup, model, tools and waiting.
    assert timing.other_ms == 53_000
    # 100-101 s, after the turn and before teardown.
    assert timing.idle_ms == 1_000
    assert (
        timing.setup_ms
        + timing.model_ms
        + timing.tool_ms
        + timing.teardown_ms
        + timing.waiting_ms
        + timing.other_ms
        + timing.idle_ms
        == timing.elapsed_ms
    )
    assert (timing.model_calls, timing.tool_calls, timing.first_model_call_ms) == (2, 1, 12_000)
    binding, finalize = timing.environment
    assert (binding.kind, binding.environment, binding.duration_ms) == (
        "binding",
        "sandbox",
        10_000,
    )
    assert [(phase.phase, phase.depth) for phase in binding.phases] == [
        ("transfer", 0),
        ("archive_preparation", 1),
        ("git_baseline", 0),
    ]
    assert (finalize.kind, finalize.duration_ms, finalize.outcome) == (
        "finalize",
        29_000,
        "completed",
    )
    assert finalize.phases[0].items == 2


def _timing_of(tmp_path: Path, events) -> Any:
    path = tmp_path / "live.db"
    asyncio.run(_write(path, events, session_id="live"))
    return asyncio.run(_read(path, lambda store: session_timing(store, "live")))


def test_a_running_session_counts_unfinished_work_and_its_open_turn(tmp_path: Path) -> None:
    # A turn still running: tool a runs 5-25 s with no end yet, tool b 15-25 s, and a model
    # call started at 30 s hasn't completed when the last event is read at 40 s.
    call = {"tool_round_id": "r"}
    timing = _timing_of(
        tmp_path,
        [
            (EventType.INTERACTION_STARTED, 0, None, {}),
            (EventType.TOOL_CALL_STARTED, 5, "shell", {**call, "tool_call_id": "a"}),
            (EventType.TOOL_CALL_STARTED, 15, "shell", {**call, "tool_call_id": "b"}),
            (EventType.TOOL_CALL_COMPLETED, 25, "shell", {**call, "tool_call_id": "b"}),
            (EventType.MODEL_STARTED, 30, None, dict(_ATTEMPT)),
            (EventType.MODEL_TEXT_DELTA, 40, None, {**_ATTEMPT, "delta": "still going"}),
        ],
    )

    assert timing.elapsed_ms == 40_000
    # Tool a counts from 5 s to the last event, but from 30 s the model call comes first.
    assert (timing.tool_ms, timing.model_ms) == (25_000, 10_000)
    # 0-5 s is the open turn's own runtime work, not idle time.
    assert (timing.other_ms, timing.idle_ms) == (5_000, 0)


def test_elapsed_time_runs_to_the_last_event_of_any_type(tmp_path: Path) -> None:
    deltas = [
        (EventType.MODEL_TEXT_DELTA, second, None, {**_ATTEMPT, "delta": "x"})
        for second in range(2, 61)
    ]
    timing = _timing_of(
        tmp_path,
        [
            (EventType.SESSION_STARTED, 0, None, {}),
            (EventType.MODEL_STARTED, 1, None, dict(_ATTEMPT)),
            *deltas,
        ],
    )

    assert (timing.elapsed_ms, timing.model_ms, timing.idle_ms) == (60_000, 59_000, 1_000)


def test_rollup_counts_teardown_apart_from_turn_time(database: Path) -> None:
    facts = asyncio.run(_read(database, lambda store: session_facts(store, "timed")))

    assert facts.teardown_ms == 29_000
    assert facts.setup_ms == 10_000


def _run_sync_session(tmp_path: Path) -> Path:
    from tests.core._workload_secret_support import FakeProvider

    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.environments.base import Environment, EnvironmentSpec
    from cayu.environments.bindings import SyncBinding
    from cayu.providers.base import ModelStreamEvent
    from cayu.workspaces.local import LocalWorkspace

    source, target, database = tmp_path / "source", tmp_path / "target", tmp_path / "run.db"
    source.mkdir()
    target.mkdir()
    (source / "input.txt").write_text("copied in")

    async def run() -> None:
        store = SQLiteSessionStore(database)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                FakeProvider(
                    [
                        ModelStreamEvent.text_delta("Done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ),
                default=True,
            )
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sandbox"),
                    workspace=LocalWorkspace(source, workspace_id="source"),
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(target, workspace_id="target"),
                        sync_back="always",
                    ),
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="run",
                    messages=[Message.text("user", "go")],
                )
            ):
                pass
        finally:
            await store.close()

    asyncio.run(run())
    return database


def test_session_show_reports_the_breakdown_as_json_and_text(tmp_path: Path, capsys) -> None:
    database = _run_sync_session(tmp_path)

    assert main(["session", "show", "run", "--sqlite", str(database), "--json"]) == 0
    timing = json.loads(capsys.readouterr().out)["timing"]
    parts = ("setup", "model", "tool", "teardown", "waiting", "other", "idle")
    assert sum(timing[f"{part}_ms"] for part in parts) == timing["elapsed_ms"]
    assert timing["model_calls"] == 1
    assert [operation["kind"] for operation in timing["environment"]] == ["binding", "finalize"]
    binding, finalize = timing["environment"]
    assert "transfer" in {phase["phase"] for phase in binding["phases"]}
    assert "copy_back_publication" in {phase["phase"] for phase in finalize["phases"]}
    assert all(operation["outcome"] == "completed" for operation in timing["environment"])

    assert main(["session", "show", "run", "--sqlite", str(database), "--table"]) == 0
    text = capsys.readouterr().out
    assert "elapsed)" in text
    assert "binding (sandbox)" in text and "finalize (sandbox)" in text
    assert "  transfer" in text

    assert main(["session", "usage", "run", "--sqlite", str(database), "--json"]) == 0
    (call,) = json.loads(capsys.readouterr().out)["calls"]
    assert type(call["duration_ms"]) is int and type(call["first_output_ms"]) is int
    assert call["first_output_ms"] <= call["duration_ms"]
    assert not any(key.startswith("_cayu") for key in call)


def test_operations_without_step_timings_still_render(capsys) -> None:
    from cayu.cli.session import _print_timing
    from cayu.sessions._rollup import SessionTiming, SessionTimingOperation

    timing = SessionTiming(
        elapsed_ms=60_000,
        setup_ms=10_000,
        other_ms=50_000,
        environment=[
            SessionTimingOperation(kind="binding", duration_ms=10_000, outcome="completed")
        ],
    )
    _print_timing(timing.model_dump(mode="json"))
    text = capsys.readouterr().out
    assert "Time (1m00s elapsed)" in text and "binding (-)" in text

    _print_timing(None)
    assert "doesn't report session timing" in capsys.readouterr().out


def test_session_usage_reports_each_calls_start_duration_and_first_output(
    database: Path, capsys
) -> None:
    assert main(["session", "usage", "timed", "--sqlite", str(database), "--json"]) == 0
    calls = json.loads(capsys.readouterr().out)["calls"]

    assert [
        (call["started_at"], call["duration_ms"], call["first_output_ms"]) for call in calls
    ] == [
        (_at(12).isoformat(), 8_000, 3_000),
        # It streamed no text, so its first output is unknown.
        (_at(50).isoformat(), 10_000, None),
    ]
    assert not any(key.startswith("_cayu") for call in calls for key in call)

    assert main(["session", "usage", "timed", "--sqlite", str(database), "--table"]) == 0
    text = capsys.readouterr().out
    assert "first_output" in text and "3.0s" in text


def test_rollup_table_shows_teardown(database: Path, capsys) -> None:
    assert main(["session", "rollup", "--sqlite", str(database), "--json"]) == 0
    (group,) = json.loads(capsys.readouterr().out)["groups"]
    assert group["durations"]["teardown_ms"] == 29_000

    assert main(["session", "rollup", "--sqlite", str(database), "--table"]) == 0
    assert "teardown" in capsys.readouterr().out
