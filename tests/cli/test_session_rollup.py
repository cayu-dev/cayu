"""`cayu session rollup` and `cayu session list --since/--until` over a local store."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.core.test_tool_completion import FinalTool

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.cli import main
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.releases import ReleaseIdentity
from cayu.sessions import InMemorySessionStore
from cayu.sessions._rollup import SessionRollupQuery, merge_groups, rollup_page
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.storage import SQLiteSessionStore
from cayu.storage import _sqlite_records as sqlite_records

_USAGE = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}


def _provider() -> ScriptedModelProvider:
    def respond(request):
        if any(
            part.type == "tool_result" for message in request.messages for part in message.content
        ):
            return [
                ModelStreamEvent.text_delta("Done"),
                ModelStreamEvent.completed({"finish_reason": "stop", "usage": _USAGE}),
            ]
        return [
            ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c1"),
            ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": _USAGE}),
        ]

    return ScriptedModelProvider(response_factory=respond, name="fixture")


async def _run(
    store, *, agent: str, session_id: str, labels: dict[str, str], fails: bool, release: str | None
) -> None:
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        release_identity=None if release is None else ReleaseIdentity(id=release),
    )
    app.register_provider(_provider(), default=True)
    app.register_agent(
        AgentSpec(name=agent, model="scripted-model"), tools=[FinalTool(fails=fails)]
    )
    async for _ in app.run(
        RunRequest(
            agent_name=agent,
            session_id=session_id,
            messages=[Message.text("user", "Help")],
            labels=labels,
            max_steps=4,
        )
    ):
        pass


_SESSIONS = (
    # id, agent, labels, failing tool, release, created_at
    ("s1", "hunter", {"repo": "a"}, False, "r1", datetime(2026, 10, 26, 9, tzinfo=UTC)),
    ("s2", "hunter", {"repo": "b"}, True, "r1", datetime(2026, 10, 27, 9, tzinfo=UTC)),
    ("s3", "verifier", {"repo": "a"}, True, "r2", datetime(2026, 10, 27, 18, tzinfo=UTC)),
)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            for session_id, agent, labels, fails, release, _ in _SESSIONS:
                await _run(
                    store,
                    agent=agent,
                    session_id=session_id,
                    labels=labels,
                    fails=fails,
                    release=release,
                )
        finally:
            await store.close()

    asyncio.run(seed())
    with sqlite3.connect(path) as connection:
        for session_id, *_, created_at in _SESSIONS:
            connection.execute(
                "UPDATE cayu_sessions SET created_at = ? WHERE id = ?",
                (sqlite_records.format_datetime(created_at), session_id),
            )
    return path


def _rollup(database: Path, capsys, *argv: str) -> dict:
    assert main(["session", "rollup", "--sqlite", str(database), "--json", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_rollup_sums_usage_time_and_failures_by_agent(database: Path, capsys) -> None:
    payload = _rollup(database, capsys)

    assert payload["sessions"] == 3 and payload["complete"] is True
    hunter, verifier = payload["groups"]
    assert (hunter["key"], hunter["sessions"], verifier["key"]) == ("hunter", 2, "verifier")
    assert hunter["statuses"] == {"completed": 2}
    assert (hunter["model_calls"], hunter["model_calls_with_usage"], hunter["tool_calls"]) == (
        4,
        4,
        2,
    )
    assert hunter["usage"]["input_tokens"] == "32" and hunter["usage"]["output_tokens"] == "8"
    assert hunter["tool_failures"] == 1
    assert hunter["failure_kinds"] == [{"tool": "ask_customer", "kind": "tool_error", "count": 1}]
    durations = hunter["durations"]
    assert durations["model_ms"] >= 0 and durations["tool_ms"] >= 0
    assert durations["active_ms"] >= durations["gap_ms"] >= 0
    assert durations["active_ms"] > 0


@pytest.mark.parametrize(
    ("group_by", "keys"),
    [
        ("day", ["2026-10-26", "2026-10-27"]),
        ("release", ["r1", "r2"]),
        ("label:repo", ["a", "b"]),
        ("label:missing", ["(none)"]),
    ],
)
def test_rollup_groups_by_day_release_and_label(database: Path, capsys, group_by, keys) -> None:
    payload = _rollup(database, capsys, "--by", group_by)

    assert [group["key"] for group in payload["groups"]] == keys
    assert sum(group["sessions"] for group in payload["groups"]) == 3


def test_rollup_window_filters_and_pages_merge_to_the_same_answer(database: Path, capsys) -> None:
    payload = _rollup(
        database, capsys, "--since", "2026-10-27", "--until", "2026-10-27T12:00:00+00:00"
    )
    assert payload["sessions"] == 1 and [g["key"] for g in payload["groups"]] == ["hunter"]

    stopped = _rollup(database, capsys, "--max-sessions", "2")
    assert stopped["sessions"] == 2 and stopped["complete"] is False
    rest = _rollup(database, capsys, "--cursor", stopped["next_cursor"])
    assert rest["sessions"] == 1 and rest["complete"] is True

    async def paged() -> tuple[list, list]:
        store = SQLiteSessionStore(database)
        try:
            whole = await rollup_page(store, SessionRollupQuery(group_by="day", max_sessions=10))
            pages, cursor = [], None
            while True:
                page = await rollup_page(
                    store, SessionRollupQuery(group_by="day", max_sessions=1, cursor=cursor)
                )
                pages.extend(page.groups)
                cursor = page.next_cursor
                if cursor is None:
                    return whole.groups, merge_groups(pages)
        finally:
            await store.close()

    whole, merged = asyncio.run(paged())
    assert merged == whole


def test_rollup_table_shows_groups_and_failure_kinds(database: Path, capsys) -> None:
    assert main(["session", "rollup", "--sqlite", str(database), "--table", "--by", "release"]) == 0
    output = capsys.readouterr().out

    assert output.splitlines()[0].split()[:5] == [
        "group",
        "sessions",
        "model_calls",
        "tool_calls",
        "tool_failures",
    ]
    assert "Tool failures by kind" in output and "tool_error" in output
    assert output.rstrip().endswith("3 sessions (complete).")


def test_rollup_rejects_unknown_grouping(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["session", "rollup", "--by", "model"])
    assert "group_by must be agent, day, release or label:KEY" in capsys.readouterr().err


def test_session_list_since_and_until_select_by_creation_time(database: Path, capsys) -> None:
    assert (
        main(
            [
                "session",
                "list",
                "--sqlite",
                str(database),
                "--since",
                "2026-10-27",
                "--until",
                "2026-10-27T12:00:00+00:00",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in payload["sessions"]] == ["s2"]
    assert payload["total_count"] == 1


def test_memory_store_matches_the_creation_window() -> None:
    async def scenario() -> list[str]:
        store = InMemorySessionStore()
        for session_id, agent, labels, fails, release, _ in _SESSIONS:
            await _run(
                store,
                agent=agent,
                session_id=session_id,
                labels=labels,
                fails=fails,
                release=release,
            )
        sessions = (
            await store.list_sessions(SessionQuery(order_by=SessionOrder.CREATED_AT_ASC))
        ).sessions
        middle = sessions[1].created_at
        result = await store.list_sessions(
            SessionQuery(created_since=middle, created_before=sessions[2].created_at)
        )
        return [session.id for session in result.sessions]

    assert asyncio.run(scenario()) == ["s2"]
    with pytest.raises(ValueError, match="created_since must be before created_before"):
        SessionQuery(
            created_since=datetime(2026, 1, 2, tzinfo=UTC),
            created_before=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_session_facts_pair_intervals_and_union_parallel_calls(tmp_path: Path) -> None:
    from datetime import timedelta

    from cayu.events import Event, EventType
    from cayu.sessions._rollup import session_facts
    from cayu.sessions.records import SessionIdentity

    t0 = datetime(2026, 10, 28, 12, tzinfo=UTC)

    def at(seconds: float) -> datetime:
        return t0 + timedelta(seconds=seconds)

    round_identity = {"model_step_id": "mstep_" + "1" * 32, "model_attempt_id": "matt_" + "1" * 32}
    events = [
        (EventType.MODEL_STARTED, 0, None, {**round_identity}),
        (
            EventType.MODEL_COMPLETED,
            2,
            None,
            {**round_identity, "usage_metrics": {"input_tokens": 5}},
        ),
        # Two parallel calls in one round, 3-6 s and 4-8 s: 5 s of tool time, not 7.
        (EventType.TOOL_CALL_STARTED, 3, "search", {"tool_round_id": "r", "tool_call_id": "a"}),
        (EventType.TOOL_CALL_STARTED, 4, "fetch", {"tool_round_id": "r", "tool_call_id": "b"}),
        (EventType.TOOL_CALL_COMPLETED, 6, "search", {"tool_round_id": "r", "tool_call_id": "a"}),
        (
            EventType.TOOL_CALL_FAILED,
            8,
            "fetch",
            {
                "tool_round_id": "r",
                "tool_call_id": "b",
                "failure_evidence": {
                    "classification": "timeout",
                    "exception_types": ["TimeoutError"],
                },
            },
        ),
        (EventType.ENVIRONMENT_MATERIALIZATION_COMPLETED, 10, None, {"elapsed_ms": 1500}),
        (EventType.MODEL_ERROR, 11, None, {"model_attempt_id": "matt_2"}),
    ]

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "facts.db")
        try:
            await store.create(
                RunRequest(
                    agent_name="a", session_id="facts", messages=[Message.text("user", "x")]
                ),
                identity=SessionIdentity(provider_name="fake", model="m"),
            )
            for event_type, seconds, tool, payload in events:
                await store.append_event(
                    "facts",
                    Event(
                        type=event_type,
                        session_id="facts",
                        timestamp=at(seconds),
                        tool_name=tool,
                        payload=payload,
                    ),
                )
            return await session_facts(store, "facts")
        finally:
            await store.close()

    facts = asyncio.run(scenario())

    assert (facts.model_calls, facts.model_calls_with_usage, facts.model_errors) == (1, 1, 1)
    assert (facts.model_ms, facts.tool_ms, facts.setup_ms) == (2000, 5000, 1500)
    assert facts.tool_calls == 2
    assert dict(facts.tool_failures) == {("fetch", "TimeoutError"): 1}
    assert facts.usage.input_tokens == 5


def test_rollup_jsonl_ends_with_the_pagination_summary(database: Path, capsys) -> None:
    assert (
        main(["session", "rollup", "--sqlite", str(database), "--jsonl", "--max-sessions", "1"])
        == 0
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert [record["record_type"] for record in records] == ["group", "summary"]
    assert records[-1]["complete"] is False and records[-1]["next_cursor"]


def test_failure_kind_overflow_survives_repeated_merges() -> None:
    from cayu.sessions._rollup import RollupFailureKind, RollupGroup, top_failure_kinds

    def group(start: int, stop: int) -> RollupGroup:
        return RollupGroup(
            key="agent",
            tool_failures=stop - start,
            failure_kinds=[
                RollupFailureKind(tool=f"t{index:04}", kind="error", count=1)
                for index in range(start, stop)
            ],
        )

    merged = (
        group(0, 250).merge(group(250, 500)).merge(group(500, 750)).merge(RollupGroup(key="agent"))
    )

    assert sum(kind.count for kind in merged.failure_kinds) == merged.tool_failures == 750
    shown = top_failure_kinds(merged)
    assert len(shown) == 201 and shown[-1].kind == "other"
    assert sum(kind.count for kind in shown) == 750


def test_failure_kinds_merge_the_same_way_whatever_the_page_size() -> None:
    from cayu.sessions._rollup import RollupFailureKind, RollupGroup, top_failure_kinds

    def group(rows: list[tuple[str, int]]) -> RollupGroup:
        return RollupGroup(
            key="agent",
            failure_kinds=[RollupFailureKind(tool=t, kind="error", count=n) for t, n in rows],
        )

    many = group([(f"a{index:03}", 2) for index in range(200)])
    one, two = group([("z", 1)]), group([("z", 2)])

    left = many.merge(one).merge(two)
    right = many.merge(one.merge(two))

    assert left == right
    assert ("z", 3) in [(kind.tool, kind.count) for kind in top_failure_kinds(left)]


def test_session_facts_count_auxiliary_and_hosted_tool_usage(tmp_path: Path) -> None:
    from cayu.events import Event, EventType
    from cayu.sessions._rollup import session_facts
    from cayu.sessions.records import SessionIdentity

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "aux.db")
        try:
            await store.create(
                RunRequest(agent_name="a", session_id="aux", messages=[Message.text("user", "x")]),
                identity=SessionIdentity(provider_name="fake", model="m"),
            )
            for event_type, payload in (
                (EventType.MODEL_COMPLETED, {"usage_metrics": {"input_tokens": 5}}),
                (EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED, {"usage_metrics": {"input_tokens": 7}}),
            ):
                await store.append_event(
                    "aux", Event(type=event_type, session_id="aux", payload=payload)
                )
            return await session_facts(store, "aux")
        finally:
            await store.close()

    facts = asyncio.run(scenario())

    assert facts.model_calls == 1 and facts.usage.input_tokens == 12


def test_gap_leaves_out_time_spent_waiting_for_approval(tmp_path: Path) -> None:
    from datetime import timedelta

    from cayu.events import Event, EventType
    from cayu.sessions._rollup import session_facts
    from cayu.sessions.records import SessionIdentity

    t0 = datetime(2026, 10, 28, 12, tzinfo=UTC)

    def at(seconds: float) -> datetime:
        return t0 + timedelta(seconds=seconds)

    first = {"model_step_id": "mstep_" + "1" * 32, "model_attempt_id": "matt_" + "1" * 32}
    second = {"model_step_id": "mstep_" + "2" * 32, "model_attempt_id": "matt_" + "2" * 32}
    call = {"tool_round_id": "r", "tool_call_id": "a"}
    hour = 3600
    events = [
        (EventType.MODEL_STARTED, 0, first),
        (EventType.MODEL_COMPLETED, 2, first),
        # The call starts, waits an hour for approval, then runs for four seconds.
        (EventType.TOOL_CALL_STARTED, 3, call),
        (EventType.TOOL_CALL_APPROVAL_REQUESTED, 4, {**call, "approval_id": "ap1"}),
        (EventType.TOOL_CALL_APPROVED, 4 + hour, {**call, "approval_id": "ap1"}),
        (EventType.TOOL_CALL_COMPLETED, 8 + hour, call),
        (EventType.MODEL_STARTED, 9 + hour, second),
        (EventType.MODEL_COMPLETED, 12 + hour, second),
        # Ten seconds of model and tool work and three seconds of runtime gaps.
        (EventType.INTERACTION_COMPLETED, 13 + hour, {"active_duration_ms": 13_000}),
    ]

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "wait.db")
        try:
            await store.create(
                RunRequest(agent_name="a", session_id="wait", messages=[Message.text("user", "x")]),
                identity=SessionIdentity(provider_name="fake", model="m"),
            )
            for event_type, seconds, payload in events:
                await store.append_event(
                    "wait",
                    Event(
                        type=event_type, session_id="wait", timestamp=at(seconds), payload=payload
                    ),
                )
            return await session_facts(store, "wait")
        finally:
            await store.close()

    facts = asyncio.run(scenario())

    assert (facts.model_ms, facts.tool_ms) == (5_000, 5_000)
    assert facts.active_ms == 13_000 and facts.gap_ms == 3_000


def test_rollup_pages_from_a_newer_agent_ignore_fields_this_cli_does_not_know() -> None:
    from cayu.sessions._rollup import RollupPage

    page = RollupPage.model_validate(
        {
            "groups": [{"key": "agent", "sessions": 1, "cost": {"USD": "1"}, "later_field": 3}],
            "sessions_read": 1,
            "next_cursor": None,
            "later_page_field": True,
        }
    )

    assert page.groups[0].sessions == 1


@pytest.mark.parametrize("resolution", ["blocked", "resumed", "unknown"])
def test_approval_waits_close_however_the_approval_resolves(
    tmp_path: Path, resolution: str
) -> None:
    from datetime import timedelta

    from cayu.events import Event, EventType
    from cayu.sessions._rollup import session_facts
    from cayu.sessions.records import SessionIdentity

    t0 = datetime(2026, 10, 28, 12, tzinfo=UTC)

    def at(seconds: float) -> datetime:
        return t0 + timedelta(seconds=seconds)

    attempt = {"model_step_id": "mstep_" + "3" * 32, "model_attempt_id": "matt_" + "3" * 32}
    call = {"tool_round_id": "r", "tool_call_id": "a", "approval_id": "ap1"}
    resolved = {
        "blocked": [
            (
                EventType.TOOL_CALL_BLOCKED,
                100,
                {
                    **call,
                    "blocked_by": "policy_evaluation_ambiguous",
                    "requested_decision": "approve",
                },
            )
        ],
        "resumed": [(EventType.SESSION_RESUMED, 100, {"approval_id": "ap1"})],
        # No event names the approval's resolution; the model running again ends the wait.
        "unknown": [],
    }[resolution]
    events = [
        (EventType.TOOL_CALL_APPROVAL_REQUESTED, 0, call),
        *resolved,
        (EventType.MODEL_STARTED, 101, attempt),
        (EventType.MODEL_COMPLETED, 105, attempt),
    ]

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "resolve.db")
        try:
            await store.create(
                RunRequest(agent_name="a", session_id="r", messages=[Message.text("user", "x")]),
                identity=SessionIdentity(provider_name="fake", model="m"),
            )
            for event_type, seconds, payload in events:
                await store.append_event(
                    "r",
                    Event(type=event_type, session_id="r", timestamp=at(seconds), payload=payload),
                )
            return await session_facts(store, "r")
        finally:
            await store.close()

    # The model call after the approval still counts.
    assert asyncio.run(scenario()).model_ms == 4_000


def test_rollup_marks_every_call_unpriced_when_the_agent_cant_price(capsys) -> None:
    import argparse
    import asyncio

    from cayu.cli import session as session_cli
    from cayu.sessions._rollup import RollupGroup, RollupPage

    class UnpricedAgent:
        rollup_pricing_unsupported = True

        async def rollup_page(self, query):
            return RollupPage(
                groups=[
                    RollupGroup(key="a", sessions=1, model_calls=5),
                    # Billed hosted usage with no completed model call.
                    RollupGroup(key="hosted-only", sessions=1, model_errors=1),
                ],
                sessions_read=1,
                next_cursor=None,
            )

    def args(output_format: str) -> argparse.Namespace:
        return argparse.Namespace(
            since=None,
            until=None,
            group_by="agent",
            agent=None,
            environment=None,
            status=None,
            label=[],
            cursor=None,
            max_sessions=100,
            price_book=None,
            output_format=output_format,
            failure_kinds=5,
        )

    assert asyncio.run(session_cli._session_rollup(args("json"), UnpricedAgent())) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["price_book"] is None
    assert payload["groups"][0]["unpriced_model_calls"] is None
    assert payload["groups"][0]["estimated_cost"] is None

    assert asyncio.run(session_cli._session_rollup(args("table"), UnpricedAgent())) == 0
    captured = capsys.readouterr()
    rows = {line.split()[0]: line for line in captured.out.splitlines() if line.split()}
    assert "unknown" in rows["a"] and "unknown" in rows["hosted-only"]
    assert payload["groups"][1]["estimated_cost"] is None
    assert "costs are left out" in captured.err
