"""`cayu session workflow` and `cayu session rollup --causal-budget/--root` over a local store."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.cli import main
from cayu.cli._workflow_run import _steps_from_journal
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.providers import ModelStreamEvent
from cayu.runners import LocalRunner
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import EventRecord
from cayu.storage import SQLiteSessionStore
from cayu.workflows import WorkflowBase, WorkflowSpec, command, gated_loop, step

_USAGE = {
    "input_tokens": 10,
    "output_tokens": 4,
    "total_tokens": 14,
    "output_tokens_details": {"reasoning_tokens": 3},
}
_RUN_ID = "team-run"


def _reply(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop", "usage": _USAGE}),
    ]


class TeamWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="team")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="planner", step_id="plan", prompt="plan")
        yield await ctx.emit_custom_event("custom.gate", payload={"gate": "plan", "passed": True})
        async for event in gated_loop(
            ctx,
            ["w1"],
            do=lambda item: step(ctx, agent="worker", step_id=f"worker-{item}", prompt="work"),
            gate=_always_pass,
            name="workers",
        ):
            yield event
        yield await ctx.emit_custom_event("custom.verdict", payload={"accepted": True})
        yield await ctx.completed({"done": True})


async def _always_pass(item, result) -> bool:
    return True


def _seed(path: Path) -> dict[str, str]:
    """Run the workflow, then a grandchild under its plan step, and an unrelated session."""

    async def seed() -> dict[str, str]:
        store = SQLiteSessionStore(path)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ScriptedModelProvider(response_factory=lambda request: _reply("ok")),
                default=True,
            )
            for name in ("planner", "worker", "helper"):
                app.register_agent(AgentSpec(name=name, model="scripted-model"))
            async for _ in TeamWorkflow(app).run(_RUN_ID):
                pass
            children = (await store.list_sessions(SessionQuery(parent_session_id=_RUN_ID))).sessions
            by_agent = {child.agent_name: child for child in children}
            plan = by_agent["planner"]
            async for _ in app.run(
                RunRequest(
                    agent_name="helper",
                    session_id="grandchild",
                    parent_session_id=plan.id,
                    causal_budget_id=plan.causal_budget_id,
                    messages=[Message.text("user", "help")],
                )
            ):
                pass
            async for _ in app.run(
                RunRequest(
                    agent_name="helper",
                    session_id="elsewhere",
                    messages=[Message.text("user", "unrelated")],
                )
            ):
                pass
            return {
                "plan": plan.id,
                "worker": by_agent["worker"].id,
                "budget": plan.causal_budget_id,
            }
        finally:
            await store.close()

    return asyncio.run(seed())


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    path = tmp_path / "cayu.db"
    return path, _seed(path)


def _json(capsys, *argv: str) -> dict:
    assert main(["session", *argv, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_workflow_lists_steps_children_usage_and_custom_events(run, capsys) -> None:
    path, ids = run
    payload = _json(capsys, "workflow", _RUN_ID, "--sqlite", str(path))

    workflow = payload["workflow"]
    assert (workflow["session_id"], workflow["name"]) == (_RUN_ID, "team")
    assert (workflow["status"], workflow["session_status"]) == ("completed", "completed")
    assert workflow["started_at"] and workflow["completed_at"]
    steps = {(row["kind"], row["step_id"]): row for row in payload["steps"]}
    plan = steps[("step", "plan")]
    assert plan["child_session_id"] == ids["plan"]
    assert (plan["agent"], plan["status"], plan["journal"]) == ("planner", "completed", "completed")
    assert plan["started_at"] <= plan["ended_at"]
    # The plan step's numbers include the helper session it spawned.
    assert (plan["sessions"], plan["model_calls"]) == (2, 2)
    assert plan["usage"]["input_tokens"] == "20"
    assert plan["usage"]["reasoning_tokens"] == "6"
    worker = steps[("step", "worker-w1")]
    assert (worker["child_session_id"], worker["model_calls"]) == (ids["worker"], 1)
    item = next(row for row in payload["steps"] if row["kind"] == "gated_loop")
    assert (item["child_session_id"], item["outcome"]) == (ids["worker"], "pass")
    assert (item["loop_name"], item["item_key"]) == ("workers", "0")
    assert (item["sessions"], item["model_calls"], item["usage"]) == (None, None, None)

    assert [event["type"] for event in payload["custom_events"]] == [
        "custom.gate",
        "custom.verdict",
    ]
    gate = json.loads(payload["custom_events"][0]["payload_preview"])
    assert (gate["gate"], gate["passed"]) == ("plan", True)

    totals = payload["totals"]
    # The workflow session, two step children and the grandchild; not the unrelated session.
    assert totals["sessions"] == 4
    assert totals["model_calls"] == 3
    assert totals["usage"]["input_tokens"] == "30"
    assert totals["usage"]["reasoning_tokens"] == "9"
    assert totals["unpriced_model_calls"] == 3 and totals["estimated_cost"] == {}
    assert payload["complete"] is True


def test_workflow_table_and_jsonl(run, capsys) -> None:
    path, ids = run
    assert main(["session", "workflow", _RUN_ID, "--sqlite", str(path), "--table"]) == 0
    table = capsys.readouterr().out
    assert "Workflow team, session team-run (completed)" in table
    assert ids["plan"] in table and "custom.verdict" in table
    assert "workers/0 [pass]" in table
    assert "Totals: 4 sessions (completed 4), 3 model calls" in table

    assert main(["session", "workflow", _RUN_ID, "--sqlite", str(path), "--jsonl"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["record_type"] for record in records][-1] == "summary"
    assert {record["record_type"] for record in records} == {"step", "custom_event", "summary"}
    assert records[-1]["totals"]["sessions"] == 4


def test_workflow_lists_children_no_step_names(run, capsys) -> None:
    path, ids = run
    payload = _json(capsys, "workflow", ids["plan"], "--sqlite", str(path))

    assert payload["workflow"]["name"] is None
    assert [(row["kind"], row["child_session_id"]) for row in payload["steps"]] == [
        ("child", "grandchild")
    ]
    assert payload["totals"]["sessions"] == 2


def test_workflow_says_when_the_session_walk_stopped_early(run, capsys) -> None:
    path, _ = run
    payload = _json(capsys, "workflow", _RUN_ID, "--sqlite", str(path), "--max-sessions", "2")

    assert payload["complete"] is False and payload["sessions_complete"] is False
    assert payload["totals"]["sessions"] == 2


def test_workflow_of_a_missing_session_fails(tmp_path: Path, capsys) -> None:
    path = tmp_path / "cayu.db"
    _seed(path)
    assert main(["session", "workflow", "nope", "--sqlite", str(path), "--json"]) == 1
    assert "Session not found" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_rollup_scopes_to_a_causal_budget_or_root(run, capsys) -> None:
    path, ids = run
    everything = _json(capsys, "rollup", "--sqlite", str(path))
    assert everything["sessions"] == 5

    budget = _json(capsys, "rollup", "--sqlite", str(path), "--causal-budget", ids["budget"])
    assert budget["sessions"] == 4
    assert sum(group["model_calls"] for group in budget["groups"]) == 3

    root = _json(capsys, "rollup", "--sqlite", str(path), "--root", "elsewhere")
    assert root["sessions"] == 1
    assert [group["key"] for group in root["groups"]] == ["helper"]


@pytest.fixture
def agent(run, monkeypatch):
    """The same store served as a Cloud Agent, and a signed-in CLI."""

    from fastapi.testclient import TestClient
    from tests.cli.test_session_cli_cloud import _FakeCloud

    from cayu.server import ServerConfig, create_server
    from cayu.server.auth import BasicAuth

    path, ids = run
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud())
    server = create_server(
        CayuApp(session_store=SQLiteSessionStore(path), enable_logging=False),
        config=ServerConfig.protected(BasicAuth(username="operator", password="s3cret-password")),
    )
    with TestClient(server, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        yield path, ids


def test_cloud_agent_workflow_and_scoped_rollup_match_the_local_store(agent, capsys) -> None:
    path, ids = agent

    for argv in (
        ("rollup", "--causal-budget", ids["budget"]),
        ("rollup", "--root", "elsewhere"),
    ):
        local = _json(capsys, *argv, "--sqlite", str(path))
        remote = _json(capsys, *argv, "--application", "bug-hunter-agent")
        assert remote == local, argv

    local = _json(capsys, "workflow", _RUN_ID, "--sqlite", str(path))
    remote = _json(capsys, "workflow", _RUN_ID, "--application", "bug-hunter-agent")
    # The Agent's public event view names each step and its child session, so the run
    # reads the same as from its store.
    assert remote == local
    assert remote["step_ids_withheld"] is False
    assert [row["step_id"] for row in remote["steps"] if row["kind"] == "step"] == [
        "plan",
        "worker-w1",
    ]
    plan = next(row for row in remote["steps"] if row["step_id"] == "plan")
    assert (plan["child_session_id"], plan["sessions"], plan["model_calls"]) == (ids["plan"], 2, 2)


def test_journal_from_an_agent_that_withholds_step_ids_lists_children_instead() -> None:
    from cayu.cli._workflow_run import _steps_from_journal
    from cayu.events import Event, EventType
    from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
    from cayu.sessions.records import EventRecord

    def record(sequence: int, event_type: EventType, **payload) -> EventRecord:
        return EventRecord(
            sequence=sequence,
            event=Event(type=event_type, session_id=_RUN_ID, agent_name="planner", payload=payload),
        )

    hidden = {"step_id": PRIVATE_EVENT_AUTHORITY, "child_session_id": PRIVATE_EVENT_AUTHORITY}
    parsed = _steps_from_journal(
        [
            record(1, EventType.WORKFLOW_STEP_STARTED, agent="planner", **hidden),
            record(2, EventType.WORKFLOW_STEP_COMPLETED, has_output=False, **hidden),
        ]
    )

    # An older Agent's public view withholds both ids; the step can't be named.
    assert parsed.ids_withheld is True
    assert parsed.steps == []


def test_agent_reader_leaves_unset_scope_out_and_names_an_agent_that_refuses_it() -> None:
    import httpx

    from cayu.cli._agent_session_reader import AgentSessionReader
    from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
    from cayu.sessions._rollup import SessionRollupQuery

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "causal_budget_id" in body or "root_session_id" in body:
            return httpx.Response(
                422, json={"detail": "body field causal_budget_id is not allowed."}
            )
        return httpx.Response(200, json={"groups": [], "sessions_read": 0, "next_cursor": None})

    reader = AgentSessionReader(
        CayuApi(
            "https://agent.example/cayu/",
            authorization="Basic secret",
            description="The Agent bug-hunter",
            http=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    asyncio.run(reader.rollup_page(SessionRollupQuery()))
    assert "causal_budget_id" not in bodies[0] and "root_session_id" not in bodies[0]
    with pytest.raises(CayuApiError, match="predate rollups scoped by --causal-budget"):
        asyncio.run(reader.rollup_page(SessionRollupQuery(causal_budget_id="run-1")))


_HOSTILE_KEY = "\x1b]52;c;YXR0YWNrZXI=\x07"


class StoppedWorkflow(WorkflowBase):
    """A gated loop whose `do` runs a child without `step()`, then a failure."""

    spec = WorkflowSpec(name="stopped")

    async def run(self, session_id):
        from cayu.workflows import StepResult

        ctx = self.context(session_id)
        yield await ctx.start()

        async def direct(item) -> StepResult:
            child_id = f"{session_id}-direct"
            async for _ in self.app.run(
                RunRequest(
                    agent_name="worker",
                    session_id=child_id,
                    parent_session_id=session_id,
                    causal_budget_id=session_id,
                    messages=[Message.text("user", "work")],
                )
            ):
                pass
            return StepResult(step_id="direct", session_id=child_id)

        async for event in gated_loop(
            ctx, ["x"], do=direct, gate=_always_pass, key=lambda item: _HOSTILE_KEY, name="items"
        ):
            yield event
        raise RuntimeError("the next step failed")


def test_workflow_status_comes_from_the_journal_and_tables_escape_journal_text(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ScriptedModelProvider(response_factory=lambda request: _reply("ok")),
                default=True,
            )
            app.register_agent(AgentSpec(name="worker", model="scripted-model"))
            with pytest.raises(RuntimeError, match="next step failed"):
                async for _ in StoppedWorkflow(app).run("stopped-run"):
                    pass
        finally:
            await store.close()

    asyncio.run(seed())
    payload = _json(capsys, "workflow", "stopped-run", "--sqlite", str(path))

    # The journal's anchor session is completed up front; the run itself never completed.
    assert payload["workflow"]["session_status"] == "completed"
    assert payload["workflow"]["status"] == "stopped"
    assert payload["workflow"]["completed_at"] is None
    (item,) = payload["steps"]
    assert (item["kind"], item["item_key"], item["outcome"]) == ("gated_loop", _HOSTILE_KEY, "pass")
    # No step row covers the item's child, so the item carries its usage.
    assert (item["child_session_id"], item["model_calls"]) == ("stopped-run-direct", 1)

    assert main(["session", "workflow", "stopped-run", "--sqlite", str(path), "--table"]) == 0
    table = capsys.readouterr().out
    assert "\x1b" not in table and "\x07" not in table
    assert "items/\\x1b]52;c;YXR0YWNrZXI=\\x07 [pass]" in table
    assert "(stopped)" in table


class CheckWorkflow(WorkflowBase):
    """A command-only workflow that completes only when its check passes."""

    spec = WorkflowSpec(name="check")

    def __init__(self, app: CayuApp, *, code: int) -> None:
        super().__init__(app)
        self.code = code

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        result = await command(
            ctx,
            step_id="check",
            argv=[sys.executable, "-c", f"import sys; sys.exit({self.code})"],
            # Run it every time, as a check of current state does, rather than replaying.
            rerun=True,
        )
        if not result.passed:
            raise RuntimeError("the check failed")
        yield await ctx.completed({"passed": True})


def _run_checks(path: Path, root: Path, *codes: int) -> None:
    """Run the check workflow under one session id once per exit code."""

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_environment(
                Environment(EnvironmentSpec(name="local"), runner=LocalRunner(root)),
                default=True,
            )
            for code in codes:
                workflow = CheckWorkflow(app, code=code)
                if code == 0:
                    async for _ in workflow.run("check-run"):
                        pass
                else:
                    with pytest.raises(RuntimeError, match="check failed"):
                        async for _ in workflow.run("check-run"):
                            pass
        finally:
            await store.close()

    asyncio.run(seed())


def test_a_command_step_is_listed_with_its_outcome(tmp_path: Path, capsys) -> None:
    path = tmp_path / "cayu.db"
    _run_checks(path, tmp_path, 3)

    payload = _json(capsys, "workflow", "check-run", "--sqlite", str(path))

    (check,) = payload["steps"]
    assert (check["kind"], check["step_id"], check["child_session_id"]) == (
        "command",
        "check",
        None,
    )
    assert (check["journal"], check["outcome"], check["exit_code"]) == ("completed", "failed", 3)
    assert check["started_at"] is not None and check["ended_at"] is not None
    assert check["model_calls"] is None
    assert payload["workflow"]["status"] == "stopped"

    assert main(["session", "workflow", "check-run", "--sqlite", str(path), "--table"]) == 0
    assert "check [failed]" in capsys.readouterr().out


def test_a_rerun_of_a_completed_run_counts_from_its_own_start(tmp_path: Path, capsys) -> None:
    path = tmp_path / "cayu.db"
    _run_checks(path, tmp_path, 0, 7)

    payload = _json(capsys, "workflow", "check-run", "--sqlite", str(path))

    # The first attempt completed; the rerun under the same session id failed its check.
    workflow = payload["workflow"]
    assert (workflow["status"], workflow["completed_at"]) == ("stopped", None)
    assert [(row["outcome"], row["exit_code"]) for row in payload["steps"]] == [
        ("passed", 0),
        ("failed", 7),
    ]
    # Its start is the rerun's, after the first attempt's check ended.
    assert payload["steps"][0]["ended_at"] <= workflow["started_at"]
    assert workflow["started_at"] <= payload["steps"][1]["started_at"]

    # Completing it again reads as completed.
    _run_checks(path, tmp_path, 0)
    payload = _json(capsys, "workflow", "check-run", "--sqlite", str(path))
    assert payload["workflow"]["status"] == "completed"


def _command_records(*events: tuple[str, str, dict]) -> list[EventRecord]:
    """Journal records of command steps: (started|completed, step id, extra payload)."""

    start = datetime(2026, 11, 1, 9, tzinfo=UTC)
    types = {
        "started": EventType.WORKFLOW_STEP_STARTED,
        "completed": EventType.WORKFLOW_STEP_COMPLETED,
    }
    return [
        EventRecord(
            sequence=index + 1,
            event=Event(
                type=types[kind],
                session_id="run",
                timestamp=start + timedelta(seconds=index),
                payload={"step_id": step_id, "kind": "command", **extra},
            ),
        )
        for index, (kind, step_id, extra) in enumerate(events)
    ]


def _rows(journal: list[EventRecord]) -> list[tuple[int | None, int | None, int | None]]:
    """Each command row as (started second, completed second, exit code)."""

    def second(moment: datetime | None) -> int | None:
        return None if moment is None else moment.second

    return [
        (second(step.started_at), second(step.completed_at), step.exit_code)
        for step in _steps_from_journal(journal).steps
    ]


def test_withheld_command_ids_pair_only_when_the_argv_tells_them_apart() -> None:
    hidden = PRIVATE_EVENT_AUTHORITY
    a, b = {"argv": ["pytest"]}, {"argv": ["ruff", "check"]}
    # Two concurrent commands: each completion finds its own start by argv.
    distinct = _command_records(
        ("started", hidden, a),
        ("started", hidden, b),
        ("completed", hidden, {**a, "exit_code": 0, "passed": True}),
        ("completed", hidden, {**b, "exit_code": 3, "passed": False}),
    )
    assert _rows(distinct) == [(0, 2, 0), (1, 3, 3)]
    assert _steps_from_journal(distinct).ids_withheld
    # The same argv twice can't be told apart, so no completion is given another's start.
    same = _command_records(
        ("started", hidden, a),
        ("started", hidden, a),
        ("completed", hidden, {**a, "exit_code": 0}),
    )
    assert _rows(same) == [(0, None, None), (1, None, None), (None, 2, 0)]


def test_a_command_completion_pairs_with_its_step_ids_latest_start() -> None:
    # An earlier attempt died mid-command; the rerun's completion is the rerun's.
    journal = _command_records(
        ("started", "check", {"argv": ["pytest"]}),
        ("started", "check", {"argv": ["pytest"]}),
        ("completed", "check", {"argv": ["pytest"], "exit_code": 0, "passed": True}),
    )
    assert _rows(journal) == [(0, None, None), (1, 2, 0)]


def test_redacted_command_names_never_swap_completions() -> None:
    from cayu.runtime._event_projection import project_persisted_runtime_event
    from cayu.vaults.redaction import SecretRedactor

    redactor = SecretRedactor(["alpha-token", "beta-token"])
    a, b = {"argv": ["pytest"]}, {"argv": ["ruff", "check"]}
    records = _command_records(
        ("started", "check-alpha-token", a),
        ("started", "check-beta-token", b),
        ("completed", "check-alpha-token", {**a, "exit_code": 0, "passed": True}),
        ("completed", "check-beta-token", {**b, "exit_code": 3, "passed": False}),
    )
    projected = [
        EventRecord(
            sequence=record.sequence,
            event=project_persisted_runtime_event(
                record.event, sequence=record.sequence, redactor=redactor
            ),
        )
        for record in records
    ]
    assert {record.event.payload["step_id"] for record in projected} == {"check-[REDACTED_SECRET]"}
    assert _rows(projected) == [(0, 2, 0), (1, 3, 3)]
    assert _steps_from_journal(projected).ids_withheld
    assert [step.outcome for step in _steps_from_journal(projected).steps] == ["passed", "failed"]
