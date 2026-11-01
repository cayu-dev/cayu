"""`cayu session workflow` and `cayu session rollup --causal-budget/--root` over a local store."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.cli import main
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions.queries import SessionQuery
from cayu.storage import SQLiteSessionStore
from cayu.workflows import WorkflowBase, WorkflowSpec, gated_loop, step

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
    assert local["step_ids_withheld"] is False
    # An Agent's public event view withholds journal step and child ids, so its run lists
    # the child sessions; the totals, gate outcomes and custom events are the same.
    assert remote["step_ids_withheld"] is True
    assert [(row["kind"], row["child_session_id"]) for row in remote["steps"]] == [
        ("gated_loop", None),
        ("child", ids["plan"]),
        ("child", ids["worker"]),
    ]
    assert remote["steps"][0]["outcome"] == "pass"
    plan = next(row for row in remote["steps"] if row["child_session_id"] == ids["plan"])
    assert (plan["agent"], plan["sessions"], plan["model_calls"]) == ("planner", 2, 2)
    for key in ("workflow", "totals", "custom_events", "complete"):
        assert remote[key] == local[key], key


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
