"""A workflow command runs in the brokered sandbox its steps share, against a fake broker."""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.sandbox_broker.fake_broker import FakeBroker
from tests.sandbox_broker.test_brokered_sandbox import _factory, _project
from tests.sandbox_broker.test_brokered_shared_allocation import _call, _done, _Solve, _WritePlan

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.storage import SQLiteSessionStore
from cayu.workflows import CommandResult, StepRunOptions, WorkflowBase, WorkflowSpec, command, step

_CHECK = (
    "import pathlib, runpy; "
    "assert pathlib.Path('plan.md').read_text().startswith('replace a - b'); "
    "add = runpy.run_path('solution.py')['add']; "
    "print('add(2, 3) =', add(2, 3)); "
    "raise SystemExit(0 if add(2, 3) == 5 else 1)"
)


def test_a_command_tests_the_solution_in_the_steps_sandbox(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    seen: dict[str, object] = {}

    class PlanSolveTest(WorkflowBase):
        spec = WorkflowSpec(name="plan-solve-test")

        async def run(self, session_id: str):
            ctx = self.context(session_id)
            yield await ctx.start()
            options = StepRunOptions(allocation_scope="parent")
            await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=options)
            await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=options)
            result = await command(
                ctx,
                step_id="tests",
                argv=[sys.executable, "-c", _CHECK],
                allocation_scope="parent",
            )
            seen["result"] = result
            # The command ran in the sandbox and left it alive for the run to release.
            seen["live_after_command"] = [sandbox.id for sandbox in broker.live()]
            seen["solution_in_source_after_command"] = (source / "solution.py").exists()
            yield await ctx.completed({"passed": result.passed})

    async def scenario():
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(
            ScriptedModelProvider([_call("write_plan"), _done(), _call("solve"), _done()]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            _factory(broker, source),
            default=True,
        )
        app.register_agent(AgentSpec(name="planner", model="scripted"), tools=[_WritePlan()])
        app.register_agent(AgentSpec(name="solver", model="scripted"), tools=[_Solve()])
        try:
            return [event async for event in PlanSolveTest(app).run("run-1")]
        finally:
            await store.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    result = seen["result"]
    assert isinstance(result, CommandResult)
    assert result.passed, (result.stdout, result.stderr)
    assert result.stdout.strip() == "add(2, 3) = 5"
    # One sandbox served both steps and the command; only completion destroyed it.
    (sandbox,) = broker.sandboxes.values()
    assert seen["live_after_command"] == [sandbox.id]
    assert seen["solution_in_source_after_command"] is False
    assert sandbox.status == "destroyed" and sandbox.destroy_reason == "requested"
    assert (source / "solution.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
