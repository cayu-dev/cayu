"""Active limits must not acquire a fresh allowance after child action close."""

import asyncio
import os
import select
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from tests.core.test_foreground_child_resolution_contention import _ContendedTool
from tests.core.test_foreground_subagent_recovery import _identity, _Provider

from cayu import (
    AgentSpec,
    CayuApp,
    IncompleteSessionRecoveryRequest,
    Message,
    RunRequest,
    SessionQuery,
    SQLiteSessionStore,
    SubagentSpec,
    SubagentTool,
    ToolApprovalDecision,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime import ToolApprovalRequest, UserInputResponse
from cayu.runtime.budgets import BudgetLimit
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.invocation import InvocationOriginClaim
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.tool_policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.user_input import UserInputTool


async def _worker(path, action, metric, phase):
    store = SQLiteSessionStore(path)
    now = datetime(2026, 9, 13, tzinfo=UTC)
    if phase == "recover" and metric == "elapsed":
        now += timedelta(seconds=30)
    limits = RunLimits(
        **{
            "tokens": {"max_total_tokens": 3},
            "tools": {"max_tool_calls": 1},
            "elapsed": {"max_elapsed_seconds": 10},
        }.get(metric, {})
    )

    def done():
        return ModelStreamEvent.completed(
            {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        )

    opening = [
        [
            ModelStreamEvent.tool_call(
                id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
            ),
            done(),
        ],
        [
            ModelStreamEvent.tool_call(
                id="action",
                name="ask_user" if action == "input" else "record",
                arguments={"question": "Continue?"} if action == "input" else {"value": 7},
            ),
            done(),
        ],
    ]
    remaining = (
        [
            [ModelStreamEvent.tool_call(id="after", name="record", arguments={"value": 8}), done()],
            [ModelStreamEvent.text_delta("parent complete"), done()],
        ]
        if metric in {"tokens", "causal", "tools"}
        else [[ModelStreamEvent.text_delta("parent complete"), done()]]
    )
    provider = _Provider(remaining if phase == "recover" else opening + remaining)
    tool = _ContendedTool()
    app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "child": SubagentSpec(
                        agent_name="child", limits=limits, max_steps=1 if metric == "steps" else 10
                    )
                },
                execution_profile_identity=_identity("restart-limits"),
            )
        ],
    )
    app.register_agent(
        AgentSpec(name="child", model="test"),
        tools=[tool, UserInputTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
    )
    try:
        if phase == "kill":
            budget = (
                ()
                if metric != "causal"
                else (
                    BudgetLimit(
                        scope="causal",
                        key="parent",
                        max_estimated_cost=Decimal("5"),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name=provider.name,
                                    model="test",
                                    input_per_million=Decimal("1000000"),
                                    output_per_million=Decimal("1000000"),
                                ),
                            )
                        ),
                    ),
                )
            )
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                        budget_limits=budget,
                        invocation_origin=InvocationOriginClaim(
                            subject="operator", tenant="tenant"
                        ),
                    )
                )
            ]
        child = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions[0]
        parent = await store.load("parent")
        assert child.parent_session_id == parent.id
        assert child.causal_budget_id == parent.causal_budget_id == "parent"
        assert child.invocation.origin == parent.invocation.origin
        assert child.invocation.origin.subject == "operator"
        assert child.invocation.root_invocation_id == parent.invocation.root_invocation_id
        assert child.invocation.source.value == "subagent"
        if phase == "recover":
            checkpoint = await store.load_checkpoint(child.id)
            marker = checkpoint["foreground_child_post_action_continuation"]
            assert marker["pending_tool_round"]["limits"] == limits.model_dump(mode="json")
            assert marker["pending_tool_round"]["max_steps"] == (1 if metric == "steps" else 10)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=child.id, inactive_for_seconds=0)
            )
            assert await app.drain_background_interruptions(timeout_s=10)
            await app.recover_persisted_event_side_effects()
            events = await store.load_events(child.id)
            stops = [
                event
                for event in events
                if event.type == "session.interrupted"
                and event.payload.get("reason") == "limit_reached"
            ]
            assert stops, [(event.type, event.payload) for event in events]
            expected = {
                "tokens": "total_tokens",
                "tools": "tool_calls",
                "elapsed": "elapsed_seconds",
                "steps": "model_steps",
                "causal": "estimated_cost",
            }[metric]
            assert stops[-1].payload["limit"] == expected
            assert tool.values == []
            assert sum(event.type == "model.started" for event in events) == (
                2 if metric in {"tokens", "causal", "tools"} else 1
            )
            assert (await store.load(child.id)).invocation == child.invocation
        else:
            publish = app._runtime_session_store.publish_runtime_publication

            async def stop(session_id, **kwargs):
                result = await publish(session_id, **kwargs)
                if kwargs["request"].kind in {"approval-close", "user-input-close"}:
                    assert "foreground_child_post_action_continuation" in (
                        await store.load_checkpoint(child.id)
                    )
                    print("CLOSED", flush=True)
                    await asyncio.Event().wait()
                return result

            app._runtime_session_store.publish_runtime_publication = stop
            events = await store.load_events(child.id)
            if action == "input":
                pending = next(
                    event for event in events if event.type == "session.awaiting_user_input"
                )
                stream = app.resolve_user_input(
                    UserInputResponse(
                        session_id=child.id, input_id=pending.payload["input_id"], answer="yes"
                    )
                )
            else:
                pending = next(
                    event for event in events if event.type == "tool.call.approval_requested"
                )
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child.id,
                        approval_id=pending.payload["approval"]["approval_id"],
                        tool_round_id=pending.payload["tool_round_id"],
                        tool_call_id=pending.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            _ = [event async for event in stream]
            raise AssertionError("Did not reach action close")
    finally:
        assert await app.drain_background_interruptions(timeout_s=10)
        await store.close()


@pytest.mark.parametrize("action", ["approval", "input"])
@pytest.mark.parametrize("metric", ["tokens", "tools", "elapsed", "steps", "causal"])
def test_child_close_restart_preserves_limit_consumption(tmp_path, action, metric):
    command = [
        sys.executable,
        "-m",
        "tests.core.test_foreground_child_restart_limits",
        str(tmp_path / "limits.sqlite"),
        action,
        metric,
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))),
    }
    process = subprocess.Popen(
        [*command, "kill"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 60)
        assert ready, "No action-close barrier"
        line = process.stdout.readline()
        if line.strip() != "CLOSED":
            output, _ = process.communicate(timeout=20)
            pytest.fail(line + output)
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode == -9
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    result = subprocess.run(
        [*command, "recover"], env=environment, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    asyncio.run(_worker(*sys.argv[1:]))
