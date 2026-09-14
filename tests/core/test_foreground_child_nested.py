"""Public three-agent foreground pauses retain one exact wait per level."""

import asyncio
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_foreground_child_restart import _RestartRecordingTool
from tests.core.test_foreground_subagent_recovery import _identity, _Provider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
)
from cayu.sessions.base import (
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    PendingActionQuery,
    RunRequest,
    SessionQuery,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool


async def _scenario(path, backend, action, phase, stop="no", *, early=False):
    from cayu.runtime import _foreground_child_wait as child_wait

    observed = asyncio.Event()
    release_observer = asyncio.Event()
    middle_running = asyncio.Event()
    release_middle = asyncio.Event()
    original_observe = child_wait.observe_foreground_child_wait

    async def delayed_observe(*args, **kwargs):
        if early and kwargs["parent"].id == "root" and not observed.is_set():
            assert await original_observe(*args, **kwargs) is not None
            observed.set()
            await release_observer.wait()
        return await original_observe(*args, **kwargs)

    class GatedProvider(_Provider):
        async def stream(self, request):
            if early and len(self.requests) == 4:
                middle_running.set()
                await release_middle.wait()
            async for event in super().stream(request):
                yield event

    store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
    opening = [
        [
            ModelStreamEvent.tool_call(
                id="same", name="subagent", arguments={"agent": target, "task": "work"}
            ),
            ModelStreamEvent.completed(),
        ]
        for target in ("middle", "leaf")
    ] + [
        [
            ModelStreamEvent.tool_call(
                id="same",
                name="record" if action == "approval" else "ask_user",
                arguments={"value": 7} if action == "approval" else {"question": "Continue?"},
            ),
            ModelStreamEvent.completed(),
        ]
    ]
    final = [
        [ModelStreamEvent.text_delta(f"{name} finished"), ModelStreamEvent.completed()]
        for name in ("leaf", "middle", "root")
    ]
    provider = GatedProvider(final if phase.startswith("recover") else opening + final)
    tool = _RestartRecordingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    for parent, child in (("root", "middle"), ("middle", "leaf")):
        app.register_agent(
            AgentSpec(name=parent, model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={child: SubagentSpec(agent_name=child)},
                    execution_profile_identity=_identity(f"nested-{parent}"),
                )
            ],
        )
    app.register_agent(
        AgentSpec(name="leaf", model="test"),
        tools=[tool, UserInputTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
    )
    tasks = []
    if early:
        child_wait.observe_foreground_child_wait = delayed_observe
    try:
        if not phase.startswith("recover"):

            async def run_root():
                return [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id="root",
                            agent_name="root",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]

            if early:
                root_task = asyncio.create_task(run_root())
                tasks.append(root_task)
                await asyncio.wait_for(observed.wait(), 30)
            else:
                await run_root()
        middle = (await store.list_sessions(SessionQuery(parent_session_id="root"))).sessions[0]
        leaf = (await store.list_sessions(SessionQuery(parent_session_id=middle.id))).sessions[0]
        if phase != "recover-close":
            for parent_id, child_id, kind in (
                ("root", middle.id, "delegated_action"),
                (middle.id, leaf.id, "tool_approval" if action == "approval" else "user_input"),
            ):
                if early and parent_id == "root":
                    continue
                parent = await store.load(parent_id)
                checkpoint = await store.load_checkpoint(parent_id)
                assert parent.status == "interrupted"
                wait = checkpoint["foreground_child_wait"]
                assert wait["child_session_id"] == child_id
                assert wait["child_action_kind"] == kind
                assert "pending_tool_approval" not in checkpoint
                assert "pending_user_input" not in checkpoint
                profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                assert active_invocation_execution_profile_is_released(
                    profile, session_id=parent.id, run_epoch=parent.run_epoch
                )
                actions = await store.query_pending_actions(
                    PendingActionQuery(session_id=parent_id)
                )
                assert len(actions.actions) == 1
                reference = actions.actions[0].delegated_action
                assert reference.child_session_id == child_id and reference.action_kind == kind
                assert not any(
                    event.type == "tool.call.completed"
                    for event in await store.load_events(parent_id)
                )
            if phase == "kill-pause":
                print("READY", flush=True)
                await asyncio.Event().wait()
        if stop == "yes":
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id="root", reason="Stop ancestor")
                )
            ]
            assert await app.drain_background_interruptions(timeout_s=10)
            stopped_events = await store.load_events("root")
        if phase == "kill-close":
            publish = app._runtime_session_store.publish_runtime_publication

            async def stop_after_close(session_id, **kwargs):
                result = await publish(session_id, **kwargs)
                if session_id == leaf.id and kwargs["request"].kind in {
                    "approval-close",
                    "user-input-close",
                }:
                    print("READY", flush=True)
                    await asyncio.Event().wait()
                return result

            app._runtime_session_store.publish_runtime_publication = stop_after_close
        if phase == "recover-close":
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=leaf.id, inactive_for_seconds=0)
            )
            await app.recover_persisted_event_side_effects()
        else:
            checkpoint = await store.load_checkpoint(leaf.id)
            if action == "approval":
                approval = checkpoint["pending_tool_approval"]
                resolution = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=leaf.id,
                        approval_id=approval["approval_id"],
                        tool_round_id=approval["tool_round_id"],
                        tool_call_id=approval["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            else:
                resolution = app.resolve_user_input(
                    UserInputResponse(
                        session_id=leaf.id,
                        input_id=checkpoint["pending_user_input"]["input_id"],
                        answer="yes",
                    )
                )

            async def resolve_leaf():
                return [event async for event in resolution]

            if early:
                resolve_task = asyncio.create_task(resolve_leaf())
                tasks.append(resolve_task)
                await asyncio.wait_for(middle_running.wait(), 30)
                assert "foreground_parent_continuation" in (await store.load_checkpoint(middle.id))
                release_observer.set()
                await asyncio.wait_for(root_task, 30)
                root_checkpoint = await store.load_checkpoint("root")
                assert (
                    root_checkpoint["foreground_child_wait"]["child_action_kind"]
                    == "delegated_action"
                )
                assert (await store.load("root")).status == "interrupted"
                assert len(provider.requests) == 4
                release_middle.set()
                await asyncio.wait_for(resolve_task, 30)
            else:
                await resolve_leaf()
        assert (await store.load(leaf.id)).status == "completed"
        assert (await store.load(middle.id)).status == "completed"
        assert (await store.load("root")).status == (
            "interrupted" if stop == "yes" else "completed"
        )
        for parent_id in (middle.id, "root"):
            events = await store.load_events(parent_id)
            if stop == "yes" and parent_id == "root":
                assert events == stopped_events
                continue
            assert sum(event.type == "tool.call.completed" for event in events) == 1
            assert sum(event.type == "interaction.started" for event in events) == 1
            assert sum(event.type == "interaction.completed" for event in events) == 1
            assert (
                sum(message.role == "tool" for message in await store.load_transcript(parent_id))
                == 1
            )
        count = len(provider.requests)
        assert count == (3 if phase.startswith("recover") else 6) - int(stop == "yes")
        assert tool.values == ([7] if action == "approval" and phase != "recover-close" else [])
        await app.recover_persisted_event_side_effects()
        assert len(provider.requests) == count
    finally:
        release_observer.set()
        release_middle.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if early:
            child_wait.observe_foreground_child_wait = original_observe
        assert await app.drain_background_interruptions(timeout_s=10)
        if isinstance(store, SQLiteSessionStore):
            await store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["approval", "input"])
@pytest.mark.parametrize("stop", ["no", "yes"])
def test_nested_foreground_actions(tmp_path, backend, action, stop):
    asyncio.run(_scenario(tmp_path / "nested.sqlite", backend, action, "control", stop))


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["approval", "input"])
def test_nested_child_resumes_before_ancestor_observes_pause(tmp_path, backend, action):
    asyncio.run(_scenario(tmp_path / "early.sqlite", backend, action, "control", early=True))


@pytest.mark.parametrize("action", ["approval", "input"])
@pytest.mark.parametrize("boundary", ["pause", "close"])
def test_nested_foreground_restart(tmp_path, action, boundary):
    command = [
        sys.executable,
        "-m",
        "tests.core.test_foreground_child_nested",
        str(tmp_path / "restart.sqlite"),
        "sqlite",
        action,
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))),
    }
    process = subprocess.Popen(
        [*command, f"kill-{boundary}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 90)
        assert ready, "Nested worker did not reach the process-loss boundary"
        line = process.stdout.readline()
        assert line.strip() == "READY", line + (
            process.communicate(timeout=10)[0] if process.poll() is not None else ""
        )
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode == -9
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    result = subprocess.run(
        [*command, f"recover-{boundary}"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    asyncio.run(_scenario(*sys.argv[1:]))
