"""A resolved parent gate may itself dispatch a child that needs a human."""

import asyncio
import os
import select
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.core.test_foreground_child_restart import _RestartRecordingTool
from tests.core.test_foreground_subagent_recovery import _identity, _Provider

from cayu import (
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
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
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.runtime.loop_policies import BeforeStopDecision, LoopPolicy
from cayu.runtime.sessions import (
    PendingActionQuery,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
)
from cayu.runtime.tool_policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.user_input import UserInputTool


class _DeclaredPolicy(LoopPolicy):
    def __init__(self):
        self.stops = 0
        self.version = "accepted"

    @property
    def execution_profile_identity(self):
        return _identity(f"gate-policy-{self.version}")

    async def before_stop(self, context):
        self.stops += 1
        return BeforeStopDecision.complete()


def _run_case(
    tmp_path,
    backend,
    parent_gate,
    child_gate,
    phase="live",
    *,
    child_count=1,
    loop_policies=(),
    expect_policy_refusal=False,
    change_policy=False,
    stop_parent=False,
    restore_policies=False,
):
    async def scenario():
        clock_offset = 301 if phase == "recover-close" else 0

        def clock():
            return datetime.now(UTC) + timedelta(seconds=clock_offset)

        store = (
            InMemorySessionStore(ownership_clock=clock)
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "gate.sqlite", ownership_clock=clock)
        )
        parent_calls = [
            ModelStreamEvent.tool_call(
                id=f"delegate-{index}",
                name="subagent",
                arguments={"agent": "child", "task": "work"},
            )
            for index in range(child_count)
        ]
        if parent_gate == "input":
            parent_calls.insert(
                0,
                ModelStreamEvent.tool_call(
                    id="answer", name="ask_user", arguments={"question": "Start?"}
                ),
            )
        batches = [
            [*parent_calls, ModelStreamEvent.completed()],
            [
                ModelStreamEvent.tool_call(
                    id="leaf",
                    name="record" if child_gate == "approval" else "ask_user",
                    arguments={"value": 7}
                    if child_gate == "approval"
                    else {"question": "Continue?"},
                ),
                ModelStreamEvent.completed(),
            ],
            [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
            [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
        ]
        provider = _Provider(
            ([batches[0]] + batches[1:3] * child_count + batches[-1:])[-1:]
            if phase == "recover-close"
            else ([batches[0]] + batches[1:3] * child_count + batches[-1:])[
                4 if phase == "recover-second-pause" else 2 :
            ]
            if phase.startswith("recover")
            else [batches[0]] + batches[1:3] * child_count + batches[-1:]
        )
        tool = _RestartRecordingTool()
        app = CayuApp(session_store=store, enable_logging=False, clock=clock)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("parent-gated-delegation"),
                ),
                UserInputTool(),
            ],
            tool_policy=AlwaysRequireApprovalToolPolicy(
                tools=["subagent"] if parent_gate == "approval" else []
            ),
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[tool, UserInputTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )

        async def resolve(session_id, kind, *, changed_controls=False):
            checkpoint = await store.load_checkpoint(session_id)
            attached = checkpoint.get("foreground_parent_continuation")
            closed_effect = (
                attached["terminal"]["wait"]["parent_effect"] if attached is not None else None
            )
            if kind == "approval":
                gate = checkpoint.get("pending_tool_approval", closed_effect)
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=gate["approval_id"],
                        tool_round_id=gate["tool_round_id"],
                        tool_call_id=gate["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                        loop_policies=loop_policies if session_id == "parent" else (),
                        **({"max_steps": 99} if changed_controls else {}),
                    )
                )
            else:
                stream = app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=checkpoint["pending_user_input"]["input_id"]
                        if "pending_user_input" in checkpoint
                        else closed_effect["pause_id"],
                        answer="yes",
                        loop_policies=loop_policies if session_id == "parent" else (),
                        **({"max_steps": 99} if changed_controls else {}),
                    )
                )
            return [event async for event in stream]

        try:
            if not phase.startswith("recover"):
                events = [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id="parent",
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                            loop_policies=loop_policies,
                        )
                    )
                ]
                if parent_gate != "none":
                    assert len(provider.requests) == 1
                    events = await resolve("parent", parent_gate)
                else:
                    assert len(provider.requests) == 2
                assert any(
                    e.type == "session.interrupted"
                    and e.payload.get("interruption_type") == "waiting_on_child_action"
                    for e in events
                ), [
                    (str(e.type), e.payload.get("error"))
                    for e in events
                    if e.type == "session.interrupted"
                ]
            children = (
                await store.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions
            if phase != "recover-close":
                actions = await store.query_pending_actions(PendingActionQuery(session_id="parent"))
                assert len(actions.actions) == 1
                assert actions.actions[0].delegated_action.child_session_id in {
                    child.id for child in children
                }
            if stop_parent:
                from cayu.runtime import InterruptSessionRequest

                owner = app._recovery_coordinator._foreground_gate_policy_owner
                assert owner._wait_policies if parent_gate == "none" else owner._policies
                _ = [
                    event
                    async for event in app.interrupt_session(
                        InterruptSessionRequest(
                            session_id="parent", reason="Stop the accepted gate."
                        )
                    )
                ]
                assert not app._recovery_coordinator._foreground_gate_policy_owner._policies
                assert not owner._wait_policies
                assert len(provider.requests) == 2
                return
            if phase == "kill-pause":
                print("READY", flush=True)
                await asyncio.Event().wait()
            if phase == "kill-close":
                publish = app._runtime_session_store.publish_runtime_publication

                async def stop_after_close(session_id, **kwargs):
                    result = await publish(session_id, **kwargs)
                    if session_id == "parent" and kwargs["request"].kind in {
                        "approval-close",
                        "user-input-close",
                    }:
                        print("READY", flush=True)
                        await asyncio.Event().wait()
                    return result

                app._runtime_session_store.publish_runtime_publication = stop_after_close
            if phase == "recover-close":
                await app.recover_persisted_event_side_effects()
            else:
                if change_policy:
                    loop_policies[0].version = "replacement"
                for index in range(1 if phase == "recover-second-pause" else child_count):
                    actions = await store.query_pending_actions(
                        PendingActionQuery(session_id="parent")
                    )
                    if not actions.actions or actions.actions[0].delegated_action is None:
                        checkpoint = await store.load_checkpoint("parent")
                        pytest.fail(
                            str(
                                {
                                    "status": (await store.load("parent")).status,
                                    "events": [
                                        (
                                            str(e.type),
                                            e.payload.get("interruption_type"),
                                            e.payload.get("error"),
                                        )
                                        for e in await store.load_events("parent")
                                        if e.type.startswith("session.")
                                    ],
                                    "wait": checkpoint.get("foreground_child_wait"),
                                    "tools": [
                                        (
                                            str(e.type),
                                            e.payload.get("tool_call_id"),
                                            e.payload.get("result"),
                                        )
                                        for e in await store.load_events("parent")
                                        if e.type.startswith("tool.call.")
                                    ],
                                    "keys": list(checkpoint),
                                }
                            )
                        )
                    delegated = actions.actions[0].delegated_action
                    assert delegated is not None
                    await resolve(delegated.child_session_id, child_gate)
                    if phase == "kill-second-pause" and index == 0:
                        checkpoint = await store.load_checkpoint("parent")
                        assert (
                            checkpoint["foreground_child_wait"]["parent_effect"]["tool_call_id"]
                            == "delegate-1"
                        )
                        assert "foreground_child_terminal" not in checkpoint
                        print("READY", flush=True)
                        await asyncio.Event().wait()
            if restore_policies:
                assert phase.startswith("recover")
                assert (await store.load("parent")).status == "interrupted"
                requests_before_retry = 0 if phase == "recover-close" else 1
                assert len(provider.requests) == requests_before_retry
                checkpoint = await store.load_checkpoint("parent")
                continuation_key = (
                    "foreground_parent_continuation"
                    if phase == "recover-close"
                    else "foreground_child_wait"
                )
                retained_continuation = checkpoint[continuation_key]
                retained_terminal = checkpoint.get("foreground_child_terminal")
                if phase != "recover-close":
                    assert retained_terminal is None
                owner = app._recovery_coordinator._foreground_gate_policy_owner
                assert not owner._policies

                # A declared identity is portable, but a different declaration
                # must not acquire the accepted gate's executable authority.
                loop_policies[0].version = "replacement"
                with suppress(ExecutionProfileMismatchError):
                    await resolve("parent", parent_gate)
                assert not owner._policies
                assert len(provider.requests) == requests_before_retry
                assert (await store.load("parent")).status == "interrupted"

                loop_policies[0].version = "accepted"
                if phase == "recover-close":
                    before_retry = await store.load_checkpoint("parent")
                    before_events = await store.load_events("parent")
                    with pytest.raises((SessionRunFenced, SessionRuntimePublicationConflict)):
                        await resolve("parent", parent_gate, changed_controls=True)
                    assert await store.load_checkpoint("parent") == before_retry
                    assert await store.load_events("parent") == before_events
                    assert not owner._policies
                await resolve("parent", parent_gate)
                # The public retry restores policy ownership, not authority to
                # select a terminal. Durable child delivery still owns selection.
                assert owner._policies
                checkpoint = await store.load_checkpoint("parent")
                assert checkpoint[continuation_key] == retained_continuation
                assert checkpoint.get("foreground_child_terminal") == retained_terminal
                assert len(provider.requests) == requests_before_retry
                # Failed event delivery has a thirty-second retry delay.
                clock_offset += 31
                await app.recover_persisted_event_side_effects()
            if expect_policy_refusal:
                assert (await store.load("parent")).status == "interrupted"
                assert len(provider.requests) == (1 if phase.startswith("recover") else 3)
                checkpoint = await store.load_checkpoint("parent")
                assert "foreground_child_wait" in checkpoint
                assert "foreground_parent_continuation" not in checkpoint
                return
            assert (await store.load("parent")).status == "completed", "\n".join(
                str(e.payload.get("error"))
                for e in await store.load_events("parent")
                if e.type == "session.interrupted"
            )
            expected_requests = (
                1
                if phase == "recover-close"
                else 2
                if phase == "recover-second-pause"
                else 2 * child_count
                if phase.startswith("recover")
                else 2 + 2 * child_count
            )
            assert len(provider.requests) == expected_requests
            assert tool.values == (
                [7] * (1 if phase == "recover-second-pause" else child_count)
                if child_gate == "approval" and phase != "recover-close"
                else []
            )
            transcript = await store.load_transcript("parent")
            results = [part for m in transcript for part in m.content if part.type == "tool_result"]
            assert len(results) == child_count + (parent_gate == "input")
            assert all(not part.is_error for part in results)
            events = await store.load_events("parent")
            for index in range(child_count):
                assert (
                    sum(
                        e.type == "tool.call.completed"
                        and e.payload.get("tool_call_id") == f"delegate-{index}"
                        for e in events
                    )
                    == 1
                )
            await app.recover_persisted_event_side_effects()
            assert len(provider.requests) == expected_requests
            assert not app._recovery_coordinator._foreground_gate_policy_owner._wait_policies
            if restore_policies:
                assert loop_policies[0].stops == 1
                assert not app._recovery_coordinator._foreground_gate_policy_owner._policies
                # Completed receipt replay does not acquire executable policy
                # ownership or run continuation again, even with new policies.
                loop_policies[0].version = "replacement"
                await resolve("parent", parent_gate)
                assert not app._recovery_coordinator._foreground_gate_policy_owner._policies
                assert len(provider.requests) == expected_requests
        finally:
            assert await app.drain_background_interruptions(timeout_s=10)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("child_gate", ["approval", "input"])
def test_ordinary_foreground_wait_preserves_request_policy(tmp_path, backend, child_gate):
    policy = _DeclaredPolicy()
    _run_case(tmp_path, backend, "none", child_gate, loop_policies=(policy,))
    assert policy.stops == 1


@pytest.mark.parametrize("child_gate", ["approval", "input"])
def test_ordinary_foreground_wait_refuses_changed_policy(tmp_path, child_gate):
    policy = _DeclaredPolicy()
    _run_case(
        tmp_path,
        "memory",
        "none",
        child_gate,
        loop_policies=(policy,),
        change_policy=True,
        expect_policy_refusal=True,
    )
    assert policy.stops == 0


def test_stopping_ordinary_wait_releases_request_policy(tmp_path):
    _run_case(
        tmp_path,
        "memory",
        "none",
        "approval",
        loop_policies=(_DeclaredPolicy(),),
        stop_parent=True,
    )


@pytest.mark.parametrize("child_gate", ["approval", "input"])
def test_restarted_ordinary_wait_refuses_missing_policy(tmp_path, child_gate):
    _crash_case(tmp_path, "none", child_gate, "pause", policy=True)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("parent_gate", ["approval", "input"])
@pytest.mark.parametrize("child_gate", ["approval", "input"])
def test_resolved_parent_gate_waits_for_child(tmp_path, backend, parent_gate, child_gate):
    _run_case(tmp_path, backend, parent_gate, child_gate)


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
@pytest.mark.parametrize("child_gate", ["approval", "input"])
def test_resolved_gate_waits_for_successive_children(tmp_path, parent_gate, child_gate):
    _run_case(tmp_path, "sqlite", parent_gate, child_gate, child_count=2)


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
def test_resolved_gate_preserves_request_policy(tmp_path, parent_gate):
    policy = _DeclaredPolicy()
    _run_case(tmp_path, "memory", parent_gate, "approval", loop_policies=(policy,))
    assert policy.stops == 1


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
def test_resolved_gate_refuses_changed_request_policy(tmp_path, parent_gate):
    policy = _DeclaredPolicy()
    _run_case(
        tmp_path,
        "memory",
        parent_gate,
        "approval",
        loop_policies=(policy,),
        change_policy=True,
        expect_policy_refusal=True,
    )
    assert policy.stops == 0


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
def test_stopping_gate_releases_request_policy(tmp_path, parent_gate):
    _run_case(
        tmp_path,
        "memory",
        parent_gate,
        "approval",
        loop_policies=(_DeclaredPolicy(),),
        stop_parent=True,
    )


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
@pytest.mark.parametrize("child_gate", ["approval", "input"])
@pytest.mark.parametrize("boundary", ["pause", "close"])
def test_resolved_parent_gate_survives_process_loss(tmp_path, parent_gate, child_gate, boundary):
    _crash_case(tmp_path, parent_gate, child_gate, boundary)


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
def test_successive_children_survive_process_loss(tmp_path, parent_gate):
    _crash_case(tmp_path, parent_gate, "approval", "second-pause", child_count=2)


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
def test_restarted_gate_refuses_missing_request_policy(tmp_path, parent_gate):
    _crash_case(tmp_path, parent_gate, "approval", "pause", policy=True)


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
@pytest.mark.parametrize("boundary", ["pause", "close"])
def test_restarted_gate_restores_explicit_request_policy(tmp_path, parent_gate, boundary):
    _crash_case(tmp_path, parent_gate, "approval", boundary, policy=2)


def _crash_case(tmp_path, parent_gate, child_gate, boundary, *, child_count=1, policy=False):
    command = [
        sys.executable,
        "-m",
        "tests.core.test_foreground_child_parent_gate",
        str(tmp_path),
        "sqlite",
        parent_gate,
        child_gate,
    ]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd())))}
    process = subprocess.Popen(
        [*command, f"kill-{boundary}", str(child_count), str(int(policy))],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 90)
        assert ready, "Parent gate worker did not reach the crash boundary"
        line = process.stdout.readline()
        if line.strip() != "READY":
            pytest.fail(line + process.communicate(timeout=30)[0])
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode == -9
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    result = subprocess.run(
        [*command, f"recover-{boundary}", str(child_count), str(int(policy))],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    phase = sys.argv[5]
    policy = int(sys.argv[7])
    _run_case(
        Path(sys.argv[1]),
        *sys.argv[2:6],
        child_count=int(sys.argv[6]),
        loop_policies=(_DeclaredPolicy(),)
        if policy and (phase.startswith("kill") or policy == 2)
        else (),
        expect_policy_refusal=policy == 1 and phase.startswith("recover"),
        restore_policies=policy == 2 and phase.startswith("recover"),
    )
