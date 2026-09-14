"""Manual input recovery keeps its exact identity through a later child pause."""

import asyncio
import os
import select
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.core.test_foreground_child_restart import _RestartRecordingTool
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_user_input import _crashed_user_input_resume_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalRecoveryOutcome
from cayu.approvals.user_input import (
    UserInputRecoveryRequest,
    UserInputResponse,
    user_input_resolution_request_digest,
)
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import (
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    PendingActionQuery,
    RunRequest,
    SessionRuntimePublicationConflict,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import ToolEffect
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool


class _UncertainTool(_RestartRecordingTool):
    spec = _RestartRecordingTool.spec.model_copy(
        update={
            "name": "uncertain",
            "effect": ToolEffect.NONE,
            "execution_profile_identity": _identity("manual-gate-tool"),
        }
    )

    def __init__(self, path):
        super().__init__()
        self.path = path

    async def run(self, ctx, args):
        # This read-only tool has no external-effect receipt protocol. The file
        # is test instrumentation counting invocations across killed processes.
        assert not self.path.exists(), "The recovered tool was dispatched twice"
        self.path.write_text("read-only tool invoked")
        print("READY", flush=True)
        await asyncio.Event().wait()


async def _worker(path, phase):
    def clock():
        return datetime.now(UTC) + timedelta(seconds=0 if phase == "dispatch" else 301)

    memory = phase == "memory"
    store = (
        InMemorySessionStore(ownership_clock=clock)
        if memory
        else SQLiteSessionStore(path / "manual.sqlite", ownership_clock=clock)
    )
    batches = {
        "dispatch": [
            [
                ModelStreamEvent.tool_call(id="external", name="uncertain", arguments={"value": 1}),
                ModelStreamEvent.tool_call(
                    id="answer", name="ask_user", arguments={"question": "Go?"}
                ),
                ModelStreamEvent.tool_call(
                    id="delegate", name="subagent", arguments={"agent": "child", "task": "work"}
                ),
                ModelStreamEvent.completed(),
            ],
        ],
        "recover": [
            [
                ModelStreamEvent.tool_call(
                    id="child-input", name="ask_user", arguments={"question": "Continue?"}
                ),
                ModelStreamEvent.completed(),
            ],
        ],
        "finish": [
            [ModelStreamEvent.text_delta("child done"), ModelStreamEvent.completed()],
            [ModelStreamEvent.text_delta("parent done"), ModelStreamEvent.completed()],
        ],
    }
    provider = _Provider(
        batches["dispatch"] + batches["recover"] + batches["finish"] if memory else batches[phase]
    )
    app = CayuApp(session_store=store, clock=clock, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            _UncertainTool(path / "effect.txt"),
            UserInputTool(),
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("manual-gate-subagent"),
            ),
        ],
    )
    app.register_agent(AgentSpec(name="child", model="test"), tools=[UserInputTool()])

    async def drain(stream):
        return [event async for event in stream]

    try:
        if phase == "dispatch" or memory:
            await drain(
                app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            )
        elif phase == "recover":
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="parent", inactive_for_seconds=0)
            )
        checkpoint = await store.load_checkpoint("parent")
        input_id = checkpoint["pending_user_input"]["input_id"]
        answer = UserInputResponse(session_id="parent", input_id=input_id, answer="yes")
        original_digest = user_input_resolution_request_digest(answer)
        original_key = f"foreground-gate:input:{input_id}:{original_digest}"
        if phase == "dispatch":
            await drain(app.resolve_user_input(answer))
            pytest.fail("The external-effect barrier was not reached")
        if memory:
            # Prime only the lost-result prefix for the memory-store case. The
            # SQLite case above obtains this prefix through real dispatch/SIGKILL.
            await store.append_events(
                "parent",
                _crashed_user_input_resume_events(
                    await store.load_events("parent"), session_id="parent", tool_call_id="external"
                ),
            )
            interrupted = await drain(app.resolve_user_input(answer))
            assert any(e.payload.get("manual_recovery_required") for e in interrupted)
        original = await store.load_session_operation("parent", original_key)
        assert original is not None
        recovery = UserInputRecoveryRequest(
            session_id="parent",
            input_id=input_id,
            answer="yes",
            tool_call_id="external",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="externally verified",
        )
        recovery_digest = user_input_resolution_request_digest(recovery)
        if phase == "recover" or memory:
            if not memory:
                interrupted = await drain(app.resolve_user_input(answer))
                assert any(e.payload.get("manual_recovery_required") for e in interrupted)
            events = await drain(app.recover_user_input(recovery))
            assert any(
                e.payload.get("interruption_type") == "waiting_on_child_action" for e in events
            ), [e.payload for e in events]
            checkpoint = await store.load_checkpoint("parent")
            assert (
                checkpoint["user_input_resolution_intent"]["resolution_stage"] == "manual-recovery"
            )
            assert await store.load_session_operation("parent", original_key) == original
            record = await store.load_session_operation(
                "parent", f"foreground-gate:input:{input_id}:{recovery_digest}"
            )
            assert record["resolution_stage"] == "manual-recovery"
            assert record["answer_request_digest"] == original["answer_request_digest"]
            if memory:
                before_events = await store.load_events("parent")
                with pytest.raises(SessionRuntimePublicationConflict):
                    await drain(
                        app.recover_user_input(
                            recovery.model_copy(update={"message": "a different claimed outcome"})
                        )
                    )
                assert await store.load_checkpoint("parent") == checkpoint
                assert await store.load_events("parent") == before_events
            assert len(provider.requests) == (2 if memory else 1)
            if not memory:
                print("READY", flush=True)
                await asyncio.Event().wait()
        actions = await store.query_pending_actions(PendingActionQuery(session_id="parent"))
        child_id = actions.actions[0].delegated_action.child_session_id
        child = await store.load_checkpoint(child_id)
        await drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=child_id,
                    input_id=child["pending_user_input"]["input_id"],
                    answer="yes",
                )
            )
        )
        assert (await store.load("parent")).status == "completed"
        receipt = await store.load_runtime_publication_receipt(
            "parent", f"user-input-close:{input_id}"
        )
        assert receipt.intent["resolution_request_digest"] == recovery_digest
        assert await store.load_session_operation("parent", original_key) == original
        if memory:
            assert not (path / "effect.txt").exists()
        else:
            assert (path / "effect.txt").read_text() == "read-only tool invoked"
        for call_id in ("external", "answer", "delegate"):
            events = await store.load_events("parent")
            assert (
                sum(
                    e.type == "tool.call.completed" and e.payload.get("tool_call_id") == call_id
                    for e in events
                )
                == 1
            )
        await drain(app.recover_user_input(recovery))
        await app.recover_persisted_event_side_effects()
        assert len(provider.requests) == (4 if memory else 2)
    finally:
        assert await app.drain_background_interruptions(timeout_s=10)
        if not memory:
            await store.close()


def test_manual_recovery_then_child_pause_in_memory(tmp_path):
    asyncio.run(_worker(tmp_path, "memory"))


def test_manual_recovery_then_child_pause_survives_process_loss(tmp_path):
    command = [
        sys.executable,
        "-m",
        "tests.core.test_foreground_gate_manual_recovery",
        str(tmp_path),
    ]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd())))}
    for phase in ("dispatch", "recover"):
        process = subprocess.Popen(
            [*command, phase], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        try:
            assert process.stdout is not None
            ready, _, _ = select.select([process.stdout], [], [], 90)
            assert ready, f"Worker did not reach {phase} barrier"
            line = process.stdout.readline()
            assert line.strip() == "READY", line + process.communicate(timeout=30)[0]
            process.kill()
            process.communicate(timeout=10)
            assert process.returncode == -9
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    result = subprocess.run(
        [*command, "finish"], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    asyncio.run(_worker(Path(sys.argv[1]), sys.argv[2]))
