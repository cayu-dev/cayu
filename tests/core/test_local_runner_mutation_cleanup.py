from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventQuery,
    ExecCommand,
    InMemorySessionStore,
    LocalArtifactStore,
    LocalRunner,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.runners import _subprocess as subprocess_module

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process group evidence")


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "nonzero",
        "timeout",
        "cancel",
        "repeat_cancel",
        "kill_failure",
        "io_failure",
        "uncertain_tree",
        "uncertain_tree_exit",
    ],
)
def test_native_local_runner_cleanup_preserves_mutation_authority(tmp_path, mode, monkeypatch):
    cancelled = mode in {"cancel", "repeat_cancel", "kill_failure", "io_failure"}
    uncertain = mode in {"kill_failure", "io_failure", "uncertain_tree", "uncertain_tree_exit"}

    async def run():
        kill_started = asyncio.Event()
        release_kill = asyncio.Event()
        original_kill = subprocess_module._kill_process

        async def delayed_kill(process, *, process_group):
            kill_started.set()
            await release_kill.wait()
            return await original_kill(process, process_group=process_group)

        if mode == "repeat_cancel":
            monkeypatch.setattr(subprocess_module, "_kill_process", delayed_kill)

        async def failed_kill(process, *, process_group):
            # Stop the synthetic child but simulate a backend that cannot prove
            # its kill completed. Reaping alone must not manufacture evidence.
            await original_kill(process, process_group=process_group)
            raise OSError("injected kill failure")

        if mode == "kill_failure":
            monkeypatch.setattr(subprocess_module, "_kill_process", failed_kill)
        if mode == "io_failure":
            original_detach = subprocess_module._detach_process

            def failed_detach(process):
                original_detach(process)
                return False

            monkeypatch.setattr(subprocess_module, "_detach_process", failed_detach)
        if mode in {"uncertain_tree", "uncertain_tree_exit"}:
            monkeypatch.setattr(subprocess_module, "_DRAIN_AFTER_KILL_S", 0.1)

        child_code = (
            "import os,time; from pathlib import Path; "
            "Path('child.pid').write_text(str(os.getpid())); "
        )
        if mode in {"uncertain_tree", "uncertain_tree_exit"}:
            child_code += (
                "import subprocess,sys; "
                "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], "
                "start_new_session=True); "
                "Path('escaped.pid').write_text(str(p.pid)); "
            )
        child_code += (
            "raise SystemExit(7)"
            if mode == "nonzero"
            else "pass"
            if mode in {"success", "uncertain_tree_exit"}
            else "time.sleep(120)"
        )

        class Probe(Tool):
            spec = ToolSpec(
                name="cleanup_probe",
                effect=ToolEffect.EXTERNAL,
                workspace_mutation=True,
                parallel_safe=False,
                input_schema={"type": "object", "properties": {}},
            )
            result = None
            cancellation_artifacts = None
            cleanup_error = None

            async def run(self, ctx, args):
                await ctx.workspace.create_bytes("temporary.txt", b"synthetic")
                try:
                    self.result = await ctx.runner.exec(
                        ExecCommand.process(
                            sys.executable,
                            "-c",
                            child_code,
                        ),
                        timeout_s=1
                        if mode in {"timeout", "uncertain_tree", "uncertain_tree_exit"}
                        else 30,
                    )
                    return ToolResult(
                        content="command outcome",
                        is_error=self.result.timed_out or self.result.exit_code != 0,
                        artifacts=self.result.artifacts,
                    )
                except asyncio.CancelledError as exc:
                    self.cancellation_artifacts = getattr(exc, "artifacts", None)
                    raise
                finally:
                    try:
                        await ctx.workspace.delete("temporary.txt")
                    except RuntimeError as exc:
                        if not uncertain:
                            raise
                        self.cleanup_error = exc

        store = InMemorySessionStore()
        runner = LocalRunner(tmp_path, inherit_env=False)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(name="cleanup_probe", arguments={}, id="probe"),
                        ModelStreamEvent.completed(),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path),
                runner=runner,
                artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
            ),
            default=True,
        )
        probe = Probe()
        app.register_agent(AgentSpec(name="probe", model="scripted"), tools=[probe])

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="probe",
                        session_id="cleanup-test",
                        messages=[Message.text("user", "Run probe")],
                    )
                )
            ]

        task = asyncio.create_task(consume())
        try:
            async with asyncio.timeout(15):
                while not (tmp_path / "child.pid").exists():
                    if task.done():
                        await task
                        pytest.fail("Child never started")
                    await asyncio.sleep(0.01)
                if cancelled:
                    task.cancel("original cancellation")
                    if mode == "repeat_cancel":
                        await kill_started.wait()
                        task.cancel("second cancellation")
                        await asyncio.sleep(0)
                        assert not task.done()
                        assert (tmp_path / "temporary.txt").exists()
                        release_kill.set()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    await task
            pid = int((tmp_path / "child.pid").read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert (tmp_path / "temporary.txt").exists() is uncertain
            records = await store.query_events(EventQuery(session_id="cleanup-test"))
            events = [record.event for record in records]
            terminals = [
                event
                for event in events
                if str(event.type)
                in {
                    "tool.call.completed",
                    "tool.call.failed",
                }
            ]
            assert len(terminals) == 1
            terminal = terminals[0]
            if uncertain:
                assert str(terminal.type) == "tool.call.failed"
                assert (
                    terminal.payload["workspace_mutation_capture_detail_code"]
                    == "mutation_settlement_unproven"
                )
                assert probe.cleanup_error is not None
                artifacts = probe.cancellation_artifacts if cancelled else probe.result.artifacts
                assert artifacts[0]["status"] == "failed"
                with pytest.raises(RuntimeError, match="cleanup could not be confirmed"):
                    await runner.exec(ExecCommand.process(sys.executable, "-c", "pass"))
                if not cancelled:
                    assert any(str(event.type) == "session.failed" for event in events)
                return
            assert (
                terminal.payload.get("workspace_mutation_capture_detail_code")
                != "mutation_settlement_unproven"
            )
            assert terminal.payload["workspace_mutation_capture_status"] == "recorded"
            assert not any(str(event.type) == "session.failed" for event in events)
            if cancelled:
                assert str(terminal.type) == "tool.call.failed"
                assert probe.result is None
                assert probe.cancellation_artifacts[0]["status"] == "completed"
            else:
                assert probe.result is not None
                assert probe.result.timed_out is (mode == "timeout")
                assert probe.result.exit_code == (
                    0 if mode == "success" else 7 if mode == "nonzero" else -9
                )
                assert str(terminal.type) == (
                    "tool.call.completed" if mode == "success" else "tool.call.failed"
                )
                assert terminal.payload["result"]["is_error"] is (mode != "success")
                if mode == "timeout":
                    assert probe.result.artifacts[0]["status"] == "completed"
                    assert terminal.payload["result"]["artifacts"] == probe.result.artifacts

        finally:
            release_kill.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await runner.close()
            escaped_path = tmp_path / "escaped.pid"
            if escaped_path.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(escaped_path.read_text()), signal.SIGKILL)

    asyncio.run(run())
