"""Real Docker disposal/retirement with a killed owner and persistent recovery."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding_live import _configuration_or_skip

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EnvironmentSpec,
    ExecutionProfileBehaviorIdentity,
    ImmutableInputStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    Tool,
    ToolResult,
    ToolSpec,
    inspect_local_immutable_input,
)

pytestmark = pytest.mark.process


async def _process(root: Path, mode: str, point: str) -> None:
    docker, image, image_id = _configuration_or_skip()
    identity = DockerImageIdentity(reference=image, content_digest=image_id)
    architecture = subprocess.check_output(
        [docker, "image", "inspect", "--format", "{{.Architecture}}", image], text=True
    ).strip()
    if mode == "seed":
        (root / "source").mkdir()
        (root / "input").mkdir()
        (root / "input" / "evidence.txt").write_text("immutable")
    inputs = ImmutableInputStore(root / "managed")
    projection = inspect_local_immutable_input(
        root / "input",
        target_path="/evidence",
        policy_fingerprint="sha256:" + "a" * 64,
        runtime_compatibility_fingerprint=identity.fingerprint,
        authorization_scope_fingerprint="sha256:" + "b" * 64,
    )

    class Write(Tool):
        spec = ToolSpec(
            name="write",
            input_schema={"type": "object", "properties": {}},
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:terminal-retirement:write",
                behavior_version="1",
                implementation_version="1",
            ),
        )

        async def run(self, ctx, args):
            await ctx.workspace.write_bytes("result.txt", b"retained")
            with (root / "effects").open("a") as stream:
                stream.write("once\n")
            return ToolResult(content="written")

    script = []
    if mode == "seed":
        script = [
            [
                ModelStreamEvent.tool_call(id="write-once", name="write", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    elif mode == "resume":
        script = [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
    provider = ScriptedModelProvider(script)
    # Expire the dead owner's lease through the supported test clock. This is
    # not a physical-quiescence assertion: Docker disposal is independently read.
    offset = timedelta() if mode == "seed" else timedelta(days=1)

    def clock():
        return datetime.now(UTC) + offset

    sessions = SQLiteSessionStore(root / "sessions.sqlite", ownership_clock=clock)
    app = CayuApp(session_store=sessions, clock=clock, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[Write()])
    app.register_environment_factory(
        EnvironmentSpec(
            name="coding",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:terminal-retirement:environment",
                behavior_version="1",
                implementation_version="1",
            ),
        ),
        DockerCodingEnvironmentFactory(
            source_workspace=LocalWorkspace(root / "source", workspace_id="source"),
            toolchain_profile=docker_toolchain_profile(
                image_identity=identity, platform_architecture=architecture
            ),
            immutable_inputs=(projection,),
            immutable_input_store=inputs,
            immutable_input_runtime_compatibility_fingerprint=identity.fingerprint,
            docker_path=docker,
        ),
        default=True,
    )
    if mode == "seed":
        original = app._environment_lifecycle._retire_disposed_allocation

        async def lose(*args, **kwargs):
            if point == "after_retirement":
                await original(*args, **kwargs)
            checkpoint = await sessions.load_checkpoint("session")
            marker = checkpoint["pending_completion_finalization"]
            assert marker["outcome"] == "interrupted" and marker["disposal_state"]
            (root / "killed-checkpoint.json").write_text(json.dumps(checkpoint))
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError("SIGKILL returned")

        app._environment_lifecycle._retire_disposed_allocation = lose
        async for _ in app.run(
            RunRequest(
                agent_name="worker",
                session_id="session",
                messages=[Message.text("user", "write once")],
                max_steps=1,
            )
        ):
            pass
        raise AssertionError("Owner did not reach the disposal boundary")
    if mode == "recover":
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(
                    session_ids=("session",),
                    inactive_for_seconds=0,
                )
            )
        )
        item = plan.items[0]
        assert RecoveryPlanAction.AUTOMATIC_REPAIR in item.allowed_actions, plan.model_dump_json()
        request = RecoveryExecutionRequest(
            plan=plan,
            decisions=(
                RecoveryDecision(
                    item_id=item.item_id,
                    action=RecoveryPlanAction.AUTOMATIC_REPAIR,
                ),
            ),
            execution_id="recover-terminal-disposal",
        )
        result = await app.execute_recovery(request)
        assert result.items[0].error_code is None, result.model_dump_json()
        checkpoint = await sessions.load_checkpoint("session")
        assert not checkpoint.get("pending_completion_finalization")
        assert not checkpoint.get("environment_factory_pending_disposals")
        assert not checkpoint.get("environment_factory_reconnect")
        replay = await app.execute_recovery(request)
        assert replay.items[0].replayed
        assert replay.items[0].event_ids == result.items[0].event_ids
        assert await sessions.load_checkpoint("session") == checkpoint
        assert not provider.requests
    else:
        async for _ in app.resume(
            ResumeRequest(
                session_id="session",
                messages=[Message.text("user", "finish")],
                max_steps=1,
            )
        ):
            pass
        assert (await sessions.load("session")).status.value == "completed"
        assert len(provider.requests) == 1
    assert (root / "source" / "result.txt").read_bytes() == b"retained"
    assert (root / "effects").read_text() == "once\n"
    assert inputs.inspect()[0].reference_count == 0
    assert await app.drain_environment_cleanups(timeout_s=10)
    await sessions.close()


@pytest.mark.parametrize("point", ["before_retirement", "after_retirement"])
def test_killed_terminal_disposal_owner_retires_before_resume(tmp_path, point, record_property):
    docker, _, _ = _configuration_or_skip()
    script = """
import asyncio, sys
from pathlib import Path
from tests.environments.test_terminal_disposal_retirement_live import _process
asyncio.run(asyncio.wait_for(_process(Path(sys.argv[1]), sys.argv[2], sys.argv[3]), 90))
"""
    for mode in ("seed", "recover", "resume"):
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode, point],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == (-signal.SIGKILL if mode == "seed" else 0), child.stderr
        if mode == "seed":
            checkpoint = json.loads((tmp_path / "killed-checkpoint.json").read_text())
            assert bool(checkpoint.get("environment_factory_pending_disposals")) == (
                point == "before_retirement"
            )
            assert bool(checkpoint.get("environment_factory_retired_disposals")) == (
                point == "after_retirement"
            )
            container = checkpoint["pending_completion_finalization"]["disposal_state"][
                "container_id"
            ]
            assert not subprocess.check_output(
                [
                    docker,
                    "ps",
                    "-a",
                    "--no-trunc",
                    "--filter",
                    "id=" + container,
                    "--format",
                    "{{.ID}}",
                ],
                text=True,
                timeout=15,
            ).strip()
    record_property("remaining_owned_containers", 0)
    record_property("immutable_input_references", 0)
