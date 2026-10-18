"""Abandoning a session that died mid tool call removes its Docker container (#2548)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding_live import _configuration_or_skip

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.environments import DockerCodingEnvironmentFactory, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runners import DockerImageIdentity
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    ResumeRequest,
    SessionExecutionConfig,
)
from cayu.sessions.records import SessionStatusConflict
from cayu.storage import SQLiteSessionStore
from cayu.tasks import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.workspaces import LocalWorkspace

pytestmark = pytest.mark.process


def _identity(name: str) -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class _CrashingEffect(Tool):
    spec = ToolSpec(
        name="record",
        description="Record one external effect.",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=_identity("tests:abandoned-docker:effect"),
    )

    async def run(self, ctx, args):
        if os.environ.get("CAYU_ABANDON_TEST_CRASH") == "1":
            os._exit(137)
        return ToolResult(content="recorded")


def _app(
    root: Path,
    docker_path: str,
    image: str,
    image_id: str,
    architecture: str,
    scripts,
    task_store=None,
):
    store = SQLiteSessionStore(root / "sessions.db")
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        enable_logging=False,
        session_execution=SessionExecutionConfig(
            heartbeat_interval_seconds=0.05, lease_seconds=1.0
        ),
    )
    app.register_provider(ScriptedModelProvider(scripts), default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[_CrashingEffect()])
    app.register_environment_factory(
        EnvironmentSpec(name="coding", execution_profile_identity=_identity("tests:abandoned")),
        DockerCodingEnvironmentFactory(
            source_workspace=LocalWorkspace(root / "source", workspace_id="abandoned-workspace"),
            toolchain_profile=docker_toolchain_profile(
                image_identity=DockerImageIdentity(reference=image, content_digest=image_id),
                platform_architecture=architecture,
            ),
            git_baseline=False,
            docker_path=docker_path,
        ),
        default=True,
    )
    return app, store


async def _crash_child(root: Path, docker_path: str, image: str, image_id: str, arch: str):
    app, _store = _app(
        root,
        docker_path,
        image,
        image_id,
        arch,
        [
            [
                ModelStreamEvent.tool_call(id="effect-once", name="record", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ],
    )
    async for _ in app.run(
        RunRequest(
            agent_name="worker", session_id="abandoned", messages=[Message.text("user", "go")]
        )
    ):
        pass
    raise AssertionError("The tool did not stop the process.")


@pytest.mark.parametrize("cleanup_interrupted", [False, True])
def test_abandon_session_removes_the_container_of_a_lost_tool_call(
    tmp_path: Path, cleanup_interrupted: bool
) -> None:
    docker_path, image, image_id = _configuration_or_skip()
    architecture = subprocess.check_output(
        [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image], text=True
    ).strip()
    (tmp_path / "source").mkdir()
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from tests.environments.test_docker_abandoned_session_live import _crash_child; "
        "asyncio.run(_crash_child(Path(sys.argv[1]), *sys.argv[2:]))"
    )
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), docker_path, image, image_id, architecture],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CAYU_ABANDON_TEST_CRASH": "1"},
    )
    assert crashed.returncode == 137, crashed.stderr[-2000:]

    def present(container_id: str) -> bool:
        return (
            subprocess.check_output(
                [
                    docker_path,
                    "ps",
                    "-a",
                    "--no-trunc",
                    "--filter",
                    f"id={container_id}",
                    "--format",
                    "{{.ID}}",
                ],
                text=True,
                timeout=15,
            ).strip()
            != ""
        )

    async def scenario():
        tasks = InMemoryTaskStore()
        app, store = _app(tmp_path, docker_path, image, image_id, architecture, [], tasks)
        container_id = None
        try:
            checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint("abandoned")
            container_id = checkpoint["environment_factory_reconnect"]["coding"]["container_id"]
            await asyncio.sleep(1.2)
            [swept] = await app.recover_abandoned_sessions(inactive_for_seconds=0)
            assert IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT in swept.actions
            assert present(container_id), "The interrupted session still holds its container."

            if cleanup_interrupted:
                # Stop after the first pass, as a crash before the cleanup pass
                # would. Nothing may continue the session; a later sweep of a
                # session whose task is terminal finishes the cleanup.
                await tasks.create_task(
                    TaskCreate(task_id="job", type="job", session_id="abandoned")
                )
                await tasks.cancel_task("job")
                first = await app._session_recovery.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="abandoned", inactive_for_seconds=0
                    ),
                    abandon_unknown_tool_effects=True,
                )
                assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in first.actions
                assert present(container_id)
                with pytest.raises(SessionStatusConflict, match="abandoned session"):
                    async for _ in app.resume(
                        ResumeRequest(
                            session_id="abandoned", messages=[Message.text("user", "again")]
                        )
                    ):
                        pass
                [result] = await app.recover_abandoned_sessions(
                    inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
                )
            else:
                result = await app.abandon_session("abandoned")
                assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions
            assert result.status is SessionStatus.FAILED, result
            assert not present(container_id)
            events = await store.load_events("abandoned")
            assert any(
                event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED for event in events
            )
            assert any(
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload.get("phase") == "release"
                and event.payload.get("release_action") == "discard"
                for event in events
            )
            assert (
                await app.recover_abandoned_sessions(
                    inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
                )
                == ()
            )
        finally:
            await store.close()
            if container_id is not None and present(container_id):
                subprocess.run([docker_path, "rm", "-f", container_id], capture_output=True)

    asyncio.run(asyncio.wait_for(scenario(), timeout=120))
