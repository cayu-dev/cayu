"""Docker coding containers of a process that exits without aclose().

Each case runs an application with an in-memory session store in a child
process. A release that fails is retried nowhere once that process ends, so the
interpreter-exit hook must remove the container; a killed process runs no hook,
so its container must carry the process-local owner label that lets the orphan
reaper remove it. The reaper here is restricted to the exact container the case
created, through its exact owner label.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from tests.environments.test_docker_coding_live import _configuration_or_skip

from cayu.environments import reap_orphaned_docker_coding_containers

pytestmark = pytest.mark.process

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CHILD = textwrap.dedent(
    """
    import asyncio, os, signal, subprocess, sys
    from pathlib import Path

    from tests.docker_toolchain import docker_toolchain_profile

    from cayu import AgentSpec, CayuApp, Message, RunRequest, Tool, ToolResult, ToolSpec
    from cayu import run_to_completion
    from cayu.environments import DockerCodingEnvironmentFactory, EnvironmentSpec
    from cayu.environments import docker_coding
    from cayu.evals import ScriptedModelProvider
    from cayu.providers import ModelStreamEvent
    from cayu.runners import DockerImageIdentity
    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
    from cayu.sessions import InMemorySessionStore
    from cayu.workspaces import LocalWorkspace

    mode, docker_path, image, image_id, architecture, root = sys.argv[1:]
    root = Path(root)
    source = root / "source"
    source.mkdir()
    (source / "README.md").write_text("hello\\n")
    identity = ExecutionProfileBehaviorIdentity(
        name="tests:orphan-cleanup", behavior_version="1", implementation_version="1"
    )

    class Factory(DockerCodingEnvironmentFactory):
        async def create_recoverable(self, request, allocation):
            result = await super().create_recoverable(request, allocation)
            (root / "container").write_text(result.metadata["container_id"])
            return result

    class Pause(Tool):
        spec = ToolSpec(
            name="pause",
            description="Wait inside the session.",
            input_schema={"type": "object", "properties": {}},
            execution_profile_identity=identity,
        )

        async def run(self, ctx, args):
            os.kill(os.getpid(), signal.SIGKILL)
            return ToolResult(content="unreachable")

    if mode == "release-fails":
        async def failing_finalize(self, bound, *, outcome=None, metadata=None):
            raise RuntimeError("injected copy-back failure")

        docker_coding.DockerCodingWorkspaceBinding.finalize = failing_finalize

    killed = mode == "killed"
    script = [
        ModelStreamEvent.tool_call(id="call", name="pause", arguments={})
        if killed
        else ModelStreamEvent.text_delta("done"),
        ModelStreamEvent.completed({"finish_reason": "tool_calls" if killed else "stop"}),
    ]
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(ScriptedModelProvider([script]), default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[Pause()])
    app.register_environment_factory(
        EnvironmentSpec(name="coding", execution_profile_identity=identity),
        Factory(
            source_workspace=LocalWorkspace(source, workspace_id="orphan-cleanup"),
            toolchain_profile=docker_toolchain_profile(
                image_identity=DockerImageIdentity(reference=image, content_digest=image_id),
                platform_architecture=architecture,
            ),
            docker_path=docker_path,
        ),
        default=True,
    )
    outcome = asyncio.run(
        run_to_completion(
            app, RunRequest(agent_name="worker", messages=[Message.text("user", "go")])
        )
    )
    print("status", outcome.status)
    # No app.aclose(): the process exits with the failed release still held.
    """
)


def _run_child(mode: str, root: Path) -> tuple[str, str, subprocess.CompletedProcess[str]]:
    docker_path, image, image_id = _configuration_or_skip()
    architecture = subprocess.check_output(
        [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image], text=True
    ).strip()
    script = root / "child.py"
    script.write_text(_CHILD)
    work = root / "work"
    work.mkdir()
    completed = subprocess.run(
        [sys.executable, str(script), mode, docker_path, image, image_id, architecture, str(work)],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    container_file = work / "container"
    assert container_file.exists(), completed.stdout + completed.stderr
    return docker_path, container_file.read_text().strip(), completed


def _labels(docker_path: str, container_id: str) -> dict[str, str] | None:
    inspected = subprocess.run(
        [docker_path, "container", "inspect", "--format", "{{json .Config.Labels}}", container_id],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if inspected.returncode != 0:
        return None
    return json.loads(inspected.stdout)


def test_failed_release_is_removed_when_the_process_exits(tmp_path):
    docker_path, container_id, completed = _run_child("release-fails", tmp_path)
    try:
        assert completed.returncode == 0, completed.stderr
        assert "status failed" in completed.stdout
        assert _labels(docker_path, container_id) is None, "container outlived its process"
    finally:
        subprocess.run([docker_path, "rm", "-f", container_id], capture_output=True, timeout=60)


def test_killed_process_container_is_reaped_once_its_owner_is_gone(tmp_path):
    docker_path, container_id, completed = _run_child("killed", tmp_path)
    try:
        assert completed.returncode == -9, completed.stderr
        labels = _labels(docker_path, container_id)
        assert labels is not None, "a killed process runs no exit hook"
        owner = labels["io.cayu.process-local-owner"]
        assert re.fullmatch(r"v1:[0-9a-f-]+:(host|[0-9]+):[1-9][0-9]*:[0-9]+", owner)
        removed = asyncio.run(
            reap_orphaned_docker_coding_containers(
                docker_path=docker_path, label=f"io.cayu.process-local-owner={owner}"
            )
        )
        assert removed == (container_id,)
        assert _labels(docker_path, container_id) is None
    finally:
        subprocess.run([docker_path, "rm", "-f", container_id], capture_output=True, timeout=60)
