"""Opt-in #860 qualification; never build images, pull, or install packages."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from uuid import uuid4

import pytest

import cayu.runners.docker as docker_module
from cayu import (
    AgentSpec,
    CayuApp,
    DockerImageIdentity,
    DockerRunner,
    DockerWorkloadRestrictions,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecCommand,
    Message,
    MicrosandboxRunner,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SearchTextTool,
    ToolExecutableRequirement,
)

pytestmark = pytest.mark.process


def _disable_docker_pulls(monkeypatch):
    dispatch = docker_module._run_docker

    async def dispatch_without_pull(docker_path, args, **kwargs):
        # A concurrent local image prune must not turn the earlier image
        # inspection into an implicit download. All actual CLI work and guest
        # evidence still pass through the real production dispatcher.
        if args and args[0] == "run":
            args = ["run", "--pull=never", *args[1:]]
        return await dispatch(docker_path, args, **kwargs)

    monkeypatch.setattr(docker_module, "_run_docker", dispatch_without_pull)


def _configured(backend: str, proof: str) -> str:
    if os.environ.get("CAYU_RUN_TOOL_ADMISSION_LIVE") != "1":
        pytest.skip("Set CAYU_RUN_TOOL_ADMISSION_LIVE=1 with prepared disposable targets.")
    suffix = "IMAGE_ID" if backend == "docker" else "SANDBOX"
    key = f"CAYU_860_{backend.upper()}_{proof.upper()}_{suffix}"
    value = os.environ.get(key)
    if not value:
        pytest.fail(f"Live qualification requires {key}; no target is prepared automatically.")
    return value


async def _qualify(runner, *, proof: str) -> None:
    directory = f"cayu-860-search-{uuid4().hex}"
    fixture = await runner.exec(
        ExecCommand.process(
            "python3",
            "-c",
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
            "(p/'example.txt').write_text('admission-needle\\n', encoding='utf-8')",
            directory,
        ),
        timeout_s=10,
        output_limit_bytes=1024,
    )
    assert fixture.exit_code == 0 and not fixture.timed_out
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="search",
                    name="search_text",
                    arguments={"pattern": "admission-needle", "path": directory, "mode": "content"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="live"), runner=runner), default=True)
    app.register_agent(AgentSpec(name="search", model="scripted"), tools=[SearchTextTool()])
    try:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="search",
                    session_id=f"live-{uuid4().hex}",
                    messages=[Message.text("user", "Search the prepared fixture.")],
                )
            )
        ]
        if proof == "present":
            assert len(provider.requests) == 2
            completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
            assert len(completed) == 1
            result = completed[0].payload["result"]
            assert result["is_error"] is False
            matches = result["structured"]["matches"]
            assert len(matches) == 1
            assert matches[0]["path"] == f"{directory}/example.txt"
            assert matches[0]["line"] == 1
            assert "admission-needle" in matches[0]["preview"]
            assert any(event.type is EventType.SESSION_COMPLETED for event in events)
        else:
            assert provider.requests == []
            assert not any(str(event.type).startswith("model.") for event in events)
            assert not any(event.type is EventType.SESSION_COMPLETED for event in events)
            failure = next(event for event in events if event.type is EventType.SESSION_FAILED)
            assert any(
                refusal["tool_name"] == "search_text" and refusal["executable"] == "rg"
                for refusal in failure.payload["execution_admission"]["refusals"]
            )
    finally:
        assert await app.drain_environment_cleanups(timeout_s=10)


@pytest.mark.parametrize("proof", ["present", "missing"])
def test_public_search_text_admission_in_live_docker(monkeypatch, proof):
    image_id = _configured("docker", proof)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image_id), "Use a local immutable Docker image ID."
    docker = shutil.which("docker")
    assert docker is not None, "Docker CLI is required for opted-in qualification."
    inspected = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image_id],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert inspected.returncode == 0 and inspected.stdout.strip() == image_id, (
        "The exact Docker image must already be loaded in the accessible local daemon."
    )
    _disable_docker_pulls(monkeypatch)

    async def run():
        async with await DockerRunner.create(
            f"cayu-860-live-{uuid4().hex[:12]}",
            image=image_id,
            image_identity=DockerImageIdentity(reference=image_id, content_digest=image_id),
            workload_restrictions=DockerWorkloadRestrictions(),
            executable_probes=(ToolExecutableRequirement(executable="rg"),),
            network="none",
            replace=False,
            close_action="remove",
            cancellation_cleanup="sandbox",
            timeout_cleanup="sandbox",
            credential_mode="trusted_tool",
            allow_raw_secret_env=False,
            docker_path=docker,
        ) as runner:
            await _qualify(runner, proof=proof)

    asyncio.run(run())


@pytest.mark.parametrize("proof", ["present", "missing"])
def test_public_search_text_admission_in_live_microsandbox(proof):
    name = _configured("microsandbox", proof)
    assert re.fullmatch(r"cayu-860-live-[a-z0-9-]+", name), (
        "Supply an explicitly disposable running sandbox named cayu-860-live-*."
    )
    # Import failure is a failed qualification after opt-in, not a silent skip.
    import microsandbox

    async def run():
        handle = await microsandbox.Sandbox.get(name)
        assert type(handle.created_at) in {int, float} and handle.created_at > 0
        async with await MicrosandboxRunner.from_existing(
            name,
            expected_created_at=handle.created_at,
            close_action="remove",
            default_cwd="/workspace",
        ) as runner:
            await _qualify(runner, proof=proof)

    asyncio.run(run())
