"""Whole-process sandbox launchers; SDKs stay in the trusted service."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar
from uuid import uuid4

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu.external_agents.base import ExternalAgentLaunch
from cayu.runners.docker import DockerRunner

_T = TypeVar("_T")


async def join_cleanup(operation: Awaitable[_T]) -> _T | None:
    """Finish owned cleanup through repeated cancellation and retain its outcome."""
    outcome = await await_shielded_task_outcome(asyncio.ensure_future(operation))
    restore_task_cancellation_requests(
        outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
    )
    if outcome.error is not None:
        raise outcome.error
    if outcome.cancellation is not None:
        raise outcome.cancellation
    return outcome.result


def scrub_external_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Override SDK inheritance without changing this process's environment."""
    return {**dict.fromkeys(os.environ, ""), **environment}


async def _require_internal_network(docker: str, network: str) -> None:
    process = await asyncio.create_subprocess_exec(
        docker,
        "network",
        "inspect",
        "--format",
        "{{json .Internal}}",
        network,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(15):
            stdout, _ = await process.communicate()
        if process.returncode != 0 or stdout.strip() != b"true":
            raise ValueError("External agents require an existing Docker --internal network.")
    finally:
        if process.returncode is None:
            process.kill()
            await join_cleanup(process.wait())


@asynccontextmanager
async def docker_agent_launch(
    *,
    image: str,
    cli: str,
    workspace: Path,
    network: str,
    environment: Mapping[str, str],
    allocation_identity: str | None = None,
    workspace_read_only: bool = False,
) -> AsyncIterator[ExternalAgentLaunch]:
    """Allocate one CLI container and tunnel SDK stdio through ``docker exec``.

    ``network`` must be a deployment-owned internal Docker network whose only
    egress is the service proxy. CLI images contain their SDK-matching binary;
    no host home, Docker socket, or service credential is mounted. The checkout
    is the sole host mount. Container removal joins all guest descendants.
    """
    if not network or network in {"host", "bridge", "default"}:
        raise ValueError("An isolated, deployment-owned Docker network is required.")
    if not Path(cli).is_absolute():
        raise ValueError("The guest CLI path must be absolute.")
    root = workspace.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("External workspace must be a prepared directory.")
    forbidden = {"ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY"}
    if forbidden.intersection(environment):
        raise ValueError("Model credentials must stay in the service proxy.")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in environment):
        raise ValueError("External environment keys must be portable variable names.")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required for the external agent launcher.")
    await _require_internal_network(docker, network)
    runner = await DockerRunner.create(
        f"cayu-external-{uuid4().hex}",
        image=image,
        mount_path=str(root),
        mount_read_only=workspace_read_only,
        network=network,
        replace=False,
        close_action="remove",
        allocation_identity=allocation_identity,
    )
    try:
        if runner.container_id is None:
            raise RuntimeError("Docker did not return an immutable container identity.")
        with TemporaryDirectory(prefix="cayu-external-") as directory:
            launcher = Path(directory) / "cli"
            command = [
                runner.docker_path,
                "exec",
                "-i",
                "--user",
                f"{root.stat().st_uid}:{root.stat().st_gid}",
                "--workdir",
                str(root),
                *[argument for key in environment for argument in ("-e", key)],
                runner.container_id,
                "sh",
                "-c",
                "mkdir -p /tmp/cayu-agent && exec env -i "
                "PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/cayu-agent "
                + " ".join(f'{key}="${{{key}}}"' for key in environment)
                + ' "$@"',
                "cayu-agent",
                cli,
            ]
            launcher.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                f"command = {command!r}\n"
                "os.execve(command[0], command + sys.argv[1:], "
                + repr(
                    {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}
                )
                + f" | {{key: os.environ[key] for key in {tuple(environment)!r}}})\n",
                encoding="utf-8",
            )
            launcher.chmod(0o700)
            yield ExternalAgentLaunch(
                str(launcher),
                str(root),
                runner.close,
                {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": directory, **environment},
                tuple(
                    value
                    for key, value in environment.items()
                    if any(word in key.upper() for word in ("KEY", "TOKEN", "AUTHORIZATION"))
                ),
            )
    finally:
        from cayu.external_agents._cleanup import close_external_guest

        await close_external_guest(runner.close)


def codex_proxy_overrides(base_url: str) -> tuple[str, ...]:
    """A credential-free custom provider, backed by an authenticating service proxy."""
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Codex requires a service model proxy URL.")
    return (
        'model_provider="cayu"',
        'model_providers.cayu.name="Cayu model proxy"',
        f"model_providers.cayu.base_url={json.dumps(base_url)}",
        'model_providers.cayu.wire_api="responses"',
        "model_providers.cayu.requires_openai_auth=false",
        'model_providers.cayu.env_key="CAYU_MODEL_PROXY_TOKEN"',
        "project_doc_max_bytes=0",
        'web_search="disabled"',
    )
