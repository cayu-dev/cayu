"""One coding scenario gives the same tool results on Docker and on a Cloud-brokered sandbox.

The brokered run uses the fake broker and always runs. The Docker run builds a
small image and runs only with ``CAYU_RUN_DOCKER_TOOLCHAIN_LIVE=1``. Both must
produce ``_EXPECTED``; backend evidence (fingerprints, image identity, timing)
is left out of the comparison and stays backend-specific.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pytest
from tests.sandbox_broker.fake_broker import FakeBroker

from cayu import ToolContext
from cayu.environments import (
    CodingAdmissionProbe,
    CodingCommandAuthority,
    CodingDependencyInput,
    CodingToolchainProfile,
    DockerCodingEnvironmentFactory,
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
)
from cayu.runners import DockerImageIdentity
from cayu.sandbox_broker import BrokeredSandboxCodingEnvironmentFactory
from cayu.tools import (
    EditFileTool,
    ExactNamedCheckCommandPolicy,
    ReadFileTool,
    RunCheckTool,
    RunCommandTool,
    ToolResult,
    WriteFileTool,
    named_checks_from_toolchain,
)
from cayu.workspaces import LocalWorkspace

_LIVE_ENVIRONMENT = "CAYU_RUN_DOCKER_TOOLCHAIN_LIVE"
_PYTHON_BASE = (
    "python:3.12-slim@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17"
)
_LOCK = b"locked\n"
_CONTEXT = "sha256:" + "c" * 64
_TOOLS = {
    "check": "#!/bin/sh\nif grep -q 'VALUE = 2' calc.py; then echo pass; else echo fail; exit 1; fi\n",
    "show": "#!/bin/sh\ncat calc.py\n",
    "version": "#!/bin/sh\necho 'conformance 1'\n",
}


def _profile() -> CodingToolchainProfile:
    return CodingToolchainProfile(
        profile_id="conformance",
        revision="1",
        read_only_support_paths=("/opt/tools",),
        command_authorities=(
            CodingCommandAuthority(
                selector="check",
                revision="1",
                description="Check that calc.py holds the fixed value.",
                exposure="named_check",
                executable="/opt/tools/bin/check",
                max_arguments=0,
                timeout_seconds=20,
                max_output_bytes=4096,
            ),
            CodingCommandAuthority(
                selector="show",
                revision="1",
                description="Print calc.py.",
                exposure="structured_command",
                executable="/opt/tools/bin/show",
                max_arguments=0,
                timeout_seconds=20,
                max_output_bytes=4096,
            ),
            CodingCommandAuthority(
                selector="version",
                revision="1",
                description="Print the toolchain version.",
                exposure="structured_command",
                executable="/opt/tools/bin/version",
                max_arguments=0,
                timeout_seconds=20,
                max_output_bytes=4096,
            ),
        ),
        dependency_inputs=(
            CodingDependencyInput(
                path="uv.lock", content_sha256="sha256:" + hashlib.sha256(_LOCK).hexdigest()
            ),
        ),
        trusted_build_context_sha256=_CONTEXT,
        admission_probes=(
            CodingAdmissionProbe(
                probe_id="version",
                argv=("/opt/tools/bin/version",),
                stdout_sha256="sha256:" + hashlib.sha256(b"conformance 1\n").hexdigest(),
                timeout_seconds=20,
                max_output_bytes=4096,
            ),
        ),
    )


def _tools(root: Path) -> Path:
    """The toolchain as a directory: the brokered guest's /opt/tools, or the image's."""

    tools = root / "opt-tools"
    (tools / "bin").mkdir(parents=True)
    for name, script in _TOOLS.items():
        (tools / "bin" / name).write_text(script, encoding="utf-8")
        (tools / "bin" / name).chmod(0o755)
    return tools


def _source(tmp_path: Path) -> LocalWorkspace:
    source = tmp_path / "source"
    source.mkdir()
    (source / "uv.lock").write_bytes(_LOCK)
    (source / "calc.py").write_text("VALUE = 1\n", encoding="utf-8")
    return LocalWorkspace(source, workspace_id="project", excluded_directory_names=(".git",))


_EXPECTED = [
    {
        "is_error": False,
        "content": "Wrote 7 bytes to notes.txt.",
        "path": "notes.txt",
        "bytes": 7,
        "mode": "create",
    },
    {
        "is_error": False,
        "content": "Check 'check' failed with exit code 1.\n\nstdout:\nfail",
        "status": "failed",
        "exit_code": 1,
        "stdout": "fail\n",
        "stderr": "",
    },
    {
        "is_error": False,
        "content": (
            "[edit_file metadata]\n"
            '{"after_revision":"sha256:'
            '3da6c4b355804b265ecef19dfe5945fe2790a8cc38dc9d0547fb1634c94cbbe7"}\n'
            "[/edit_file metadata]\n"
            "Edited calc.py: 1 edit(s), 1 replacement(s), 10 -> 10 bytes.\n\n"
            "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
        ),
        "path": "calc.py",
    },
    {
        "is_error": False,
        "content": "Check 'check' passed.\n\nstdout:\npass",
        "status": "passed",
        "exit_code": 0,
        "stdout": "pass\n",
        "stderr": "",
    },
    {
        "is_error": False,
        "content": "Command selector 'show' succeeded.\n\nstdout:\nVALUE = 2",
        "status": "succeeded",
        "exit_code": 0,
        "stdout": "VALUE = 2\n",
        "stderr": "",
    },
]

# Structured fields that describe what happened rather than where it ran.
_COMPARED_FIELDS = ("status", "exit_code", "stdout", "stderr", "path", "bytes", "mode", "error")


def _comparable(result: ToolResult) -> dict[str, Any]:
    structured = result.structured or {}
    return {
        "is_error": result.is_error,
        "content": result.content.split("\nRevision: ")[0],
        **{name: structured[name] for name in _COMPARED_FIELDS if name in structured},
    }


async def _scenario(
    factory: EnvironmentFactory, profile: CodingToolchainProfile
) -> list[dict[str, Any]]:
    created = await factory.create(
        EnvironmentFactoryRequest(
            session_id="conformance", agent_name="coder", environment_name="coding"
        )
    )
    runner = created.environment.runner
    binding = created.environment.binding
    assert runner is not None and binding is not None
    try:
        bound = await binding.bind(
            created.environment.workspace,
            runner,
            session_id="conformance",
            agent_name="coder",
            environment_name="coding",
        )
        assert bound.workspace is not None
        ctx = ToolContext(
            session_id="conformance",
            agent_name="coder",
            environment_name="coding",
            workspace_id=bound.workspace.id,
            workspace=bound.workspace,
            runner=runner,
        )
        checks = named_checks_from_toolchain(profile)
        run_check = RunCheckTool(
            checks=checks,
            command_policy=ExactNamedCheckCommandPolicy(checks),
            toolchain_profile=profile,
        )
        run_command = RunCommandTool(toolchain_profile=profile)
        results = [
            await WriteFileTool().run(
                ctx, {"path": "notes.txt", "content": "edited\n", "mode": "create"}
            ),
            await run_check.run(ctx, {"check": "check"}),
        ]
        read = await ReadFileTool().run(ctx, {"path": "calc.py"})
        assert read.structured is not None
        results.append(
            await EditFileTool().run(
                ctx,
                {
                    "path": "calc.py",
                    "expected_revision": read.structured["revision"],
                    "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}],
                },
            )
        )
        results.append(await run_check.run(ctx, {"check": "check"}))
        results.append(await run_command.run(ctx, {"selector": "show"}))
        await binding.finalize(bound, outcome="completed")
        return [_comparable(result) for result in results]
    finally:
        if created.release is not None:
            await created.release(EnvironmentFactoryReleaseAction.DISCARD)


def test_brokered_sandbox_runs_the_conformance_scenario(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.catalog.add("release")
    broker.guest_paths["/opt/tools"] = _tools(tmp_path)
    broker.image_identities["release"] = {
        "name": "release",
        "content_digest": "sha256:" + "d" * 64,
        "trusted_build_context_sha256": _CONTEXT,
        "platform_architecture": "arm64",
    }
    profile = _profile()
    factory = BrokeredSandboxCodingEnvironmentFactory(
        broker=broker.client(),
        source_workspace=_source(tmp_path),
        egress="none",
        image="release",
        toolchain_profile=profile,
    )

    assert asyncio.run(_scenario(factory, profile)) == _EXPECTED
    assert broker.live() == []


def _require_live_docker() -> str:
    if os.environ.get(_LIVE_ENVIRONMENT) != "1":
        pytest.skip(f"set {_LIVE_ENVIRONMENT}=1 to run the Docker half of the conformance scenario")
    docker = shutil.which("docker")
    if docker is None:
        pytest.fail("Docker CLI is required by the opted-in conformance scenario")
    return docker


def test_docker_runs_the_conformance_scenario(tmp_path: Path) -> None:
    docker = _require_live_docker()
    context = tmp_path / "image"
    context.mkdir()
    shutil.copytree(_tools(tmp_path), context / "opt-tools")
    (context / "Dockerfile").write_text(
        f"FROM {_PYTHON_BASE}\n"
        # Docker coding records its baseline with git inside the container.
        "RUN apt-get update && apt-get install -y --no-install-recommends git"
        " && rm -rf /var/lib/apt/lists/*\n"
        "COPY opt-tools /opt/tools\n",
        encoding="utf-8",
    )
    tag = f"cayu-conformance:{uuid4().hex}"
    try:
        built = subprocess.run(
            [docker, "build", "--tag", tag, str(context)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=300,
            check=False,
        )
        assert built.returncode == 0, built.stderr.decode(errors="replace")[-2000:]
        inspected = subprocess.run(
            [docker, "image", "inspect", "--format", "{{.Id}} {{.Architecture}}", tag],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=20,
            check=True,
        )
        digest, architecture = inspected.stdout.decode().split()
        profile = _profile()
        factory = DockerCodingEnvironmentFactory(
            source_workspace=_source(tmp_path),
            toolchain_profile=profile,
            image_identity=DockerImageIdentity(reference=tag, content_digest=digest),
            platform_architecture=cast("Literal['amd64', 'arm64']", architecture),
            docker_path=docker,
        )

        assert asyncio.run(_scenario(factory, profile)) == _EXPECTED
    finally:
        subprocess.run(
            [docker, "image", "rm", "--force", tag],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            check=False,
        )
