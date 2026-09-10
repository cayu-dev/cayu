"""Local driver checks; these are not Docker or Microsandbox qualification."""

from __future__ import annotations

import asyncio
import shutil
import sys
from types import SimpleNamespace

import pytest
from tests.runners import test_tool_admission_live as live_driver
from tests.runners.test_tool_admission_live import _configured, _disable_docker_pulls, _qualify

import cayu.runners.docker as docker_module
from cayu import LocalRunner


@pytest.mark.parametrize("proof", ["present", "missing"])
def test_live_docker_driver_passes_real_creation_validation(monkeypatch, proof):
    image_id = "sha256:" + "a" * 64
    monkeypatch.setenv("CAYU_RUN_TOOL_ADMISSION_LIVE", "1")
    monkeypatch.setenv(f"CAYU_860_DOCKER_{proof.upper()}_IMAGE_ID", image_id)
    monkeypatch.setattr(live_driver.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        live_driver.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=image_id),
    )
    calls = []

    class DispatchReached(Exception):
        pass

    async def dispatch(path, args, **kwargs):
        calls.append(args)
        raise DispatchReached("Validated creation reached Docker dispatch")

    monkeypatch.setattr(docker_module, "_run_docker", dispatch)
    with pytest.raises(DispatchReached):
        live_driver.test_public_search_text_admission_in_live_docker(monkeypatch, proof)
    assert calls[0][:2] == ["run", "--pull=never"]


@pytest.mark.parametrize("proof", ["present", "missing"])
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable symlink fixture")
def test_live_qualification_driver_exercises_real_local_search(tmp_path, monkeypatch, proof):
    rg = shutil.which("rg")
    if proof == "present" and rg is None:
        pytest.skip("Local driver validation requires ripgrep.")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python3").symlink_to(sys.executable)
    if proof == "present":
        (binaries / "rg").symlink_to(rg)
    monkeypatch.setenv("PATH", str(binaries))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    asyncio.run(_qualify(LocalRunner(workspace, inherit_env=False), proof=proof))


@pytest.mark.parametrize("backend", ["docker", "microsandbox"])
def test_opted_in_qualification_requires_explicit_target(monkeypatch, backend):
    monkeypatch.setenv("CAYU_RUN_TOOL_ADMISSION_LIVE", "1")
    suffix = "IMAGE_ID" if backend == "docker" else "SANDBOX"
    monkeypatch.delenv(f"CAYU_860_{backend.upper()}_PRESENT_{suffix}", raising=False)
    with pytest.raises(pytest.fail.Exception, match="no target is prepared automatically"):
        _configured(backend, "present")


def test_live_docker_driver_forbids_implicit_pull_without_replacing_dispatch(monkeypatch):
    calls = []

    async def dispatch(path, args, **kwargs):
        calls.append((path, args, kwargs))
        return "original-result"

    monkeypatch.setattr(docker_module, "_run_docker", dispatch)
    _disable_docker_pulls(monkeypatch)

    async def run():
        for operation in ("run", "exec"):
            assert (
                await docker_module._run_docker(
                    "docker", [operation, "target"], docker_cli_env_allowlist=()
                )
                == "original-result"
            )

    asyncio.run(run())
    assert calls == [
        ("docker", ["run", "--pull=never", "target"], {"docker_cli_env_allowlist": ()}),
        ("docker", ["exec", "target"], {"docker_cli_env_allowlist": ()}),
    ]
