from __future__ import annotations

import asyncio

import pytest

from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.docker import DockerRunner


@pytest.mark.anyio
@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("signal", ["cancel", "timeout"])
async def test_none_cleanup_requires_start_evidence(monkeypatch, started, signal):
    accepted = asyncio.Event()
    release = asyncio.Event()
    effects = []

    async def remote():
        await release.wait()
        effects.append("original")

    remote_task = asyncio.create_task(remote())

    async def dispatch(*args, **kwargs):
        accepted.set()
        if signal == "timeout":
            try:
                async with asyncio.timeout(0.02):
                    await asyncio.shield(remote_task)
            except TimeoutError:
                return ExecResult(timed_out=True)
        await asyncio.shield(remote_task)
        return ExecResult()

    async def probe(*args, **kwargs):
        return ExecResult(exit_code=0 if started else 1)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", dispatch)
    monkeypatch.setattr("cayu.runners.docker._run_docker", probe)
    runner = DockerRunner(
        "owned", docker_path="/usr/bin/docker", cancellation_cleanup="none", timeout_cleanup="none"
    )
    task = asyncio.create_task(runner.exec(ExecCommand.process("true"), timeout_s=1))
    try:
        await accepted.wait()
        if signal == "cancel":
            task.cancel("caller cancelled")
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
        else:
            assert (await task).timed_out
        assert runner.lifecycle_state == ("reusable" if started else "poisoned")
        if not started:
            with pytest.raises(RuntimeError):
                runner.reopen_exec()
            with pytest.raises(RuntimeError):
                await runner.exec(ExecCommand.process("true"))
        assert effects == []
        release.set()
        await remote_task
        assert effects == ["original"]
        if not started:
            assert runner.lifecycle_state == "poisoned"
    finally:
        release.set()
        await asyncio.gather(task, remote_task, return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_setup", [False, True])
async def test_constructor_rollback_owns_exact_container(monkeypatch, cancel_setup):
    setup = asyncio.Event()
    removing = asyncio.Event()
    release = asyncio.Event()
    container_id = "a" * 64
    removals = []

    async def docker(path, args, **kwargs):
        if args[0] == "run":
            return ExecResult(stdout=container_id)
        if args[0] == "exec":
            setup.set()
            if cancel_setup:
                await asyncio.Event().wait()
            raise RuntimeError("setup failed")
        if args[:2] == ["rm", "-f"]:
            removals.append(args[2])
            removing.set()
            await release.wait()
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker._run_docker", docker)
    task = asyncio.create_task(
        DockerRunner.create(
            "replaceable-name", replace=False, docker_path="/usr/bin/docker", cancel_timeout_s=0.03
        )
    )
    try:
        await setup.wait()
        if cancel_setup:
            task.cancel("setup cancellation")
        await removing.wait()
        if cancel_setup:
            task.cancel("cleanup cancellation")
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 2
        else:
            with pytest.raises(ExceptionGroup):
                await task
        assert await DockerRunner.drain_failed_creations(timeout_s=0.01) == 1
        assert removals == [container_id]
        release.set()
        assert await DockerRunner.drain_failed_creations(timeout_s=1) == 0
        assert removals == [container_id]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await DockerRunner.drain_failed_creations(timeout_s=1)
