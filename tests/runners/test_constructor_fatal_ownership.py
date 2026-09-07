from __future__ import annotations

import asyncio

import pytest
from tests.runners.test_e2b import (
    FakeAsyncSandbox,
    FakeE2BModule,
    reset_fake_e2b,
)
from tests.runners.test_e2b import (
    FakeSandbox as E2BSandbox,
)
from tests.runners.test_microsandbox import (
    FakeMicrosandboxModule,
    FakeSandboxApi,
    reset_fake_module,
)
from tests.runners.test_microsandbox import (
    FakeSandbox as MicroSandbox,
)

from cayu.runners import DockerRunner, E2BRunner, ExecResult, MicrosandboxRunner


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", ["docker", "e2b", "microsandbox"])
@pytest.mark.parametrize("cleanup_state", ["success", "failed", "pending"])
@pytest.mark.parametrize("grouped", [False, True])
async def test_constructor_fatal_setup_survives_rollback_cancellation(
    monkeypatch, adapter, cleanup_state, grouped
):
    class FatalSetupSignal(BaseException):
        pass

    original = (
        BaseExceptionGroup("fatal setup", [SystemExit(7), KeyboardInterrupt()])
        if grouped
        else FatalSetupSignal("fatal setup")
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    cleanup_calls = []
    repaired = False
    container_id = "b" * 64

    async def setup(*args, **kwargs):
        raise original

    async def cleanup(*args, **kwargs):
        cleanup_calls.append(args)
        entered.set()
        await release.wait()
        if cleanup_state == "failed" and not repaired:
            raise PermissionError("rollback denied")
        return True

    if adapter == "docker":

        async def docker(path, args, **kwargs):
            if args[0] == "run":
                return ExecResult(stdout=container_id)
            if args[0] == "exec":
                await setup()
            assert args == ["rm", "-f", container_id]
            await cleanup(container_id)
            return ExecResult()

        monkeypatch.setattr("cayu.runners.docker._run_docker", docker)
        creation = DockerRunner.create(
            "fatal-setup", replace=False, docker_path="docker", cancel_timeout_s=0.1
        )
        runner_type = DockerRunner
    elif adapter == "e2b":
        reset_fake_e2b()
        sandbox = E2BSandbox()
        FakeAsyncSandbox.next_sandbox = sandbox
        sandbox.commands.run = setup
        sandbox.kill = cleanup
        creation = E2BRunner.create(e2b_module=FakeE2BModule, cancel_timeout_s=0.1)
        runner_type = E2BRunner
    else:
        reset_fake_module()
        monkeypatch.setattr(MicroSandbox, "exec", setup)
        monkeypatch.setattr(FakeSandboxApi, "remove", classmethod(cleanup))
        creation = MicrosandboxRunner.create(
            "fatal-setup",
            replace=False,
            sandbox_module=FakeMicrosandboxModule,
            cancel_timeout_s=0.1,
            remove_timeout_s=0.1,
        )
        runner_type = MicrosandboxRunner

    task = asyncio.create_task(creation)
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel("cancel during rollback")
        await asyncio.sleep(0)
        assert not task.done()
        if cleanup_state != "pending":
            release.set()
        with pytest.raises(BaseExceptionGroup) as caught:
            await task
        errors = caught.value.exceptions
        assert errors[0] is original
        assert len(errors) == (2 if cleanup_state == "success" else 3)
        assert isinstance(errors[-1], asyncio.CancelledError)
        assert str(errors[-1]) == "cancel during rollback"
        assert not task.cancelled() and task.cancelling() == 1
        if cleanup_state == "pending":
            assert isinstance(errors[1], TimeoutError)
            assert await runner_type.drain_failed_creations(timeout_s=0.01) == 1
            assert len(cleanup_calls) == 1
            if adapter == "microsandbox":
                with pytest.raises(RuntimeError, match="pending"):
                    await MicrosandboxRunner.create(
                        "fatal-setup", sandbox_module=FakeMicrosandboxModule
                    )
        elif cleanup_state == "failed":
            assert isinstance(errors[1], Exception)
            assert await runner_type.drain_failed_creations(timeout_s=1) == 1
        repaired = True
        release.set()
        assert await runner_type.drain_failed_creations(timeout_s=1) == 0
        if cleanup_state != "failed":
            assert len(cleanup_calls) == 1
    finally:
        repaired = True
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await runner_type.drain_failed_creations(timeout_s=1) == 0
