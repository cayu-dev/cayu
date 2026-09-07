from __future__ import annotations

import asyncio
import os
import sys
import threading

import pytest

from cayu import ExecCommand, LocalRunner
from cayu.runners import _subprocess as subprocess_module
from cayu.runners.base import runner_workspace_mutation_settlement

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process group evidence")


@pytest.mark.parametrize("fail_stream", [False, True])
def test_local_cleanup_waits_for_delegated_stream_after_repeated_cancellation(
    tmp_path, fail_stream
):
    started = threading.Event()
    release = threading.Event()

    class BlockingInput:
        def read(self, size):
            started.set()
            if not release.wait(10):
                raise TimeoutError("test stream not released")
            if fail_stream:
                raise OSError("injected stream failure")
            return b""

    async def run():
        runner = LocalRunner(tmp_path, inherit_env=False)
        task = asyncio.create_task(
            runner.exec_stream(
                ExecCommand.process(sys.executable, "-c", "import time; time.sleep(120)"),
                stdin=BlockingInput(),
            )
        )
        try:
            async with asyncio.timeout(10):
                while not started.is_set():
                    await asyncio.sleep(0.01)
                task.cancel("first cancellation")
                await asyncio.sleep(0.05)
                task.cancel("second cancellation")
                await asyncio.sleep(0.05)
                assert not task.done()
                release.set()
                with pytest.raises(asyncio.CancelledError, match="first cancellation") as caught:
                    await task
            artifact = caught.value.artifacts[0]
            assert artifact["adapter"] == "local"
            assert artifact["status"] == ("failed" if fail_stream else "completed")
            assert runner_workspace_mutation_settlement(result=None, error=caught.value) == (
                "uncertain" if fail_stream else "complete"
            )
            if fail_stream:
                with pytest.raises(RuntimeError, match="cleanup could not be confirmed"):
                    await runner.exec(ExecCommand.process(sys.executable, "-c", "pass"))
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await runner.close()

    asyncio.run(run())


@pytest.mark.parametrize("group_gone", [False, True])
def test_child_reaped_does_not_prove_owned_group_gone(monkeypatch, group_gone):
    class Process:
        pid = 123
        returncode = -9

    async def kill(process, *, process_group):
        return True

    def probe_group(pid, sig):
        assert (pid, sig) == (123, 0)
        if group_gone:
            raise ProcessLookupError

    monkeypatch.setattr(subprocess_module, "_kill_process", kill)
    monkeypatch.setattr(subprocess_module.os, "killpg", probe_group)

    async def run():
        async def waited():
            return -9

        waiter = asyncio.create_task(waited())
        return await subprocess_module._kill_process_and_wait(
            Process(),
            process_group=True,
            wait_task=waiter,
        )

    assert asyncio.run(run()) is group_gone
