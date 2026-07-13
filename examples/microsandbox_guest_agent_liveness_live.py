from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from typing import Any

from cayu import ExecCommand, MicrosandboxRunner, MicrosandboxUnavailableError


async def main() -> None:
    import microsandbox  # ty: ignore[unresolved-import]

    name = f"cayu-agent-liveness-{uuid.uuid4().hex[:10]}"
    started_path = f"/tmp/{name}.started"
    memory_mib = int(os.environ.get("CAYU_MICROSANDBOX_LIVENESS_MEMORY_MIB", "256"))
    if memory_mib < 128:
        raise ValueError("CAYU_MICROSANDBOX_LIVENESS_MEMORY_MIB must be at least 128.")
    runner: MicrosandboxRunner | None = None
    command_task: asyncio.Task[Any] | None = None
    try:
        runner = await MicrosandboxRunner.create(
            name,
            image=os.environ.get("CAYU_MICROSANDBOX_IMAGE", "alpine"),
            memory=memory_mib,
            close_action="remove",
            replace=True,
        )
        command_task = asyncio.create_task(
            runner.exec(ExecCommand.bash(f"touch {started_path}\nexec sleep 30"))
        )
        await _wait_for_command_start(runner.filesystem(), started_path, command_task)
        handle = await microsandbox.Sandbox.get(name)
        await handle.kill()
        try:
            result = await command_task
        except MicrosandboxUnavailableError as exc:
            _assert_unavailable_contract(exc)
            try:
                await runner.exec(ExecCommand.process("true"))
            except MicrosandboxUnavailableError:
                pass
            else:
                raise AssertionError("Confirmed-dead runner accepted another command.") from exc
            outcome = "agent-unavailable"
        else:
            if result.exit_code != -9 or result.timed_out:
                raise AssertionError(
                    "Expected a typed unavailable result after killing the microVM, "
                    f"got {result.model_dump(mode='json')}."
                )
            raise AssertionError("Killed microVM still passed its guest-agent ping.")
        print(f"sandbox={name} memory_mib={memory_mib} outcome={outcome}")
    finally:
        if command_task is not None and not command_task.done():
            command_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await command_task
        await _remove_sandbox(name, runner)


def _assert_unavailable_contract(exc: MicrosandboxUnavailableError) -> None:
    diagnostic = exc.diagnostic
    if diagnostic.get("reason") not in {
        "guest_agent_unavailable_after_signal_9",
        "guest_agent_unavailable_after_incomplete_exec",
    }:
        raise AssertionError(f"Unexpected unavailable diagnostic: {diagnostic}")
    if "oom" in str(diagnostic).lower():
        raise AssertionError("The diagnostic must not infer OOM from signal 9.")


async def _wait_for_command_start(fs: Any, path: str, command_task: asyncio.Task[Any]) -> None:
    try:
        async with asyncio.timeout(5.0):
            while True:
                if command_task.done():
                    await command_task
                    raise AssertionError(
                        "Guest command exited before its start marker was observed."
                    )
                if await fs.exists(path):
                    return
                await asyncio.sleep(0.05)
    except TimeoutError as exc:
        raise RuntimeError("Timed out waiting for the guest command start marker.") from exc


async def _remove_sandbox(name: str, runner: MicrosandboxRunner | None) -> None:
    import microsandbox  # ty: ignore[unresolved-import]

    if runner is not None:
        with contextlib.suppress(Exception):
            await runner.close()
    for attempt in range(20):
        try:
            handle = await microsandbox.Sandbox.get(name)
            if str(handle.status) not in {"stopped", "crashed"}:
                try:
                    sandbox: Any = await handle.connect()
                except microsandbox.SandboxNotRunningError:
                    pass
                else:
                    with contextlib.suppress(Exception):
                        await sandbox.stop_and_wait()
            await microsandbox.Sandbox.remove(name)
            return
        except microsandbox.SandboxNotFoundError:
            return
        except microsandbox.SandboxStillRunningError:
            if attempt == 19:
                raise
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
