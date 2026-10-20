"""Managed egress runners forward interactive processes under their dispatch fences."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cayu.egress.runtime import _EgressManagedRunner
from cayu.runners import (
    ExecCommand,
    ExecResult,
    RunnerInteractiveProcess,
    RunnerInteractiveProcessCapability,
)


class _Process(RunnerInteractiveProcess):
    def __init__(self) -> None:
        self.closes = 0
        self.fail_close = False

    @property
    def secret_values(self) -> tuple[str, ...]:
        return ("inner-process-secret",)

    async def write(self, data: bytes) -> None:
        return None

    async def close_stdin(self) -> None:
        return None

    async def read(self) -> bytes:
        return b""

    async def wait(self) -> int:
        return 0

    async def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("unconfirmed")
        self.closes += 1


class _Capability(RunnerInteractiveProcessCapability):
    def __init__(self) -> None:
        self.process = _Process()
        self.fail_open = False

    async def open_interactive_process(self, command, *, cwd=None, env=None, timeout_s=None):
        if self.fail_open:
            raise RuntimeError("start refused")
        return self.process


class _Dispatches:
    def __init__(self) -> None:
        self.active: set[int] = set()
        self.finished: list[tuple[int, ExecResult | None, BaseException | None]] = []

    def begin(self) -> int:
        dispatch_id = len(self.finished) + len(self.active) + 1
        self.active.add(dispatch_id)
        return dispatch_id

    def finish(self, *, dispatch_id, result=None, error=None) -> None:
        self.active.remove(dispatch_id)
        self.finished.append((dispatch_id, result, error))


def _managed(capability: RunnerInteractiveProcessCapability | None):
    dispatches = _Dispatches()
    runner = object.__new__(_EgressManagedRunner)
    runner._runner = SimpleNamespace(interactive_process_capability=lambda: capability)
    runner._egress_grants = (SimpleNamespace(presented_value="cayu_vc_presented"),)
    runner._begin_workspace_dispatch = dispatches.begin
    runner._finish_workspace_dispatch = dispatches.finish
    return runner, dispatches


def test_open_process_holds_a_workspace_dispatch_until_confirmed_closed() -> None:
    inner = _Capability()
    runner, dispatches = _managed(inner)

    async def scenario() -> None:
        capability = runner.interactive_process_capability()
        assert capability.secret_values == ("cayu_vc_presented",)
        process = await capability.open_interactive_process(ExecCommand.process("cli"))
        assert set(process.secret_values) == {"cayu_vc_presented", "inner-process-secret"}
        assert dispatches.active == {1}

        inner.process.fail_close = True
        with pytest.raises(RuntimeError, match="unconfirmed"):
            await process.close()
        # Unconfirmed: the dispatch settles as uncertain, the stop is retried.
        [(_, result, error)] = dispatches.finished
        assert result is None and isinstance(error, RuntimeError)
        inner.process.fail_close = False
        await process.close()
        assert inner.process.closes == 1
        assert len(dispatches.finished) == 1

    asyncio.run(scenario())


def test_confirmed_close_settles_the_dispatch_as_complete() -> None:
    inner = _Capability()
    runner, dispatches = _managed(inner)

    async def scenario() -> None:
        process = await runner.interactive_process_capability().open_interactive_process(
            ExecCommand.process("cli")
        )
        await process.close()
        [(_, result, error)] = dispatches.finished
        assert error is None and result.exit_code == 0 and not result.cancelled

    asyncio.run(scenario())


def test_failed_open_releases_the_dispatch() -> None:
    inner = _Capability()
    inner.fail_open = True
    runner, dispatches = _managed(inner)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="start refused"):
            await runner.interactive_process_capability().open_interactive_process(
                ExecCommand.process("cli")
            )
        assert dispatches.active == set()

    asyncio.run(scenario())


def test_runner_without_interactive_processes_reports_none() -> None:
    runner, _ = _managed(None)

    assert runner.interactive_process_capability() is None


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_start_retains_egress_dispatch_until_cleanup(cancelled):
    error = asyncio.CancelledError("cancelled") if cancelled else RuntimeError("lost start")

    class PendingCapability(_Capability):
        async def open_interactive_process(self, *args, **kwargs):
            self.retain_failed_start(error, self.process)
            raise error

    inner = PendingCapability()
    runner, dispatches = _managed(inner)

    async def scenario():
        capability = runner.interactive_process_capability()
        with pytest.raises(type(error)) as raised:
            await capability.open_interactive_process(ExecCommand.process("cli"))
        assert raised.value is error
        assert dispatches.active == {1}
        pending = capability.failed_start_process(error)
        assert pending is not inner.process
        await pending.close()
        assert not dispatches.active
        [(_, result, failure)] = dispatches.finished
        assert failure is None and result.exit_code == 0

    asyncio.run(scenario())
