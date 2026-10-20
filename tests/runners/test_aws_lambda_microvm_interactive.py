"""LambdaMicroVMRunner interactive processes over sidecar protocol 6 (fake AWS, real supervisor)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.runners.lambda_microvm_harness import ConformanceLambdaClient, SupervisorTransport

from cayu.runners import (
    ExecCommand,
    LambdaMicroVMOwnershipSuperseded,
    LambdaMicroVMRunner,
    RunnerInteractiveProcessCapability,
)
from cayu.runners.aws_lambda_microvm import (
    LambdaMicroVMEndpointTransientError,
    LambdaMicroVMError,
    LambdaMicroVMProtocolError,
)

_ECHO_UPPER = (
    "import sys\n"
    "for line in sys.stdin:\n"
    "    sys.stdout.write(line.upper())\n"
    "    sys.stdout.flush()\n"
    "sys.exit(5)\n"
)


class InteractiveTransport(SupervisorTransport):
    """The real supervisor's stdio protocol behind the sidecar's owner fence.

    Hosts without Linux PID namespaces run commands in a process group, which
    the supervisor truthfully reports as unconfirmed. ``isolated`` stands in for
    the guest kernel's guarantee by reporting the namespace the first-party
    image provides.
    """

    def __init__(self, root: Path, *, isolated: bool = True) -> None:
        super().__init__(root)
        self.isolated = isolated
        self.stdin_calls: list[tuple[int, bytes]] = []
        self.fail_next_write = 0

    def _isolate(self, response: dict[str, Any]) -> dict[str, Any]:
        if self.isolated and response.get("interactive"):
            response = dict(response)
            response["process_tree"] = "pid_namespace"
            if "process_tree_stopped" in response:
                response["process_tree_stopped"] = True
        return response

    async def start_command(self, **kwargs: Any) -> dict[str, Any]:
        return self._isolate(await super().start_command(**kwargs))

    async def get_command(self, **kwargs: Any) -> dict[str, Any]:
        return self._isolate(await super().get_command(**kwargs))

    async def cancel_command(self, **kwargs: Any) -> dict[str, Any]:
        return self._isolate(await super().cancel_command(**kwargs))

    async def write_command_stdin(
        self, *, command_id: str, owner_claim: str, offset: int, data_base64: str, eof: bool, **_
    ) -> dict[str, Any]:
        self.admit_start({"owner_claim": owner_claim})
        self.stdin_calls.append((offset, data_base64.encode()))
        response = await asyncio.to_thread(
            self.supervisor.write_stdin,
            command_id,
            offset=offset,
            data_base64=data_base64,
            eof=eof,
        )
        if self.fail_next_write:
            # The sidecar applied the write but the response was lost.
            self.fail_next_write -= 1
            raise LambdaMicroVMEndpointTransientError("lost response")
        return response

    async def read_command_stdout(
        self, *, command_id: str, owner_claim: str, offset: int, max_bytes: int, wait_s: float, **_
    ) -> dict[str, Any]:
        self.admit_start({"owner_claim": owner_claim})
        return await asyncio.to_thread(
            self.supervisor.read_stdout,
            command_id,
            offset=offset,
            max_bytes=max_bytes,
            wait_s=wait_s,
        )


def _runner(root: Path, transport: Any) -> LambdaMicroVMRunner:
    return LambdaMicroVMRunner(
        ConformanceLambdaClient(),
        microvm_id="mvm-interactive",
        endpoint="interactive.lambda-microvm.invalid",
        region_name="us-east-1",
        default_cwd=str(root),
        endpoint_transport=transport,
        poll_interval_s=0,
        env_overlay={"HTTPS_PROXY": "http://10.0.0.5:3128"},
    )


async def _read_all(process) -> bytes:
    output = b""
    while chunk := await process.read():
        output += chunk
    return output


def _python(script: str) -> ExecCommand:
    return ExecCommand.process(sys.executable, "-c", script)


def test_interactive_process_streams_stdio_and_confirms_teardown(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = InteractiveTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        capability = runner.interactive_process_capability()
        assert isinstance(capability, RunnerInteractiveProcessCapability)

        process = await capability.open_interactive_process(
            _python(_ECHO_UPPER), env={"A": "1"}, timeout_s=60
        )
        await process.write(b"one\n")
        assert await process.read() == b"ONE\n"
        await process.write(b"two\n")
        await process.close_stdin()
        assert await _read_all(process) == b"TWO\n"
        assert await process.wait() == 5
        await process.close()
        await process.close()

        [payload] = transport.payloads
        assert payload["interactive"] is True
        assert payload["execution_profile"] == "agent"
        assert payload["timeout_s"] == 60
        # The runner's egress overlay reaches the process like any command.
        assert payload["env"] == {"A": "1", "HTTPS_PROXY": "http://10.0.0.5:3128"}
        assert len(transport.released) == 1

    asyncio.run(scenario())


def test_lost_write_response_is_retried_without_duplicating_input(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = InteractiveTransport(tmp_path)
        process = (
            await _runner(tmp_path, transport)
            .interactive_process_capability()
            .open_interactive_process(_python(_ECHO_UPPER))
        )
        transport.fail_next_write = 1
        await process.write(b"once\n")
        await process.close_stdin()

        assert await _read_all(process) == b"ONCE\n"
        assert [offset for offset, _ in transport.stdin_calls] == [0, 0, 5]
        await process.close()

    asyncio.run(scenario())


def test_close_stops_a_running_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = InteractiveTransport(tmp_path)
        process = (
            await _runner(tmp_path, transport)
            .interactive_process_capability()
            .open_interactive_process(
                _python("import time\nprint('up', flush=True)\ntime.sleep(60)\n")
            )
        )
        assert await process.read() == b"up\n"

        await process.close()

        [command_id] = transport.released
        assert transport.supervisor.get(command_id)["state"] == "released"
        with pytest.raises(BrokenPipeError):
            await process.write(b"late")

    asyncio.run(scenario())


def test_unconfirmed_process_tree_fails_close(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = InteractiveTransport(tmp_path)
        process = (
            await _runner(tmp_path, transport)
            .interactive_process_capability()
            .open_interactive_process(_python("pass\n"))
        )
        assert await _read_all(process) == b""
        transport.isolated = False

        with pytest.raises(LambdaMicroVMError, match="could not confirm"):
            await process.close()
        assert transport.released == []
        transport.isolated = True
        await process.close()
        assert transport.released == [process.command_id]
        await process.close()
        assert transport.released == [process.command_id]

    asyncio.run(scenario())


def test_sidecar_without_pid_namespace_is_refused_and_stopped(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = InteractiveTransport(tmp_path, isolated=False)
        runner = _runner(tmp_path, transport)

        with pytest.raises(LambdaMicroVMProtocolError, match="rebuild the image"):
            await runner.interactive_process_capability().open_interactive_process(
                _python("import time\ntime.sleep(60)\n")
            )

        [payload] = transport.payloads
        [record] = transport.supervisor._records.values()
        assert record.state == "cancelled"
        # The unconfirmed cleanup fences the runner instead of reporting success.
        with pytest.raises(RuntimeError):
            await runner.exec(ExecCommand.process("true"))
        assert payload["interactive"] is True

    asyncio.run(scenario())


def test_superseded_owner_cannot_move_stdio(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = InteractiveTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        process = await runner.interactive_process_capability().open_interactive_process(
            _python(_ECHO_UPPER)
        )
        await transport.claim_owner(claim_id="n" * 40)

        with pytest.raises(LambdaMicroVMOwnershipSuperseded):
            await process.write(b"x\n")
        assert runner.owner_fence_generation is None

    asyncio.run(scenario())


def test_transport_without_stdio_reports_no_capability(tmp_path: Path) -> None:
    assert _runner(tmp_path, SupervisorTransport(tmp_path)).interactive_process_capability() is None


def test_interactive_process_declares_captured_overlay_secrets(tmp_path):
    async def scenario():
        transport = InteractiveTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        runner.env_overlay["CAYU_PROXY_TOKEN"] = "overlay-credential-canary"
        capability = runner.interactive_process_capability()
        assert "overlay-credential-canary" in capability.secret_values
        process = await capability.open_interactive_process(_python("pass"))
        runner.env_overlay["CAYU_PROXY_TOKEN"] = "rotated-credential"
        assert "overlay-credential-canary" in process.secret_values
        assert "rotated-credential" not in process.secret_values
        await process.close()

    asyncio.run(scenario())


def test_superseded_process_can_close_but_cannot_wait(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = InteractiveTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        process = await runner.interactive_process_capability().open_interactive_process(
            _python(_ECHO_UPPER)
        )
        await transport.claim_owner(claim_id="n" * 40)
        with pytest.raises(LambdaMicroVMOwnershipSuperseded):
            await process.wait()
        await process.close()
        await process.close()
        assert runner.owner_fence_generation is None
        with pytest.raises(LambdaMicroVMOwnershipSuperseded):
            await process.wait()

    asyncio.run(scenario())


class FailedStartTransport(InteractiveTransport):
    def __init__(self, root: Path, error: BaseException) -> None:
        super().__init__(root)
        self.start_error = error
        self.fail_cancel = True

    async def start_command(self, **kwargs):
        await super().start_command(**kwargs)
        raise self.start_error

    async def cancel_command(self, **kwargs):
        if self.fail_cancel:
            raise LambdaMicroVMEndpointTransientError("cancel unavailable")
        return await super().cancel_command(**kwargs)


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_start_retains_cleanup_and_original_error(tmp_path, cancelled):
    async def scenario():
        error = (
            asyncio.CancelledError("caller cancelled")
            if cancelled
            else LambdaMicroVMEndpointTransientError("lost start response")
        )
        transport = FailedStartTransport(tmp_path, error)
        runner = _runner(tmp_path, transport)
        capability = runner.interactive_process_capability()
        try:
            with pytest.raises(type(error)) as raised:
                await capability.open_interactive_process(_python("import time; time.sleep(60)"))
            assert raised.value is error
            pending = capability.failed_start_process(error)
            assert pending is not None
            assert not transport.released
            with pytest.raises(LambdaMicroVMEndpointTransientError):
                await pending.close()
            transport.fail_cancel = False
            await pending.close()
            assert transport.released == [pending.command_id]
        finally:
            transport.fail_cancel = False
            for command_id in tuple(transport.supervisor._records):
                await transport.cancel_command(command_id=command_id)

    asyncio.run(scenario())
