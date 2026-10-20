"""runner_agent_launch: the vendor CLI in any runner with interactive processes."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.runners.lambda_microvm_harness import ConformanceLambdaClient
from tests.runners.test_aws_lambda_microvm_interactive import InteractiveTransport

from cayu.external_agents import runner_agent_launch
from cayu.external_agents._cleanup import ExternalCleanupIncomplete
from cayu.external_agents.environment import scrub_external_environment
from cayu.runners import (
    ExecCommand,
    LambdaMicroVMRunner,
    LocalRunner,
    RunnerInteractiveProcess,
    RunnerInteractiveProcessCapability,
)

_ECHO = (
    "import os, sys\n"
    "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'home': os.environ['HOME'],"
    " 'mode': os.environ.get('MODE')}), flush=True)\n"
    "for line in sys.stdin:\n"
    "    sys.stdout.write(line.upper())\n"
    "    sys.stdout.flush()\n"
    "sys.exit(3)\n"
)
_SCRIPT = "import json\n" + _ECHO


class _LocalProcess(RunnerInteractiveProcess):
    """A host subprocess standing in for a guest process."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.closed = 0
        self.fail_close = 0

    @property
    def secret_values(self) -> tuple[str, ...]:
        return ("process-secret",)

    async def write(self, data: bytes) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def close_stdin(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()

    async def read(self) -> bytes:
        assert self.process.stdout is not None
        return await self.process.stdout.read(65536)

    async def wait(self) -> int:
        return await self.process.wait()

    async def close(self) -> None:
        if self.fail_close:
            self.fail_close -= 1
            raise RuntimeError("guest teardown not confirmed")
        if self.process.returncode is None:
            self.process.kill()
        await self.process.wait()
        self.closed += 1


class _LocalCapability(RunnerInteractiveProcessCapability):
    def __init__(self) -> None:
        self.opened: list[tuple[ExecCommand, dict[str, Any]]] = []
        self.processes: list[_LocalProcess] = []
        self.fail_close = 0

    @property
    def secret_values(self) -> tuple[str, ...]:
        return ("cayu_vc_overlay",)

    async def open_interactive_process(
        self, command: ExecCommand, *, cwd=None, env=None, timeout_s=None
    ) -> RunnerInteractiveProcess:
        self.opened.append((command, {"cwd": cwd, "env": dict(env or {}), "timeout_s": timeout_s}))
        assert command.argv is not None
        process = _LocalProcess(
            await asyncio.create_subprocess_exec(
                *command.argv,
                cwd=cwd,
                env={"PATH": os.environ["PATH"], **(env or {})},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        )
        process.fail_close = self.fail_close
        self.processes.append(process)
        return process


class _Runner(LocalRunner):
    def __init__(self, root: Path, capability: RunnerInteractiveProcessCapability | None) -> None:
        super().__init__(root=str(root))
        self.capability = capability

    def interactive_process_capability(self) -> RunnerInteractiveProcessCapability | None:
        return self.capability


async def _run_shim(launch, *args: str, stdin: bytes, protect_secret=None) -> tuple[int, bytes]:
    """Start the stand-in the way an SDK starts its CLI."""
    launch.protect_secrets(protect_secret or (lambda secret: None))
    process = await asyncio.create_subprocess_exec(
        launch.executable,
        *args,
        env=scrub_external_environment(launch.environment),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(stdin), 30)
    return process.returncode, stdout


def test_stand_in_relays_arguments_stdio_and_exit_code(tmp_path: Path) -> None:
    capability = _LocalCapability()
    home = tmp_path / "home"

    async def scenario() -> None:
        async with runner_agent_launch(
            runner=_Runner(tmp_path, capability),
            cli=sys.executable,
            environment={"HOME": str(home), "MODE": "review", "CAYU_MODEL_PROXY_TOKEN": "tok"},
            cwd=str(tmp_path),
            timeout_s=120,
        ) as launch:
            # The SDK starts the stand-in in a host directory; the guest maps
            # the same path to its working directory.
            assert Path(launch.cwd).is_dir()
            assert Path(launch.cwd).parent == Path(launch.executable).parent
            assert set(launch.secret_values) == {"tok", "cayu_vc_overlay"}
            # The SDK-facing environment carries no guest configuration.
            assert set(launch.environment) == {"PATH", "HOME"}
            code, stdout = await _run_shim(launch, "-c", _SCRIPT, "--flag", stdin=b"a\nb\n")
            assert code == 3
            first, rest = stdout.split(b"\n", 1)
            assert json.loads(first) == {
                "argv": ["--flag"],
                "cwd": os.path.realpath(tmp_path),
                "home": str(home),
                "mode": "review",
            }
            assert rest == b"A\nB\n"
            assert home.is_dir()

        [(command, options)] = capability.opened
        state = str(Path(command.argv[5]).parent)
        assert command.argv[:7] == [
            "/bin/sh",
            "-c",
            'mkdir -p -- "$HOME" "$1" && { [ -e "$2" ] || ln -s -- "$PWD" "$2"; } '
            '&& shift 2 && exec "$@"',
            "cayu-agent",
            state,
            f"{state}/workspace",
            sys.executable,
        ]
        assert command.argv[7:] == ["-c", _SCRIPT, "--flag"]
        assert options["cwd"] == str(tmp_path)
        assert options["timeout_s"] == 120
        assert capability.processes[0].closed == 1

    asyncio.run(scenario())


def test_close_stops_guest_processes_and_retries_unconfirmed_ones(tmp_path: Path) -> None:
    capability = _LocalCapability()
    capability.fail_close = 1

    async def scenario() -> None:
        launched = runner_agent_launch(
            runner=_Runner(tmp_path, capability), cli=sys.executable, environment={}
        )
        launch = await launched.__aenter__()
        launch.protect_secrets(lambda secret: None)
        shim = await asyncio.create_subprocess_exec(
            launch.executable,
            "-c",
            "import time\nprint('up', flush=True)\ntime.sleep(60)\n",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        assert await asyncio.wait_for(shim.stdout.readline(), 30) == b"up\n"

        with pytest.raises(ExceptionGroup):
            await launch.close()
        await launch.close()
        [process] = capability.processes
        assert process.process.returncode is not None
        assert process.closed == 1
        # The stand-in sees its connection end and exits instead of hanging.
        assert await asyncio.wait_for(shim.wait(), 30) != 0
        await launched.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_failed_cleanup_on_exit_is_reported_for_retry(tmp_path: Path) -> None:
    capability = _LocalCapability()
    capability.fail_close = 1

    async def scenario() -> None:
        with pytest.raises(ExternalCleanupIncomplete) as failure:
            async with runner_agent_launch(
                runner=_Runner(tmp_path, capability), cli=sys.executable, environment={}
            ) as launch:
                launch.protect_secrets(lambda secret: None)
                shim = await asyncio.create_subprocess_exec(
                    launch.executable,
                    "-c",
                    "import time\nprint('up', flush=True)\ntime.sleep(60)\n",
                    stdout=asyncio.subprocess.PIPE,
                )
                assert await asyncio.wait_for(shim.stdout.readline(), 30) == b"up\n"
        await failure.value.retry()
        assert capability.processes[0].closed == 1
        assert await asyncio.wait_for(shim.wait(), 30) != 0

    asyncio.run(scenario())


def test_stand_in_socket_refuses_a_wrong_token(tmp_path: Path) -> None:
    capability = _LocalCapability()

    async def scenario() -> None:
        async with runner_agent_launch(
            runner=_Runner(tmp_path, capability), cli=sys.executable, environment={}
        ) as launch:
            socket_path = str(Path(launch.executable).parent / "s")
            assert oct(os.stat(socket_path).st_mode & 0o777) == "0o600"

            def probe() -> bytes:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(socket_path)
                    hello = json.dumps({"token": "0" * 64, "argv": []}).encode()
                    connection.sendall(b"h" + struct.pack(">I", len(hello)) + hello)
                    return connection.recv(16)

            assert await asyncio.to_thread(probe) == b""
        assert capability.opened == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"cli": "claude"}, "absolute"),
        ({"environment": {"ANTHROPIC_API_KEY": "sk"}}, "service proxy"),
        ({"environment": {"BAD-NAME": "x"}}, "portable"),
        ({"capability": None}, "interactive processes"),
    ],
)
def test_launcher_refuses_unsafe_configuration(
    tmp_path: Path, options: dict[str, Any], error: str
) -> None:
    capability = options.pop("capability", _LocalCapability())
    arguments = {"cli": sys.executable, "environment": {}, **options}

    async def scenario() -> None:
        with pytest.raises(ValueError, match=error):
            async with runner_agent_launch(runner=_Runner(tmp_path, capability), **arguments):
                pass

    asyncio.run(scenario())


def test_lambda_microvm_runner_hosts_the_cli(tmp_path: Path) -> None:
    transport = InteractiveTransport(tmp_path)
    runner = LambdaMicroVMRunner(
        ConformanceLambdaClient(),
        microvm_id="mvm-agent",
        endpoint="agent.lambda-microvm.invalid",
        region_name="us-east-1",
        default_cwd=str(tmp_path),
        endpoint_transport=transport,
        poll_interval_s=0,
    )

    async def scenario() -> None:
        async with runner_agent_launch(
            runner=runner,
            cli=sys.executable,
            environment={"HOME": str(tmp_path / "home")},
            timeout_s=300,
        ) as launch:
            code, stdout = await _run_shim(launch, "-c", _SCRIPT, stdin=b"hi\n")
        assert code == 3
        assert stdout.split(b"\n", 1)[1] == b"HI\n"
        [payload] = transport.payloads
        assert payload["interactive"] is True
        assert payload["timeout_s"] == 300
        assert len(transport.released) == 1

    asyncio.run(scenario())


def test_process_secrets_registered_before_output(tmp_path):
    recorded = []
    capability = _LocalCapability()
    original = capability.open_interactive_process

    async def open_process(*args, **kwargs):
        process = await original(*args, **kwargs)
        read = process.read

        async def guarded_read():
            assert "process-secret" in recorded
            return await read()

        process.read = guarded_read
        return process

    capability.open_interactive_process = open_process

    async def scenario():
        async with runner_agent_launch(
            runner=_Runner(tmp_path, capability),
            cli=sys.executable,
            environment={"HOME": str(tmp_path / "home")},
        ) as launch:
            code, output = await _run_shim(
                launch, "-c", "print('ok')", stdin=b"", protect_secret=recorded.append
            )
            assert code == 0 and output == b"ok\n"

    asyncio.run(scenario())
    assert "process-secret" in recorded
    assert "cayu_vc_overlay" in recorded


def test_stdin_failure_keeps_guest_output_and_exit_status(tmp_path):
    capability = _LocalCapability()
    original = capability.open_interactive_process

    async def open_process(*args, **kwargs):
        process = await original(*args, **kwargs)

        async def broken_write(data):
            raise BrokenPipeError("guest stopped reading")

        process.write = broken_write
        return process

    capability.open_interactive_process = open_process

    async def scenario():
        async with runner_agent_launch(
            runner=_Runner(tmp_path, capability),
            cli=sys.executable,
            environment={"HOME": str(tmp_path / "home")},
        ) as launch:
            code, output = await _run_shim(
                launch,
                "-c",
                "import time,sys; time.sleep(.1); print('last output'); sys.exit(7)",
                stdin=b"input",
            )
            assert code == 7 and output == b"last output\n"

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["write", "close_stdin"])
def test_stdin_transport_failure_terminates_the_shim(tmp_path, operation):
    capability = _LocalCapability()
    original = capability.open_interactive_process

    async def open_process(*args, **kwargs):
        process = await original(*args, **kwargs)

        async def fail(*args):
            raise RuntimeError("stdin delivery failed")

        setattr(process, operation, fail)
        return process

    capability.open_interactive_process = open_process

    async def scenario():
        async with runner_agent_launch(
            runner=_Runner(tmp_path, capability),
            cli=sys.executable,
            environment={"HOME": str(tmp_path / "home")},
        ) as launch:
            code, _ = await asyncio.wait_for(
                _run_shim(launch, "-c", "import sys; sys.stdin.read()", stdin=b"input"),
                timeout=5,
            )
            assert code != 0
        assert capability.processes[0].process.returncode is not None

    asyncio.run(scenario())


def test_lost_interactive_start_keeps_launch_cleanup_retryable(tmp_path):
    from tests.runners.test_aws_lambda_microvm_interactive import FailedStartTransport, _runner

    from cayu.runners.aws_lambda_microvm import LambdaMicroVMEndpointTransientError

    async def scenario():
        transport = FailedStartTransport(
            tmp_path, LambdaMicroVMEndpointTransientError("lost start response")
        )
        runner = _runner(tmp_path, transport)
        try:
            with pytest.raises(ExternalCleanupIncomplete) as failure:
                async with runner_agent_launch(
                    runner=runner, cli=sys.executable, environment={}
                ) as launch:
                    code, _ = await _run_shim(
                        launch, "-c", "import time; time.sleep(60)", stdin=b""
                    )
                    assert code != 0
            [record] = transport.supervisor._records.values()
            assert record.process is not None and record.process.poll() is None
            assert not transport.released
            transport.fail_cancel = False
            await failure.value.retry()
            assert record.process.poll() is not None
            assert len(transport.released) == 1
        finally:
            transport.fail_cancel = False
            for command_id in tuple(transport.supervisor._records):
                await transport.cancel_command(command_id=command_id)

    asyncio.run(scenario())
