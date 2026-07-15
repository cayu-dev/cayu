from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from examples._runner_conformance import verify_bounded_output_drain
from examples.aws.lambda_microvm_sidecar.supervisor import CommandSupervisor
from tests.runners.conformance import (
    CapabilityClaim,
    ConformanceEvidence,
    RunnerCapabilities,
    RunnerConformanceRegistration,
    RunnerHarness,
)

import cayu.runners as runners_module
from cayu.runners import (
    DockerRunner,
    E2BRunner,
    ExecCommand,
    LambdaMicroVMProtocolError,
    LambdaMicroVMRunner,
    LocalRunner,
    MicrosandboxRunner,
    Runner,
    RunnerCleanupPolicy,
    attach_cancellation_artifacts,
)
from cayu.runners._subprocess import SubprocessCommand, run_subprocess

ORPHAN_WRITE_DELAY_SECONDS = 0.5
ORPHAN_OBSERVATION_DELAY_SECONDS = 0.6
SETSID_PROBE = "if command -v setsid >/dev/null 2>&1; then "


async def _local_factory(root: Path, _monkeypatch: pytest.MonkeyPatch) -> RunnerHarness:
    return RunnerHarness(LocalRunner(root, inherit_env=False), root)


async def _run_cli_guest_locally(command: SubprocessCommand, **kwargs: object):
    argv = command.argv
    assert argv is not None
    if len(argv) < 2 or argv[-2] != "-c":
        return await run_subprocess(
            SubprocessCommand(argv=[sys.executable, "-c", "pass"]),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
    is_supervised_command = "-w" in argv
    cwd = argv[argv.index("-w") + 1] if is_supervised_command else "/"
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if "--env-file" in argv:
        env_path = Path(argv[argv.index("--env-file") + 1])
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split("=", 1)
            environment[key] = value
    assert argv[-2] == "-c"
    guest_script = argv[-1]
    if is_supervised_command:
        assert guest_script.count(SETSID_PROBE) == 1, (
            "CLI guest wrapper setsid probe drifted; update the deterministic fixture rewrite."
        )
        guest_script = guest_script.replace(SETSID_PROBE, "if false; then ", 1)
    # run_subprocess already owns a fresh host process group. Force the guest
    # wrapper's portable fallback so Linux `setsid` cannot fork away from that
    # group merely because this deterministic fixture is not a real container.
    return await run_subprocess(
        SubprocessCommand(shell=guest_script),
        cwd=cwd,
        env=environment,
        timeout_s=kwargs.get("timeout_s"),  # type: ignore[arg-type]
        stdin=kwargs.get("stdin"),  # type: ignore[arg-type]
        output_limit_bytes=kwargs.get("output_limit_bytes"),  # type: ignore[arg-type]
    )


async def _docker_factory(root: Path, monkeypatch: pytest.MonkeyPatch) -> RunnerHarness:
    return await _docker_cleanup_factory(root, monkeypatch, "command")


async def _docker_cleanup_factory(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_policy: RunnerCleanupPolicy,
) -> RunnerHarness:
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", _run_cli_guest_locally)
    return RunnerHarness(
        DockerRunner(
            "cayu-conformance",
            default_cwd=str(root),
            docker_path="/conformance/docker",
            close_action="none",
            cancellation_cleanup=cleanup_policy,
            timeout_cleanup=cleanup_policy,
        ),
        root,
    )


class _LocalE2BHandle:
    def __init__(self, script: str, options: dict[str, Any]) -> None:
        self.script = script
        self.options = options
        self.stdin_parts: list[str] = []
        self._process: asyncio.subprocess.Process | None = None

    async def send_stdin(self, data: str) -> None:
        self.stdin_parts.append(data)

    async def close_stdin(self) -> None:
        return None

    async def wait(self) -> Any:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            **self.options.get("envs", {}),
        }
        self._process = await asyncio.create_subprocess_shell(
            self.script,
            cwd=self.options["cwd"],
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if self._process.stdin is not None:
            if self.options.get("stdin"):
                self._process.stdin.write("".join(self.stdin_parts).encode("utf-8"))
                await self._process.stdin.drain()
            self._process.stdin.close()
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        async def pump(
            stream: asyncio.StreamReader | None,
            callback: Any,
            parts: list[str],
        ) -> None:
            if stream is None:
                return
            while chunk := await stream.read(8192):
                text = chunk.decode("utf-8", errors="replace")
                parts.append(text)
                if callback is not None:
                    await callback(text)

        stdout_task = asyncio.create_task(
            pump(self._process.stdout, self.options.get("on_stdout"), stdout_parts)
        )
        stderr_task = asyncio.create_task(
            pump(self._process.stderr, self.options.get("on_stderr"), stderr_parts)
        )
        try:
            exit_code = await self._process.wait()
            await asyncio.gather(stdout_task, stderr_task)
        except asyncio.CancelledError:
            _kill_local_process(self._process)
            await self._process.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        return SimpleNamespace(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            exit_code=exit_code,
        )

    async def kill(self) -> bool:
        if self._process is not None and self._process.returncode is None:
            _kill_local_process(self._process)
            await self._process.wait()
        return True


class _LocalE2BCommands:
    async def run(self, script: str, **options: Any) -> Any:
        if options.get("background"):
            return _LocalE2BHandle(script, dict(options))
        result = await run_subprocess(
            SubprocessCommand(shell=script),
            cwd=options.get("cwd"),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **options.get("envs", {})},
        )
        return SimpleNamespace(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )


class _LocalE2BSandbox:
    def __init__(self) -> None:
        self.sandbox_id = "conformance-e2b"
        self.commands = _LocalE2BCommands()
        self.files = object()

    async def kill(self) -> bool:
        return True


async def _e2b_factory(root: Path, _monkeypatch: pytest.MonkeyPatch) -> RunnerHarness:
    return await _e2b_cleanup_factory(root, _monkeypatch, "command")


async def _e2b_cleanup_factory(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
    cleanup_policy: RunnerCleanupPolicy,
) -> RunnerHarness:
    return RunnerHarness(
        E2BRunner(
            _LocalE2BSandbox(),
            default_cwd=str(root),
            close_action="none",
            e2b_module=SimpleNamespace(),
            cancellation_cleanup=cleanup_policy,
            timeout_cleanup=cleanup_policy,
        ),
        root,
    )


class _LocalMicrosandboxHandle:
    def __init__(
        self,
        command: SubprocessCommand,
        *,
        cwd: str,
        env: dict[str, str],
        stdin: bytes | None,
    ) -> None:
        self.command = command
        self.cwd = cwd
        self.env = env
        self.stdin = stdin
        self._process: asyncio.subprocess.Process | None = None
        self._producer: asyncio.Task[None] | None = None
        self._events: asyncio.Queue[Any | None] = asyncio.Queue()
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._exit_code: int | None = None

    def __aiter__(self) -> _LocalMicrosandboxHandle:
        return self

    async def __anext__(self) -> Any:
        if self._producer is None:
            self._producer = asyncio.create_task(self._produce())
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event

    async def _produce(self) -> None:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            **self.env,
        }
        if self.command.argv is not None:
            self._process = await asyncio.create_subprocess_exec(
                *self.command.argv,
                cwd=self.cwd,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        else:
            assert self.command.shell is not None
            self._process = await asyncio.create_subprocess_shell(
                self.command.shell,
                cwd=self.cwd,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        if self._process.stdin is not None:
            if self.stdin is not None:
                self._process.stdin.write(self.stdin)
                await self._process.stdin.drain()
            self._process.stdin.close()

        async def pump(
            stream: asyncio.StreamReader | None,
            event_type: str,
            content: bytearray,
        ) -> None:
            if stream is None:
                return
            while chunk := await stream.read(8192):
                content.extend(chunk)
                await self._events.put(SimpleNamespace(event_type=event_type, data=chunk))

        stdout_task = asyncio.create_task(pump(self._process.stdout, "stdout", self._stdout))
        stderr_task = asyncio.create_task(pump(self._process.stderr, "stderr", self._stderr))
        self._exit_code = await self._process.wait()
        await asyncio.gather(stdout_task, stderr_task)
        await self._events.put(SimpleNamespace(event_type="exited", code=self._exit_code))
        await self._events.put(None)

    async def collect(self) -> Any:
        if self._producer is not None:
            await self._producer
        return SimpleNamespace(
            exit_code=self._exit_code if self._exit_code is not None else -9,
            stdout_bytes=bytes(self._stdout),
            stderr_bytes=bytes(self._stderr),
        )

    async def kill(self) -> None:
        if self._process is not None and self._process.returncode is None:
            _kill_local_process(self._process)
            await self._process.wait()
        if self._producer is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._producer


def _kill_local_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
            return
    process.kill()


class _LocalMicrosandbox:
    def __init__(self) -> None:
        self.name = "conformance-microsandbox"
        self.handles: list[_LocalMicrosandboxHandle] = []

    async def exec_stream(self, cmd: str, args: list[str], **options: Any) -> Any:
        handle = _LocalMicrosandboxHandle(
            SubprocessCommand(argv=[cmd, *args]),
            cwd=options["cwd"],
            env=options["env"],
            stdin=options.get("stdin"),
        )
        self.handles.append(handle)
        return handle

    async def shell_stream(self, script: str, **options: Any) -> Any:
        handle = _LocalMicrosandboxHandle(
            SubprocessCommand(shell=script),
            cwd=options["cwd"],
            env=options["env"],
            stdin=options.get("stdin"),
        )
        self.handles.append(handle)
        return handle

    async def stop(self) -> None:
        return None

    async def stop_and_wait(self) -> None:
        return None

    async def kill(self) -> None:
        await asyncio.gather(*(handle.kill() for handle in self.handles))

    async def detach(self) -> None:
        return None

    async def finalize(self) -> None:
        await self.kill()


async def _microsandbox_factory(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
) -> RunnerHarness:
    return await _microsandbox_cleanup_factory(root, _monkeypatch, "command")


async def _microsandbox_cleanup_factory(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
    cleanup_policy: RunnerCleanupPolicy,
) -> RunnerHarness:
    sandbox = _LocalMicrosandbox()
    return RunnerHarness(
        MicrosandboxRunner(
            sandbox,
            name="conformance-microsandbox",
            default_cwd=str(root),
            close_action="none",
            sandbox_module=SimpleNamespace(),
            cancellation_cleanup=cleanup_policy,
            timeout_cleanup=cleanup_policy,
        ),
        root,
        finalize=sandbox.finalize,
    )


class _ConformanceLambdaClient:
    def __init__(self) -> None:
        self.suspend_calls = 0
        self.resume_calls = 0
        self.terminate_calls = 0

    def run_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "microvmId": "mvm-conformance",
            "endpoint": "conformance.lambda-microvm.invalid",
            "state": "PENDING",
            "imageArn": "arn:aws:lambda:us-east-1:123:microvm-image:conformance",
            "imageVersion": "1",
        }

    def create_microvm_auth_token(self, **_kwargs: Any) -> dict[str, Any]:
        return {"authToken": {"X-aws-proxy-auth": "conformance-token"}}

    def get_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "microvmId": _kwargs.get("microvmIdentifier", "mvm-conformance"),
            "endpoint": "conformance.lambda-microvm.invalid",
            "state": "RUNNING",
            "imageArn": "arn:aws:lambda:us-east-1:123:microvm-image:conformance",
            "imageVersion": "1",
        }

    def suspend_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.suspend_calls += 1
        return {}

    def resume_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.resume_calls += 1
        return {}

    def terminate_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.terminate_calls += 1
        return {}


class _SupervisorTransport:
    def __init__(self, root: Path) -> None:
        self.supervisor = CommandSupervisor(root=root)

    async def health(self, **_kwargs: Any) -> dict[str, str]:
        return {"status": "ok", "protocol_version": "1"}

    async def start_command(self, *, command_id: str, payload: dict[str, Any], **_kwargs: Any):
        return await asyncio.to_thread(self.supervisor.start, command_id, payload)

    async def get_command(self, *, command_id: str, **_kwargs: Any):
        return await asyncio.to_thread(self.supervisor.get, command_id)

    async def cancel_command(self, *, command_id: str, **_kwargs: Any):
        return await asyncio.to_thread(self.supervisor.cancel, command_id)


async def _lambda_microvm_factory(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
) -> RunnerHarness:
    return await _lambda_microvm_cleanup_factory(root, _monkeypatch, "command")


async def _lambda_microvm_cleanup_factory(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
    cleanup_policy: RunnerCleanupPolicy,
) -> RunnerHarness:
    return RunnerHarness(
        LambdaMicroVMRunner(
            _ConformanceLambdaClient(),
            microvm_id="mvm-conformance",
            endpoint="conformance.lambda-microvm.invalid",
            image_identifier="arn:aws:lambda:us-east-1:123:microvm-image:conformance",
            region_name="us-east-1",
            default_cwd=str(root),
            close_action="none",
            endpoint_transport=_SupervisorTransport(root),
            poll_interval_s=0,
            cancellation_cleanup=cleanup_policy,
            timeout_cleanup=cleanup_policy,
        ),
        root,
    )


class _DelayedE2BCommands(_LocalE2BCommands):
    def __init__(self) -> None:
        self.start_received = asyncio.Event()
        self._delay_next_start = True

    async def run(self, script: str, **options: Any) -> Any:
        if options.get("background") and self._delay_next_start:
            self._delay_next_start = False
            self.start_received.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(0.15)
                return _LocalE2BHandle(script, dict(options))
        return await super().run(script, **options)


async def _probe_e2b_ambiguous_start(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _LocalE2BSandbox()
    commands = _DelayedE2BCommands()
    sandbox.commands = commands
    runner = E2BRunner(
        sandbox,
        default_cwd=str(root),
        close_action="none",
        cancel_timeout_s=0.05,
        e2b_module=SimpleNamespace(),
    )
    task = asyncio.create_task(
        runner.exec(ExecCommand.process(sys.executable, "-c", "print('late')"))
    )
    await commands.start_received.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task
    assert getattr(exc_info.value, "artifacts", [])[0]["status"] == "deferred"
    await asyncio.sleep(0.25)
    result = await runner.exec(ExecCommand.process(sys.executable, "-c", "print('safe')"))
    assert result.stdout == "safe\n"
    await runner.close()


class _DelayedMicrosandbox(_LocalMicrosandbox):
    def __init__(self) -> None:
        super().__init__()
        self.start_received = asyncio.Event()
        self._delay_next_start = True

    async def exec_stream(self, cmd: str, args: list[str], **options: Any) -> Any:
        if self._delay_next_start:
            self._delay_next_start = False
            self.start_received.set()
            await asyncio.sleep(30)
        return await super().exec_stream(cmd, args, **options)


async def _probe_microsandbox_ambiguous_start(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _DelayedMicrosandbox()
    runner = MicrosandboxRunner(
        sandbox,
        name="conformance-microsandbox-ambiguous",
        default_cwd=str(root),
        close_action="none",
        sandbox_module=SimpleNamespace(),
    )
    task = asyncio.create_task(
        runner.exec(ExecCommand.process(sys.executable, "-c", "print('uncertain')"))
    )
    await sandbox.start_received.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task
    assert getattr(exc_info.value, "artifacts", [])[0]["status"] == "unsupported"
    with pytest.raises(RuntimeError, match="command state is unknown"):
        await runner.exec(ExecCommand.process(sys.executable, "-c", "print('unsafe')"))
    await runner.close()


class _DelayedSupervisorTransport(_SupervisorTransport):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.start_received = asyncio.Event()
        self.release_start = asyncio.Event()
        self.late_start_tasks: list[asyncio.Task[Any]] = []
        self._delay_next_start = True

    async def start_command(self, *, command_id: str, payload: dict[str, Any], **kwargs: Any):
        if not self._delay_next_start:
            return await super().start_command(command_id=command_id, payload=payload, **kwargs)
        self._delay_next_start = False

        async def land_start() -> Any:
            await self.release_start.wait()
            return await super(_DelayedSupervisorTransport, self).start_command(
                command_id=command_id,
                payload=payload,
                **kwargs,
            )

        task = asyncio.create_task(land_start())
        self.late_start_tasks.append(task)
        self.start_received.set()
        return await asyncio.shield(task)

    async def cancel_command(self, *, command_id: str, **kwargs: Any):
        result = await super().cancel_command(command_id=command_id, **kwargs)
        self.release_start.set()
        return result


async def _probe_lambda_microvm_ambiguous_start(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _DelayedSupervisorTransport(root)
    runner = LambdaMicroVMRunner(
        _ConformanceLambdaClient(),
        microvm_id="mvm-conformance-ambiguous",
        endpoint="conformance.lambda-microvm.invalid",
        default_cwd=str(root),
        close_action="none",
        endpoint_transport=transport,
        poll_interval_s=0,
    )
    marker = root / "late.txt"
    task = asyncio.create_task(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import pathlib; pathlib.Path('late.txt').write_text('late')",
            )
        )
    )
    await transport.start_received.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task
    assert getattr(exc_info.value, "artifacts", [])[0]["status"] == "completed"
    await asyncio.gather(*transport.late_start_tasks)
    await asyncio.sleep(0.05)
    assert not marker.exists()
    result = await runner.exec(ExecCommand.process(sys.executable, "-c", "print('safe')"))
    assert result.stdout == "safe\n"
    await runner.close()


class _ProtocolTransport:
    def __init__(
        self,
        *,
        health: dict[str, Any] | None = None,
        command: dict[str, Any] | None = None,
    ) -> None:
        self.health_response = health or {"status": "ok", "protocol_version": "1"}
        self.command_response = command or {"state": "not_found"}

    async def health(self, **_kwargs: Any) -> dict[str, Any]:
        return dict(self.health_response)

    async def start_command(self, *, command_id: str, **_kwargs: Any) -> dict[str, Any]:
        return {"command_id": command_id, "state": "accepted"}

    async def get_command(self, *, command_id: str, **_kwargs: Any) -> dict[str, Any]:
        return {"command_id": command_id, **self.command_response}

    async def cancel_command(self, *, command_id: str, **_kwargs: Any) -> dict[str, Any]:
        return {"command_id": command_id, "state": "cancelled"}


async def _probe_lambda_microvm_protocol(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
) -> dict[str, bool]:
    observed: dict[str, bool] = {}
    mismatch_client = _ConformanceLambdaClient()
    try:
        await LambdaMicroVMRunner.create(
            "arn:aws:lambda:us-east-1:123:microvm-image:conformance",
            client=mismatch_client,
            endpoint_transport=_ProtocolTransport(health={"status": "ok", "protocol_version": "2"}),
            default_cwd=str(root),
            ready_timeout_s=0.2,
            poll_interval_s=0,
            cancel_timeout_s=0.2,
        )
    except LambdaMicroVMProtocolError as exc:
        observed["protocol_mismatch"] = "protocol version" in str(exc)
    else:
        observed["protocol_mismatch"] = False
    observed["protocol_mismatch_cleanup"] = mismatch_client.terminate_calls == 1

    for key, response, message in (
        ("unknown_command", {"state": "not_found"}, "unsupported state"),
        ("malformed_state", {"state": 3}, "state must be a string"),
        (
            "incomplete_terminal_snapshot",
            {"state": "completed"},
            "exit_code must be an integer",
        ),
    ):
        runner = LambdaMicroVMRunner(
            _ConformanceLambdaClient(),
            microvm_id="mvm-conformance-protocol",
            endpoint="conformance.lambda-microvm.invalid",
            default_cwd=str(root),
            endpoint_transport=_ProtocolTransport(command=response),
            poll_interval_s=0,
        )
        try:
            await runner.exec(ExecCommand.process(sys.executable, "-c", "print('probe')"))
        except LambdaMicroVMProtocolError as exc:
            observed[key] = message in str(exc)
        else:
            observed[key] = False
        await runner.close()
    return observed


class _FailFirstSuspendClient(_ConformanceLambdaClient):
    def __init__(self) -> None:
        super().__init__()
        self._fail_suspend = True

    def suspend_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.suspend_calls += 1
        if self._fail_suspend:
            self._fail_suspend = False
            raise RuntimeError("transient suspend failure")
        return {}


async def _probe_lambda_microvm_lifecycle(
    root: Path,
    _monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FailFirstSuspendClient()
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-conformance-lifecycle",
        endpoint="conformance.lambda-microvm.invalid",
        default_cwd=str(root),
        close_action="terminate",
        endpoint_transport=_ProtocolTransport(),
        poll_interval_s=0,
    )
    with pytest.raises(RuntimeError, match="transient suspend failure"):
        await runner.suspend()
    await asyncio.gather(runner.suspend(), runner.suspend())
    assert client.suspend_calls == 2
    await asyncio.gather(runner.resume(), runner.resume())
    assert client.resume_calls == 1
    await runner.close()
    await runner.close()
    assert client.terminate_calls == 1


LOCAL = RunnerConformanceRegistration(
    name="local",
    runner_type=LocalRunner,
    factory=_local_factory,
    capabilities=RunnerCapabilities(
        command_cleanup=CapabilityClaim.not_applicable(
            "LocalRunner owns and kills its subprocess directly."
        ),
        sandbox_cleanup=CapabilityClaim.not_applicable("LocalRunner has no sandbox lifecycle."),
        no_cleanup=CapabilityClaim.not_applicable(
            "LocalRunner cannot intentionally leave a cancelled subprocess running."
        ),
        ambiguous_start=CapabilityClaim.not_applicable(
            "Local subprocess creation has no remote acknowledgement phase."
        ),
        remote_protocol=CapabilityClaim.not_applicable(
            "LocalRunner does not cross a Cayu-owned remote protocol."
        ),
        suspend_resume=CapabilityClaim.not_applicable(
            "LocalRunner does not own a suspendable sandbox."
        ),
    ),
)


CLI_CAPABILITIES = RunnerCapabilities(
    command_cleanup=CapabilityClaim.supported(),
    sandbox_cleanup=CapabilityClaim.supported(),
    no_cleanup=CapabilityClaim.supported(),
    ambiguous_start=CapabilityClaim.not_applicable(
        "The supervised CLI call has no separately acknowledged remote start phase."
    ),
    remote_protocol=CapabilityClaim.not_applicable(
        "The runner uses an installed CLI rather than a Cayu-owned remote protocol."
    ),
    suspend_resume=CapabilityClaim.not_applicable(
        "The CLI runner does not expose suspend/resume lifecycle operations."
    ),
)

DOCKER = RunnerConformanceRegistration(
    name="docker",
    runner_type=DockerRunner,
    factory=_docker_factory,
    capabilities=CLI_CAPABILITIES,
    cleanup_factory=_docker_cleanup_factory,
)

REMOTE_SANDBOX_CAPABILITIES = RunnerCapabilities(
    command_cleanup=CapabilityClaim.supported(),
    sandbox_cleanup=CapabilityClaim.supported(),
    no_cleanup=CapabilityClaim.supported(),
    ambiguous_start=CapabilityClaim.supported(),
    remote_protocol=CapabilityClaim.not_applicable(
        "The vendor SDK is tested separately and is not a Cayu-owned command protocol."
    ),
    suspend_resume=CapabilityClaim.not_applicable(
        "The adapter does not expose suspend/resume through the Runner interface."
    ),
)

E2B = RunnerConformanceRegistration(
    name="e2b",
    runner_type=E2BRunner,
    factory=_e2b_factory,
    capabilities=REMOTE_SANDBOX_CAPABILITIES,
    cleanup_factory=_e2b_cleanup_factory,
    ambiguous_start_probe=_probe_e2b_ambiguous_start,
)

MICROSANDBOX = RunnerConformanceRegistration(
    name="microsandbox",
    runner_type=MicrosandboxRunner,
    factory=_microsandbox_factory,
    capabilities=REMOTE_SANDBOX_CAPABILITIES,
    cleanup_factory=_microsandbox_cleanup_factory,
    ambiguous_start_probe=_probe_microsandbox_ambiguous_start,
)

LAMBDA_MICROVM = RunnerConformanceRegistration(
    name="lambda-microvm",
    runner_type=LambdaMicroVMRunner,
    factory=_lambda_microvm_factory,
    capabilities=RunnerCapabilities(
        command_cleanup=CapabilityClaim.supported(),
        sandbox_cleanup=CapabilityClaim.supported(),
        no_cleanup=CapabilityClaim.supported(),
        ambiguous_start=CapabilityClaim.supported(),
        remote_protocol=CapabilityClaim.supported(),
        suspend_resume=CapabilityClaim.supported(),
    ),
    cleanup_factory=_lambda_microvm_cleanup_factory,
    ambiguous_start_probe=_probe_lambda_microvm_ambiguous_start,
    remote_protocol_probe=_probe_lambda_microvm_protocol,
    suspend_resume_probe=_probe_lambda_microvm_lifecycle,
)

REGISTRATIONS = (LOCAL, DOCKER, E2B, LAMBDA_MICROVM, MICROSANDBOX)
CLEANUP_CASES = tuple(
    (registration, policy)
    for registration in REGISTRATIONS
    for policy, claim in (
        ("command", registration.capabilities.command_cleanup),
        ("sandbox", registration.capabilities.sandbox_cleanup),
        ("none", registration.capabilities.no_cleanup),
    )
    if claim.state == "supported"
)
AMBIGUOUS_START_REGISTRATIONS = tuple(
    registration
    for registration in REGISTRATIONS
    if registration.capabilities.ambiguous_start.state == "supported"
)
REMOTE_PROTOCOL_REGISTRATIONS = tuple(
    registration
    for registration in REGISTRATIONS
    if registration.capabilities.remote_protocol.state == "supported"
)
SUSPEND_RESUME_REGISTRATIONS = tuple(
    registration
    for registration in REGISTRATIONS
    if registration.capabilities.suspend_resume.state == "supported"
)


def test_runner_conformance_registry_covers_every_exported_builtin_runner() -> None:
    registered = {registration.runner_type for registration in REGISTRATIONS}
    exported = {
        value
        for name in runners_module.__all__
        if isinstance((value := getattr(runners_module, name)), type)
        and value is not Runner
        and issubclass(value, Runner)
    }

    assert registered == exported
    assert len({registration.name for registration in REGISTRATIONS}) == len(REGISTRATIONS)


def test_runner_capability_claims_require_bounded_skip_reasons() -> None:
    with pytest.raises(ValueError, match="require a reason"):
        CapabilityClaim.unsupported(" ")
    with pytest.raises(ValueError, match="cannot define"):
        CapabilityClaim("supported", "silently skipped")
    with pytest.raises(ValueError, match="at most 240"):
        CapabilityClaim.unsupported("x" * 241)


@pytest.mark.parametrize(
    "guest_script",
    (
        "if command -v setsid; then :; fi",
        "printf wrapper-without-probe",
    ),
)
def test_cli_guest_fixture_rejects_setsid_wrapper_drift(
    tmp_path: Path,
    guest_script: str,
) -> None:
    command = SubprocessCommand(
        argv=[
            "/conformance/cli",
            "exec",
            "-w",
            str(tmp_path),
            "runner",
            "sh",
            "-c",
            guest_script,
        ]
    )

    with pytest.raises(AssertionError, match="setsid probe drifted"):
        asyncio.run(_run_cli_guest_locally(command))


def test_runner_conformance_failure_output_carries_required_evidence() -> None:
    evidence = ConformanceEvidence(
        "seeded-scenario",
        "seeded-adapter",
        "supported",
        observed={"state": "wrong"},
        cleanup_artifact={"status": "failed"},
    )
    with pytest.raises(AssertionError) as exc_info, evidence.reporting():
        raise AssertionError("seeded failure")

    message = str(exc_info.value)
    assert "scenario=seeded-scenario" in message
    assert "adapter=seeded-adapter" in message
    assert "capability=supported" in message
    assert "observed={'state': 'wrong'}" in message
    assert "cleanup_artifact={'status': 'failed'}" in message


@pytest.mark.parametrize(
    "registration",
    AMBIGUOUS_START_REGISTRATIONS,
    ids=lambda item: item.name,
)
def test_runner_conformance_ambiguous_start_proves_cleanup_or_latches_closed(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert registration.ambiguous_start_probe is not None
    evidence = ConformanceEvidence(
        "ambiguous-start",
        registration.name,
        registration.capabilities.ambiguous_start.state,
    )
    with evidence.reporting():
        asyncio.run(registration.ambiguous_start_probe(tmp_path, monkeypatch))
        evidence.observed = "cleanup proved or exec path latched"


@pytest.mark.parametrize(
    "registration",
    REMOTE_PROTOCOL_REGISTRATIONS,
    ids=lambda item: item.name,
)
def test_runner_conformance_remote_protocol_failures_are_explicit(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert registration.remote_protocol_probe is not None
    evidence = ConformanceEvidence(
        "remote-protocol-failures",
        registration.name,
        registration.capabilities.remote_protocol.state,
    )
    with evidence.reporting():
        observed = asyncio.run(registration.remote_protocol_probe(tmp_path, monkeypatch))
        evidence.observed = observed
        assert set(observed) == {
            "protocol_mismatch",
            "protocol_mismatch_cleanup",
            "unknown_command",
            "malformed_state",
            "incomplete_terminal_snapshot",
        }
        assert all(observed.values())


@pytest.mark.parametrize(
    "registration",
    SUSPEND_RESUME_REGISTRATIONS,
    ids=lambda item: item.name,
)
def test_runner_conformance_lifecycle_is_serialized_idempotent_and_retryable(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert registration.suspend_resume_probe is not None
    evidence = ConformanceEvidence(
        "serialized-lifecycle",
        registration.name,
        registration.capabilities.suspend_resume.state,
    )
    with evidence.reporting():
        asyncio.run(registration.suspend_resume_probe(tmp_path, monkeypatch))
        evidence.observed = "retry, serialization, idempotency, and termination passed"


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_runner_conformance_preserves_process_argv_and_intentional_shell_expansion(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = ConformanceEvidence("process-vs-shell", registration.name, "required")

    async def run() -> None:
        harness = await registration.factory(tmp_path, monkeypatch)
        try:
            literal = "$CAYU_CONFORMANCE_VALUE;printf injected"
            process_result = await harness.runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    "import sys; print(sys.argv[1])",
                    literal,
                ),
                env={"CAYU_CONFORMANCE_VALUE": "expanded"},
            )
            shell_result = await harness.runner.exec(
                ExecCommand.bash('printf "%s" "$CAYU_CONFORMANCE_VALUE"'),
                env={"CAYU_CONFORMANCE_VALUE": "expanded"},
            )
            evidence.observed = {
                "process": process_result.model_dump(),
                "shell": shell_result.model_dump(),
            }

            assert process_result.exit_code == 0
            assert process_result.stdout == f"{literal}\n"
            assert shell_result.exit_code == 0
            assert shell_result.stdout == "expanded"
        finally:
            await harness.aclose()

    with evidence.reporting():
        asyncio.run(run())


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_runner_conformance_executes_each_submitted_command_once(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = ConformanceEvidence("execute-once", registration.name, "required")

    async def run() -> None:
        harness = await registration.factory(tmp_path, monkeypatch)
        try:
            result = await harness.runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib; "
                        "path=pathlib.Path('count.txt'); "
                        "count=int(path.read_text()) + 1 if path.exists() else 1; "
                        "path.write_text(str(count)); "
                        "print(count)"
                    ),
                )
            )
            evidence.observed = {
                "result": result.model_dump(),
                "side_effect": (tmp_path / "count.txt").read_text(encoding="utf-8"),
            }
            assert result.stdout == "1\n"
            assert (tmp_path / "count.txt").read_text(encoding="utf-8") == "1"
        finally:
            await harness.aclose()

    with evidence.reporting():
        asyncio.run(run())


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_runner_conformance_resolves_cwd_idempotently_and_rejects_escape(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "nested"
    work.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    evidence = ConformanceEvidence("canonical-cwd", registration.name, "required")

    async def run() -> None:
        harness = await registration.factory(tmp_path, monkeypatch)
        try:
            canonical = harness.runner.resolve_cwd("nested")
            evidence.observed = {
                "default": harness.runner.resolve_cwd(),
                "canonical": canonical,
            }
            assert canonical == str(work)
            assert harness.runner.resolve_cwd(canonical) == canonical
            with pytest.raises(ValueError, match="escapes the runner root"):
                harness.runner.resolve_cwd("../outside")
            with pytest.raises(ValueError, match="outside the runner root"):
                harness.runner.resolve_cwd(str(outside))
        finally:
            await harness.aclose()

    with evidence.reporting():
        asyncio.run(run())


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_runner_conformance_reports_success_failure_cwd_env_stdin_and_output(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_CONFORMANCE_HOST_SECRET", "must-not-leak")
    work = tmp_path / "nested"
    work.mkdir()
    evidence = ConformanceEvidence("core-exec-result", registration.name, "required")

    async def run() -> None:
        harness = await registration.factory(tmp_path, monkeypatch)
        try:
            script = (
                "import os,pathlib,sys; "
                "print(pathlib.Path.cwd().name); "
                "print(os.environ['VISIBLE']); "
                "print(os.environ.get('CAYU_CONFORMANCE_HOST_SECRET', '')); "
                "print(sys.stdin.read()); "
                "print('warning', file=sys.stderr); "
                "sys.exit(7)"
            )
            result = await harness.runner.exec(
                ExecCommand.process(sys.executable, "-c", script),
                cwd="nested",
                env={"VISIBLE": "explicit"},
                stdin="input",
            )
            evidence.observed = result.model_dump()

            assert result.exit_code == 7
            assert result.timed_out is False
            assert result.stdout == "nested\nexplicit\n\ninput\n"
            assert result.stderr == "warning\n"
            assert result.stdout_bytes == len(b"nested\nexplicit\n\ninput\n")
            assert result.stderr_bytes == len(b"warning\n")
        finally:
            await harness.aclose()

    with evidence.reporting():
        asyncio.run(run())


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_runner_conformance_bounds_output_and_retains_total_byte_counts(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        harness = await registration.factory(tmp_path, monkeypatch)
        try:
            await verify_bounded_output_drain(
                harness.runner,
                adapter=registration.name,
            )
        finally:
            await harness.aclose()

    evidence = ConformanceEvidence("bounded-output-drain", registration.name, "required")
    with evidence.reporting():
        asyncio.run(run())
        evidence.observed = "pipe-scale output drained with bounded capture and exact totals"


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_runner_conformance_times_out_with_honest_partial_result(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = ConformanceEvidence("honest-timeout", registration.name, "required")

    async def run() -> None:
        harness = await registration.factory(tmp_path, monkeypatch)
        try:
            result = await harness.runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    "import time; print('before', flush=True); time.sleep(30)",
                ),
                timeout_s=1,
                output_limit_bytes=20,
            )
            evidence.observed = result.model_dump()
            evidence.cleanup_artifact = result.artifacts

            assert result.timed_out is True
            assert result.exit_code != 0
            assert result.stdout == "before\n"
            assert result.stdout_bytes == len(b"before\n")
            if registration.capabilities.command_cleanup.state == "supported":
                assert result.artifacts
                assert result.artifacts[-1]["type"] == "cayu.runner_cleanup.v1"
        finally:
            await harness.aclose()

    with evidence.reporting():
        asyncio.run(run())


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_runner_conformance_cancellation_cannot_leave_command_running(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = ConformanceEvidence("cancellation-no-orphan", registration.name, "required")

    async def run() -> None:
        harness = await registration.factory(tmp_path, monkeypatch)
        started = tmp_path / "started.txt"
        orphan = tmp_path / "orphan.txt"
        try:
            task = asyncio.create_task(
                harness.runner.exec(
                    ExecCommand.process(
                        sys.executable,
                        "-c",
                        (
                            "import pathlib,time; "
                            "pathlib.Path('started.txt').write_text('started'); "
                            f"time.sleep({ORPHAN_WRITE_DELAY_SECONDS}); "
                            "pathlib.Path('orphan.txt').write_text('orphan')"
                        ),
                    )
                )
            )
            for _ in range(100):
                if started.exists():
                    break
                await asyncio.sleep(0.02)
            assert started.exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            evidence.cleanup_artifact = getattr(exc_info.value, "artifacts", None)
            await asyncio.sleep(ORPHAN_OBSERVATION_DELAY_SECONDS)
            evidence.observed = {
                "started": started.exists(),
                "orphan": orphan.exists(),
            }
            assert not orphan.exists()

            after = await harness.runner.exec(
                ExecCommand.process(sys.executable, "-c", "print('reusable')")
            )
            evidence.observed = {
                "started": started.exists(),
                "orphan": orphan.exists(),
                "reuse": after.model_dump(),
            }
            assert after.stdout == "reusable\n"
        finally:
            await harness.aclose()

    with evidence.reporting():
        asyncio.run(run())


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_runner_conformance_close_is_bounded_idempotent_and_terminal(
    registration: RunnerConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = ConformanceEvidence("bounded-idempotent-close", registration.name, "required")

    async def run() -> None:
        harness = await registration.factory(tmp_path, monkeypatch)
        await asyncio.wait_for(harness.runner.close(), timeout=2)
        await asyncio.wait_for(harness.runner.close(), timeout=2)
        with pytest.raises(RuntimeError, match="closed"):
            await harness.runner.exec(
                ExecCommand.process(sys.executable, "-c", "print('must not run')")
            )
        if harness.finalize is not None:
            await harness.finalize()
        evidence.observed = "two closes completed and post-close exec was rejected"

    with evidence.reporting():
        asyncio.run(run())


@pytest.mark.parametrize(
    ("registration", "cleanup_policy"),
    CLEANUP_CASES,
    ids=lambda item: item.name if isinstance(item, RunnerConformanceRegistration) else item,
)
def test_runner_conformance_cleanup_policies_have_distinct_observable_outcomes(
    registration: RunnerConformanceRegistration,
    cleanup_policy: RunnerCleanupPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = {
        "command": registration.capabilities.command_cleanup,
        "sandbox": registration.capabilities.sandbox_cleanup,
        "none": registration.capabilities.no_cleanup,
    }[cleanup_policy]
    evidence = ConformanceEvidence(
        "cancellation-cleanup-policy",
        registration.name,
        capability.state,
    )

    async def run() -> None:
        assert registration.cleanup_factory is not None
        harness = await registration.cleanup_factory(tmp_path, monkeypatch, cleanup_policy)
        started = tmp_path / f"started-{cleanup_policy}.txt"
        try:
            task = asyncio.create_task(
                harness.runner.exec(
                    ExecCommand.process(
                        sys.executable,
                        "-c",
                        (
                            "import pathlib,time; "
                            f"pathlib.Path({started.name!r}).write_text('started'); "
                            "time.sleep(0.3)"
                        ),
                    )
                )
            )
            for _ in range(100):
                if started.exists():
                    break
                await asyncio.sleep(0.01)
            assert started.exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task

            artifacts = getattr(exc_info.value, "artifacts", None)
            evidence.cleanup_artifact = artifacts
            assert isinstance(artifacts, list) and len(artifacts) == 1
            artifact = artifacts[0]
            evidence.observed = {
                "policy": cleanup_policy,
                "action": artifact.get("action"),
                "status": artifact.get("status"),
            }
            assert artifact["type"] == "cayu.runner_cleanup.v1"
            assert (
                artifact["action"]
                == {
                    "command": "kill_command",
                    "sandbox": "kill_sandbox",
                    "none": "none",
                }[cleanup_policy]
            )
            assert artifact["status"] == ("skipped" if cleanup_policy == "none" else "completed")

            if cleanup_policy == "none":
                await asyncio.sleep(0.4)
            if cleanup_policy == "sandbox":
                with pytest.raises(RuntimeError, match="closed"):
                    await harness.runner.exec(
                        ExecCommand.process(sys.executable, "-c", "print('closed')")
                    )
            else:
                after = await harness.runner.exec(
                    ExecCommand.process(sys.executable, "-c", "print('open')")
                )
                assert after.stdout == "open\n"
        finally:
            await harness.aclose()

    with evidence.reporting():
        asyncio.run(run())


@pytest.mark.parametrize(
    ("registration", "cleanup_policy"),
    CLEANUP_CASES,
    ids=lambda item: item.name if isinstance(item, RunnerConformanceRegistration) else item,
)
def test_runner_conformance_timeout_cleanup_policies_have_distinct_observable_outcomes(
    registration: RunnerConformanceRegistration,
    cleanup_policy: RunnerCleanupPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = {
        "command": registration.capabilities.command_cleanup,
        "sandbox": registration.capabilities.sandbox_cleanup,
        "none": registration.capabilities.no_cleanup,
    }[cleanup_policy]
    evidence = ConformanceEvidence(
        "timeout-cleanup-policy",
        registration.name,
        capability.state,
    )

    async def run() -> None:
        assert registration.cleanup_factory is not None
        harness = await registration.cleanup_factory(tmp_path, monkeypatch, cleanup_policy)
        try:
            result = await harness.runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    "import sys,time; print('partial', flush=True); time.sleep(30)",
                ),
                timeout_s=1,
            )
            evidence.observed = {
                "timed_out": result.timed_out,
                "stdout": result.stdout,
                "exit_code": result.exit_code,
            }
            evidence.cleanup_artifact = result.artifacts
            assert result.timed_out is True
            assert result.stdout == "partial\n"
            assert len(result.artifacts) == 1
            artifact = result.artifacts[0]
            assert artifact["type"] == "cayu.runner_cleanup.v1"
            assert (
                artifact["action"]
                == {
                    "command": "kill_command",
                    "sandbox": "kill_sandbox",
                    "none": "none",
                }[cleanup_policy]
            )
            assert artifact["status"] == ("skipped" if cleanup_policy == "none" else "completed")

            if cleanup_policy == "sandbox":
                with pytest.raises(RuntimeError, match="closed"):
                    await harness.runner.exec(
                        ExecCommand.process(sys.executable, "-c", "print('closed')")
                    )
            else:
                after = await harness.runner.exec(
                    ExecCommand.process(sys.executable, "-c", "print('open')")
                )
                assert after.stdout == "open\n"
        finally:
            await harness.aclose()

    with evidence.reporting():
        asyncio.run(run())


class _BrokenFixtureRunner(Runner):
    isolation = "broken-fixture"

    def __init__(self, root: Path, failure: str) -> None:
        self.root = root
        self.failure = failure
        self.delegate = LocalRunner(root, inherit_env=False)
        self._closed = False
        self._exec_closed = False
        self._exec_closed_reason = None
        self._background: set[asyncio.Task[None]] = set()

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = 1024 * 1024,
    ):
        if self.failure == "duplicate_start":
            await self.delegate.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                stdin=stdin,
                output_limit_bytes=output_limit_bytes,
            )
            return await self.delegate.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                stdin=stdin,
                output_limit_bytes=output_limit_bytes,
            )
        if self.failure == "output_overflow":
            return runners_module.ExecResult(
                stdout="abcdef",
                stderr="uvwxyz",
                exit_code=0,
                stdout_bytes=6,
                stderr_bytes=6,
            )
        if self.failure == "dishonest_timeout":
            return runners_module.ExecResult(
                stdout="before\n",
                exit_code=0,
                timed_out=False,
                stdout_bytes=7,
                stderr_bytes=0,
            )
        if self.failure in {"lost_cancellation", "false_cleanup_completion"}:
            (self.root / "started.txt").write_text("started", encoding="utf-8")

            async def leave_orphan() -> None:
                await asyncio.sleep(ORPHAN_WRITE_DELAY_SECONDS)
                (self.root / "orphan.txt").write_text("orphan", encoding="utf-8")

            background = asyncio.create_task(leave_orphan())
            self._background.add(background)
            background.add_done_callback(self._background.discard)
            try:
                return await asyncio.shield(background)
            except asyncio.CancelledError as exc:
                if self.failure == "false_cleanup_completion":
                    attach_cancellation_artifacts(
                        exc,
                        [
                            {
                                "type": "cayu.runner_cleanup.v1",
                                "adapter": "broken-fixture",
                                "action": "kill_command",
                                "status": "completed",
                                "timeout_s": 1.0,
                            }
                        ],
                    )
                raise
        return await self.delegate.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )

    async def close(self) -> None:
        if self.failure == "close_race":
            return
        await self.delegate.close()
        self._closed = True
        for task in tuple(self._background):
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)


def _broken_registration(failure: str) -> RunnerConformanceRegistration:
    async def factory(root: Path, _monkeypatch: pytest.MonkeyPatch) -> RunnerHarness:
        return RunnerHarness(_BrokenFixtureRunner(root, failure), root)

    return RunnerConformanceRegistration(
        name=f"broken-{failure}",
        runner_type=_BrokenFixtureRunner,
        factory=factory,
        capabilities=RunnerCapabilities(
            command_cleanup=CapabilityClaim.not_applicable("Seeded broken fixture."),
            sandbox_cleanup=CapabilityClaim.not_applicable("Seeded broken fixture."),
            no_cleanup=CapabilityClaim.not_applicable("Seeded broken fixture."),
            ambiguous_start=CapabilityClaim.not_applicable("Seeded broken fixture."),
            remote_protocol=CapabilityClaim.not_applicable("Seeded broken fixture."),
            suspend_resume=CapabilityClaim.not_applicable("Seeded broken fixture."),
        ),
    )


def test_runner_conformance_guard_detects_duplicate_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AssertionError):
        test_runner_conformance_executes_each_submitted_command_once(
            _broken_registration("duplicate_start"), tmp_path, monkeypatch
        )


def test_runner_conformance_guard_detects_output_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AssertionError):
        test_runner_conformance_bounds_output_and_retains_total_byte_counts(
            _broken_registration("output_overflow"), tmp_path, monkeypatch
        )


def test_runner_conformance_guard_detects_dishonest_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AssertionError):
        test_runner_conformance_times_out_with_honest_partial_result(
            _broken_registration("dishonest_timeout"), tmp_path, monkeypatch
        )


def test_runner_conformance_guard_detects_orphaned_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AssertionError):
        test_runner_conformance_cancellation_cannot_leave_command_running(
            _broken_registration("lost_cancellation"), tmp_path, monkeypatch
        )


def test_runner_conformance_guard_detects_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AssertionError):
        test_runner_conformance_cancellation_cannot_leave_command_running(
            _broken_registration("false_cleanup_completion"), tmp_path, monkeypatch
        )


def test_runner_conformance_guard_detects_protocol_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken_protocol_probe(
        _root: Path,
        _monkeypatch: pytest.MonkeyPatch,
    ) -> dict[str, bool]:
        return {
            "protocol_mismatch": False,
            "protocol_mismatch_cleanup": False,
            "unknown_command": False,
            "malformed_state": False,
            "incomplete_terminal_snapshot": False,
        }

    registration = RunnerConformanceRegistration(
        name="broken-protocol-mismatch",
        runner_type=_BrokenFixtureRunner,
        factory=_broken_registration("protocol_mismatch").factory,
        capabilities=RunnerCapabilities(
            command_cleanup=CapabilityClaim.not_applicable("Seeded broken fixture."),
            sandbox_cleanup=CapabilityClaim.not_applicable("Seeded broken fixture."),
            no_cleanup=CapabilityClaim.not_applicable("Seeded broken fixture."),
            ambiguous_start=CapabilityClaim.not_applicable("Seeded broken fixture."),
            remote_protocol=CapabilityClaim.supported(),
            suspend_resume=CapabilityClaim.not_applicable("Seeded broken fixture."),
        ),
        remote_protocol_probe=broken_protocol_probe,
    )
    with pytest.raises(AssertionError):
        test_runner_conformance_remote_protocol_failures_are_explicit(
            registration,
            tmp_path,
            monkeypatch,
        )


def test_runner_conformance_guard_detects_close_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(BaseException, match="DID NOT RAISE"):
        test_runner_conformance_close_is_bounded_idempotent_and_terminal(
            _broken_registration("close_race"), tmp_path, monkeypatch
        )
