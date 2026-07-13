from __future__ import annotations

import asyncio
import base64
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

import cayu.runners.lambda_microvm as lambda_microvm_module
from cayu import ExecCommand, LambdaMicroVMRunner, RunnerWorkspace
from cayu.runners import (
    LambdaMicroVMEndpointUnauthorized,
    LambdaMicroVMError,
    LambdaMicroVMProtocolError,
)

SUPERVISOR_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "lambda_microvm_sidecar" / "supervisor.py"
)
SUPERVISOR_SPEC = importlib.util.spec_from_file_location(
    "cayu_lambda_microvm_runner_supervisor", SUPERVISOR_PATH
)
assert SUPERVISOR_SPEC is not None and SUPERVISOR_SPEC.loader is not None
SUPERVISOR_MODULE = importlib.util.module_from_spec(SUPERVISOR_SPEC)
sys.modules[SUPERVISOR_SPEC.name] = SUPERVISOR_MODULE
SUPERVISOR_SPEC.loader.exec_module(SUPERVISOR_MODULE)
CommandSupervisor = SUPERVISOR_MODULE.CommandSupervisor
HEALTH_RESPONSE = {"status": "ok", "protocol_version": "1"}


class FakeLambdaMicroVMClient:
    def __init__(self) -> None:
        self.run_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.token_calls: list[dict[str, Any]] = []
        self.suspend_calls: list[dict[str, Any]] = []
        self.resume_calls: list[dict[str, Any]] = []
        self.terminate_calls: list[dict[str, Any]] = []

    def run_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.run_calls.append(kwargs)
        return {
            "microvmId": "mvm-123",
            "endpoint": "mvm-123.lambda-microvm.us-west-2.on.aws",
            "state": "PENDING",
            "imageArn": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
            "imageVersion": "7",
        }

    def get_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        return {
            "microvmId": "mvm-123",
            "endpoint": "mvm-123.lambda-microvm.us-west-2.on.aws",
            "state": "RUNNING",
            "imageArn": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
            "imageVersion": "7",
        }

    def create_microvm_auth_token(self, **kwargs: Any) -> dict[str, Any]:
        self.token_calls.append(kwargs)
        token = "token-123" if len(self.token_calls) == 1 else "token-456"
        return {"authToken": {"X-aws-proxy-auth": token}}

    def suspend_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.suspend_calls.append(kwargs)
        return {}

    def resume_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.resume_calls.append(kwargs)
        return {}

    def terminate_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.terminate_calls.append(kwargs)
        return {}


class SuspendingLambdaMicroVMClient(FakeLambdaMicroVMClient):
    def get_microvm(self, **kwargs: Any) -> dict[str, Any]:
        response = super().get_microvm(**kwargs)
        response["state"] = "SUSPENDING" if len(self.get_calls) == 1 else "SUSPENDED"
        return response


class FakeEndpointTransport:
    def __init__(self, *, result_overrides: dict[str, Any] | None = None) -> None:
        self.health_calls: list[dict[str, Any]] = []
        self.start_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self.closed = False
        self.result_overrides = dict(result_overrides or {})

    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> dict[str, str]:
        self.health_calls.append({"endpoint": endpoint, "token": token, "timeout_s": timeout_s})
        return dict(HEALTH_RESPONSE)

    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        self.start_calls.append(
            {
                "endpoint": endpoint,
                "token": token,
                "command_id": command_id,
                "payload": payload,
                "timeout_s": timeout_s,
            }
        )
        return {"command_id": command_id, "state": "running"}

    async def get_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.get_calls.append(
            {
                "endpoint": endpoint,
                "token": token,
                "command_id": command_id,
                "timeout_s": timeout_s,
            }
        )
        result = {
            "command_id": command_id,
            "state": "completed",
            "exit_code": 7,
            "timed_out": False,
            "stdout_base64": base64.b64encode(b"hello\xff").decode("ascii"),
            "stderr_base64": base64.b64encode(b"warning").decode("ascii"),
            "stdout_bytes": 6,
            "stderr_bytes": 7,
            "stdout_truncated": True,
            "stderr_truncated": False,
        }
        result.update(self.result_overrides)
        return result

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.cancel_calls.append(
            {
                "endpoint": endpoint,
                "token": token,
                "command_id": command_id,
                "timeout_s": timeout_s,
            }
        )
        return {"command_id": command_id, "state": "cancelled"}

    async def aclose(self) -> None:
        self.closed = True


class BlockingEndpointTransport(FakeEndpointTransport):
    def __init__(self) -> None:
        super().__init__()
        self.get_started = asyncio.Event()
        self.release_get = asyncio.Event()

    async def get_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.get_started.set()
        await self.release_get.wait()
        return await super().get_command(
            endpoint=endpoint,
            token=token,
            command_id=command_id,
            timeout_s=timeout_s,
        )

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.release_get.set()
        return await super().cancel_command(
            endpoint=endpoint,
            token=token,
            command_id=command_id,
            timeout_s=timeout_s,
        )


class UnauthorizedOnceEndpointTransport(FakeEndpointTransport):
    def __init__(self) -> None:
        super().__init__()
        self.start_attempt_tokens: list[str] = []

    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        self.start_attempt_tokens.append(token)
        if len(self.start_attempt_tokens) == 1:
            raise LambdaMicroVMEndpointUnauthorized("expired")
        return await super().start_command(
            endpoint=endpoint,
            token=token,
            command_id=command_id,
            payload=payload,
            timeout_s=timeout_s,
        )


class FailingHealthEndpointTransport(FakeEndpointTransport):
    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> dict[str, str]:
        raise RuntimeError("not ready")


class MismatchedProtocolEndpointTransport(FakeEndpointTransport):
    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> dict[str, str]:
        return {"status": "ok", "protocol_version": "2"}


class LegacyEndpointTransport(FakeEndpointTransport):
    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> dict[str, str]:
        return {"status": "ok"}


class RunningForeverEndpointTransport(FakeEndpointTransport):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    async def get_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.get_calls.append(
            {
                "endpoint": endpoint,
                "token": token,
                "command_id": command_id,
                "timeout_s": timeout_s,
            }
        )
        if self.cancelled:
            return {
                "command_id": command_id,
                "state": "cancelled",
                "exit_code": -15,
                "timed_out": False,
                "cancelled": True,
                "stdout_base64": base64.b64encode(b"partial").decode("ascii"),
                "stderr_base64": "",
                "stdout_bytes": 7,
                "stderr_bytes": 0,
                "stdout_truncated": False,
                "stderr_truncated": False,
            }
        return {"command_id": command_id, "state": "running"}

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        result = await super().cancel_command(
            endpoint=endpoint,
            token=token,
            command_id=command_id,
            timeout_s=timeout_s,
        )
        self.cancelled = True
        return result


class FailingStartEndpointTransport(FakeEndpointTransport):
    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        await super().start_command(
            endpoint=endpoint,
            token=token,
            command_id=command_id,
            payload=payload,
            timeout_s=timeout_s,
        )
        raise LambdaMicroVMError("connection lost after start")


class SupervisorEndpointTransport:
    def __init__(self, root: Path) -> None:
        self.supervisor = CommandSupervisor(root=root)

    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> dict[str, str]:
        return dict(HEALTH_RESPONSE)

    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.supervisor.start, command_id, payload)

    async def get_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.supervisor.get, command_id)

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.supervisor.cancel, command_id)


class DelayedStartSupervisorEndpointTransport(SupervisorEndpointTransport):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.start_received = asyncio.Event()
        self.release_start = asyncio.Event()
        self.late_start_tasks: list[asyncio.Task[dict[str, Any]]] = []

    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        async def land_start() -> dict[str, Any]:
            await self.release_start.wait()
            return await asyncio.to_thread(self.supervisor.start, command_id, payload)

        task = asyncio.create_task(land_start())
        self.late_start_tasks.append(task)
        self.start_received.set()
        return await asyncio.shield(task)

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(self.supervisor.cancel, command_id)
        self.release_start.set()
        return result


@pytest.mark.anyio
async def test_lambda_microvm_runner_creates_and_executes_without_host_env_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_HOST_SECRET_SHOULD_NOT_LEAK", "hidden")
    client = FakeLambdaMicroVMClient()
    transport = FakeEndpointTransport()
    runner = await LambdaMicroVMRunner.create(
        "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
        region_name="us-west-2",
        image_version="7",
        idle_policy={
            "autoResumeEnabled": True,
            "maxIdleDurationSeconds": 900,
            "suspendedDurationSeconds": 300,
        },
        close_action="none",
        client=client,
        endpoint_transport=transport,
        poll_interval_s=0,
    )

    result = await runner.exec(
        ExecCommand.process("python3", "-V"),
        cwd="src",
        env={"VISIBLE": "yes"},
        timeout_s=10,
        stdin="input",
        output_limit_bytes=5,
    )

    assert client.run_calls == [
        {
            "imageIdentifier": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
            "imageVersion": "7",
            "idlePolicy": {
                "autoResumeEnabled": True,
                "maxIdleDurationSeconds": 900,
                "suspendedDurationSeconds": 300,
            },
        }
    ]
    assert client.token_calls == [
        {
            "microvmIdentifier": "mvm-123",
            "expirationInMinutes": 30,
            "allowedPorts": [{"port": 8080}],
        }
    ]
    assert transport.start_calls[0]["payload"] == {
        "kind": "process",
        "argv": ["python3", "-V"],
        "cwd": "/workspace/src",
        "env": {"VISIBLE": "yes"},
        "stdin_base64": base64.b64encode(b"input").decode("ascii"),
        "timeout_s": 10,
        "output_limit_bytes": 5,
    }
    assert "CAYU_HOST_SECRET_SHOULD_NOT_LEAK" not in transport.start_calls[0]["payload"]["env"]
    assert result.stdout == "hello�"
    assert result.stderr == "warning"
    assert result.exit_code == 7
    assert result.stdout_truncated is True
    assert result.stderr_truncated is False
    assert result.stdout_bytes == 6
    assert result.stderr_bytes == 7
    assert runner.microvm_id == "mvm-123"
    assert runner.image_version == "7"


@pytest.mark.anyio
async def test_lambda_microvm_runner_attaches_to_existing_microvm() -> None:
    client = FakeLambdaMicroVMClient()
    transport = FakeEndpointTransport()

    runner = await LambdaMicroVMRunner.from_existing(
        "mvm-123",
        region_name="us-west-2",
        client=client,
        endpoint_transport=transport,
        poll_interval_s=0,
    )

    assert client.get_calls == [{"microvmIdentifier": "mvm-123"}]
    assert client.run_calls == []
    assert runner.endpoint == "mvm-123.lambda-microvm.us-west-2.on.aws"
    assert runner.image_identifier == "arn:aws:lambda:us-west-2:123:microvm-image:cayu"
    assert runner.image_version == "7"
    assert transport.health_calls[0]["token"] == "token-123"


@pytest.mark.anyio
async def test_lambda_microvm_runner_waits_for_suspend_before_resuming() -> None:
    client = SuspendingLambdaMicroVMClient()

    await LambdaMicroVMRunner.from_existing(
        "mvm-123",
        client=client,
        endpoint_transport=FakeEndpointTransport(),
        poll_interval_s=0,
    )

    assert len(client.get_calls) == 2
    assert client.resume_calls == [{"microvmIdentifier": "mvm-123"}]


@pytest.mark.anyio
async def test_lambda_microvm_runner_cancels_guest_command_and_preserves_diagnostics() -> None:
    client = FakeLambdaMicroVMClient()
    transport = BlockingEndpointTransport()
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
        cancel_timeout_s=1,
        cancellation_cleanup="command",
        poll_interval_s=0,
    )

    task = asyncio.create_task(runner.exec(ExecCommand.bash("sleep 30")))
    await asyncio.wait_for(transport.get_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task

    assert len(transport.cancel_calls) == 1
    assert transport.cancel_calls[0]["command_id"] == transport.start_calls[0]["command_id"]
    artifacts = getattr(excinfo.value, "artifacts", [])
    assert artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "lambda-microvm",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 1.0,
        }
    ]


@pytest.mark.anyio
async def test_lambda_microvm_runner_tombstones_cancelled_in_flight_start(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "orphan-marker"
    transport = DelayedStartSupervisorEndpointTransport(tmp_path)
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="local.test",
        default_cwd=str(tmp_path),
        endpoint_transport=transport,
        cancel_timeout_s=1,
        cancellation_cleanup="command",
        poll_interval_s=0,
    )

    task = asyncio.create_task(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            )
        )
    )
    await asyncio.wait_for(transport.start_received.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    await asyncio.gather(*transport.late_start_tasks)

    command_id = transport.late_start_tasks[0].result()["command_id"]
    assert transport.supervisor.get(command_id)["state"] == "cancelled"
    assert marker.exists() is False
    assert getattr(excinfo.value, "artifacts", [])[0]["status"] == "completed"

    result = await runner.exec(ExecCommand.process(sys.executable, "-c", "print('reusable')"))
    assert result.stdout == "reusable\n"


@pytest.mark.anyio
@pytest.mark.parametrize("close_action", ["terminate", "suspend", "none"])
async def test_lambda_microvm_runner_applies_close_action_once(close_action: str) -> None:
    client = FakeLambdaMicroVMClient()
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        close_action=close_action,  # type: ignore[arg-type]
        endpoint_transport=FakeEndpointTransport(),
    )

    await runner.close()
    await runner.close()
    await runner.kill()

    assert len(client.terminate_calls) == (1 if close_action == "terminate" else 0)
    assert len(client.suspend_calls) == (1 if close_action == "suspend" else 0)
    with pytest.raises(RuntimeError, match="closed"):
        await runner.exec(ExecCommand.process("true"))


@pytest.mark.anyio
async def test_lambda_microvm_runner_lifecycle_methods_are_idempotent() -> None:
    client = FakeLambdaMicroVMClient()
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=FakeEndpointTransport(),
        poll_interval_s=0,
    )

    await runner.suspend()
    await runner.suspend()
    with pytest.raises(RuntimeError, match="suspended"):
        await runner.exec(ExecCommand.process("true"))
    await runner.resume()
    await runner.resume()
    await runner.terminate()
    await runner.terminate()

    assert client.suspend_calls == [{"microvmIdentifier": "mvm-123"}]
    assert client.resume_calls == [{"microvmIdentifier": "mvm-123"}]
    assert client.terminate_calls == [{"microvmIdentifier": "mvm-123"}]


@pytest.mark.anyio
async def test_lambda_microvm_runner_serializes_concurrent_lifecycle_calls() -> None:
    client = FakeLambdaMicroVMClient()
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        close_action="terminate",
        endpoint_transport=FakeEndpointTransport(),
    )

    await asyncio.gather(*(runner.terminate() for _ in range(8)))
    await asyncio.gather(*(runner.close() for _ in range(8)))

    assert client.terminate_calls == [{"microvmIdentifier": "mvm-123"}]


@pytest.mark.anyio
async def test_lambda_microvm_runner_discards_cached_token_on_close() -> None:
    client = FakeLambdaMicroVMClient()
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=FakeEndpointTransport(),
    )
    await runner._endpoint_token()

    await runner.close()

    assert runner._auth_token is None
    assert runner._auth_token_expires_at == 0.0


@pytest.mark.anyio
async def test_lambda_microvm_runner_refreshes_rejected_endpoint_token_once() -> None:
    client = FakeLambdaMicroVMClient()
    transport = UnauthorizedOnceEndpointTransport()
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
        poll_interval_s=0,
    )

    await runner.exec(ExecCommand.process("true"))

    assert transport.start_attempt_tokens == ["token-123", "token-456"]
    assert len(client.token_calls) == 2


@pytest.mark.anyio
async def test_lambda_microvm_runner_terminates_new_microvm_when_readiness_fails() -> None:
    client = FakeLambdaMicroVMClient()

    with pytest.raises(RuntimeError, match="did not become ready"):
        await LambdaMicroVMRunner.create(
            "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
            client=client,
            endpoint_transport=FailingHealthEndpointTransport(),
            ready_timeout_s=0.01,
            poll_interval_s=0,
        )

    assert client.terminate_calls == [{"microvmIdentifier": "mvm-123"}]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "transport",
    [MismatchedProtocolEndpointTransport(), LegacyEndpointTransport()],
)
async def test_lambda_microvm_runner_rejects_sidecar_protocol_mismatch(
    transport: FakeEndpointTransport,
) -> None:
    client = FakeLambdaMicroVMClient()

    with pytest.raises(LambdaMicroVMProtocolError, match="expected 1"):
        await asyncio.wait_for(
            LambdaMicroVMRunner.create(
                "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
                client=client,
                endpoint_transport=transport,
                ready_timeout_s=30,
            ),
            timeout=0.2,
        )

    assert client.terminate_calls == [{"microvmIdentifier": "mvm-123"}]


@pytest.mark.anyio
async def test_lambda_microvm_runner_composes_with_runner_workspace(tmp_path: Path) -> None:
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="local.test",
        default_cwd=str(tmp_path),
        endpoint_transport=SupervisorEndpointTransport(tmp_path),
        poll_interval_s=0.001,
    )
    workspace = RunnerWorkspace(runner, python_executable=sys.executable)

    await workspace.write_bytes("nested/file.txt", b"hello")
    read = await workspace.read_bytes("nested/file.txt")
    listed = await workspace.list("**/*.txt")
    await workspace.delete("nested/file.txt")

    assert read.content == b"hello"
    assert read.truncated is False
    assert listed.paths == ("nested/file.txt",)
    assert (await workspace.list()).paths == ()


@pytest.mark.anyio
async def test_lambda_microvm_runner_preserves_guest_spawn_error_detail(tmp_path: Path) -> None:
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="local.test",
        default_cwd=str(tmp_path),
        endpoint_transport=SupervisorEndpointTransport(tmp_path),
        poll_interval_s=0,
    )

    result = await runner.exec(
        ExecCommand.process("/definitely/missing-cayu-binary"),
        output_limit_bytes=100,
    )

    assert result.exit_code == -1
    assert "FileNotFoundError" in result.stderr
    assert "missing-cayu-binary" in result.stderr


@pytest.mark.anyio
async def test_lambda_microvm_runner_honors_sandbox_cleanup_on_command_timeout() -> None:
    client = FakeLambdaMicroVMClient()
    transport = FakeEndpointTransport(result_overrides={"timed_out": True, "exit_code": -9})
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
        timeout_cleanup="sandbox",
        cancel_timeout_s=1,
        poll_interval_s=0,
    )

    result = await runner.exec(ExecCommand.bash("sleep 30"), timeout_s=1)

    assert result.timed_out is True
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "lambda-microvm",
            "action": "kill_sandbox",
            "status": "completed",
            "timeout_s": 1.0,
        }
    ]
    assert client.terminate_calls == [{"microvmIdentifier": "mvm-123"}]
    with pytest.raises(RuntimeError, match="closed"):
        await runner.exec(ExecCommand.process("true"))


@pytest.mark.anyio
async def test_lambda_microvm_runner_enforces_host_deadline_for_wedged_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lambda_microvm_module,
        "DEFAULT_LAMBDA_MICROVM_EXEC_TIMEOUT_GRACE_SECONDS",
        0.01,
    )
    transport = RunningForeverEndpointTransport()
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
        timeout_cleanup="command",
        cancel_timeout_s=1,
        poll_interval_s=0,
    )

    result = await runner.exec(ExecCommand.bash("sleep 30"), timeout_s=1)

    assert result.exit_code == -9
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.stdout == "partial"
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "lambda-microvm",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 1.0,
        }
    ]
    assert len(transport.cancel_calls) == 1
    assert len(transport.get_calls) < 200


@pytest.mark.anyio
async def test_lambda_microvm_runner_cleans_up_ambiguous_command_start() -> None:
    transport = FailingStartEndpointTransport()
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
        cancellation_cleanup="command",
        cancel_timeout_s=1,
    )

    with pytest.raises(LambdaMicroVMError, match="connection lost"):
        await runner.exec(ExecCommand.process("true"))

    assert len(transport.cancel_calls) == 1
    assert transport.cancel_calls[0]["command_id"] == transport.start_calls[0]["command_id"]


@pytest.mark.anyio
async def test_lambda_microvm_runner_latches_after_ambiguous_start_without_cleanup() -> None:
    transport = FailingStartEndpointTransport()
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
        cancellation_cleanup="none",
    )

    with pytest.raises(LambdaMicroVMError, match="connection lost"):
        await runner.exec(ExecCommand.process("true"))

    assert transport.cancel_calls == []
    with pytest.raises(RuntimeError, match="command start was not acknowledged"):
        await runner.exec(ExecCommand.process("true"))
