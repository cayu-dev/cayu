from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

import cayu.runners.aws_lambda_microvm as lambda_microvm_module
from cayu import ExecCommand, LambdaMicroVMRunner, RunnerWorkspace
from cayu.runners import (
    HttpxLambdaMicroVMEndpointTransport,
    LambdaMicroVMEndpointTransientError,
    LambdaMicroVMEndpointUnauthorized,
    LambdaMicroVMError,
    LambdaMicroVMProtocolError,
)
from cayu.testing import verify_provider_credential_isolation
from cayu.vaults import REDACTED_SECRET, SecretRedactor

SUPERVISOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "aws"
    / "lambda_microvm_sidecar"
    / "supervisor.py"
)
SUPERVISOR_SPEC = importlib.util.spec_from_file_location(
    "cayu_lambda_microvm_runner_supervisor", SUPERVISOR_PATH
)
assert SUPERVISOR_SPEC is not None and SUPERVISOR_SPEC.loader is not None


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["suspend", "terminate"])
@pytest.mark.parametrize("late", [False, True])
async def test_public_lifecycle_retains_dispatched_thread(action, late):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    client = FakeLambdaMicroVMClient()
    original = getattr(client, action + "_microvm")

    def blocked(**kwargs):
        started.set()
        assert release.wait(5)
        try:
            return original(**kwargs)
        finally:
            finished.set()

    setattr(client, action + "_microvm", blocked)
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=FakeEndpointTransport(),
        cancel_timeout_s=0.05 if late else 1,
    )
    task = asyncio.create_task(getattr(runner, action)())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel("owner")
        await asyncio.sleep(0)
        task.cancel("second")
        with pytest.raises(RuntimeError):
            await runner.exec(ExecCommand.process("true"))
        with pytest.raises(RuntimeError):
            await runner.resume()
        with pytest.raises(RuntimeError):
            runner.reopen_exec()
        if not late:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 2
        if late:
            assert runner.lifecycle_state == "poisoned"
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        while runner._public_lifecycle_task is not None:
            await asyncio.sleep(0)
        if late:
            with pytest.raises(RuntimeError, match="poisoned"):
                runner.reopen_exec()
        elif action == "suspend":
            await runner.resume()
            assert runner.lifecycle_state == "reusable"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize("cleanup_failure", ["none", "terminate", "transport"])
async def test_cancelled_allocation_reclaims_late_vm_and_owned_client(
    monkeypatch, caplog, cleanup_failure
):
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    client = FakeLambdaMicroVMClient()
    original = client.run_microvm

    def blocked(**kwargs):
        started.set()
        assert release.wait(5)
        return original(**kwargs)

    client.run_microvm = blocked
    client.close = closed.set
    terminate = client.terminate_microvm
    previous_cleanups = set(lambda_microvm_module._ABANDONED_LAMBDA_ALLOCATIONS)
    if cleanup_failure == "terminate":

        def fail_termination(**kwargs):
            client.terminate_calls.append(kwargs)
            raise RuntimeError("private-provider-cleanup-canary")

        client.terminate_microvm = fail_termination
    elif cleanup_failure == "transport":

        def fail_close():
            raise RuntimeError("private-provider-cleanup-canary")

        client.close = fail_close
    monkeypatch.setattr(lambda_microvm_module, "_control_client", lambda **kwargs: (client, True))
    task = asyncio.create_task(
        LambdaMicroVMRunner.create(
            "image",
            endpoint_transport=FakeEndpointTransport(),
            cancel_timeout_s=0.02,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel("allocation owner")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert not client.terminate_calls
        release.set()
        owners = lambda_microvm_module._ABANDONED_LAMBDA_ALLOCATIONS - previous_cleanups
        assert len(owners) == 1
        owner = next(iter(owners))
        assert owner.task is not None
        await asyncio.wait_for(asyncio.shield(owner.task), 2)
        assert client.terminate_calls == [{"microvmIdentifier": "mvm-123"}]
        assert not client.token_calls
        if cleanup_failure != "none":
            assert not closed.is_set()
            assert "drain_abandoned_allocations is required" in caplog.text
            assert "private-provider-cleanup-canary" not in caplog.text
            client.terminate_microvm = terminate
            client.close = closed.set
            reads = len(client.get_calls)
            assert await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2) == 0
            assert len(client.terminate_calls) == (2 if cleanup_failure == "terminate" else 1)
            if cleanup_failure == "transport":
                assert len(client.get_calls) == reads
        assert closed.is_set()
        assert len(client.run_calls) == 1
        assert owner not in lambda_microvm_module._ABANDONED_LAMBDA_ALLOCATIONS
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        client.terminate_microvm = terminate
        client.close = closed.set
        await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2)


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_drain", [False, True])
async def test_abandoned_allocation_drains_join_in_flight_termination(cancel_drain):
    allocating = threading.Event()
    allocate_release = threading.Event()
    terminating = threading.Event()
    terminate_release = threading.Event()
    client = FakeLambdaMicroVMClient()
    create = client.run_microvm
    terminate = client.terminate_microvm
    calls = []

    def allocate(**kwargs):
        allocating.set()
        assert allocate_release.wait(5)
        return create(**kwargs)

    def blocked_terminate(**kwargs):
        calls.append(kwargs)
        terminating.set()
        assert terminate_release.wait(5)
        return terminate(**kwargs)

    client.run_microvm = allocate
    client.terminate_microvm = blocked_terminate
    task = asyncio.create_task(
        LambdaMicroVMRunner.create(
            "image",
            client=client,
            endpoint_transport=FakeEndpointTransport(),
            cancel_timeout_s=0.02,
        )
    )
    try:
        assert await asyncio.to_thread(allocating.wait, 2)
        task.cancel("allocation")
        with pytest.raises(asyncio.CancelledError):
            await task
        allocate_release.set()
        assert await asyncio.to_thread(terminating.wait, 2)
        drain = asyncio.create_task(
            LambdaMicroVMRunner.drain_abandoned_allocations(
                timeout_s=2 if cancel_drain else 0.02,
            )
        )
        if cancel_drain:
            await asyncio.sleep(0)
            drain.cancel("drain")
            with pytest.raises(asyncio.CancelledError):
                await drain
            assert drain.cancelled() and drain.cancelling() == 1
        else:
            assert await drain == 1
        assert await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=0.02) == 1
        assert len(calls) == 1
        terminate_release.set()
        assert await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2) == 0
        assert len(calls) == len(client.run_calls) == 1
    finally:
        allocate_release.set()
        terminate_release.set()
        await asyncio.gather(task, return_exceptions=True)
        await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2)


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_failed_allocation_retries_unbound_client_close(monkeypatch, cancel_owner):
    started = threading.Event()
    release = threading.Event()
    client = FakeLambdaMicroVMClient()
    close_calls = []
    allocation_calls = []
    repaired = False

    def allocate(**kwargs):
        allocation_calls.append(kwargs)
        started.set()
        assert release.wait(5)
        raise RuntimeError("allocation rejected")

    def close():
        close_calls.append(True)
        if not repaired:
            raise RuntimeError("client close failed")

    client.run_microvm = allocate
    client.close = close
    monkeypatch.setattr(lambda_microvm_module, "_control_client", lambda **kwargs: (client, True))
    previous = set(lambda_microvm_module._ABANDONED_LAMBDA_ALLOCATIONS)
    task = asyncio.create_task(LambdaMicroVMRunner.create("image", cancel_timeout_s=0.02))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if cancel_owner:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
        else:
            release.set()
            with pytest.raises(ExceptionGroup, match="allocation failed"):
                await task
        release.set()
        (owner,) = lambda_microvm_module._ABANDONED_LAMBDA_ALLOCATIONS - previous
        assert owner.task is not None
        await asyncio.wait_for(asyncio.shield(owner.task), 2)
        assert owner.runner is None
        assert owner in lambda_microvm_module._ABANDONED_LAMBDA_ALLOCATIONS
        calls_before_retry = len(close_calls)
        repaired = True
        assert await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2) == 0
        assert len(close_calls) == calls_before_retry + 1
        assert len(allocation_calls) == 1
        assert not client.terminate_calls
    finally:
        repaired = True
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2)


@pytest.mark.anyio
@pytest.mark.parametrize("factory", ["create", "from_existing"])
@pytest.mark.parametrize(
    "options",
    [
        {"cancel_timeout_s": 0},
        {"cancel_timeout_s": True},
        {"close_action": "invalid"},
        {"ready_timeout_s": 0},
        {"ready_timeout_s": float("nan")},
        {"poll_interval_s": float("inf")},
        {"poll_interval_s": float("nan")},
        {"request_timeout_s": float("inf")},
        {"request_timeout_s": -1},
        {"auth_token_expiration_minutes": True},
        {"auth_token_expiration_minutes": 0},
        {"cancellation_cleanup": "invalid"},
        {"timeout_cleanup": "invalid"},
        {"region_name": ""},
        {"env_overlay": object()},
    ],
)
async def test_lambda_constructor_validation_precedes_provider_access(
    factory: str,
    options: dict[str, Any],
) -> None:
    client = FakeLambdaMicroVMClient()
    with pytest.raises((TypeError, ValueError)):
        if factory == "create":
            await LambdaMicroVMRunner.create("image", client=client, **options)
        else:
            await LambdaMicroVMRunner.from_existing("mvm-123", client=client, **options)
    assert client.run_calls == []
    assert client.get_calls == []
    assert client.token_calls == []


@pytest.mark.parametrize("invalid_text", ("/workspace\x00bad", "/workspace\ud800bad"))
def test_lambda_microvm_runner_rejects_nonportable_default_cwd(invalid_text: str) -> None:
    with pytest.raises(ValueError, match="default_cwd"):
        LambdaMicroVMRunner(
            object(),
            microvm_id="mvm-test",
            endpoint="local.test",
            default_cwd=invalid_text,
            endpoint_transport=object(),  # type: ignore[arg-type]
        )

    runner = LambdaMicroVMRunner(
        object(),
        microvm_id="mvm-test",
        endpoint="local.test",
        endpoint_transport=object(),  # type: ignore[arg-type]
    )
    runner.default_cwd = invalid_text
    with pytest.raises(ValueError, match="default_cwd"):
        runner.resolve_cwd()


SUPERVISOR_MODULE = importlib.util.module_from_spec(SUPERVISOR_SPEC)
sys.modules[SUPERVISOR_SPEC.name] = SUPERVISOR_MODULE
SUPERVISOR_SPEC.loader.exec_module(SUPERVISOR_MODULE)
CommandSupervisor = SUPERVISOR_MODULE.CommandSupervisor
HEALTH_RESPONSE = {
    "status": "ok",
    "protocol_version": lambda_microvm_module.LAMBDA_MICROVM_PROTOCOL_VERSION,
}


@pytest.mark.anyio
@pytest.mark.parametrize("before_resume", [False, True])
async def test_cancelled_attachment_retains_resume_and_restores_suspension(
    monkeypatch, before_resume
):
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    client = FakeLambdaMicroVMClient()
    client.state = "SUSPENDED"
    resume = client.get_microvm if before_resume else client.resume_microvm
    suspend = client.suspend_microvm

    def blocked(**kwargs):
        started.set()
        assert release.wait(5)
        return resume(**kwargs)

    if before_resume:
        client.get_microvm = blocked
    else:
        client.resume_microvm = blocked
    client.close = closed.set
    monkeypatch.setattr(lambda_microvm_module, "_control_client", lambda **kwargs: (client, True))
    task = asyncio.create_task(
        LambdaMicroVMRunner.from_existing(
            "mvm-123",
            endpoint_transport=FakeEndpointTransport(),
            cancel_timeout_s=0.02,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel("attach")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert not closed.is_set() and not client.terminate_calls
        reads = len(client.get_calls)
        with pytest.raises(LambdaMicroVMError, match="still pending"):
            await LambdaMicroVMRunner.from_existing("mvm-123", client=FakeLambdaMicroVMClient())
        assert len(client.get_calls) == reads
        assert await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=0.02) == 1
        if not before_resume:

            def fail_suspend(**kwargs):
                raise RuntimeError("suspension unavailable")

            client.suspend_microvm = fail_suspend
        release.set()
        if not before_resume:
            assert await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2) == 1
            with pytest.raises(LambdaMicroVMError, match="still pending"):
                await LambdaMicroVMRunner.from_existing("mvm-123", client=FakeLambdaMicroVMClient())
            client.suspend_microvm = suspend
        assert await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2) == 0
        assert client.state == "SUSPENDED"
        assert len(client.resume_calls) == len(client.suspend_calls) == (0 if before_resume else 1)
        assert closed.is_set() and not client.terminate_calls and not client.run_calls
        fresh_client = FakeLambdaMicroVMClient()
        fresh_client.state = "SUSPENDED"
        monkeypatch.setattr(
            lambda_microvm_module, "_control_client", lambda **kwargs: (fresh_client, False)
        )
        replacement = await LambdaMicroVMRunner.from_existing(
            "mvm-123", endpoint_transport=FakeEndpointTransport(), close_action="none"
        )
        assert replacement.lifecycle_state == "reusable"
        assert fresh_client.state == "RUNNING" and len(fresh_client.resume_calls) == 1
        assert await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2) == 0
        assert fresh_client.state == "RUNNING"
        await replacement.close()
    finally:
        client.suspend_microvm = suspend
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2)


@pytest.mark.anyio
@pytest.mark.parametrize("allocation", [False, True])
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_known_failure_survives_later_cleanup_timeout(monkeypatch, allocation, cancel_owner):
    from cayu.runners._cleanup import runner_cancellation_failure

    started = threading.Event()
    release = threading.Event()
    primary = RuntimeError("known failure")
    client = FakeLambdaMicroVMClient()
    terminate = client.terminate_microvm
    runner = None
    if allocation:

        async def fail_ready(self, timeout):
            raise primary

        monkeypatch.setattr(LambdaMicroVMRunner, "_wait_until_ready", fail_ready)

        def blocked(**kwargs):
            started.set()
            assert release.wait(5)
            return terminate(**kwargs)

        client.terminate_microvm = blocked
        operation = LambdaMicroVMRunner.create(
            "image",
            client=client,
            endpoint_transport=FakeEndpointTransport(),
            ready_timeout_s=0.05,
            request_timeout_s=0.05,
            cancel_timeout_s=0.05,
        )
    else:

        def failed(**kwargs):
            raise primary

        client.terminate_microvm = failed

        class Transport(FakeEndpointTransport):
            async def aclose(self):
                started.set()
                await asyncio.to_thread(release.wait, 5)

        runner = LambdaMicroVMRunner(
            client,
            microvm_id="mvm-123",
            endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
            endpoint_transport=Transport(),
            cancel_timeout_s=0.05,
        )
        runner._owns_endpoint_transport = True
        operation = runner.kill()
    task = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if cancel_owner:
            task.cancel("owner")
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert task.cancelled() and task.cancelling() == 1
            failure = runner_cancellation_failure(caught.value)
        else:
            with pytest.raises(BaseExceptionGroup) as caught:
                await task
            failure = caught.value
        assert isinstance(failure, BaseExceptionGroup)
        assert failure.exceptions[0] is primary
        assert len(failure.exceptions) == 2
        assert isinstance(failure.exceptions[1], TimeoutError)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if allocation:
            await LambdaMicroVMRunner.drain_abandoned_allocations(timeout_s=2)
        elif runner is not None and runner._terminal_lifecycle_task is not None:
            await asyncio.wait_for(asyncio.shield(runner._terminal_lifecycle_task), 2)


class FakeLambdaMicroVMClient:
    def __init__(self) -> None:
        self.run_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.token_calls: list[dict[str, Any]] = []
        self.suspend_calls: list[dict[str, Any]] = []
        self.resume_calls: list[dict[str, Any]] = []
        self.terminate_calls: list[dict[str, Any]] = []
        self.state = "RUNNING"

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
            "state": self.state,
            "imageArn": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
            "imageVersion": "7",
        }

    def create_microvm_auth_token(self, **kwargs: Any) -> dict[str, Any]:
        self.token_calls.append(kwargs)
        token = "token-123" if len(self.token_calls) == 1 else "token-456"
        return {"authToken": {"X-aws-proxy-auth": token}}

    def suspend_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.suspend_calls.append(kwargs)
        self.state = "SUSPENDED"
        return {}

    def resume_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.resume_calls.append(kwargs)
        self.state = "RUNNING"
        return {}

    def terminate_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.terminate_calls.append(kwargs)
        self.state = "TERMINATED"
        return {}


@pytest.mark.anyio
@pytest.mark.parametrize("factory", ("create", "from_existing"))
@pytest.mark.parametrize("invalid_text", ("/workspace\x00bad", "/workspace\ud800bad"))
async def test_lambda_microvm_factory_rejects_nonportable_default_cwd_before_provider_call(
    factory: str,
    invalid_text: str,
) -> None:
    client = FakeLambdaMicroVMClient()

    with pytest.raises(ValueError, match="default_cwd"):
        if factory == "create":
            await LambdaMicroVMRunner.create(
                "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
                client=client,
                default_cwd=invalid_text,
            )
        else:
            await LambdaMicroVMRunner.from_existing(
                "mvm-123",
                client=client,
                default_cwd=invalid_text,
            )

    assert client.run_calls == []
    assert client.get_calls == []
    assert client.token_calls == []


class SuspendingLambdaMicroVMClient(FakeLambdaMicroVMClient):
    def get_microvm(self, **kwargs: Any) -> dict[str, Any]:
        response = super().get_microvm(**kwargs)
        response["state"] = "SUSPENDING" if len(self.get_calls) == 1 else "SUSPENDED"
        return response


class TerminatingLambdaMicroVMClient(FakeLambdaMicroVMClient):
    def get_microvm(self, **kwargs: Any) -> dict[str, Any]:
        response = super().get_microvm(**kwargs)
        response["state"] = "TERMINATING" if len(self.get_calls) == 1 else "TERMINATED"
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


@pytest.mark.anyio
async def test_lambda_http_transport_omits_untrusted_error_body() -> None:
    secret = "lambda-http-error-boundary-secret"

    class ClientOwner:
        def __init__(self) -> None:
            self.client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        500,
                        request=request,
                        text="x" * 995 + secret,
                    )
                )
            )

        def get(self) -> httpx.AsyncClient:
            return self.client

        async def aclose(self) -> None:
            await self.client.aclose()

    transport = HttpxLambdaMicroVMEndpointTransport()
    owner = ClientOwner()
    transport._client = owner
    try:
        with pytest.raises(LambdaMicroVMError) as exc_info:
            await transport.health(
                endpoint="example.lambda-microvm.us-east-1.on.aws",
                token="endpoint-token",
                timeout_s=1,
            )
    finally:
        await transport.aclose()

    message = str(exc_info.value)
    assert message == "Lambda MicroVM endpoint returned HTTP 500; response body omitted"
    assert secret not in message
    assert secret[:10] not in message


@pytest.mark.anyio
async def test_lambda_runner_redacts_complete_output_and_omits_pretruncated_output() -> None:
    secret = "lambda-output-boundary-secret"
    complete = f"prefix:{secret}:suffix".encode()
    complete_transport = FakeEndpointTransport(
        result_overrides={
            "stdout_base64": base64.b64encode(complete).decode("ascii"),
            "stdout_bytes": len(complete),
            "stdout_truncated": False,
        }
    )
    complete_runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-complete",
        endpoint="complete.lambda-microvm.us-east-1.on.aws",
        endpoint_transport=complete_transport,
        poll_interval_s=0,
    )

    complete_result = await complete_runner.exec_redacted(
        ExecCommand.process("echo", "ignored"),
        redactor=SecretRedactor(secret),
        output_limit_bytes=128,
    )

    assert complete_result.stdout == f"prefix:{REDACTED_SECRET}:suffix"
    assert complete_result.stdout_bytes == len(complete)
    assert complete_result.stdout_truncated is False
    assert complete_transport.start_calls[0]["payload"]["omit_truncated_output"] is True

    truncated_transport = FakeEndpointTransport(
        result_overrides={
            "stdout_base64": base64.b64encode(secret[:12].encode()).decode("ascii"),
            "stdout_bytes": len(secret),
            "stdout_truncated": True,
        }
    )
    truncated_runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-truncated",
        endpoint="truncated.lambda-microvm.us-east-1.on.aws",
        endpoint_transport=truncated_transport,
        poll_interval_s=0,
    )

    truncated_result = await truncated_runner.exec_redacted(
        ExecCommand.process("echo", "ignored"),
        redactor=SecretRedactor(secret),
        output_limit_bytes=12,
    )

    assert truncated_result.stdout == ""
    assert truncated_result.stdout_bytes == len(secret)
    assert truncated_result.stdout_truncated is True


@pytest.mark.anyio
@pytest.mark.parametrize("failure_count", [1, 2, 3])
async def test_command_state_transient_retry_is_bounded_without_redispatch(failure_count: int):
    class Transport(FakeEndpointTransport):
        polls = 0

        async def get_command(self, **kwargs: Any) -> dict[str, Any]:
            self.polls += 1
            if self.polls <= failure_count:
                raise LambdaMicroVMEndpointTransientError("temporary transport failure")
            return await super().get_command(**kwargs)

    transport = Transport()
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
    )
    if failure_count == 3:
        with pytest.raises(LambdaMicroVMEndpointTransientError):
            await runner.exec(ExecCommand.process("true"))
        assert len(transport.cancel_calls) == 1
    else:
        result = await runner.exec(ExecCommand.process("true"))
        assert result.exit_code == 7
        assert not transport.cancel_calls
    assert transport.polls == min(failure_count + 1, 3)
    assert len(transport.start_calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("entrance", ["start_command", "cancel_command"])
async def test_mutating_endpoint_operations_do_not_retry_transient_failure(entrance: str):
    transport = FakeEndpointTransport()
    calls = 0

    async def reject(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise LambdaMicroVMEndpointTransientError("temporary transport failure")

    setattr(transport, entrance, reject)
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
    )
    if entrance == "start_command":
        with pytest.raises(LambdaMicroVMEndpointTransientError):
            await runner.exec(ExecCommand.process("true"))
    else:
        # A public timeout reaches cancellation after a successfully acknowledged start.
        async def timeout_poll(**kwargs: Any) -> dict[str, Any]:
            raise TimeoutError

        transport.get_command = timeout_poll
        result = await runner.exec(ExecCommand.process("true"))
        assert result.timed_out
        assert runner.lifecycle_state == "poisoned"
    assert calls == 1


@pytest.mark.anyio
async def test_opaque_poll_is_bounded_and_cannot_start_a_late_retry():
    release = threading.Event()
    started = asyncio.Event()
    completed = asyncio.Event()
    loop = asyncio.get_running_loop()

    class Transport(FakeEndpointTransport):
        polls = 0

        async def get_command(self, **kwargs: Any) -> dict[str, Any]:
            self.polls += 1

            def read():
                loop.call_soon_threadsafe(started.set)
                release.wait(timeout=2)
                loop.call_soon_threadsafe(completed.set)
                raise LambdaMicroVMEndpointTransientError("late unavailable result")

            return await asyncio.to_thread(read)

    transport = Transport()
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
        request_timeout_s=0.02,
    )
    task = asyncio.create_task(runner.exec(ExecCommand.process("true")))
    try:
        await started.wait()
        result = await asyncio.wait_for(task, timeout=1)
        assert result.timed_out
        assert not completed.is_set()
        assert transport.polls == 1
        assert len(transport.cancel_calls) == 1
        assert len(runner._command_poll_tasks) == 1
        pending_task = next(iter(runner._command_poll_tasks.values()))[1]
        release.set()
        await asyncio.wait_for(asyncio.shield(pending_task), 1)
        assert transport.polls == 1
        assert not runner._command_poll_tasks
    finally:
        release.set()


@pytest.mark.anyio
async def test_poll_does_not_dispatch_after_token_refresh_consumes_deadline(monkeypatch):
    release = asyncio.Event()
    refreshing = asyncio.Event()
    transport = FakeEndpointTransport()
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
        request_timeout_s=0.02,
    )
    original = runner._endpoint_token
    calls = 0
    polls = 0

    async def token(*, force_refresh=False):
        nonlocal calls
        calls += 1
        if calls == 2:
            refreshing.set()
            await release.wait()
        return await original(force_refresh=force_refresh)

    async def read(**kwargs):
        nonlocal polls
        polls += 1
        raise AssertionError("expired poll must not dispatch")

    monkeypatch.setattr(runner, "_endpoint_token", token)
    monkeypatch.setattr(transport, "get_command", read)
    task = asyncio.create_task(runner.exec(ExecCommand.process("true")))
    try:
        await asyncio.wait_for(refreshing.wait(), 1)
        result = await asyncio.wait_for(task, 1)
        assert result.timed_out
        pending = next(iter(runner._command_poll_tasks.values()))[1]
        release.set()
        await asyncio.wait_for(asyncio.shield(pending), 1)
        assert polls == 0
        assert not runner._command_poll_tasks
    finally:
        release.set()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error_type",
    [LambdaMicroVMProtocolError, LambdaMicroVMError, LambdaMicroVMEndpointUnauthorized],
)
async def test_command_state_permanent_failure_is_not_retried(error_type):
    transport = FakeEndpointTransport()
    calls = 0

    async def reject(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise error_type("permanent failure")

    transport.get_command = reject
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
    )
    with pytest.raises(error_type):
        await runner.exec(ExecCommand.process("true"))
    assert calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("extra_bytes", [0, 1])
async def test_endpoint_stream_enforces_exact_body_ceiling(monkeypatch, extra_bytes):
    wire = b'{"status":"ok"}'
    ceiling = len(wire)
    monkeypatch.setattr(lambda_microvm_module, "LAMBDA_MICROVM_MAX_RESPONSE_BYTES", ceiling)
    closed = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield wire
            if extra_bytes:
                yield b" "

        async def aclose(self):
            closed.set()

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))
    )

    class Owner:
        def get(self):
            return client

        async def aclose(self):
            await client.aclose()

    transport = HttpxLambdaMicroVMEndpointTransport()
    transport._client = Owner()
    try:
        if extra_bytes:
            with pytest.raises(LambdaMicroVMProtocolError, match="byte ceiling"):
                await transport.health(endpoint="example.on.aws", token="token", timeout_s=1)
        else:
            assert await transport.health(
                endpoint="example.on.aws", token="token", timeout_s=1
            ) == {"status": "ok"}
        assert closed.is_set()
    finally:
        await transport.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 401, 429, 501, 503])
async def test_endpoint_http_transient_classification_is_allowlisted(status):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, text="untrusted failure body")
        )
    )

    class Owner:
        def get(self):
            return client

        async def aclose(self):
            await client.aclose()

    transport = HttpxLambdaMicroVMEndpointTransport()
    transport._client = Owner()
    try:
        with pytest.raises(LambdaMicroVMError) as raised:
            await transport.health(endpoint="example.on.aws", token="token", timeout_s=1)
        assert isinstance(raised.value, LambdaMicroVMEndpointTransientError) == (
            status in {429, 503}
        )
        assert "untrusted failure body" not in str(raised.value)
    finally:
        await transport.aclose()


@pytest.mark.anyio
async def test_kill_waits_for_terminal_state_not_only_termination_acknowledgement():
    probing = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class Client(FakeLambdaMicroVMClient):
        def get_microvm(self, **kwargs: Any) -> dict[str, Any]:
            loop.call_soon_threadsafe(probing.set)
            release.wait(timeout=2)
            return super().get_microvm(**kwargs)

    client = Client()
    runner = LambdaMicroVMRunner(
        client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=FakeEndpointTransport(),
    )
    task = asyncio.create_task(runner.kill())
    try:
        await probing.wait()
        assert len(client.terminate_calls) == 1
        assert not task.done()
        assert not runner.is_closed
        assert runner.lifecycle_state == "closing"
        release.set()
        await task
        assert runner.is_closed
    finally:
        release.set()


@pytest.mark.anyio
@pytest.mark.parametrize("output_size", [4, 5, 7])
async def test_runner_rejects_oversized_output_before_base64_decode(monkeypatch, output_size):
    raw = base64.b64encode(b"x" * output_size).decode("ascii")
    monkeypatch.setattr(lambda_microvm_module, "LAMBDA_MICROVM_MAX_OUTPUT_BYTES", 4)
    monkeypatch.setattr(lambda_microvm_module, "LAMBDA_MICROVM_MAX_ENCODED_OUTPUT_BYTES", 8)
    decode = base64.b64decode
    decoded_calls = []

    def bounded_decode(value, **kwargs):
        decoded_calls.append(value)
        return decode(value, **kwargs)

    monkeypatch.setattr(base64, "b64decode", bounded_decode)
    transport = FakeEndpointTransport(
        result_overrides={
            "stdout_base64": raw,
            "stdout_bytes": output_size,
            "stderr_base64": "",
            "stderr_bytes": 0,
        }
    )
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=transport,
    )
    if output_size > 4:
        with pytest.raises(LambdaMicroVMProtocolError, match="byte ceiling"):
            await runner.exec(ExecCommand.process("true"), output_limit_bytes=None)
        assert not decoded_calls
    else:
        result = await runner.exec(ExecCommand.process("true"), output_limit_bytes=None)
        assert result.stdout == "xxxx"
        assert result.stdout_bytes == 4
        assert result.stdout_truncated
    assert transport.start_calls[0]["payload"]["output_limit_bytes"] == 4


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
        return {"status": "ok", "protocol_version": "1"}


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
        "execution_profile": "agent",
        "kind": "process",
        "argv": ["python3", "-V"],
        "cwd": "/workspace/src",
        "env": {"VISIBLE": "yes"},
        "stdin_base64": base64.b64encode(b"input").decode("ascii"),
        "timeout_s": 10,
        "output_limit_bytes": 5,
        "omit_truncated_output": False,
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
async def test_lambda_microvm_runner_marks_trusted_control_commands() -> None:
    transport = FakeEndpointTransport()
    runner = LambdaMicroVMRunner(
        FakeLambdaMicroVMClient(),
        microvm_id="mvm-123",
        endpoint="local.test",
        endpoint_transport=transport,
        poll_interval_s=0,
    )

    await runner.exec_system(ExecCommand.process("mount", "-a"))

    assert transport.start_calls[0]["payload"]["execution_profile"] == "trusted"


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
async def test_lambda_microvm_reconnect_passes_provider_credential_isolation_probe(
    provider_credential_canaries,
) -> None:
    class ProbeEndpointTransport(FakeEndpointTransport):
        async def get_command(self, **kwargs: Any) -> dict[str, Any]:
            self.get_calls.append(dict(kwargs))
            environment = self.start_calls[-1]["payload"]["env"]
            stdout = json.dumps(
                {
                    "environment": environment,
                    "auth_paths": {},
                    "auth_scan_complete": True,
                    "provider_canary_matches": [],
                    "detector_control_match": True,
                },
                sort_keys=True,
            ).encode()
            return {
                "command_id": kwargs["command_id"],
                "state": "completed",
                "exit_code": 0,
                "timed_out": False,
                "stdout_base64": base64.b64encode(stdout).decode("ascii"),
                "stderr_base64": "",
                "stdout_bytes": len(stdout),
                "stderr_bytes": 0,
                "stdout_truncated": False,
                "stderr_truncated": False,
            }

    client = FakeLambdaMicroVMClient()
    transport = ProbeEndpointTransport()
    runner = await LambdaMicroVMRunner.from_existing(
        "mvm-123",
        region_name="us-west-2",
        client=client,
        endpoint_transport=transport,
        poll_interval_s=0,
    )

    evidence = await verify_provider_credential_isolation(
        runner,
        adapter="lambda_microvm",
        scope="isolated_guest",
        provider_canaries=provider_credential_canaries.values,
        operational_env={
            "CAYU_PROBE_VISIBLE": provider_credential_canaries.positive_env["CAYU_PROBE_VISIBLE"]
        },
        workload_env={
            "CAYU_WORKLOAD_TOKEN": provider_credential_canaries.positive_env["CAYU_WORKLOAD_TOKEN"]
        },
        guest_cwd="/workspace",
        guest_auth_search_paths={"mounted_workspace": "/workspace"},
    )

    assert evidence.status == "verified"
    assert "os.walk(root" in repr(transport.start_calls[-1]["payload"])
    assert provider_credential_canaries.positive_env.items() <= (
        transport.start_calls[-1]["payload"]["env"].items()
    )
    assert all(
        value not in repr(transport.start_calls)
        for value in provider_credential_canaries.values.values()
    )


@pytest.mark.anyio
async def test_lambda_microvm_runner_applies_trusted_env_overlay_last() -> None:
    client = FakeLambdaMicroVMClient()
    transport = FakeEndpointTransport()
    runner = await LambdaMicroVMRunner.create(
        "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
        client=client,
        endpoint_transport=transport,
        poll_interval_s=0,
        close_action="none",
        env_overlay={
            "HTTPS_PROXY": "http://10.0.1.10:8443",
            "VIRTUAL_TOKEN": "cayu_virtual",
        },
    )

    await runner.exec(
        ExecCommand.process("true"),
        env={"HTTPS_PROXY": "http://attacker.invalid", "CALLER": "kept"},
    )

    assert transport.start_calls[0]["payload"]["env"] == {
        "HTTPS_PROXY": "http://10.0.1.10:8443",
        "VIRTUAL_TOKEN": "cayu_virtual",
        "CALLER": "kept",
    }


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
    await asyncio.wait_for(transport.get_started.wait(), timeout=10)
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
async def test_lambda_microvm_runner_waits_for_positive_lifecycle_quiescence() -> None:
    suspended_client = SuspendingLambdaMicroVMClient()
    suspended_runner = LambdaMicroVMRunner(
        suspended_client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=FakeEndpointTransport(),
        poll_interval_s=0,
    )
    await suspended_runner.suspend()
    assert len(suspended_client.get_calls) == 2
    await suspended_runner.wait_until_suspended(timeout_s=1)
    assert len(suspended_client.get_calls) == 3

    terminated_client = TerminatingLambdaMicroVMClient()
    terminated_runner = LambdaMicroVMRunner(
        terminated_client,
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        endpoint_transport=FakeEndpointTransport(),
        poll_interval_s=0,
    )
    await terminated_runner.terminate()
    assert len(terminated_client.get_calls) == 2
    await terminated_runner.wait_until_terminated(timeout_s=1)
    assert len(terminated_client.get_calls) == 3


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
@pytest.mark.parametrize("action", ["close", "kill"])
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_lambda_finalization_preserves_failures_and_closes_every_transport(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    cancel_owner: bool,
) -> None:
    primary = RuntimeError("termination failed")
    http_failure = RuntimeError("endpoint close failed")
    client_failure = RuntimeError("client close failed")
    started = asyncio.Event()
    release = threading.Event()
    calls: list[str] = []
    loop = asyncio.get_running_loop()

    class Client(FakeLambdaMicroVMClient):
        def terminate_microvm(self, **kwargs: Any) -> dict[str, Any]:
            calls.append("terminate")
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            raise primary

        def close(self) -> None:
            calls.append("client_close")
            raise client_failure

    class Transport(FakeEndpointTransport):
        async def aclose(self) -> None:
            calls.append("http_close")
            raise http_failure

    monkeypatch.setattr(lambda_microvm_module, "HttpxLambdaMicroVMEndpointTransport", Transport)
    runner = LambdaMicroVMRunner(
        Client(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        owns_client=True,
        close_action="terminate",
    )
    task = asyncio.create_task(getattr(runner, action)())
    try:
        await started.wait()
        assert runner.lifecycle_state == "closing"
        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec(ExecCommand.process("true"))
        if cancel_owner:
            task.cancel("original cancellation")
            await asyncio.sleep(0)
            task.cancel("another cancellation")
        release.set()
        if cancel_owner:
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert task.cancelled()
            assert task.cancelling() == 2
            assert raised.value.args == ("original cancellation",)
            failure = raised.value.__cause__
        else:
            with pytest.raises(BaseExceptionGroup) as raised_group:
                await task
            failure = raised_group.value
        assert isinstance(failure, BaseExceptionGroup)
        assert failure.exceptions[0] is primary
        nested = failure.exceptions[1]
        assert isinstance(nested, BaseExceptionGroup)
        assert nested.exceptions == (http_failure, client_failure)
        assert calls == ["terminate", "http_close", "client_close"]
        assert runner.lifecycle_state == "poisoned"
        with pytest.raises(RuntimeError, match="permanently poisoned"):
            runner.reopen_exec()
    finally:
        release.set()


@pytest.mark.anyio
async def test_lambda_close_timeout_retains_single_owner_until_thread_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    calls: list[str] = []

    class Client(FakeLambdaMicroVMClient):
        def terminate_microvm(self, **kwargs: Any) -> dict[str, Any]:
            calls.append("terminate")
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            return super().terminate_microvm(**kwargs)

        def close(self) -> None:
            calls.append("client_close")

    transport = FakeEndpointTransport()
    monkeypatch.setattr(
        lambda_microvm_module, "HttpxLambdaMicroVMEndpointTransport", lambda: transport
    )
    runner = LambdaMicroVMRunner(
        Client(),
        microvm_id="mvm-123",
        endpoint="mvm-123.lambda-microvm.us-west-2.on.aws",
        owns_client=True,
        close_action="terminate",
        cancel_timeout_s=0.02,
    )
    task = asyncio.create_task(runner.close())
    try:
        await started.wait()
        with pytest.raises(TimeoutError):
            await task
        with pytest.raises(TimeoutError):
            await runner.close()
        assert calls == ["terminate"]
        assert runner.lifecycle_state == "poisoned"
        assert not transport.closed
        release.set()
        await runner.close()
        assert runner.is_closed
        assert transport.closed
        assert calls == ["terminate", "client_close"]
    finally:
        release.set()


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

    with pytest.raises(LambdaMicroVMProtocolError, match="expected 2"):
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
