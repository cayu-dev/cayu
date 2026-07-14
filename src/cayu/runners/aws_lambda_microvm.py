from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import uuid
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import httpx

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.providers._http import SharedAsyncClient
from cayu.runners._cleanup import (
    DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
    DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
    DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
    RunnerCleanupPolicy,
    RunnerCleanupResult,
    cleanup_runner_command_with_diagnostic,
    validate_cancel_timeout,
    validate_runner_cleanup_policy,
)
from cayu.runners._subprocess import (
    copy_runner_env,
    validate_output_limit,
    validate_stdin,
    validate_timeout,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    attach_cancellation_artifacts,
)

DEFAULT_LAMBDA_MICROVM_CWD = "/workspace"
DEFAULT_LAMBDA_MICROVM_PORT = 8080
DEFAULT_LAMBDA_MICROVM_AUTH_TOKEN_MINUTES = 30
DEFAULT_LAMBDA_MICROVM_POLL_INTERVAL_SECONDS = 0.1
DEFAULT_LAMBDA_MICROVM_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS = 60.0
DEFAULT_LAMBDA_MICROVM_TOKEN_REFRESH_SKEW_SECONDS = 60.0
DEFAULT_LAMBDA_MICROVM_EXEC_TIMEOUT_GRACE_SECONDS = 5.0
DEFAULT_LAMBDA_MICROVM_MIN_POLL_INTERVAL_SECONDS = 0.01
LAMBDA_MICROVM_PROTOCOL_VERSION = "1"

LambdaMicroVMCloseAction = Literal["terminate", "suspend", "none"]


class LambdaMicroVMError(RuntimeError):
    """Base error for AWS Lambda MicroVM runner failures."""


class LambdaMicroVMProtocolError(LambdaMicroVMError):
    """A control-plane or sidecar response violated the runner contract."""


class _LambdaMicroVMProtocolVersionMismatch(LambdaMicroVMProtocolError):
    """The sidecar is healthy but speaks an incompatible protocol version."""


class LambdaMicroVMEndpointUnauthorized(LambdaMicroVMError):
    """The endpoint JWE token was rejected and should be refreshed."""


class LambdaMicroVMEndpointTransport(Protocol):
    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> Mapping[str, Any]: ...

    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    async def get_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...


class HttpxLambdaMicroVMEndpointTransport:
    """Authenticated HTTPS transport for the Cayu MicroVM sidecar."""

    def __init__(self) -> None:
        self._client = SharedAsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> Mapping[str, Any]:
        response = await self._request(
            "GET", endpoint=endpoint, token=token, path="/health", timeout_s=timeout_s
        )
        return response

    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/commands",
            payload={"command_id": command_id, **payload},
            timeout_s=timeout_s,
        )

    async def get_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "GET",
            endpoint=endpoint,
            token=token,
            path=f"/v1/commands/{command_id}",
            timeout_s=timeout_s,
        )

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "DELETE",
            endpoint=endpoint,
            token=token,
            path=f"/v1/commands/{command_id}",
            timeout_s=timeout_s,
        )

    async def _request(
        self,
        method: str,
        *,
        endpoint: str,
        token: str,
        path: str,
        timeout_s: float,
        payload: dict[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        headers = {
            "X-aws-proxy-auth": require_clean_nonblank(token, "endpoint token"),
            "X-aws-proxy-port": str(DEFAULT_LAMBDA_MICROVM_PORT),
        }
        try:
            request_options: dict[str, Any] = {"headers": headers, "timeout": timeout_s}
            if payload is not None:
                request_options["json"] = payload
            response = await self._client.get().request(
                method, f"{_endpoint_base_url(endpoint)}{path}", **request_options
            )
        except httpx.RequestError as exc:
            raise LambdaMicroVMError(f"Lambda MicroVM endpoint request failed: {exc}") from exc
        if response.status_code in {401, 403}:
            raise LambdaMicroVMEndpointUnauthorized("Lambda MicroVM endpoint token was rejected.")
        if response.status_code >= 400:
            raise LambdaMicroVMError(
                f"Lambda MicroVM endpoint returned HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        try:
            decoded = response.json()
        except ValueError as exc:
            raise LambdaMicroVMProtocolError(
                "Lambda MicroVM endpoint returned invalid JSON."
            ) from exc
        if not isinstance(decoded, Mapping):
            raise LambdaMicroVMProtocolError("Lambda MicroVM endpoint response must be an object.")
        return decoded


class LambdaMicroVMRunner(Runner):
    """Execute commands through a Cayu sidecar in an AWS Lambda MicroVM."""

    isolation = "lambda-microvm"
    default_cwd = DEFAULT_LAMBDA_MICROVM_CWD

    def __init__(
        self,
        client: Any,
        *,
        microvm_id: str,
        endpoint: str,
        image_identifier: str | None = None,
        image_version: str | None = None,
        region_name: str | None = None,
        default_cwd: str = DEFAULT_LAMBDA_MICROVM_CWD,
        close_action: LambdaMicroVMCloseAction = "none",
        endpoint_transport: LambdaMicroVMEndpointTransport | None = None,
        owns_client: bool = False,
        poll_interval_s: float = DEFAULT_LAMBDA_MICROVM_POLL_INTERVAL_SECONDS,
        request_timeout_s: float = DEFAULT_LAMBDA_MICROVM_REQUEST_TIMEOUT_SECONDS,
        auth_token_expiration_minutes: int = DEFAULT_LAMBDA_MICROVM_AUTH_TOKEN_MINUTES,
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        env_overlay: Mapping[str, str] | None = None,
    ) -> None:
        if client is None:
            raise TypeError("LambdaMicroVMRunner client cannot be None.")
        self._client = client
        self._owns_client = owns_client
        self.microvm_id = require_clean_nonblank(microvm_id, "microvm_id")
        self.endpoint = require_clean_nonblank(endpoint, "endpoint")
        self.image_identifier = _optional_clean_string(image_identifier, "image_identifier")
        self.image_version = _optional_clean_string(image_version, "image_version")
        self.region_name = _optional_clean_string(region_name, "region_name")
        self.default_cwd = _validate_guest_root(default_cwd)
        self.close_action = _validate_close_action(close_action)
        self.poll_interval_s = _nonnegative_float(poll_interval_s, "poll_interval_s")
        self.request_timeout_s = _positive_float(request_timeout_s, "request_timeout_s")
        if type(auth_token_expiration_minutes) is not int or auth_token_expiration_minutes <= 0:
            raise ValueError("auth_token_expiration_minutes must be a positive integer.")
        self.auth_token_expiration_minutes = auth_token_expiration_minutes
        self.cancel_timeout_s = validate_cancel_timeout(cancel_timeout_s)
        self.cancellation_cleanup = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        self.timeout_cleanup = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        self.env_overlay = dict(env_overlay) if env_overlay else {}
        self._endpoint_transport = (
            endpoint_transport
            if endpoint_transport is not None
            else HttpxLambdaMicroVMEndpointTransport()
        )
        self._owns_endpoint_transport = endpoint_transport is None
        self._auth_token: str | None = None
        self._auth_token_expires_at = 0.0
        self._auth_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False
        self._exec_closed = False
        self._exec_closed_reason = None
        self._suspended = False
        self._termination_requested = False

    @classmethod
    async def create(
        cls,
        image_identifier: str,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
        image_version: str | None = None,
        execution_role_arn: str | None = None,
        ingress_network_connectors: list[str] | None = None,
        egress_network_connectors: list[str] | None = None,
        idle_policy: dict[str, Any] | None = None,
        maximum_duration_in_seconds: int | None = None,
        run_hook_payload: str | None = None,
        default_cwd: str = DEFAULT_LAMBDA_MICROVM_CWD,
        close_action: LambdaMicroVMCloseAction = "terminate",
        client: Any | None = None,
        endpoint_transport: LambdaMicroVMEndpointTransport | None = None,
        ready_timeout_s: float = DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS,
        poll_interval_s: float = DEFAULT_LAMBDA_MICROVM_POLL_INTERVAL_SECONDS,
        request_timeout_s: float = DEFAULT_LAMBDA_MICROVM_REQUEST_TIMEOUT_SECONDS,
        auth_token_expiration_minutes: int = DEFAULT_LAMBDA_MICROVM_AUTH_TOKEN_MINUTES,
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        env_overlay: Mapping[str, str] | None = None,
    ) -> LambdaMicroVMRunner:
        image = require_clean_nonblank(image_identifier, "image_identifier")
        control_client, owns_client = _control_client(
            client=client,
            region_name=region_name,
            profile_name=profile_name,
            endpoint_url=endpoint_url,
        )
        run_options: dict[str, Any] = {"imageIdentifier": image}
        _put_optional(run_options, "imageVersion", image_version)
        _put_optional(run_options, "executionRoleArn", execution_role_arn)
        if ingress_network_connectors is not None:
            run_options["ingressNetworkConnectors"] = _copy_string_list(
                ingress_network_connectors, "ingress_network_connectors"
            )
        if egress_network_connectors is not None:
            run_options["egressNetworkConnectors"] = _copy_string_list(
                egress_network_connectors, "egress_network_connectors"
            )
        if idle_policy is not None:
            run_options["idlePolicy"] = copy_json_value(idle_policy, "idle_policy")
        if maximum_duration_in_seconds is not None:
            if type(maximum_duration_in_seconds) is not int or maximum_duration_in_seconds <= 0:
                raise ValueError("maximum_duration_in_seconds must be a positive integer.")
            run_options["maximumDurationInSeconds"] = maximum_duration_in_seconds
        _put_optional(run_options, "runHookPayload", run_hook_payload)

        response = await asyncio.to_thread(control_client.run_microvm, **run_options)
        microvm_id, endpoint = _microvm_identity(response)
        runner = cls(
            control_client,
            microvm_id=microvm_id,
            endpoint=endpoint,
            image_identifier=_response_string(response, "imageArn") or image,
            image_version=_response_string(response, "imageVersion") or image_version,
            region_name=region_name,
            default_cwd=default_cwd,
            close_action=close_action,
            endpoint_transport=endpoint_transport,
            owns_client=owns_client,
            poll_interval_s=poll_interval_s,
            request_timeout_s=request_timeout_s,
            auth_token_expiration_minutes=auth_token_expiration_minutes,
            cancel_timeout_s=cancel_timeout_s,
            cancellation_cleanup=cancellation_cleanup,
            timeout_cleanup=timeout_cleanup,
            env_overlay=env_overlay,
        )
        try:
            await runner._wait_until_ready(ready_timeout_s)
        except BaseException:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(runner._terminate(), timeout=runner.cancel_timeout_s)
            await runner._close_transports()
            raise
        return runner

    @classmethod
    async def from_existing(
        cls,
        microvm_id: str,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
        default_cwd: str = DEFAULT_LAMBDA_MICROVM_CWD,
        close_action: LambdaMicroVMCloseAction = "none",
        client: Any | None = None,
        endpoint_transport: LambdaMicroVMEndpointTransport | None = None,
        ready_timeout_s: float = DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS,
        poll_interval_s: float = DEFAULT_LAMBDA_MICROVM_POLL_INTERVAL_SECONDS,
        request_timeout_s: float = DEFAULT_LAMBDA_MICROVM_REQUEST_TIMEOUT_SECONDS,
        auth_token_expiration_minutes: int = DEFAULT_LAMBDA_MICROVM_AUTH_TOKEN_MINUTES,
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        env_overlay: Mapping[str, str] | None = None,
    ) -> LambdaMicroVMRunner:
        identifier = require_clean_nonblank(microvm_id, "microvm_id")
        control_client, owns_client = _control_client(
            client=client,
            region_name=region_name,
            profile_name=profile_name,
            endpoint_url=endpoint_url,
        )
        response = await asyncio.to_thread(control_client.get_microvm, microvmIdentifier=identifier)
        response_id, endpoint = _microvm_identity(response)
        if response_id != identifier:
            raise LambdaMicroVMProtocolError("get_microvm returned the wrong MicroVM id.")
        state = _required_response_string(response, "state")
        if state in {"TERMINATING", "TERMINATED"}:
            raise LambdaMicroVMError(f"Cannot attach to Lambda MicroVM in state {state}.")
        runner = cls(
            control_client,
            microvm_id=identifier,
            endpoint=endpoint,
            image_identifier=_response_string(response, "imageArn"),
            image_version=_response_string(response, "imageVersion"),
            region_name=region_name,
            default_cwd=default_cwd,
            close_action=close_action,
            endpoint_transport=endpoint_transport,
            owns_client=owns_client,
            poll_interval_s=poll_interval_s,
            request_timeout_s=request_timeout_s,
            auth_token_expiration_minutes=auth_token_expiration_minutes,
            cancel_timeout_s=cancel_timeout_s,
            cancellation_cleanup=cancellation_cleanup,
            timeout_cleanup=timeout_cleanup,
            env_overlay=env_overlay,
        )
        try:
            await runner._prepare_existing_for_attach(state, ready_timeout_s)
        except BaseException:
            await runner._close_transports()
            raise
        return runner

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        if type(command) is not ExecCommand:
            raise TypeError("LambdaMicroVMRunner command must be an ExecCommand.")
        self._ensure_exec_open()
        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=False)
        if self.env_overlay:
            # Applied last: enforced egress configuration must win over model env.
            environment.update(self.env_overlay)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        output_limit = validate_output_limit(output_limit_bytes)
        command_id = f"cmd-{uuid.uuid4()}"
        payload: dict[str, Any] = {
            "kind": command.kind,
            "cwd": working_dir,
            "env": environment,
            "stdin_base64": (
                base64.b64encode(standard_input.encode("utf-8")).decode("ascii")
                if standard_input is not None
                else None
            ),
            "timeout_s": timeout,
            "output_limit_bytes": output_limit,
        }
        if command.kind == "process":
            payload["argv"] = list(command.argv or [])
        else:
            payload["shell"] = command.shell
        handle = _LambdaMicroVMCommandHandle(self, command_id)
        start_acknowledged = False
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + timeout + DEFAULT_LAMBDA_MICROVM_EXEC_TIMEOUT_GRACE_SECONDS
            if timeout is not None
            else None
        )
        try:
            async with asyncio.timeout_at(deadline):
                await self._endpoint_start(command_id, payload)
                start_acknowledged = True
                while True:
                    response = await self._endpoint_get(command_id)
                    state = _required_response_string(response, "state")
                    if state in {"completed", "cancelled", "failed"}:
                        result = _exec_result(response)
                        if result.timed_out:
                            cleanup = await self._cleanup_exec_command(
                                handle=handle,
                                policy=self.timeout_cleanup,
                                start_acknowledged=start_acknowledged,
                            )
                            result.artifacts.append(cleanup.artifact)
                        return result
                    if state not in {"accepted", "running"}:
                        raise LambdaMicroVMProtocolError(
                            f"Lambda MicroVM command returned unsupported state: {state}"
                        )
                    await asyncio.sleep(
                        max(
                            self.poll_interval_s,
                            DEFAULT_LAMBDA_MICROVM_MIN_POLL_INTERVAL_SECONDS,
                        )
                    )
        except asyncio.CancelledError as exc:
            cleanup = await self._cleanup_exec_command(
                handle=handle,
                policy=self.cancellation_cleanup,
                start_acknowledged=start_acknowledged,
            )
            attach_cancellation_artifacts(exc, [cleanup.artifact])
            raise
        except TimeoutError:
            cleanup = await self._cleanup_exec_command(
                handle=handle,
                policy=self.timeout_cleanup,
                start_acknowledged=start_acknowledged,
            )
            result = await self._host_timeout_result(command_id, cleanup)
            result.artifacts.append(cleanup.artifact)
            return result
        except Exception:
            await self._cleanup_exec_command(
                handle=handle,
                policy=self.cancellation_cleanup,
                start_acknowledged=start_acknowledged,
            )
            raise

    async def _cleanup_exec_command(
        self,
        *,
        handle: _LambdaMicroVMCommandHandle,
        policy: RunnerCleanupPolicy,
        start_acknowledged: bool,
    ) -> RunnerCleanupResult:
        cleanup = await cleanup_runner_command_with_diagnostic(
            self,
            handle=handle,
            adapter="lambda-microvm",
            timeout_s=self.cancel_timeout_s,
            policy=policy,
        )
        self._apply_cleanup_result(cleanup)
        if not start_acknowledged and policy == "none":
            self._close_exec(
                "Lambda MicroVM command start was not acknowledged; command state is unknown"
            )
        return cleanup

    async def _host_timeout_result(
        self,
        command_id: str,
        cleanup: RunnerCleanupResult,
    ) -> ExecResult:
        result = ExecResult(exit_code=-9, timed_out=True)
        artifact = cleanup.artifact
        if artifact.get("action") != "kill_command" or artifact.get("status") != "completed":
            return result
        try:
            response = await asyncio.wait_for(
                self._endpoint_get(command_id),
                timeout=self.cancel_timeout_s,
            )
            state = _required_response_string(response, "state")
            if state not in {"completed", "cancelled", "failed"}:
                return result
            terminal = _exec_result(response)
        except Exception:
            return result
        return ExecResult(
            stdout=terminal.stdout,
            stderr=terminal.stderr,
            exit_code=-9,
            timed_out=True,
            stdout_truncated=terminal.stdout_truncated,
            stderr_truncated=terminal.stderr_truncated,
            stdout_bytes=terminal.stdout_bytes,
            stderr_bytes=terminal.stderr_bytes,
        )

    async def suspend(self) -> None:
        async with self._lifecycle_lock:
            await self._suspend()

    async def _suspend(self) -> None:
        if self._suspended or self._termination_requested:
            return
        self._ensure_lifecycle_open()
        await asyncio.to_thread(self._client.suspend_microvm, microvmIdentifier=self.microvm_id)
        self._suspended = True
        self._close_exec("Lambda MicroVM is suspended")

    async def resume(self) -> None:
        async with self._lifecycle_lock:
            self._ensure_lifecycle_open()
            if self._termination_requested:
                raise RuntimeError("Cannot resume a terminated Lambda MicroVM.")
            if not self._suspended:
                response = await asyncio.to_thread(
                    self._client.get_microvm, microvmIdentifier=self.microvm_id
                )
                response_id, endpoint = _microvm_identity(response)
                if response_id != self.microvm_id or endpoint != self.endpoint:
                    raise LambdaMicroVMProtocolError(
                        "get_microvm returned different identity on resume."
                    )
                state = _required_response_string(response, "state")
                if state in {"PENDING", "RUNNING"}:
                    return
                if state == "SUSPENDING":
                    await self._prepare_existing_for_attach(
                        state, DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS
                    )
                    self._suspended = False
                    self._open_exec()
                    return
                if state != "SUSPENDED":
                    raise LambdaMicroVMError(f"Cannot resume Lambda MicroVM in state {state}.")
            await asyncio.to_thread(self._client.resume_microvm, microvmIdentifier=self.microvm_id)
            self._auth_token = None
            self._auth_token_expires_at = 0.0
            await self._wait_until_ready(DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS)
            self._suspended = False
            self._open_exec()

    async def terminate(self) -> None:
        async with self._lifecycle_lock:
            if self._termination_requested:
                return
            self._ensure_lifecycle_open()
            await self._terminate()

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            if self.close_action == "terminate":
                await self._terminate()
            elif self.close_action == "suspend":
                await self._suspend()
            elif self.close_action != "none":
                raise AssertionError(
                    f"Unsupported Lambda MicroVM close action: {self.close_action}"
                )
            self._closed = True
            await self._close_transports()

    async def kill(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            await self._terminate()
            self._closed = True
            await self._close_transports()

    async def _terminate(self) -> None:
        if self._termination_requested:
            return
        await asyncio.to_thread(self._client.terminate_microvm, microvmIdentifier=self.microvm_id)
        self._termination_requested = True
        self._close_exec("Lambda MicroVM termination was requested")

    async def _close_transports(self) -> None:
        self._auth_token = None
        self._auth_token_expires_at = 0.0
        if self._owns_endpoint_transport:
            close = getattr(self._endpoint_transport, "aclose", None)
            if callable(close):
                await close()
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    async def _wait_until_ready(self, ready_timeout_s: float) -> None:
        timeout = _positive_float(ready_timeout_s, "ready_timeout_s")
        deadline = asyncio.get_running_loop().time() + timeout
        last_error: Exception | None = None
        while True:
            try:
                await self._endpoint_health()
                return
            except _LambdaMicroVMProtocolVersionMismatch:
                raise
            except Exception as exc:
                last_error = exc
            if asyncio.get_running_loop().time() >= deadline:
                raise LambdaMicroVMError(
                    f"Lambda MicroVM did not become ready within {timeout:g} seconds: {last_error}"
                ) from last_error
            await asyncio.sleep(max(self.poll_interval_s, 0.05))

    async def _prepare_existing_for_attach(
        self, initial_state: str, ready_timeout_s: float
    ) -> None:
        timeout = _positive_float(ready_timeout_s, "ready_timeout_s")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        state = initial_state
        while state == "SUSPENDING":
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise LambdaMicroVMError(
                    f"Lambda MicroVM did not finish suspending within {timeout:g} seconds."
                )
            await asyncio.sleep(min(max(self.poll_interval_s, 0.05), remaining))
            response = await asyncio.to_thread(
                self._client.get_microvm, microvmIdentifier=self.microvm_id
            )
            response_id, endpoint = _microvm_identity(response)
            if response_id != self.microvm_id or endpoint != self.endpoint:
                raise LambdaMicroVMProtocolError(
                    "get_microvm returned different identity while attaching."
                )
            state = _required_response_string(response, "state")
        if state == "SUSPENDED":
            await asyncio.to_thread(self._client.resume_microvm, microvmIdentifier=self.microvm_id)
        elif state not in {"PENDING", "RUNNING"}:
            raise LambdaMicroVMError(f"Cannot attach to Lambda MicroVM in state {state}.")
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise LambdaMicroVMError(
                f"Lambda MicroVM did not become ready within {timeout:g} seconds."
            )
        await self._wait_until_ready(remaining)

    async def _endpoint_health(self) -> None:
        response = await self._endpoint_call("health")
        status = _required_response_string(response, "status")
        if status != "ok":
            raise LambdaMicroVMProtocolError("Lambda MicroVM health response was not ready.")
        version = response.get("protocol_version")
        if version != LAMBDA_MICROVM_PROTOCOL_VERSION:
            reported_version = version if type(version) is str else repr(version)
            raise _LambdaMicroVMProtocolVersionMismatch(
                "Lambda MicroVM sidecar protocol version mismatch: "
                f"expected {LAMBDA_MICROVM_PROTOCOL_VERSION}, got {reported_version}."
            )

    async def _endpoint_start(self, command_id: str, payload: dict[str, Any]) -> Mapping[str, Any]:
        response = await self._endpoint_call(
            "start_command", command_id=command_id, payload=payload
        )
        returned_id = _required_response_string(response, "command_id")
        if returned_id != command_id:
            raise LambdaMicroVMProtocolError("Lambda MicroVM start returned the wrong command id.")
        return response

    async def _endpoint_get(self, command_id: str) -> Mapping[str, Any]:
        return await self._endpoint_call("get_command", command_id=command_id)

    async def _endpoint_cancel(self, command_id: str) -> Mapping[str, Any]:
        return await self._endpoint_call("cancel_command", command_id=command_id)

    async def _endpoint_call(self, method_name: str, **kwargs: Any) -> Mapping[str, Any]:
        method = getattr(self._endpoint_transport, method_name)
        for attempt in range(2):
            token = await self._endpoint_token(force_refresh=attempt == 1)
            try:
                result = await method(
                    endpoint=self.endpoint,
                    token=token,
                    timeout_s=self.request_timeout_s,
                    **kwargs,
                )
            except LambdaMicroVMEndpointUnauthorized:
                if attempt == 0:
                    continue
                raise
            if not isinstance(result, Mapping):
                raise LambdaMicroVMProtocolError(
                    f"Lambda MicroVM {method_name} response must be an object."
                )
            return result
        raise AssertionError("unreachable endpoint retry loop")

    async def _endpoint_token(self, *, force_refresh: bool = False) -> str:
        loop = asyncio.get_running_loop()
        if (
            not force_refresh
            and self._auth_token is not None
            and loop.time() < self._auth_token_expires_at
        ):
            return self._auth_token
        async with self._auth_lock:
            if (
                not force_refresh
                and self._auth_token is not None
                and loop.time() < self._auth_token_expires_at
            ):
                return self._auth_token
            response = await asyncio.to_thread(
                self._client.create_microvm_auth_token,
                microvmIdentifier=self.microvm_id,
                expirationInMinutes=self.auth_token_expiration_minutes,
                allowedPorts=[{"port": DEFAULT_LAMBDA_MICROVM_PORT}],
            )
            token = _auth_token(response)
            lifetime_s = self.auth_token_expiration_minutes * 60
            self._auth_token = token
            self._auth_token_expires_at = loop.time() + max(
                1.0, lifetime_s - DEFAULT_LAMBDA_MICROVM_TOKEN_REFRESH_SKEW_SECONDS
            )
            return token

    def _ensure_lifecycle_open(self) -> None:
        if self._closed:
            raise RuntimeError("LambdaMicroVMRunner is closed.")


class _LambdaMicroVMCommandHandle:
    def __init__(self, runner: LambdaMicroVMRunner, command_id: str) -> None:
        self.runner = runner
        self.command_id = command_id

    async def kill(self) -> None:
        response = await self.runner._endpoint_cancel(self.command_id)
        state = _required_response_string(response, "state")
        if state not in {"cancelled", "completed", "failed", "not_found"}:
            raise LambdaMicroVMProtocolError(
                f"Lambda MicroVM cancellation did not reach a terminal state: {state}"
            )


def _exec_result(response: Mapping[str, Any]) -> ExecResult:
    exit_code = response.get("exit_code")
    if type(exit_code) is not int:
        raise LambdaMicroVMProtocolError("Lambda MicroVM result exit_code must be an integer.")
    stdout_bytes = _output_byte_count(response, "stdout_bytes")
    stderr_bytes = _output_byte_count(response, "stderr_bytes")
    return ExecResult(
        stdout=_decode_output(response, "stdout_base64", stdout_bytes),
        stderr=_decode_output(response, "stderr_base64", stderr_bytes),
        exit_code=exit_code,
        timed_out=_required_bool(response, "timed_out"),
        cancelled=_optional_bool(response, "cancelled", False),
        stdout_truncated=_required_bool(response, "stdout_truncated"),
        stderr_truncated=_required_bool(response, "stderr_truncated"),
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
    )


def _output_byte_count(response: Mapping[str, Any], key: str) -> int:
    value = response.get(key)
    if type(value) is not int or value < 0:
        raise LambdaMicroVMProtocolError(
            f"Lambda MicroVM result {key} must be a nonnegative integer."
        )
    return value


def _decode_output(response: Mapping[str, Any], key: str, byte_count: int) -> str:
    raw = response.get(key, "")
    if type(raw) is not str:
        raise LambdaMicroVMProtocolError(f"Lambda MicroVM result {key} must be a string.")
    try:
        decoded = base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise LambdaMicroVMProtocolError(
            f"Lambda MicroVM result {key} was invalid base64."
        ) from exc
    if byte_count < len(decoded):
        raise LambdaMicroVMProtocolError(
            f"Lambda MicroVM result byte total for {key} must cover decoded output bytes."
        )
    return decoded.decode("utf-8", errors="replace")


def _control_client(
    *,
    client: Any | None,
    region_name: str | None,
    profile_name: str | None,
    endpoint_url: str | None,
) -> tuple[Any, bool]:
    if client is not None:
        if profile_name is not None or endpoint_url is not None:
            raise ValueError(
                "An injected client cannot be combined with profile_name or endpoint_url."
            )
        return client, False
    boto3 = _boto3_module()
    session_options: dict[str, Any] = {}
    if profile_name is not None:
        session_options["profile_name"] = require_clean_nonblank(profile_name, "profile_name")
    session = boto3.Session(**session_options)
    client_options: dict[str, Any] = {}
    if region_name is not None:
        client_options["region_name"] = require_clean_nonblank(region_name, "region_name")
    if endpoint_url is not None:
        client_options["endpoint_url"] = require_clean_nonblank(endpoint_url, "endpoint_url")
    return session.client("lambda-microvms", **client_options), True


def _boto3_module() -> Any:
    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError as exc:
        if exc.name != "boto3":
            raise
        raise RuntimeError(
            "LambdaMicroVMRunner requires the optional AWS dependencies; install cayu[aws]."
        ) from exc


def _microvm_identity(response: Any) -> tuple[str, str]:
    if not isinstance(response, Mapping):
        raise LambdaMicroVMProtocolError("run_microvm response must be an object.")
    return (
        _required_response_string(response, "microvmId"),
        _required_response_string(response, "endpoint"),
    )


def _auth_token(response: Any) -> str:
    if not isinstance(response, Mapping):
        raise LambdaMicroVMProtocolError("auth-token response must be an object.")
    value = response.get("authToken")
    if isinstance(value, Mapping):
        value = value.get("X-aws-proxy-auth")
    if type(value) is not str or not value.strip():
        raise LambdaMicroVMProtocolError("auth-token response omitted X-aws-proxy-auth.")
    return value


def _endpoint_base_url(endpoint: str) -> str:
    value = require_clean_nonblank(endpoint, "endpoint").rstrip("/")
    if value.startswith("https://"):
        return value
    if "://" in value:
        raise ValueError("Lambda MicroVM endpoint must use HTTPS.")
    return f"https://{value}"


def _validate_guest_root(value: str) -> str:
    root = require_clean_nonblank(value, "default_cwd")
    if not root.startswith("/"):
        raise ValueError("LambdaMicroVMRunner default_cwd must be an absolute guest path.")
    return root.rstrip("/") or "/"


def _validate_close_action(value: LambdaMicroVMCloseAction) -> LambdaMicroVMCloseAction:
    if value not in {"terminate", "suspend", "none"}:
        raise ValueError("Lambda MicroVM close_action must be one of: terminate, suspend, none.")
    return value


def _required_response_string(response: Mapping[str, Any], key: str) -> str:
    value = response.get(key)
    if type(value) is not str or not value.strip():
        raise LambdaMicroVMProtocolError(f"Lambda MicroVM response {key} must be a string.")
    return value


def _response_string(response: Any, key: str) -> str | None:
    if not isinstance(response, Mapping):
        return None
    value = response.get(key)
    return value if type(value) is str and value.strip() else None


def _required_bool(response: Mapping[str, Any], key: str) -> bool:
    value = response.get(key)
    if type(value) is not bool:
        raise LambdaMicroVMProtocolError(f"Lambda MicroVM response {key} must be a boolean.")
    return value


def _optional_bool(response: Mapping[str, Any], key: str, default: bool) -> bool:
    value = response.get(key, default)
    if type(value) is not bool:
        raise LambdaMicroVMProtocolError(f"Lambda MicroVM response {key} must be a boolean.")
    return value


def _put_optional(target: dict[str, Any], key: str, value: str | None) -> None:
    if value is not None:
        target[key] = require_clean_nonblank(value, key)


def _copy_string_list(values: list[str], field_name: str) -> list[str]:
    if type(values) is not list:
        raise TypeError(f"{field_name} must be a list.")
    return [require_clean_nonblank(value, field_name) for value in values]


def _optional_clean_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_clean_nonblank(value, field_name)


def _positive_float(value: float, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return float(value)


def _nonnegative_float(value: float, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return float(value)
