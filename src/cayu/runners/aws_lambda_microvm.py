from __future__ import annotations

import asyncio
import base64
import importlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, Protocol

import httpx

from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import (
    copy_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.providers._http import SharedAsyncClient
from cayu.runners._cleanup import (
    DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
    DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
    DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
    RunnerCleanupPolicy,
    RunnerCleanupResult,
    RunnerFailureProgress,
    attach_runner_cancellation_failure,
    cleanup_runner_command_with_diagnostic,
    validate_cancel_timeout,
    validate_runner_cleanup_policy,
)
from cayu.runners._redacted_output import redact_completed_exec_result
from cayu.runners._subprocess import (
    copy_runner_env,
    remove_runner_env,
    validate_output_limit,
    validate_stdin,
    validate_timeout,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerSystemExecutionMode,
    _clean_runner_preflight,
    _clear_preflight_traceback_frames,
    _contains_runner_fatal_signal,
    attach_cancellation_artifacts,
    copy_exec_command,
)
from cayu.vaults import SecretRedactor

DEFAULT_LAMBDA_MICROVM_CWD = "/workspace"
DEFAULT_LAMBDA_MICROVM_PORT = 8080
DEFAULT_LAMBDA_MICROVM_AUTH_TOKEN_MINUTES = 30
DEFAULT_LAMBDA_MICROVM_POLL_INTERVAL_SECONDS = 0.1
DEFAULT_LAMBDA_MICROVM_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS = 60.0
DEFAULT_LAMBDA_MICROVM_TOKEN_REFRESH_SKEW_SECONDS = 60.0
DEFAULT_LAMBDA_MICROVM_EXEC_TIMEOUT_GRACE_SECONDS = 5.0
DEFAULT_LAMBDA_MICROVM_MIN_POLL_INTERVAL_SECONDS = 0.01
LAMBDA_MICROVM_PROTOCOL_VERSION = "2"
LAMBDA_MICROVM_MAX_OUTPUT_BYTES = 16 * 1024 * 1024
LAMBDA_MICROVM_MAX_ENCODED_OUTPUT_BYTES = 4 * ((LAMBDA_MICROVM_MAX_OUTPUT_BYTES + 2) // 3)
LAMBDA_MICROVM_MAX_RESPONSE_BYTES = 2 * LAMBDA_MICROVM_MAX_ENCODED_OUTPUT_BYTES + 64 * 1024
_LAMBDA_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_LAMBDA_POLL_ATTEMPTS = 3
_CommandPollTask = asyncio.Task[CapturedAwaitableOutcome[Mapping[str, Any]]]
_PENDING_LAMBDA_POLLS: set[_CommandPollTask] = set()
_PENDING_LAMBDA_LIFECYCLES: set[asyncio.Task[CapturedAwaitableOutcome[None]]] = set()
_PENDING_LAMBDA_ALLOCATIONS: set[asyncio.Task[CapturedAwaitableOutcome[LambdaMicroVMRunner]]] = (
    set()
)
_LOGGER = logging.getLogger(__name__)
_ABANDONED_LAMBDA_ALLOCATIONS: set[_LambdaAllocationReclamation] = set()
_LAMBDA_ATTACHMENT_OWNERS: dict[str, _LambdaAllocationReclamation] = {}


@dataclass(eq=False)
class _LambdaAllocationReclamation:
    runner_type: type[LambdaMicroVMRunner]
    client: Any = None
    owns_client: bool = False
    client_closed: bool = False
    allocation: asyncio.Task[CapturedAwaitableOutcome[LambdaMicroVMRunner]] | None = None
    runner: LambdaMicroVMRunner | None = None
    task: asyncio.Task[CapturedAwaitableOutcome[None]] | None = None
    termination_confirmed: bool = False
    observed: bool = False
    progress: RunnerFailureProgress = field(default_factory=RunnerFailureProgress)
    attachment: bool = False
    restore_suspended: bool = False
    abandoned: bool = False
    attachment_identifier: str | None = None

    def release_attachment(self) -> None:
        identifier = self.attachment_identifier
        if identifier is not None and _LAMBDA_ATTACHMENT_OWNERS.get(identifier) is self:
            del _LAMBDA_ATTACHMENT_OWNERS[identifier]

    async def reclaim(self) -> None:
        assert self.allocation is not None
        allocation = await self.allocation
        if self.runner is None and allocation.error is not None:
            await self.close_unbound_client()
            return
        await self.reclaim_runner()

    async def close_unbound_client(self) -> None:
        if self.owns_client and not self.client_closed:
            close = getattr(self.client, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        self.client_closed = True

    async def reclaim_runner(self) -> None:
        runner = self.runner
        assert runner is not None
        if runner.is_closed:
            return
        # This is an outstanding allocation owner, not a terminal close(). Keep
        # the control client usable until deletion is positively reconciled.
        if not self.termination_confirmed:
            if not self.attachment or self.restore_suspended:
                if self.attachment:
                    await runner._suspend()
                else:
                    await runner._terminate()
                await runner._wait_for_lifecycle_state(
                    terminal_state="SUSPENDED" if self.attachment else "TERMINATED",
                    transitional_states={"RUNNING", "SUSPENDING", "SUSPENDED", "TERMINATING"},
                    timeout_s=runner.cancel_timeout_s,
                )
            self.termination_confirmed = True
        await runner._close_transports(progress=self.progress)
        runner._closed = True

    def start(self) -> asyncio.Task[CapturedAwaitableOutcome[None]]:
        if (
            self.task is not None
            and self.task.done()
            and not self.task.cancelled()
            and self.task.result().error is None
        ):
            self.settled(self.task)
            return self.task
        if self.task is None or self.task.done():
            self.observed = False
            self.task = asyncio.create_task(capture_awaitable_outcome(self.reclaim))
            self.task.add_done_callback(self.settled)
        return self.task

    def settled(self, task: asyncio.Task[CapturedAwaitableOutcome[None]]) -> None:
        if task is not self.task or self.observed:
            return
        self.observed = True
        if task.cancelled() or task.result().error is not None:
            _LOGGER.error(
                "Lambda MicroVM late allocation cleanup failed; drain_abandoned_allocations is required."
            )
            return
        _ABANDONED_LAMBDA_ALLOCATIONS.discard(self)
        self.release_attachment()


LambdaMicroVMCloseAction = Literal["terminate", "suspend", "none"]


class LambdaMicroVMError(RuntimeError):
    """Base error for AWS Lambda MicroVM runner failures."""


class LambdaMicroVMProtocolError(LambdaMicroVMError):
    """A control-plane or sidecar response violated the runner contract."""


class _LambdaMicroVMProtocolVersionMismatch(LambdaMicroVMProtocolError):
    """The sidecar is healthy but speaks an incompatible protocol version."""


class LambdaMicroVMEndpointUnauthorized(LambdaMicroVMError):
    """The endpoint JWE token was rejected and should be refreshed."""


class LambdaMicroVMEndpointTransientError(LambdaMicroVMError):
    """A transport failure eligible for bounded command-state read retry only."""


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
            "Accept-Encoding": "identity",
        }
        try:
            request_options: dict[str, Any] = {"headers": headers, "timeout": timeout_s}
            if payload is not None:
                request_options["json"] = payload
            async with asyncio.timeout(timeout_s):
                async with self._client.get().stream(
                    method, f"{_endpoint_base_url(endpoint)}{path}", **request_options
                ) as response:
                    if response.status_code in {401, 403}:
                        raise LambdaMicroVMEndpointUnauthorized(
                            "Lambda MicroVM endpoint token was rejected."
                        )
                    if response.status_code >= 400:
                        error_type = (
                            LambdaMicroVMEndpointTransientError
                            if response.status_code in _LAMBDA_TRANSIENT_HTTP_STATUSES
                            else LambdaMicroVMError
                        )
                        raise error_type(
                            f"Lambda MicroVM endpoint returned HTTP {response.status_code}; "
                            "response body omitted"
                        )
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise LambdaMicroVMProtocolError(
                            "Lambda MicroVM endpoint returned unsupported content encoding."
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes(
                        chunk_size=min(64 * 1024, LAMBDA_MICROVM_MAX_RESPONSE_BYTES + 1)
                    ):
                        if len(chunk) > LAMBDA_MICROVM_MAX_RESPONSE_BYTES - len(body):
                            raise LambdaMicroVMProtocolError(
                                "Lambda MicroVM endpoint response exceeds its byte ceiling."
                            )
                        body.extend(chunk)
        except (httpx.TimeoutException, httpx.NetworkError, TimeoutError) as exc:
            raise LambdaMicroVMEndpointTransientError(
                "Lambda MicroVM endpoint request temporarily failed."
            ) from exc
        except httpx.RequestError as exc:
            raise LambdaMicroVMError("Lambda MicroVM endpoint request failed.") from exc
        try:
            decoded = json.loads(body)
        except ValueError:
            decoded = None
        finally:
            body.clear()
        if decoded is None:
            raise LambdaMicroVMProtocolError("Lambda MicroVM endpoint returned invalid JSON.")
        if not isinstance(decoded, Mapping):
            raise LambdaMicroVMProtocolError("Lambda MicroVM endpoint response must be an object.")
        return decoded


class LambdaMicroVMRunner(Runner):
    """Execute commands through a Cayu sidecar in an AWS Lambda MicroVM."""

    isolation = "lambda-microvm"
    system_execution_mode: RunnerSystemExecutionMode = "separate"

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("lambda-microvm", self.microvm_id)

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
        self._public_lifecycle_task: asyncio.Task[CapturedAwaitableOutcome[None]] | None = None
        self._public_lifecycle_action: str | None = None
        self._command_poll_tasks: dict[str, tuple[float, _CommandPollTask]] = {}
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
        guest_root = _validate_guest_root(default_cwd)
        environment_overlay = _preflight_runner_creation(
            region_name=region_name,
            close_action=close_action,
            ready_timeout_s=ready_timeout_s,
            poll_interval_s=poll_interval_s,
            request_timeout_s=request_timeout_s,
            auth_token_expiration_minutes=auth_token_expiration_minutes,
            cancel_timeout_s=cancel_timeout_s,
            cancellation_cleanup=cancellation_cleanup,
            timeout_cleanup=timeout_cleanup,
            env_overlay=env_overlay,
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

        control_client, owns_client = _control_client(
            client=client,
            region_name=region_name,
            profile_name=profile_name,
            endpoint_url=endpoint_url,
        )
        reclamation = _LambdaAllocationReclamation(cls, control_client, owns_client)

        async def allocate() -> LambdaMicroVMRunner:
            runner: LambdaMicroVMRunner | None = None
            try:
                response = await asyncio.to_thread(control_client.run_microvm, **run_options)
                microvm_id, endpoint = _microvm_identity(response)
                runner = cls(
                    control_client,
                    microvm_id=microvm_id,
                    endpoint=endpoint,
                    image_identifier=_response_string(response, "imageArn") or image,
                    image_version=_response_string(response, "imageVersion") or image_version,
                    region_name=region_name,
                    default_cwd=guest_root,
                    close_action=close_action,
                    endpoint_transport=endpoint_transport,
                    owns_client=owns_client,
                    poll_interval_s=poll_interval_s,
                    request_timeout_s=request_timeout_s,
                    auth_token_expiration_minutes=auth_token_expiration_minutes,
                    cancel_timeout_s=cancel_timeout_s,
                    cancellation_cleanup=cancellation_cleanup,
                    timeout_cleanup=timeout_cleanup,
                    env_overlay=environment_overlay,
                )
                reclamation.runner = runner
                if not reclamation.abandoned:
                    await runner._wait_until_ready(ready_timeout_s)
                return runner
            except BaseException as primary:
                reclamation.progress.failures = (primary,)
                try:
                    if runner is not None:
                        await reclamation.reclaim_runner()
                    else:
                        await reclamation.close_unbound_client()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "Lambda MicroVM allocation failed.", [primary, cleanup]
                    ) from None
                raise

        return await cls._observe_binding(
            reclamation, allocate, ready_timeout_s + request_timeout_s, cancel_timeout_s
        )

    @staticmethod
    async def _observe_binding(
        reclamation: _LambdaAllocationReclamation,
        operation: Callable[[], Awaitable[LambdaMicroVMRunner]],
        timeout_s: float,
        cancel_timeout_s: float | None,
    ) -> LambdaMicroVMRunner:
        try:
            task = asyncio.create_task(capture_awaitable_outcome(operation))
        except BaseException:
            reclamation.release_attachment()
            raise
        reclamation.allocation = task
        _PENDING_LAMBDA_ALLOCATIONS.add(task)
        task.add_done_callback(_PENDING_LAMBDA_ALLOCATIONS.discard)
        outcome = await await_shielded_task_outcome(
            task,
            timeout_s=timeout_s,
            timeout_after_cancellation_s=validate_cancel_timeout(cancel_timeout_s),
        )
        failure = outcome.error if outcome.result is None else outcome.result.error
        if (
            outcome.cancellation is not None
            or outcome.timed_out
            or (
                failure is not None
                and (
                    (reclamation.runner is not None and not reclamation.runner.is_closed)
                    or (reclamation.runner is None and not reclamation.client_closed)
                )
            )
        ):
            reclamation.abandoned = True
            _ABANDONED_LAMBDA_ALLOCATIONS.add(reclamation)
            reclamation.start()
            if outcome.timed_out:
                failure = reclamation.progress.with_timeout(
                    "Lambda MicroVM allocation has not settled."
                )
        else:
            reclamation.release_attachment()
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
        if failure is not None and _contains_runner_fatal_signal(failure):
            raise failure
        if outcome.cancellation is not None:
            if failure is not None:
                attach_runner_cancellation_failure(outcome.cancellation, failure)
            raise outcome.cancellation
        if failure is not None:
            raise failure
        assert outcome.result is not None and outcome.result.result is not None
        return outcome.result.result

    @classmethod
    async def drain_abandoned_allocations(
        cls, *, timeout_s: float = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS
    ) -> int:
        """Retry process-local abandoned allocations; return the unsettled count.

        Each call retries settled failures once and joins still-running cleanup.
        Timeout or caller cancellation never cancels provider work. Applications
        should drain again when a provider failure has been repaired.
        """
        timeout = validate_cancel_timeout(timeout_s)
        owners = [
            owner for owner in _ABANDONED_LAMBDA_ALLOCATIONS if issubclass(owner.runner_type, cls)
        ]
        if owners:
            tasks = {owner.start() for owner in owners}
            done, _ = await asyncio.wait(tasks, timeout=timeout)
            for owner in owners:
                if owner.task in done:
                    assert owner.task is not None
                    owner.settled(owner.task)
        return sum(issubclass(owner.runner_type, cls) for owner in _ABANDONED_LAMBDA_ALLOCATIONS)

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
        guest_root = _validate_guest_root(default_cwd)
        environment_overlay = _preflight_runner_creation(
            region_name=region_name,
            close_action=close_action,
            ready_timeout_s=ready_timeout_s,
            poll_interval_s=poll_interval_s,
            request_timeout_s=request_timeout_s,
            auth_token_expiration_minutes=auth_token_expiration_minutes,
            cancel_timeout_s=cancel_timeout_s,
            cancellation_cleanup=cancellation_cleanup,
            timeout_cleanup=timeout_cleanup,
            env_overlay=env_overlay,
        )
        if identifier in _LAMBDA_ATTACHMENT_OWNERS:
            raise LambdaMicroVMError("A Lambda MicroVM attachment or its cleanup is still pending.")
        reclamation = _LambdaAllocationReclamation(
            cls, attachment=True, attachment_identifier=identifier
        )
        _LAMBDA_ATTACHMENT_OWNERS[identifier] = reclamation
        try:
            control_client, owns_client = _control_client(
                client=client,
                region_name=region_name,
                profile_name=profile_name,
                endpoint_url=endpoint_url,
            )
        except BaseException:
            reclamation.release_attachment()
            raise
        reclamation.client = control_client
        reclamation.owns_client = owns_client

        async def attach() -> LambdaMicroVMRunner:
            try:
                response = await asyncio.to_thread(
                    control_client.get_microvm, microvmIdentifier=identifier
                )
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
                    default_cwd=guest_root,
                    close_action=close_action,
                    endpoint_transport=endpoint_transport,
                    owns_client=owns_client,
                    poll_interval_s=poll_interval_s,
                    request_timeout_s=request_timeout_s,
                    auth_token_expiration_minutes=auth_token_expiration_minutes,
                    cancel_timeout_s=cancel_timeout_s,
                    cancellation_cleanup=cancellation_cleanup,
                    timeout_cleanup=timeout_cleanup,
                    env_overlay=environment_overlay,
                )
                reclamation.runner = runner

                def mark_resume_dispatched() -> None:
                    reclamation.restore_suspended = True

                if not reclamation.abandoned:
                    await runner._prepare_existing_for_attach(
                        state, ready_timeout_s, before_resume=mark_resume_dispatched
                    )
                return runner
            except BaseException as primary:
                reclamation.progress.failures = (primary,)
                try:
                    if reclamation.runner is not None:
                        await reclamation.reclaim_runner()
                    else:
                        await reclamation.close_unbound_client()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "Lambda MicroVM attachment failed.", [primary, cleanup]
                    ) from None
                raise

        return await cls._observe_binding(
            reclamation, attach, ready_timeout_s + request_timeout_s, cancel_timeout_s
        )

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        operation = self._exec(
            command,
            execution_profile="agent",
            output_redactor=None,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        return await operation

    async def exec_redacted(
        self,
        command: ExecCommand,
        *,
        redactor: SecretRedactor,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("LambdaMicroVMRunner redactor must be a SecretRedactor.")
        operation = self._exec(
            command,
            execution_profile="agent",
            output_redactor=redactor,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        del command, redactor, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        return await operation

    async def exec_system(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        """Run a control-plane lifecycle command outside the unprivileged agent profile."""
        operation = self._exec(
            command,
            execution_profile="trusted",
            output_redactor=None,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        return await operation

    @_clean_runner_preflight
    def preflight_exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> None:
        """Validate the complete sidecar request without provider activity."""

        prepared = self._prepare_exec_request(
            command,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        del prepared

    def _prepare_exec_request(
        self,
        command: ExecCommand,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        env_remove: tuple[str, ...],
        timeout_s: int | None,
        stdin: str | None,
        output_limit_bytes: int | None,
    ) -> tuple[ExecCommand, str, dict[str, str], int | None, str | None, int | None]:
        """Own one complete sidecar request before dispatch admission."""

        if type(command) is not ExecCommand:
            raise TypeError("LambdaMicroVMRunner command must be an ExecCommand.")
        self._ensure_exec_open()
        owned_command = copy_exec_command(command)
        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=False)
        env_overlay = copy_runner_env(self.env_overlay, inherit_env=False)
        environment = remove_runner_env(environment, env_remove)
        if env_overlay:
            # Applied last: enforced egress configuration must win over model env.
            environment.update(env_overlay)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        output_limit = validate_output_limit(output_limit_bytes)
        output_limit = min(
            LAMBDA_MICROVM_MAX_OUTPUT_BYTES,
            LAMBDA_MICROVM_MAX_OUTPUT_BYTES if output_limit is None else output_limit,
        )
        return owned_command, working_dir, environment, timeout, standard_input, output_limit

    async def _exec(
        self,
        command: ExecCommand,
        *,
        execution_profile: Literal["agent", "trusted"],
        output_redactor: SecretRedactor | None,
        cwd: str | None,
        env: dict[str, str] | None,
        env_remove: tuple[str, ...],
        timeout_s: int | None,
        stdin: str | None,
        output_limit_bytes: int | None,
    ) -> ExecResult:
        try:
            (
                owned_command,
                working_dir,
                environment,
                timeout,
                standard_input,
                output_limit,
            ) = self._prepare_exec_request(
                command,
                cwd=cwd,
                env=env,
                env_remove=env_remove,
                timeout_s=timeout_s,
                stdin=stdin,
                output_limit_bytes=output_limit_bytes,
            )
            command_id = f"cmd-{uuid.uuid4()}"
            payload: dict[str, Any] = {
                "execution_profile": execution_profile,
                "kind": owned_command.kind,
                "cwd": working_dir,
                "env": environment,
                "stdin_base64": (
                    base64.b64encode(standard_input.encode("utf-8")).decode("ascii")
                    if standard_input is not None
                    else None
                ),
                "timeout_s": timeout,
                "output_limit_bytes": output_limit,
                # The sidecar does not receive the workload-secret registry. When
                # host-side redaction is required, it must therefore omit a
                # pre-truncated channel whose missing suffix could complete a
                # secret. Ordinary/system executions retain their established
                # bounded-output contract.
                "omit_truncated_output": output_redactor is not None,
            }
            if owned_command.kind == "process":
                payload["argv"] = list(owned_command.argv or [])
            else:
                payload["shell"] = owned_command.shell
        except BaseException as error:
            _clear_preflight_traceback_frames(error)
            owned_command = None
            working_dir = ""
            environment = {}
            standard_input = None
            payload = {}
            cwd = None
            env = None
            env_remove = ()
            stdin = None
            output_redactor = None
            raise
        finally:
            del command
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
                        if output_redactor is not None:
                            result = redact_completed_exec_result(
                                result,
                                redactor=output_redactor,
                                output_limit_bytes=output_limit,
                                omit_pretruncated=True,
                            )
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
                cancellation=exc,
            )
            attach_cancellation_artifacts(exc, [cleanup.artifact])
            raise
        except TimeoutError:
            cleanup = await self._cleanup_exec_command(
                handle=handle,
                policy=self.timeout_cleanup,
                start_acknowledged=start_acknowledged,
            )
            result = await self._host_timeout_result(
                command_id,
                cleanup,
                output_redactor=output_redactor,
                output_limit_bytes=output_limit,
            )
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
        cancellation: asyncio.CancelledError | None = None,
    ) -> RunnerCleanupResult:
        if not start_acknowledged and policy == "none":
            self._poison_exec()
        cleanup = await self._settle_command_cleanup(
            lambda: cleanup_runner_command_with_diagnostic(
                self,
                handle=handle,
                adapter="lambda-microvm",
                timeout_s=self.cancel_timeout_s,
                policy=policy,
            ),
            adapter="lambda-microvm",
            timeout_s=self.cancel_timeout_s,
            policy=policy,
            cancellation=cancellation,
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
        *,
        output_redactor: SecretRedactor | None,
        output_limit_bytes: int | None,
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
            if output_redactor is not None:
                terminal = redact_completed_exec_result(
                    terminal,
                    redactor=output_redactor,
                    output_limit_bytes=output_limit_bytes,
                    omit_pretruncated=True,
                )
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
        await self._settle_public_lifecycle(self._suspend, action="suspend")

    async def wait_until_suspended(
        self,
        timeout_s: float = DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS,
    ) -> None:
        """Wait for positive control-plane evidence that guest execution is suspended."""

        async with self._lifecycle_lock:
            if not self._suspended:
                raise RuntimeError("Lambda MicroVM suspension has not been requested.")
            await self._wait_for_lifecycle_state(
                terminal_state="SUSPENDED",
                transitional_states={"RUNNING", "SUSPENDING"},
                timeout_s=timeout_s,
            )

    async def _suspend(self) -> None:
        if self._suspended or self._termination_requested:
            return
        await asyncio.to_thread(self._client.suspend_microvm, microvmIdentifier=self.microvm_id)
        self._suspended = True
        self._close_exec("Lambda MicroVM is suspended")

    async def resume(self) -> None:
        await self._settle_public_lifecycle(self._resume, action="resume")

    async def _resume(self) -> None:
        # The public lifecycle owner holds the lock across all provider calls.
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
                return
            if state != "SUSPENDED":
                raise LambdaMicroVMError(f"Cannot resume Lambda MicroVM in state {state}.")
        await asyncio.to_thread(self._client.resume_microvm, microvmIdentifier=self.microvm_id)
        self._auth_token = None
        self._auth_token_expires_at = 0.0
        await self._wait_until_ready(DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS)
        self._suspended = False

    async def terminate(self) -> None:
        await self._settle_public_lifecycle(self._terminate, action="terminate")

    async def _settle_public_lifecycle(
        self, operation: Callable[[], Awaitable[None]], *, action: str
    ) -> None:
        task = self._public_lifecycle_task
        if task is None:
            self._ensure_lifecycle_open()
        elif self._public_lifecycle_action != action or self._exec_poisoned or self._closing:
            raise RuntimeError("Lambda MicroVM has a conflicting lifecycle mutation.")

        async def owned() -> None:
            async with self._lifecycle_lock:
                if self._closed or self._closing:
                    raise RuntimeError("LambdaMicroVMRunner is closed.")
                await operation()
                if action in {"suspend", "terminate"}:
                    await self._wait_for_lifecycle_state(
                        terminal_state="SUSPENDED" if action == "suspend" else "TERMINATED",
                        transitional_states=(
                            {"RUNNING", "SUSPENDING"}
                            if action == "suspend"
                            else {"RUNNING", "SUSPENDING", "SUSPENDED", "TERMINATING"}
                        ),
                        timeout_s=self.request_timeout_s,
                    )

        def settled(task: asyncio.Task[CapturedAwaitableOutcome[None]]) -> None:
            _PENDING_LAMBDA_LIFECYCLES.discard(task)
            if task is not self._public_lifecycle_task:
                return
            self._public_lifecycle_task = None
            self._public_lifecycle_action = None
            self._command_cleanups_pending -= 1
            if task.cancelled() or task.result().error is not None:
                self._poison_exec("Lambda MicroVM lifecycle mutation failed")
            elif action == "resume" and not self._exec_poisoned and not self._closing:
                self._open_exec()
            elif not self._exec_poisoned and not self._closing:
                self._close_exec(
                    "Lambda MicroVM is suspended"
                    if self._suspended
                    else "Lambda MicroVM termination was requested"
                )

        if task is None:
            self._command_cleanups_pending += 1
            self._close_exec("Lambda MicroVM lifecycle mutation is pending")
            try:
                task = asyncio.create_task(capture_awaitable_outcome(owned))
            except BaseException:
                self._command_cleanups_pending -= 1
                self._poison_exec()
                raise
            self._public_lifecycle_task = task
            self._public_lifecycle_action = action
            _PENDING_LAMBDA_LIFECYCLES.add(task)
            task.add_done_callback(settled)
        outcome = await await_shielded_task_outcome(
            task,
            timeout_s=self.request_timeout_s
            + (DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS if action == "resume" else 0),
            timeout_after_cancellation_s=self.cancel_timeout_s,
        )
        failure = outcome.error if outcome.result is None else outcome.result.error
        if outcome.timed_out:
            self._poison_exec("Lambda MicroVM lifecycle mutation has not settled")
            failure = TimeoutError("Lambda MicroVM lifecycle mutation has not settled.")
        elif task.done():
            settled(task)
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
        if failure is not None and _contains_runner_fatal_signal(failure):
            raise failure
        if outcome.cancellation is not None:
            if failure is not None:
                attach_runner_cancellation_failure(outcome.cancellation, failure)
            raise outcome.cancellation
        if failure is not None:
            if isinstance(failure, asyncio.CancelledError):
                raise RuntimeError("Lambda MicroVM lifecycle dependency was cancelled.") from None
            raise failure

    async def wait_until_terminated(
        self,
        timeout_s: float = DEFAULT_LAMBDA_MICROVM_READY_TIMEOUT_SECONDS,
    ) -> None:
        """Wait for positive control-plane evidence that guest execution terminated."""

        async with self._lifecycle_lock:
            if not self._termination_requested:
                raise RuntimeError("Lambda MicroVM termination has not been requested.")
            await self._wait_for_lifecycle_state(
                terminal_state="TERMINATED",
                transitional_states={"RUNNING", "SUSPENDING", "SUSPENDED", "TERMINATING"},
                timeout_s=timeout_s,
            )

    async def close(self) -> None:
        action = self.close_action
        progress = RunnerFailureProgress()
        await self._settle_terminal_lifecycle(
            lambda: self._close_owned(action, progress=progress),
            action=action,
            timeout_s=self.cancel_timeout_s,
            progress=progress,
        )

    async def kill(self) -> None:
        progress = RunnerFailureProgress()
        await self._settle_terminal_lifecycle(
            lambda: self._close_owned("terminate", progress=progress),
            action="terminate",
            timeout_s=self.cancel_timeout_s,
            progress=progress,
        )

    async def _close_owned(
        self, action: LambdaMicroVMCloseAction, *, progress: RunnerFailureProgress | None = None
    ) -> None:
        async with self._lifecycle_lock:
            failures: list[BaseException] = []
            try:
                if action == "terminate":
                    await self._terminate()
                    await self._wait_for_lifecycle_state(
                        terminal_state="TERMINATED",
                        transitional_states={"RUNNING", "SUSPENDING", "SUSPENDED", "TERMINATING"},
                        timeout_s=self.cancel_timeout_s,
                    )
                elif action == "suspend":
                    await self._suspend()
                    await self._wait_for_lifecycle_state(
                        terminal_state="SUSPENDED",
                        transitional_states={"RUNNING", "SUSPENDING"},
                        timeout_s=self.cancel_timeout_s,
                    )
                elif action != "none":
                    raise AssertionError(f"Unsupported Lambda MicroVM close action: {action}")
            except BaseException as error:
                failures.append(error)
                if progress is not None:
                    progress.failures = tuple(failures)
            try:
                await self._close_transports(progress=progress)
            except BaseException as error:
                failures.append(error)
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("Lambda MicroVM finalization failed.", failures)

    async def _terminate(self) -> None:
        if self._termination_requested:
            return
        await asyncio.to_thread(self._client.terminate_microvm, microvmIdentifier=self.microvm_id)
        self._termination_requested = True
        self._close_exec("Lambda MicroVM termination was requested")

    async def _wait_for_lifecycle_state(
        self,
        *,
        terminal_state: str,
        transitional_states: set[str],
        timeout_s: float,
    ) -> None:
        timeout = _positive_float(timeout_s, "timeout_s")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            response = await asyncio.to_thread(
                self._client.get_microvm,
                microvmIdentifier=self.microvm_id,
            )
            response_id, endpoint = _microvm_identity(response)
            if response_id != self.microvm_id or endpoint != self.endpoint:
                raise LambdaMicroVMProtocolError(
                    "get_microvm returned different identity while waiting for lifecycle "
                    "quiescence."
                )
            state = _required_response_string(response, "state")
            if state == terminal_state:
                return
            if state not in transitional_states:
                raise LambdaMicroVMError(
                    f"Lambda MicroVM entered unexpected state {state} while waiting for "
                    f"{terminal_state}."
                )
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise LambdaMicroVMError(
                    f"Lambda MicroVM did not reach {terminal_state} within {timeout:g} seconds."
                )
            await asyncio.sleep(min(max(self.poll_interval_s, 0.05), remaining))

    async def _close_transports(self, *, progress: RunnerFailureProgress | None = None) -> None:
        self._auth_token = None
        self._auth_token_expires_at = 0.0
        failures: list[BaseException] = []
        prior_failures = () if progress is None else progress.failures
        if self._owns_endpoint_transport:
            close = getattr(self._endpoint_transport, "aclose", None)
            if callable(close):
                try:
                    await close()
                except BaseException as error:
                    failures.append(error)
                    if progress is not None:
                        progress.failures = (*prior_failures, *failures)
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    await asyncio.to_thread(close)
                except BaseException as error:
                    failures.append(error)
                    if progress is not None:
                        progress.failures = (*prior_failures, *failures)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Lambda MicroVM transport cleanup failed.", failures)

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
        self,
        initial_state: str,
        ready_timeout_s: float,
        *,
        before_resume: Callable[[], None] | None = None,
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
            if before_resume is not None:
                before_resume()
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
        loop = asyncio.get_running_loop()
        pending = self._command_poll_tasks.get(command_id)
        if pending is None:
            deadline = loop.time() + self.request_timeout_s
            task = asyncio.create_task(
                capture_awaitable_outcome(
                    lambda: self._poll_command_until_deadline(command_id, deadline)
                )
            )
            self._command_poll_tasks[command_id] = (deadline, task)
            _PENDING_LAMBDA_POLLS.add(task)

            def retire(completed: _CommandPollTask) -> None:
                _PENDING_LAMBDA_POLLS.discard(completed)
                if self._command_poll_tasks.get(command_id) == (deadline, completed):
                    del self._command_poll_tasks[command_id]

            task.add_done_callback(retire)
        else:
            deadline, task = pending
        outcome = await await_shielded_task_outcome(
            task,
            timeout_s=max(0, deadline - loop.time()),
            timeout_after_cancellation_s=0,
        )
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if outcome.timed_out:
            raise TimeoutError("Lambda MicroVM command-state poll exceeded its deadline.")
        captured = outcome.result
        failure = outcome.error if captured is None else captured.error
        if isinstance(failure, asyncio.CancelledError):
            raise LambdaMicroVMError(
                "Lambda MicroVM command-state read was cancelled by its dependency."
            )
        if failure is not None:
            raise failure
        if captured is None or captured.result is None:
            raise LambdaMicroVMProtocolError(
                "Lambda MicroVM command-state read returned no result."
            )
        return captured.result

    async def _poll_command_until_deadline(
        self, command_id: str, deadline: float
    ) -> Mapping[str, Any]:
        loop = asyncio.get_running_loop()
        for attempt in range(_LAMBDA_POLL_ATTEMPTS):
            if loop.time() >= deadline:
                raise TimeoutError("Lambda MicroVM command-state poll exceeded its deadline.")
            try:
                result = await self._endpoint_call(
                    "get_command", deadline=deadline, command_id=command_id
                )
            except LambdaMicroVMEndpointTransientError:
                if attempt == _LAMBDA_POLL_ATTEMPTS - 1:
                    raise
            else:
                if loop.time() >= deadline:
                    raise TimeoutError("Lambda MicroVM command-state poll exceeded its deadline.")
                return result
            await asyncio.sleep(min(0.05 * (2**attempt), max(0, deadline - loop.time())))
        raise AssertionError("unreachable command-state retry loop")

    async def _endpoint_cancel(self, command_id: str) -> Mapping[str, Any]:
        return await self._endpoint_call("cancel_command", command_id=command_id)

    async def _endpoint_call(
        self, method_name: str, *, deadline: float | None = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        method = getattr(self._endpoint_transport, method_name)
        attempts = 1 if method_name == "get_command" else 2
        for attempt in range(attempts):
            token = await self._endpoint_token(force_refresh=attempt == 1)
            timeout_s = self.request_timeout_s
            if deadline is not None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Lambda MicroVM command-state poll exceeded its deadline.")
                timeout_s = min(timeout_s, remaining)
            try:
                result = await method(
                    endpoint=self.endpoint,
                    token=token,
                    timeout_s=timeout_s,
                    **kwargs,
                )
            except LambdaMicroVMEndpointUnauthorized:
                if attempt + 1 < attempts:
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
        if self._closed or self._closing or self._command_cleanups_pending or self._exec_poisoned:
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
    if len(raw) > LAMBDA_MICROVM_MAX_ENCODED_OUTPUT_BYTES:
        raise LambdaMicroVMProtocolError("Lambda MicroVM encoded output exceeds its byte ceiling.")
    padding = 2 if raw.endswith("==") else 1 if raw.endswith("=") else 0
    if (len(raw) // 4) * 3 - padding > LAMBDA_MICROVM_MAX_OUTPUT_BYTES:
        raise LambdaMicroVMProtocolError("Lambda MicroVM decoded output exceeds its byte ceiling.")
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
    root = require_durable_clean_nonblank(value, "default_cwd")
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


def _preflight_runner_creation(
    *,
    region_name: str | None,
    close_action: LambdaMicroVMCloseAction,
    ready_timeout_s: float,
    poll_interval_s: float,
    request_timeout_s: float,
    auth_token_expiration_minutes: int,
    cancel_timeout_s: float | None,
    cancellation_cleanup: RunnerCleanupPolicy,
    timeout_cleanup: RunnerCleanupPolicy,
    env_overlay: Mapping[str, str] | None,
) -> dict[str, str]:
    """Reject local configuration before allocating or attaching provider resources."""

    _optional_clean_string(region_name, "region_name")
    _validate_close_action(close_action)
    _positive_float(ready_timeout_s, "ready_timeout_s")
    _nonnegative_float(poll_interval_s, "poll_interval_s")
    _positive_float(request_timeout_s, "request_timeout_s")
    if type(auth_token_expiration_minutes) is not int or auth_token_expiration_minutes <= 0:
        raise ValueError("auth_token_expiration_minutes must be a positive integer.")
    validate_cancel_timeout(cancel_timeout_s)
    validate_runner_cleanup_policy(cancellation_cleanup, "cancellation_cleanup")
    validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
    return dict(env_overlay) if env_overlay else {}


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
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero.")
    return float(value)


def _nonnegative_float(value: float, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return float(value)
