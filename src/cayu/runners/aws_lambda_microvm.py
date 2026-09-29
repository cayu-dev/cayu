from __future__ import annotations

import asyncio
import base64
import calendar
import contextlib
import importlib
import json
import logging
import re
import secrets
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

import httpx

from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.providers._http import SharedAsyncClient
from cayu.runners._admission_probes import EXECUTABLE_AVAILABILITY_SCRIPT
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
    RemoteWorkspaceBranchCapability,
    Runner,
    RunnerExecutionAdmissionObserver,
    RunnerSystemExecutionMode,
    RunnerWorkloadAuthority,
    RunnerWorkspaceCapabilityT,
    _clean_runner_preflight,
    _clear_preflight_traceback_frames,
    _contains_runner_fatal_signal,
    attach_cancellation_artifacts,
    copy_exec_command,
)
from cayu.runners.workloads import (
    BROWSER_FETCH_WORKLOAD_NAME,
    BROWSER_SESSION_WORKLOAD_NAME,
    BROWSER_WORKER_DIRECTORY,
    BROWSER_WORKER_FILES,
    BROWSER_WORKER_PLAYWRIGHT_BROWSERS_PATH,
    BROWSER_WORKER_PLAYWRIGHT_VERSION,
    BROWSER_WORKER_WEBSOCKETS_VERSION,
    PINNED_BROWSER_FETCH_WORKLOAD,
    PINNED_BROWSER_SESSION_WORKLOAD,
    browser_worker_source_digests,
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
LAMBDA_MICROVM_PROTOCOL_VERSION = "5"
# Sized for the built-in browser worker: a default session response envelope
# plus a browser-profile checkpoint, and a default upload batch on stdin. Both
# match the first-party sidecar's own ceilings.
LAMBDA_MICROVM_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
LAMBDA_MICROVM_MAX_STDIN_BYTES = 24 * 1024 * 1024
# A MicroVM fetches its root filesystem lazily, so a first Chromium start
# measured from a few seconds to about a minute.
LAMBDA_MICROVM_BROWSER_LAUNCH_TIMEOUT_SECONDS = 120
#: The agent namespace reaches the Cayu control server only through the
#: sidecar's control relay, as ``wss://cayu-control:18443/...``.
LAMBDA_MICROVM_CONTROL_HOSTNAME = "cayu-control"
LAMBDA_MICROVM_CONTROL_RELAY_PORT = 18443
_BROWSER_RECORDING_FINALIZE_TIMEOUT_SECONDS = 6
_CONTROL_CA_MAX_BYTES = 64 * 1024
LAMBDA_MICROVM_MAX_ENCODED_OUTPUT_BYTES = 4 * ((LAMBDA_MICROVM_MAX_OUTPUT_BYTES + 2) // 3)
LAMBDA_MICROVM_MAX_RESPONSE_BYTES = 2 * LAMBDA_MICROVM_MAX_ENCODED_OUTPUT_BYTES + 64 * 1024
_LAMBDA_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_LAMBDA_OWNER_SUPERSEDED_HTTP_STATUS = 412
_LAMBDA_OWNER_LIFECYCLE_LEASED_HTTP_STATUS = 423
_LAMBDA_OWNER_SUPERSEDED_CANCEL_REASON = "owner_superseded"
# A lifecycle lease authorizes exactly one provider request, and only if it is
# signed within this window of the ownership check. The window bounds how stale
# that check may be; it is not a claim about when the request finishes.
_LAMBDA_LIFECYCLE_DISPATCH_WINDOW_SECONDS = 60.0
_LAMBDA_LIFECYCLE_OPERATIONS = ("SuspendMicrovm", "TerminateMicrovm")
_LAMBDA_AMBIGUOUS_PROVIDER_ERROR_CODES = frozenset(
    {"InternalServerException", "ServiceUnavailableException"}
)
_LIFECYCLE_DISPATCH = threading.local()
LAMBDA_MICROVM_CLIENT_TOKEN_MAX_LENGTH = 128
# Only a client-token submission may be retried: AWS returns the original
# allocation for a replay of identical parameters instead of creating another.
_LAMBDA_CLIENT_TOKEN_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)
#: AWS rejects a SigV4-signed request more than this long after its signing
#: time, so no request signed by a deadline can reach AWS after deadline + this.
LAMBDA_SIGV4_REQUEST_VALIDITY_SECONDS = 900
_RUN_SUBMISSION_GUARD_EVENT = "before-send.lambda-microvms.RunMicrovm"
_RUN_SUBMISSION_GUARD_ID = "cayu.lambda-microvm.run-submission-deadline"
_RUN_SUBMISSION_DEADLINE = threading.local()
_submission_clock: Callable[[], float] = time.time
_LAMBDA_TRANSIENT_CONTROL_ERROR_CODES = frozenset(
    {
        "InternalServerException",
        "ServiceUnavailableException",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)
_LAMBDA_TRANSIENT_CONTROL_ERROR_TYPES = frozenset(
    {
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "EndpointConnectionError",
        "ReadTimeoutError",
    }
)
_LAMBDA_UNUSABLE_ALLOCATION_STATES = frozenset({"FAILED", "TERMINATING", "TERMINATED"})
_LAMBDA_POLL_ATTEMPTS = 3
LAMBDA_MICROVM_ADMISSION_PROBE_TIMEOUT_SECONDS = 5
_LAMBDA_ADMISSION_PROBE_OUTPUT_LIMIT_BYTES = 1024
_LAMBDA_ADMISSION_RUNNING_STATES = frozenset({"PENDING", "RUNNING"})
# Read through the agent lane with a shell builtin, so the identity read does
# not depend on any guest utility beyond /bin/sh.
_LAMBDA_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
_LAMBDA_BOOT_ID_SCRIPT = f'read -r boot_id < {_LAMBDA_BOOT_ID_PATH} && printf "%s" "$boot_id"'
_LAMBDA_BOOT_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
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
    # The MicroVM identifier can reach another host: a client-token submission
    # (a replay returns the same MicroVM) or an attachment by identifier.
    shared_identity: bool = False
    cleanup_deferred: bool = False

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
                if not runner._owner_claimed and self.shared_identity:
                    # Not having claimed proves only that this runner never
                    # owned the MicroVM, not that nobody else does: another
                    # creator replaying the token, or the attachment's source,
                    # may already hold the claim. Leave disposal to the
                    # allocation reap or the MicroVM's maximum duration.
                    self.cleanup_deferred = True
                    _LOGGER.warning(
                        "Lambda MicroVM %s was not claimed by its failed constructor; its "
                        "cleanup is deferred to the allocation reap or its maximum duration.",
                        runner.microvm_id,
                    )
                else:
                    # A claimed runner mutates only under the sidecar's lease.
                    # An unclaimed one here made a tokenless create whose
                    # identifier it never returned, so no other host can hold
                    # the MicroVM and it is disposed of without the fence.
                    fenced = runner._owner_claimed
                    if self.attachment:
                        await runner._suspend(fenced=fenced)
                    else:
                        await runner._terminate(fenced=fenced)
                    await runner._wait_for_lifecycle_state(
                        terminal_state="SUSPENDED" if self.attachment else "TERMINATED",
                        transitional_states={"RUNNING", "SUSPENDING", "SUSPENDED", "TERMINATING"},
                        timeout_s=runner.cancel_timeout_s,
                    )
            # Local cleanup is settled; a deferred MicroVM is not this owner's to dispose of.
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


class LambdaMicroVMOwnershipSuperseded(LambdaMicroVMError):
    """A later host claimed this MicroVM; this runner may no longer act on it.

    Execution is permanently closed and lifecycle mutations are skipped so a
    stale owner cannot run commands in, suspend, or terminate its successor's
    MicroVM.
    """


class LambdaMicroVMOwnershipUnverified(LambdaMicroVMError):
    """The sidecar could not confirm this runner still owns the MicroVM.

    No suspend or terminate was sent. An unreachable sidecar is not proof that
    no successor owns the MicroVM, so the mutation is refused and cleanup stays
    pending for an authoritative path: the runtime's allocation reap, which
    runs under the durable allocation fence, or the MicroVM's maximum duration.
    """


class LambdaMicroVMLifecycleInProgress(LambdaMicroVMError):
    """The current owner is suspending or terminating the MicroVM.

    The sidecar refuses new claims and commands until the lease is settled:
    the owner releases it after a provider request that was never sent or was
    definitively rejected, or the guest's matching suspend/terminate hook
    observes the request's effect. The lease never expires by time.
    """


class _LambdaMicroVMAdmissionIdentityDrift(LambdaMicroVMError):
    """The exact MicroVM changed while admission evidence was being observed."""


class LambdaMicroVMBrowserWorkloadError(LambdaMicroVMError):
    """The MicroVM image does not contain this Cayu release's browser worker."""


class LambdaMicroVMBrowserControlError(LambdaMicroVMError):
    """The guest could not reach the Cayu control server through the relay."""


class LambdaMicroVMClientTokenConflict(LambdaMicroVMError):
    """AWS rejected a client-token replay because its request parameters changed.

    No MicroVM is created by the rejected request. The earlier allocation, if
    any, remains owned by the original request and must not be replaced.
    """


class LambdaMicroVMSubmissionClosed(LambdaMicroVMError):
    """A ``RunMicrovm`` request was not sent because its submission deadline passed.

    Nothing reached AWS, so the refused request cannot create an allocation.
    """


class LambdaMicroVMAllocationTerminated(LambdaMicroVMError):
    """An idempotent replay resolved an allocation that is no longer usable."""

    def __init__(self, message: str, *, microvm_id: str, state: str) -> None:
        super().__init__(message)
        self.microvm_id = microvm_id
        self.state = state


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

    async def claim_owner(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    async def check_owner(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    async def acquire_lifecycle(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        action: str,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    async def release_lifecycle(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    async def configure_control_relay(
        self,
        *,
        endpoint: str,
        token: str,
        owner_claim: str,
        host: str,
        port: int,
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

    async def release_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/commands/{command_id}/release",
            timeout_s=timeout_s,
        )

    async def claim_owner(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/owner",
            payload={"claim_id": claim_id},
            timeout_s=timeout_s,
        )

    async def check_owner(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/owner/check",
            payload={"claim_id": claim_id},
            timeout_s=timeout_s,
        )

    async def acquire_lifecycle(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        action: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/owner/lifecycle",
            payload={"claim_id": claim_id, "action": action},
            timeout_s=timeout_s,
        )

    async def release_lifecycle(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/owner/lifecycle/release",
            payload={"claim_id": claim_id},
            timeout_s=timeout_s,
        )

    async def configure_control_relay(
        self,
        *,
        endpoint: str,
        token: str,
        owner_claim: str,
        host: str,
        port: int,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/control-relay",
            payload={"owner_claim": owner_claim, "host": host, "port": port},
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
                    if response.status_code == _LAMBDA_OWNER_SUPERSEDED_HTTP_STATUS:
                        raise LambdaMicroVMOwnershipSuperseded(
                            "Lambda MicroVM sidecar reports a newer owner."
                        )
                    if response.status_code == _LAMBDA_OWNER_LIFECYCLE_LEASED_HTTP_STATUS:
                        raise LambdaMicroVMLifecycleInProgress(
                            "Lambda MicroVM owner is suspending or terminating it."
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
        # Memory-only bearer claim; the sidecar keeps only its digest.
        self._owner_claim = secrets.token_hex(32)
        self._owner_claimed = False
        self._owner_superseded = False
        self._owner_generation: int | None = None
        # Advanced by every lifecycle transition so retained admission evidence
        # never outlives the running incarnation it observed.
        self._admission_epoch = 0
        # Set only by a trusted probe of this MicroVM's image; never configured.
        self._browser_workload_verified = False
        # Set only after the agent profile completed a verified TLS handshake
        # with the control server through the sidecar relay. The sidecar drops
        # the relay on suspension, so every lifecycle transition clears it.
        self._browser_control_relay_verified = False
        # Fail closed: an overlay may carry secret values unless the adapter that
        # built it declares otherwise after verifying the guest boundary.
        self._env_overlay_secret_values_present = bool(self.env_overlay)

    def execution_admission_observer(self, requirements):
        """Own live executable evidence for this exact MicroVM and requirement set."""

        if not requirements.executable_names():
            return super().execution_admission_observer(requirements)
        return _LambdaAdmissionObserver(self, requirements)

    def execution_admission_candidate_for(self, requirements):
        if not requirements.executable_names():
            return self.execution_admission_candidate()
        return _LambdaAdmissionObserver(self, requirements).snapshot()

    async def refresh_execution_admission(self) -> None:
        """Refuse renewal for a MicroVM that cannot execute; claims are request-scoped.

        Executable evidence is renewed by the request's admission observer,
        which re-reads the complete identity. This hook adds no claim.
        """

        self._require_admission_open()

    def _require_admission_open(self) -> None:
        if self._owner_superseded:
            raise LambdaMicroVMOwnershipSuperseded("Lambda MicroVM has a newer owner.")
        self._ensure_exec_open()
        if (
            self._suspended
            or self._termination_requested
            or self._public_lifecycle_task is not None
        ):
            raise LambdaMicroVMError("Lambda MicroVM is not running; admission is unavailable.")

    def _admission_local_identity(self) -> tuple[object, ...]:
        return (
            self.microvm_id,
            self.endpoint,
            self.image_identifier,
            self.image_version,
            self.default_cwd,
            tuple(sorted(copy_runner_env(self.env_overlay, inherit_env=False).items())),
            self._admission_epoch,
        )

    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None:
        """Declare the browser workload only after this MicroVM's image proved it."""

        if not self._browser_workload_verified:
            return None
        if name == BROWSER_FETCH_WORKLOAD_NAME:
            return PINNED_BROWSER_FETCH_WORKLOAD
        if name == BROWSER_SESSION_WORKLOAD_NAME:
            return PINNED_BROWSER_SESSION_WORKLOAD
        return None

    def output_secret_values_present(self) -> bool:
        """Declare whether command output can contain runner-owned secret values."""

        return self._env_overlay_secret_values_present

    def browser_control_endpoint_reachable(self, endpoint: str) -> bool:
        """Report whether the guest can dial ``endpoint`` for browser control or recording.

        Only the verified control relay leaves the agent namespace, so only its
        exact ``wss://cayu-control:18443`` authority is reachable. Anything else
        would carry a control credential toward an address the guest cannot
        reach, so it is reported unreachable.
        """

        if not self._browser_control_relay_verified or type(endpoint) is not str:
            return False
        try:
            parsed = urlsplit(endpoint)
            port = parsed.port
        except ValueError:
            return False
        return (
            parsed.scheme == "wss"
            and parsed.hostname == LAMBDA_MICROVM_CONTROL_HOSTNAME
            and port == LAMBDA_MICROVM_CONTROL_RELAY_PORT
        )

    def browser_recording_supported(self) -> bool:
        """Declare recording once the worker and its control relay are both verified.

        Frames leave through the control relay, and the adapter finalizes
        recordings in the guest before the MicroVM is suspended or terminated.
        """

        return self._browser_workload_verified and self._browser_control_relay_verified

    async def configure_browser_control_relay(
        self,
        *,
        host: str,
        port: int,
        ca_certificate_pem: bytes,
        timeout_s: int = 30,
    ) -> None:
        """Give the agent namespace one verified path to the Cayu control server, or raise.

        A trusted command installs the control server's public CA where the
        browser worker reads control roots. The owner-fenced sidecar then relays
        ``cayu-control:18443`` in the agent namespace to ``host:port`` (a private
        IPv4 target). Finally the agent profile completes a TLS handshake through
        the relay with the worker's own control trust, which proves the route,
        the name mapping, and certificate verification without sending any
        credential. Only then does :meth:`browser_control_endpoint_reachable`
        report the relay endpoint.
        """

        from cayu.tools._browser_control_transport import CONTROL_CA_PATH

        certificate = _control_ca_certificate(ca_certificate_pem)
        if type(host) is not str or not host:
            raise ValueError("control relay host must be a private IPv4 literal.")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("control relay port must be an integer from 1 to 65535.")
        self._browser_control_relay_verified = False
        installed = await self.exec_system(
            ExecCommand.process("python3", "-c", _CONTROL_CA_INSTALL_SCRIPT, CONTROL_CA_PATH),
            stdin=certificate.decode("ascii"),
            timeout_s=timeout_s,
            output_limit_bytes=_BROWSER_WORKLOAD_PROBE_OUTPUT_LIMIT_BYTES,
        )
        if installed.timed_out or installed.exit_code != 0:
            raise LambdaMicroVMBrowserControlError(
                "Lambda MicroVM could not install the control server CA."
            )
        try:
            if not self._owner_claimed:
                await self._claim_owner()
            await self._endpoint_call(
                "configure_control_relay",
                owner_claim=self._owner_claim,
                host=host,
                port=port,
            )
        except LambdaMicroVMOwnershipSuperseded:
            self._mark_superseded()
            raise
        probe = await self.exec(
            ExecCommand.process(
                PINNED_BROWSER_SESSION_WORKLOAD.command[0],
                "-I",
                "-c",
                _CONTROL_RELAY_PROBE_SCRIPT,
                BROWSER_WORKER_DIRECTORY,
                LAMBDA_MICROVM_CONTROL_HOSTNAME,
                str(LAMBDA_MICROVM_CONTROL_RELAY_PORT),
            ),
            timeout_s=timeout_s,
            output_limit_bytes=_BROWSER_WORKLOAD_PROBE_OUTPUT_LIMIT_BYTES,
        )
        if probe.timed_out or probe.exit_code != 0:
            reason = _CONTROL_RELAY_PROBE_FAILURES.get(
                probe.exit_code, "the relay probe did not complete"
            )
            raise LambdaMicroVMBrowserControlError(
                f"Lambda MicroVM cannot reach the control server: {reason}."
            )
        self._browser_control_relay_verified = True

    async def _finalize_browser_recordings(self, *, normal: bool) -> None:
        """Settle optional media before suspension or termination; never browser input."""

        if not self.browser_recording_supported():
            return
        with contextlib.suppress(Exception):
            await self.exec(
                ExecCommand.process(
                    *PINNED_BROWSER_SESSION_WORKLOAD.command,
                    "--finalize-recordings",
                    "normal" if normal else "partial",
                ),
                timeout_s=_BROWSER_RECORDING_FINALIZE_TIMEOUT_SECONDS,
                output_limit_bytes=_BROWSER_WORKLOAD_PROBE_OUTPUT_LIMIT_BYTES,
            )
        # A missing acknowledgement is reconciled as partial or unavailable.
        # Browser input or business work is never repeated to fill a media gap.

    async def verify_browser_workload(
        self,
        *,
        timeout_s: int = 30,
        launch_timeout_s: int = LAMBDA_MICROVM_BROWSER_LAUNCH_TIMEOUT_SECONDS,
    ) -> None:
        """Prove the image carries this Cayu release's browser worker, or raise.

        A trusted-profile probe hashes the root-owned worker files and reads the
        pinned component versions through the exact interpreter the worker
        command names. Then the agent profile launches Chromium once with its
        sandbox enabled and renders a page, which proves the sandbox works on
        this MicroVM and pays the first, lazily loaded Chromium start before any
        tool can run. Only when both pass does :meth:`workload_authority` report
        the browser workloads; the image is immutable for the MicroVM's lifetime
        and unwritable by agent commands, so the proof holds until this runner
        is discarded.
        """

        expected = browser_worker_source_digests()
        result = await self.exec_system(
            ExecCommand.process(
                PINNED_BROWSER_SESSION_WORKLOAD.command[0],
                "-I",
                "-c",
                _BROWSER_WORKLOAD_PROBE_SCRIPT,
                BROWSER_WORKER_DIRECTORY,
                BROWSER_WORKER_PLAYWRIGHT_BROWSERS_PATH,
                *(name for name, _source in BROWSER_WORKER_FILES),
            ),
            timeout_s=timeout_s,
            output_limit_bytes=_BROWSER_WORKLOAD_PROBE_OUTPUT_LIMIT_BYTES,
        )
        if result.timed_out or result.exit_code != 0 or result.stdout_truncated:
            raise LambdaMicroVMBrowserWorkloadError(
                "Lambda MicroVM image has no browser worker interpreter at "
                f"{PINNED_BROWSER_SESSION_WORKLOAD.command[0]}; build it from the browser "
                "variant of the first-party sidecar image."
            )
        try:
            observed = json.loads(result.stdout)
        except ValueError:
            observed = None
        mismatch = _browser_workload_mismatch(observed, expected)
        if mismatch is not None:
            raise LambdaMicroVMBrowserWorkloadError(
                f"Lambda MicroVM browser workload is not this Cayu release's: {mismatch}."
            )
        launched = await self.exec(
            ExecCommand.process(
                PINNED_BROWSER_SESSION_WORKLOAD.command[0], "-I", "-c", _BROWSER_LAUNCH_PROBE_SCRIPT
            ),
            env={"PLAYWRIGHT_BROWSERS_PATH": BROWSER_WORKER_PLAYWRIGHT_BROWSERS_PATH},
            timeout_s=launch_timeout_s,
            output_limit_bytes=_BROWSER_WORKLOAD_PROBE_OUTPUT_LIMIT_BYTES,
        )
        if launched.timed_out or launched.exit_code != 0:
            reason = (
                "did not start within its launch timeout"
                if launched.timed_out
                else "could not start with its sandbox as the agent user"
            )
            raise LambdaMicroVMBrowserWorkloadError(f"Lambda MicroVM Chromium {reason}.")
        self._browser_workload_verified = True

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
        client_token: str | None = None,
        client_token_not_after: float | None = None,
    ) -> LambdaMicroVMRunner:
        """Allocate a MicroVM and wait until its sidecar is ready.

        With ``client_token``, submission is idempotent: transient control-plane
        failures replay the same token and parameters, and a replay adopts the
        allocation AWS already created for that token. The adopted state is read
        back with ``get_microvm`` because a replay response repeats the original
        acknowledgement. A token-owned allocation that is already failed or
        terminated raises ``LambdaMicroVMAllocationTerminated``; changed
        parameters raise ``LambdaMicroVMClientTokenConflict``. Neither creates a
        replacement. ``client_token_not_after`` bounds when any submission for
        the token may be sent (see ``run_microvm_with_client_token``).
        """
        image = require_clean_nonblank(image_identifier, "image_identifier")
        token = None if client_token is None else _validate_client_token(client_token)
        if client_token_not_after is not None and token is None:
            raise ValueError("client_token_not_after requires client_token.")
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
        run_options = lambda_microvm_run_options(
            image,
            image_version=image_version,
            execution_role_arn=execution_role_arn,
            ingress_network_connectors=ingress_network_connectors,
            egress_network_connectors=egress_network_connectors,
            idle_policy=idle_policy,
            maximum_duration_in_seconds=maximum_duration_in_seconds,
            run_hook_payload=run_hook_payload,
        )

        control_client, owns_client = _control_client(
            client=client,
            region_name=region_name,
            profile_name=profile_name,
            endpoint_url=endpoint_url,
        )
        reclamation = _LambdaAllocationReclamation(
            cls, control_client, owns_client, shared_identity=token is not None
        )

        async def allocate() -> LambdaMicroVMRunner:
            runner: LambdaMicroVMRunner | None = None
            try:
                if token is None:
                    response = await asyncio.to_thread(control_client.run_microvm, **run_options)
                else:
                    response = await run_microvm_with_client_token(
                        control_client,
                        run_options,
                        client_token=token,
                        submit_not_after=client_token_not_after,
                    )
                microvm_id, endpoint = _microvm_identity(response)
                adopted_state: str | None = None
                if token is not None:
                    # A replay repeats the original acknowledgement, including its
                    # stale state, so trust only a fresh control-plane read.
                    adopted_state = await _read_adoptable_state(
                        control_client,
                        microvm_id=microvm_id,
                        endpoint=endpoint,
                        image_arn=_response_string(response, "imageArn"),
                        image_version=image_version,
                    )
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
                    if adopted_state is None:
                        await runner._wait_until_ready(ready_timeout_s)
                    else:
                        await runner._prepare_existing_for_attach(adopted_state, ready_timeout_s)
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
                _note_deferred_cleanup(primary, reclamation)
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
            cls, attachment=True, attachment_identifier=identifier, shared_identity=True
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
                _note_deferred_cleanup(primary, reclamation)
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
        if (
            standard_input is not None
            and len(standard_input.encode("utf-8")) > LAMBDA_MICROVM_MAX_STDIN_BYTES
        ):
            raise ValueError(
                f"Lambda MicroVM stdin exceeds the sidecar limit of "
                f"{LAMBDA_MICROVM_MAX_STDIN_BYTES} bytes."
            )
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
        release_after_read = False
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
                        if response.get("cancel_reason") == _LAMBDA_OWNER_SUPERSEDED_CANCEL_REASON:
                            self._mark_superseded()
                            raise LambdaMicroVMOwnershipSuperseded(
                                "A newer Lambda MicroVM owner cancelled this command."
                            )
                        result = _exec_result(response)
                        del response
                        release_after_read = True
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
                        break
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
        if release_after_read:
            await self._release_command(command_id)
        return result

    async def _release_command(self, command_id: str) -> None:
        """Ask the sidecar to drop a terminal result the host has now read.

        Browser profile and upload traffic carries cookies and file contents, so
        the delivered output should not stay in sidecar memory (or in a suspend
        snapshot) until the result TTL expires. Failure leaves that TTL as the
        bound and never changes the command's outcome.
        """

        if not callable(getattr(self._endpoint_transport, "release_command", None)):
            return
        try:
            await self._endpoint_call("release_command", command_id=command_id)
        except Exception:
            _LOGGER.debug("Lambda MicroVM command release failed", exc_info=True)

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
            del response
            if output_redactor is not None:
                terminal = redact_completed_exec_result(
                    terminal,
                    redactor=output_redactor,
                    output_limit_bytes=output_limit_bytes,
                    omit_pretruncated=True,
                )
        except Exception:
            return result
        await self._release_command(command_id)
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

    async def _suspend(self, *, fenced: bool = True) -> None:
        if self._suspended or self._termination_requested:
            return
        self._admission_epoch += 1
        self._browser_control_relay_verified = False
        if fenced:
            await self._owned_provider_mutation("suspend", self._client.suspend_microvm)
        else:
            await asyncio.to_thread(self._client.suspend_microvm, microvmIdentifier=self.microvm_id)
        self._suspended = True
        self._close_exec("Lambda MicroVM is suspended")

    async def resume(self) -> None:
        await self._settle_public_lifecycle(self._resume, action="resume")

    async def _resume(self) -> None:
        # The public lifecycle owner holds the lock across all provider calls.
        if self._termination_requested:
            raise RuntimeError("Cannot resume a terminated Lambda MicroVM.")
        self._admission_epoch += 1
        self._browser_control_relay_verified = False
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

    async def _terminate(self, *, fenced: bool = True) -> None:
        if self._termination_requested:
            return
        self._admission_epoch += 1
        self._browser_control_relay_verified = False
        if not fenced:
            await asyncio.to_thread(
                self._client.terminate_microvm, microvmIdentifier=self.microvm_id
            )
            self._termination_requested = True
            self._close_exec("Lambda MicroVM termination was requested")
            return
        if self._suspended:
            # A suspended guest cannot answer for its owner. Resuming is not
            # destructive, and readiness re-checks the claim, so a successor
            # that suspended this MicroVM after taking it over is never
            # terminated by this runner.
            await self._resume()
        await self._owned_provider_mutation("terminate", self._client.terminate_microvm)
        self._termination_requested = True
        self._close_exec("Lambda MicroVM termination was requested")

    async def _owned_provider_mutation(
        self, action: Literal["suspend", "terminate"], operation: Callable[..., Any]
    ) -> None:
        """Send one provider suspend or terminate under the sidecar's lifecycle lease.

        The lease atomically confirms this runner is the current owner and
        refuses any claim until it ends, so no successor can take over between
        the check and the provider call. It authorizes exactly one request,
        signed within the dispatch window; SDK retries are refused before they
        are sent. The lease is released here only when no request was sent or
        AWS definitively rejected the one that was. Otherwise (success, a
        timeout, a server error, or cancellation while the send is in flight)
        it stays held until the guest's lifecycle hook settles it.
        """

        started = time.monotonic()
        await self._acquire_lifecycle_lease(action)
        dispatch = _LifecycleDispatch(deadline=started + _LAMBDA_LIFECYCLE_DISPATCH_WINDOW_SECONDS)
        observed = _install_lifecycle_send_guard(self._client)

        def send() -> None:
            if time.monotonic() > dispatch.deadline:
                raise _LifecycleSendRefused(
                    f"Lambda MicroVM {action} was not sent within its lifecycle lease."
                )
            if not observed:
                # Without SDK events the one call is the one request.
                dispatch.sent = 1
            _LIFECYCLE_DISPATCH.value = dispatch
            try:
                operation(microvmIdentifier=self.microvm_id)
            except BaseException as error:
                if not observed and _is_definitive_provider_rejection(error):
                    dispatch.rejected = 1
                raise
            finally:
                _LIFECYCLE_DISPATCH.value = None

        try:
            await asyncio.to_thread(send)
        except asyncio.CancelledError:
            # The send may still be running in its thread; keep the lease.
            raise
        except BaseException:
            if dispatch.unapplied:
                await self._release_lifecycle_lease()
            raise

    async def _acquire_lifecycle_lease(self, action: str) -> None:
        if self._owner_superseded:
            raise LambdaMicroVMOwnershipSuperseded("Lambda MicroVM has a newer owner.")
        try:
            if not self._owner_claimed:
                # A directly constructed runner attaches before it acts.
                await self._claim_owner()
            response = await self._endpoint_call(
                "acquire_lifecycle", claim_id=self._owner_claim, action=action
            )
        except LambdaMicroVMOwnershipSuperseded:
            self._mark_superseded()
            raise
        except (LambdaMicroVMLifecycleInProgress, LambdaMicroVMProtocolError):
            raise
        except Exception as exc:
            raise LambdaMicroVMOwnershipUnverified(
                f"Lambda MicroVM sidecar could not confirm ownership; {action} was not sent "
                "and cleanup remains pending."
            ) from exc
        generation = response.get("generation")
        if type(generation) is not int or generation <= 0 or response.get("action") != action:
            raise LambdaMicroVMProtocolError("Lambda MicroVM lifecycle lease was malformed.")

    async def _release_lifecycle_lease(self) -> None:
        try:
            await self._endpoint_call("release_lifecycle", claim_id=self._owner_claim)
        except Exception:
            # The lease stays held until the guest settles it; the original
            # failure is reported.
            _LOGGER.debug("Lambda MicroVM lifecycle lease release failed.", exc_info=True)

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
                await self._claim_owner()
                return
            except (_LambdaMicroVMProtocolVersionMismatch, LambdaMicroVMOwnershipSuperseded):
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

    async def _claim_owner(self) -> None:
        """Become the MicroVM's only command owner; later claims fence this runner.

        A runner claims once. Readiness after resume only confirms the existing
        claim, so a stale runner can never take the MicroVM back from its
        successor; a sidecar that lost its fence also reports it as superseded.
        """

        if self._owner_superseded:
            raise LambdaMicroVMOwnershipSuperseded("Lambda MicroVM has a newer owner.")
        if self._owner_claimed:
            response = await self._endpoint_call("check_owner", claim_id=self._owner_claim)
            if response.get("current") is not True:
                self._mark_superseded()
                raise LambdaMicroVMOwnershipSuperseded("Lambda MicroVM has a newer owner.")
            return
        response = await self._endpoint_call("claim_owner", claim_id=self._owner_claim)
        generation = response.get("generation")
        if type(generation) is not int or generation <= 0:
            raise LambdaMicroVMProtocolError("Lambda MicroVM owner claim returned no generation.")
        self._owner_generation = generation
        self._owner_claimed = True

    def workspace_capability(
        self,
        capability_type: type[RunnerWorkspaceCapabilityT],
    ) -> RunnerWorkspaceCapabilityT | None:
        if type(self) is LambdaMicroVMRunner and capability_type is RemoteWorkspaceBranchCapability:
            # The MicroVM disk retains branch state across Cayu process loss and
            # suspend/resume for the MicroVM's lifetime; the sidecar guarantees
            # python3 for the guest guard.
            capability = _LambdaRemoteWorkspaceBranchCapability(self)
            return cast("RunnerWorkspaceCapabilityT", capability)
        return super().workspace_capability(capability_type)

    @property
    def owner_fence_generation(self) -> str | None:
        """Non-secret identity of this runner's current sidecar owner claim.

        ``None`` until the runner has claimed the MicroVM or once any guest
        operation reported that a successor superseded it. Because every guest
        operation is fenced, a successor that claims inside a caller's window
        makes some operation in that window fail and clears this value.
        """

        if not self._owner_claimed or self._owner_superseded or self._owner_generation is None:
            return None
        return f"{self.microvm_id}:{self._owner_generation}"

    def _mark_superseded(self) -> None:
        self._owner_superseded = True
        self._poison_exec("Lambda MicroVM ownership was superseded")

    async def _endpoint_start(self, command_id: str, payload: dict[str, Any]) -> Mapping[str, Any]:
        try:
            if not self._owner_claimed:
                # A directly constructed runner claims before its first command.
                await self._claim_owner()
            response = await self._endpoint_call(
                "start_command",
                command_id=command_id,
                payload={**payload, "owner_claim": self._owner_claim},
            )
        except LambdaMicroVMOwnershipSuperseded:
            self._mark_superseded()
            raise
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


class _LambdaRemoteWorkspaceBranchCapability(RemoteWorkspaceBranchCapability):
    """The retained MicroVM filesystem for the guest branch protocol."""

    def __init__(self, runner: LambdaMicroVMRunner) -> None:
        self._runner = runner

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("lambda-microvm", self._runner.microvm_id)

    @property
    def allocation_fingerprint(self) -> str:
        encoded = f"lambda-microvm\0{self._runner.microvm_id}".encode()
        return "sha256:" + sha256(encoded).hexdigest()


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


@dataclass(frozen=True, slots=True)
class _LambdaAdmissionIdentity:
    """One complete observation of the exact MicroVM an admission probe ran in."""

    local: tuple[object, ...]
    image_arn: str
    image_version: str | None
    protocol_version: str
    boot_id: str


@dataclass
class _LambdaAdmissionState:
    candidate: Any = None
    identity: _LambdaAdmissionIdentity | None = None


@dataclass(frozen=True, repr=False)
class _LambdaAdmissionObserver(RunnerExecutionAdmissionObserver):
    """Probe executables through the agent lane of one exact, unchanged MicroVM.

    Identity is read before and after the probes from the control plane
    (``get_microvm``), the sidecar health endpoint, and the guest kernel boot
    id. Any change fails closed. Probes use ``runner.exec``, so every dispatched
    command is settled, or the runner is fenced, by the runner's own command
    cleanup before this observer returns or raises.
    """

    runner: LambdaMicroVMRunner
    state: _LambdaAdmissionState = field(default_factory=_LambdaAdmissionState)

    def _fingerprint(self, identity: _LambdaAdmissionIdentity | None) -> str:
        runner = self.runner
        return (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    {
                        "microvm_id": runner.microvm_id,
                        "endpoint": runner.endpoint,
                        "image_arn": None if identity is None else identity.image_arn,
                        "image_version": None if identity is None else identity.image_version,
                        "root": runner.default_cwd,
                        "environment": copy_runner_env(runner.env_overlay, inherit_env=False),
                        "protocol_version": None if identity is None else identity.protocol_version,
                        "boot_id": None if identity is None else identity.boot_id,
                    },
                    "lambda_microvm_execution_identity",
                )
            ).hexdigest()
        )

    def _candidate(self, identity: _LambdaAdmissionIdentity | None, claims=()):
        from cayu.environments.admission import (
            ExecutionAdmissionCandidate,
            ExecutionCapabilityEvidence,
            ExecutionToolRequirementEvidence,
        )

        fingerprint = self._fingerprint(identity)
        return ExecutionAdmissionCandidate(
            candidate="lambda-microvm",
            evidence=ExecutionCapabilityEvidence(
                subject="lambda-microvm",
                unclaimed_reason_code="security_unclaimed",
                environment_fingerprint=fingerprint,
                tool_requirements=ExecutionToolRequirementEvidence(
                    environment_fingerprint=fingerprint,
                    executables=claims,
                ),
            ),
        )

    def snapshot(self):
        from cayu.environments.admission import ExecutionExecutableEvidence

        self.runner._require_admission_open()
        identity = self.state.identity
        candidate = self.state.candidate
        if (
            candidate is not None
            and identity is not None
            and identity.local == self.runner._admission_local_identity()
        ):
            return candidate
        probes = {probe.executable: probe for probe in self.requirements.executable_probes()}
        return self._candidate(
            None,
            tuple(
                ExecutionExecutableEvidence(
                    executable=name,
                    state="declared",
                    requirement_fingerprint=None
                    if name not in probes
                    else probes[name].fingerprint,
                )
                for name in self.requirements.executable_names()
            ),
        )

    async def _observe_identity(self) -> _LambdaAdmissionIdentity:
        runner = self.runner
        runner._require_admission_open()
        local = runner._admission_local_identity()
        # A read-only control-plane call; a timed-out worker thread cannot
        # mutate the MicroVM and its late answer is discarded.
        response = await asyncio.wait_for(
            asyncio.to_thread(runner._client.get_microvm, microvmIdentifier=runner.microvm_id),
            timeout=runner.request_timeout_s,
        )
        response_id, endpoint = _microvm_identity(response)
        if response_id != runner.microvm_id or endpoint != runner.endpoint:
            raise _LambdaMicroVMAdmissionIdentityDrift(
                "get_microvm returned a different MicroVM identity during admission."
            )
        state = _required_response_string(response, "state")
        if state not in _LAMBDA_ADMISSION_RUNNING_STATES:
            raise LambdaMicroVMError(
                f"Lambda MicroVM is {state}; executable admission requires a running MicroVM."
            )
        image_arn = _response_string(response, "imageArn")
        if image_arn is None:
            raise LambdaMicroVMProtocolError("get_microvm omitted the MicroVM image ARN.")
        image_version = _response_string(response, "imageVersion")
        if (
            runner.image_identifier is not None
            and runner.image_identifier.startswith("arn:")
            and image_arn != runner.image_identifier
        ) or (runner.image_version is not None and image_version != runner.image_version):
            raise _LambdaMicroVMAdmissionIdentityDrift(
                "Lambda MicroVM reports a different image than this runner was bound to."
            )
        # Proves the running sidecar speaks the exact protocol this runner uses.
        await runner._endpoint_health()
        result = await runner.exec(
            ExecCommand.process("/bin/sh", "-c", _LAMBDA_BOOT_ID_SCRIPT),
            timeout_s=LAMBDA_MICROVM_ADMISSION_PROBE_TIMEOUT_SECONDS,
            output_limit_bytes=_LAMBDA_ADMISSION_PROBE_OUTPUT_LIMIT_BYTES,
        )
        boot_id = result.stdout.strip()
        if (
            result.timed_out
            or result.cancelled
            or result.exit_code != 0
            or _LAMBDA_BOOT_ID_PATTERN.fullmatch(boot_id) is None
        ):
            raise LambdaMicroVMError("Lambda MicroVM guest boot identity is unavailable.")
        return _LambdaAdmissionIdentity(
            local=local,
            image_arn=image_arn,
            image_version=image_version,
            protocol_version=LAMBDA_MICROVM_PROTOCOL_VERSION,
            boot_id=boot_id,
        )

    async def _probe(self, command: ExecCommand) -> int:
        result = await self.runner.exec(
            command,
            timeout_s=LAMBDA_MICROVM_ADMISSION_PROBE_TIMEOUT_SECONDS,
            output_limit_bytes=_LAMBDA_ADMISSION_PROBE_OUTPUT_LIMIT_BYTES,
        )
        if result.timed_out or result.cancelled:
            raise LambdaMicroVMError("Lambda MicroVM admission probe did not complete.")
        return result.exit_code

    async def collect(self):
        from cayu.environments.admission import (
            EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS,
            ExecutionExecutableEvidence,
        )

        self.state.candidate = None
        self.state.identity = None
        # Timestamp the start of the complete observation, not its end.
        observed_at = datetime.now(UTC)
        valid_until = observed_at + timedelta(seconds=EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS)
        initial = await self._observe_identity()
        probes = {probe.executable: probe for probe in self.requirements.executable_probes()}
        claims = []
        for name in self.requirements.executable_names():
            probe = probes.get(name)
            arguments = None if probe is None else probe.probe_arguments
            command = (
                ExecCommand.process(
                    "/bin/sh", "-c", EXECUTABLE_AVAILABILITY_SCRIPT, "cayu-admission", name
                )
                if arguments is None
                else ExecCommand.process(name, *arguments)
            )
            code = await self._probe(command)
            available = code in ((0,) if probe is None else probe.accepted_exit_codes)
            claims.append(
                ExecutionExecutableEvidence(
                    executable=name,
                    state="live_verified" if available else "unavailable",
                    observed_at=observed_at if available else None,
                    valid_until=valid_until if available else None,
                    requirement_fingerprint=None if probe is None else probe.fingerprint,
                    reason_code=None if available else "executable_unavailable",
                    remediation_code=None if available else "install_executable",
                )
            )
        final = await self._observe_identity()
        if final != initial:
            raise _LambdaMicroVMAdmissionIdentityDrift(
                "Lambda MicroVM identity changed during admission observation."
            )
        self.runner._require_admission_open()
        candidate = self._candidate(final, tuple(claims))
        self.state.identity = final
        self.state.candidate = candidate
        return candidate

    async def refresh(self):
        await self.collect()


_BROWSER_WORKLOAD_PROBE_OUTPUT_LIMIT_BYTES = 4096
# Runs in the agent profile. Exit status is the whole protocol.
_BROWSER_LAUNCH_PROBE_SCRIPT = """
import asyncio, shutil, tempfile
from playwright.async_api import async_playwright

async def launch(home):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(chromium_sandbox=True, env={'HOME': home})
        page = await browser.new_page()
        await page.set_content('<main>cayu</main>')
        await browser.close()

home = tempfile.mkdtemp(prefix='cayu-browser-probe-')
try:
    asyncio.run(launch(home))
finally:
    shutil.rmtree(home, ignore_errors=True)
"""
# Runs in the trusted profile with the worker's own interpreter. It reports
# digests and versions only; the host compares them for equality and never
# interprets other guest text.
_BROWSER_WORKLOAD_PROBE_SCRIPT = """
import glob, hashlib, importlib.metadata as metadata, json, os, stat, sys
root, browsers, names = sys.argv[1], sys.argv[2], sys.argv[3:]
def owned(path, directory=False):
    try:
        info = os.lstat(path)
    except OSError:
        return False
    kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    return kind and info.st_uid == 0 and not info.st_mode & 0o022
files = {}
for name in names:
    path = os.path.join(root, name)
    if owned(path):
        with open(path, 'rb') as handle:
            files[name] = hashlib.sha256(handle.read()).hexdigest()
    else:
        files[name] = None
versions = {}
for name in ('playwright', 'websockets'):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
shells = [
    path
    for pattern in ('headless_shell', 'chrome-headless-shell')
    for path in glob.glob(os.path.join(browsers, 'chromium_headless_shell-*', '**', pattern),
                          recursive=True)
]
print(json.dumps({
    'directory': owned(root, directory=True),
    'files': files,
    'versions': versions,
    'interpreter': owned(os.path.realpath(sys.executable)),
    'headless_shell': any(owned(path) and os.access(path, os.X_OK) for path in shells),
    'certutil': owned('/usr/bin/certutil') and os.access('/usr/bin/certutil', os.X_OK),
}, sort_keys=True))
"""


# Runs in the trusted profile. Installs the control server's public roots where
# the browser worker's control transport reads them, root-owned and read-only.
_CONTROL_CA_INSTALL_SCRIPT = """
import os, sys
path = sys.argv[1]
directory = os.path.dirname(path)
os.makedirs(directory, mode=0o755, exist_ok=True)
os.chown(directory, 0, 0)
os.chmod(directory, 0o755)
temporary = path + '.tmp'
with open(temporary, 'wb') as handle:
    handle.write(sys.stdin.buffer.read())
os.chown(temporary, 0, 0)
os.chmod(temporary, 0o644)
os.replace(temporary, path)
"""
# Runs in the agent profile with the worker's interpreter and control trust.
# It completes a TLS handshake through the relay and sends nothing else. Exit
# status is the whole protocol.
_CONTROL_RELAY_PROBE_SCRIPT = """
import socket, ssl, sys
sys.path.insert(0, sys.argv[1])
from _browser_control_transport import control_tls_context
host, port = sys.argv[2], int(sys.argv[3])
try:
    context = control_tls_context()
except Exception:
    sys.exit(20)
try:
    connection = socket.create_connection((host, port), timeout=10)
except OSError:
    sys.exit(21)
try:
    with context.wrap_socket(connection, server_hostname=host):
        pass
except ssl.SSLCertVerificationError:
    sys.exit(22)
except (OSError, ssl.SSLError):
    sys.exit(23)
"""
_CONTROL_RELAY_PROBE_FAILURES = {
    20: "the installed control CA is unusable",
    21: "the relay did not connect to the control server",
    22: "the control server certificate is not trusted for cayu-control",
    23: "the TLS handshake with the control server failed",
}


def _control_ca_certificate(value: object) -> bytes:
    """Return only the PEM certificates from ``value``; refuse anything else."""

    if type(value) is not bytes or not value or len(value) > _CONTROL_CA_MAX_BYTES:
        raise ValueError("control CA must be PEM certificate bytes of at most 64 KiB.")
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding

    try:
        certificates = x509.load_pem_x509_certificates(value)
    except ValueError:
        raise ValueError("control CA must contain PEM certificates.") from None
    if not certificates:
        raise ValueError("control CA must contain PEM certificates.")
    return b"".join(certificate.public_bytes(Encoding.PEM) for certificate in certificates)


def _browser_workload_mismatch(observed: object, expected: Mapping[str, str]) -> str | None:
    if not isinstance(observed, dict):
        return "the probe returned no report"
    observed = cast("dict[str, Any]", observed)
    if observed.get("directory") is not True:
        return f"{BROWSER_WORKER_DIRECTORY} is not a root-owned, read-only directory"
    if observed.get("interpreter") is not True:
        return "the worker interpreter is not root-owned and read-only"
    files = observed.get("files")
    if not isinstance(files, dict):
        return "the probe reported no worker files"
    for name, digest in expected.items():
        if files.get(name) != digest:
            return f"worker file {name} is missing, writable, or from another release"
    versions = observed.get("versions")
    if not isinstance(versions, dict):
        return "the probe reported no component versions"
    pins = {
        "playwright": BROWSER_WORKER_PLAYWRIGHT_VERSION,
        "websockets": BROWSER_WORKER_WEBSOCKETS_VERSION,
    }
    for name, version in pins.items():
        if versions.get(name) != version:
            return f"{name} is not the pinned {version}"
    if observed.get("headless_shell") is not True:
        return (
            f"no root-owned Chromium headless shell under {BROWSER_WORKER_PLAYWRIGHT_BROWSERS_PATH}"
        )
    if observed.get("certutil") is not True:
        return "/usr/bin/certutil (NSS tools) is not installed"
    return None


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


def lambda_microvm_run_options(
    image_identifier: str,
    *,
    image_version: str | None = None,
    execution_role_arn: str | None = None,
    ingress_network_connectors: list[str] | None = None,
    egress_network_connectors: list[str] | None = None,
    idle_policy: dict[str, Any] | None = None,
    maximum_duration_in_seconds: int | None = None,
    run_hook_payload: str | None = None,
) -> dict[str, Any]:
    """Return the exact ``RunMicrovm`` parameters Cayu submits for one allocation."""

    run_options: dict[str, Any] = {
        "imageIdentifier": require_clean_nonblank(image_identifier, "image_identifier")
    }
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
    return run_options


async def run_microvm_with_client_token(
    client: Any,
    run_options: Mapping[str, Any],
    *,
    client_token: str,
    submit_not_after: float | None = None,
) -> Mapping[str, Any]:
    """Submit or replay one idempotent ``RunMicrovm`` request.

    Transient control-plane failures are retried with the same token and
    parameters a bounded number of times. The returned acknowledgement may be a
    replay of the original response; callers must read current state separately.

    With ``submit_not_after`` (epoch seconds), no attempt is sent after that
    instant: a botocore client refuses, at its send boundary, any request whose
    SigV4 signing time is later, including botocore's own re-signed retries;
    other clients are checked in the calling thread immediately before the call.
    A refused attempt raises ``LambdaMicroVMSubmissionClosed`` and is never
    retried, so no request for the token can reach AWS later than
    ``submit_not_after + LAMBDA_SIGV4_REQUEST_VALIDITY_SECONDS``.
    """

    token = _validate_client_token(client_token)
    if submit_not_after is not None and (
        type(submit_not_after) not in {int, float} or not isfinite(submit_not_after)
    ):
        raise ValueError("submit_not_after must be finite epoch seconds.")
    options = {**dict(run_options), "clientToken": token}
    delays = _LAMBDA_CLIENT_TOKEN_RETRY_DELAYS_SECONDS
    for attempt in range(len(delays) + 1):
        try:
            response = await asyncio.to_thread(
                _submit_run_microvm, client, options, submit_not_after
            )
        except LambdaMicroVMSubmissionClosed:
            raise
        except Exception as exc:
            if _is_client_token_conflict(exc):
                raise LambdaMicroVMClientTokenConflict(
                    "Lambda MicroVM client token was already used with different run "
                    "parameters; refusing to create a replacement allocation."
                ) from exc
            if attempt >= len(delays) or not _is_transient_control_error(exc):
                raise
            await asyncio.sleep(delays[attempt])
            continue
        if not isinstance(response, Mapping):
            raise LambdaMicroVMProtocolError("run_microvm response must be an object.")
        return response
    raise AssertionError("unreachable")


async def terminate_microvm_confirmed(
    client: Any,
    microvm_id: str,
    *,
    timeout_s: float,
    poll_interval_s: float = 0.5,
) -> str:
    """Terminate one exact MicroVM and return only after terminal evidence.

    ``TerminateMicrovm`` is idempotent. The result is ``"terminated"`` or
    ``"absent"`` when the control plane no longer knows the identifier. A
    timeout raises and leaves termination retryable.
    """

    identifier = require_clean_nonblank(microvm_id, "microvm_id")
    timeout = _positive_float(timeout_s, "timeout_s")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    try:
        await asyncio.to_thread(client.terminate_microvm, microvmIdentifier=identifier)
    except Exception as exc:
        if _client_error_code(exc) != "ResourceNotFoundException":
            raise
        return "absent"
    while True:
        try:
            response = await asyncio.to_thread(client.get_microvm, microvmIdentifier=identifier)
        except Exception as exc:
            if _client_error_code(exc) != "ResourceNotFoundException":
                raise
            return "absent"
        if not isinstance(response, Mapping):
            raise LambdaMicroVMProtocolError("get_microvm response must be an object.")
        if _required_response_string(response, "microvmId") != identifier:
            raise LambdaMicroVMProtocolError("get_microvm returned the wrong MicroVM id.")
        if _required_response_string(response, "state") == "TERMINATED":
            return "terminated"
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise LambdaMicroVMError(
                f"Lambda MicroVM {identifier} did not reach TERMINATED within {timeout:g} seconds."
            )
        await asyncio.sleep(min(max(poll_interval_s, 0.05), remaining))


async def read_microvm_state(client: Any, microvm_id: str) -> str | None:
    """Return one exact MicroVM's control-plane state, or ``None`` when not found."""

    identifier = require_clean_nonblank(microvm_id, "microvm_id")
    try:
        response = await asyncio.to_thread(client.get_microvm, microvmIdentifier=identifier)
    except Exception as exc:
        if _client_error_code(exc) != "ResourceNotFoundException":
            raise
        return None
    if not isinstance(response, Mapping):
        raise LambdaMicroVMProtocolError("get_microvm response must be an object.")
    if _required_response_string(response, "microvmId") != identifier:
        raise LambdaMicroVMProtocolError("get_microvm returned the wrong MicroVM id.")
    return _required_response_string(response, "state")


async def _read_adoptable_state(
    client: Any,
    *,
    microvm_id: str,
    endpoint: str,
    image_arn: str | None,
    image_version: str | None,
) -> str:
    response = await asyncio.to_thread(client.get_microvm, microvmIdentifier=microvm_id)
    response_id, response_endpoint = _microvm_identity(response)
    if response_id != microvm_id or response_endpoint != endpoint:
        raise LambdaMicroVMProtocolError(
            "get_microvm returned a different identity for the client-token allocation."
        )
    if image_arn is not None and _response_string(response, "imageArn") != image_arn:
        raise LambdaMicroVMProtocolError("Client-token allocation reports a different image.")
    if image_version is not None and _response_string(response, "imageVersion") != image_version:
        raise LambdaMicroVMProtocolError(
            "Client-token allocation reports a different image version."
        )
    state = _required_response_string(response, "state")
    if state in _LAMBDA_UNUSABLE_ALLOCATION_STATES:
        raise LambdaMicroVMAllocationTerminated(
            f"Client-token allocation {microvm_id} is {state}; it cannot be adopted and "
            "will not be replaced.",
            microvm_id=microvm_id,
            state=state,
        )
    return state


def _submit_run_microvm(
    client: Any, options: Mapping[str, Any], submit_not_after: float | None
) -> Any:
    if submit_not_after is None:
        return client.run_microvm(**options)
    _install_run_submission_guard(client)
    if _submission_clock() > submit_not_after:
        raise _submission_closed(submit_not_after)
    _RUN_SUBMISSION_DEADLINE.not_after = submit_not_after
    try:
        return client.run_microvm(**options)
    finally:
        _RUN_SUBMISSION_DEADLINE.not_after = None


def _install_run_submission_guard(client: Any) -> None:
    events = getattr(getattr(client, "meta", None), "events", None)
    register = getattr(events, "register", None)
    if callable(register):
        # unique_id makes registration idempotent per client.
        register(
            _RUN_SUBMISSION_GUARD_EVENT,
            _refuse_late_run_submission,
            unique_id=_RUN_SUBMISSION_GUARD_ID,
        )


def _refuse_late_run_submission(request: Any, **_kwargs: Any) -> None:
    # botocore signs every attempt, including its internal retries, before this
    # event; the handler runs in the thread that called run_microvm.
    not_after = getattr(_RUN_SUBMISSION_DEADLINE, "not_after", None)
    if not_after is None:
        return None
    signed_at = _sigv4_signing_time(getattr(request, "headers", None))
    if signed_at is None or signed_at > not_after:
        raise _submission_closed(not_after)
    return None


def _sigv4_signing_time(headers: Any) -> float | None:
    value = headers.get("X-Amz-Date") if isinstance(headers, Mapping) else None
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    if type(value) is not str:
        return None
    try:
        return float(calendar.timegm(time.strptime(value, "%Y%m%dT%H%M%SZ")))
    except ValueError:
        return None


def _submission_closed(not_after: float) -> LambdaMicroVMSubmissionClosed:
    return LambdaMicroVMSubmissionClosed(
        "Lambda MicroVM RunMicrovm submission deadline "
        f"{not_after:.0f} (epoch seconds) has passed; the request was not sent."
    )


def _validate_client_token(value: str) -> str:
    token = require_clean_nonblank(value, "client_token")
    if len(token) > LAMBDA_MICROVM_CLIENT_TOKEN_MAX_LENGTH or not token.isascii():
        raise ValueError(
            "Lambda MicroVM client_token must be ASCII and at most "
            f"{LAMBDA_MICROVM_CLIENT_TOKEN_MAX_LENGTH} characters."
        )
    return token


def _client_error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return code if type(code) is str else None


def _client_error_message(error: BaseException) -> str:
    response = getattr(error, "response", None)
    details = response.get("Error") if isinstance(response, Mapping) else None
    message = details.get("Message") if isinstance(details, Mapping) else None
    return message if type(message) is str else ""


def _is_client_token_conflict(error: BaseException) -> bool:
    # AWS reports token reuse with different parameters as a validation error.
    return (
        _client_error_code(error) in {"ValidationException", "ConflictException"}
        and "clienttoken" in _client_error_message(error).replace(" ", "").lower()
    )


def _is_definitive_provider_rejection(error: BaseException) -> bool:
    """AWS answered with an error that means the request was not applied."""

    code = _client_error_code(error)
    return code is not None and code not in _LAMBDA_AMBIGUOUS_PROVIDER_ERROR_CODES


@dataclass
class _LifecycleDispatch:
    """Send accounting for the one provider request a lifecycle lease authorizes."""

    deadline: float
    sent: int = 0
    rejected: int = 0

    @property
    def unapplied(self) -> bool:
        """Every request that crossed the send boundary was definitively rejected."""

        return self.sent == self.rejected


class _LifecycleSendRefused(LambdaMicroVMOwnershipUnverified):
    """The send guard stopped a lifecycle request before it left the host."""


def _note_deferred_cleanup(error: BaseException, reclamation: _LambdaAllocationReclamation) -> None:
    runner = reclamation.runner
    if reclamation.cleanup_deferred and runner is not None:
        error.add_note(
            f"Lambda MicroVM {runner.microvm_id} was not claimed before this failure and may "
            "belong to another host; it was not suspended or terminated. Its cleanup is "
            "deferred to the allocation reap or its maximum duration."
        )


def _guard_lifecycle_send(**_kwargs: Any) -> None:
    dispatch = getattr(_LIFECYCLE_DISPATCH, "value", None)
    if dispatch is None:
        return
    if dispatch.sent:
        raise _LifecycleSendRefused(
            "Lambda MicroVM lifecycle retry was refused: a lifecycle lease authorizes one "
            "request, and the earlier one stays unresolved under the lease."
        )
    if time.monotonic() > dispatch.deadline:
        raise _LifecycleSendRefused(
            "Lambda MicroVM lifecycle request was not sent within its lifecycle lease."
        )
    dispatch.sent += 1


def _observe_lifecycle_attempt(
    response: Any = None, caught_exception: BaseException | None = None, **_kwargs: Any
) -> bool | None:
    dispatch = getattr(_LIFECYCLE_DISPATCH, "value", None)
    if dispatch is None:
        return None
    if caught_exception is None and response is not None:
        status = getattr(response[0], "status_code", None)
        if type(status) is int and 400 <= status < 500:
            # AWS answered and refused the request; it was not applied.
            dispatch.rejected += 1
    # A lease authorizes one request: never let the SDK retry it.
    return False


def _install_lifecycle_send_guard(client: Any) -> bool:
    """Count and bound every lifecycle send; return whether sends are observable.

    The before-send guard refuses a request signed after the lease window and
    any second request in the same call, so an SDK retry is never sent. The
    needs-retry observer, registered on the exact operation so it runs before
    the service-wide retry handler, records definitive rejections and vetoes
    retries.
    """

    events = getattr(getattr(client, "meta", None), "events", None)
    register = getattr(events, "register", None)
    if not callable(register):
        return False
    for operation in _LAMBDA_LIFECYCLE_OPERATIONS:
        register(
            f"before-send.lambda-microvms.{operation}",
            _guard_lifecycle_send,
            unique_id=f"cayu-lifecycle-lease-{operation}",
        )
        register(
            f"needs-retry.lambda-microvms.{operation}",
            _observe_lifecycle_attempt,
            unique_id=f"cayu-lifecycle-attempt-{operation}",
        )
    return True


def _is_transient_control_error(error: BaseException) -> bool:
    return (
        _client_error_code(error) in _LAMBDA_TRANSIENT_CONTROL_ERROR_CODES
        or type(error).__name__ in _LAMBDA_TRANSIENT_CONTROL_ERROR_TYPES
    )


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
