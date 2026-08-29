from __future__ import annotations

import asyncio
import contextlib
import json
import math
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import require_clean_nonblank
from cayu.egress._resolution import (
    InvalidResolvedAddressError,
    ProhibitedResolvedAddressError,
)
from cayu.egress._resolution import (
    resolve_destination_owned as _resolve_destination_owned,
)
from cayu.egress._resolution import (
    validated_resolved_address as _validated_resolved_address,
)
from cayu.egress.credential_kinds import (
    extract_presented_credential,
    supported_credential_kind_descriptor,
    uses_virtual_credential_namespace,
)
from cayu.egress.destinations import (
    ApprovedEgressDestination,
    EgressProtocol,
    normalize_egress_hostname,
    validate_approved_destinations,
)
from cayu.egress.errors import VirtualCredentialError
from cayu.egress.grants import VirtualCredentialGrant, VirtualCredentialRegistry
from cayu.egress.policy import BrowserEgressPolicy, EgressPolicy, EgressRequest
from cayu.vaults import (
    REDACTED_SECRET,
    ResolvedSecret,
    SecretRedactor,
    SecretRef,
    SecretResolver,
    validate_secret_resolver,
)

CAYU_EGRESS_ERROR_HEADER = "X-Cayu-Egress-Error"
DEFAULT_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
DEFAULT_EGRESS_MAX_ACTIVE_UPSTREAM_OPERATIONS = 16
MAX_EGRESS_MAX_ACTIVE_UPSTREAM_OPERATIONS = 64
_MAX_ACTIVE_CREDENTIAL_RESOLUTIONS = 16
DEFAULT_EGRESS_UPSTREAM_TOTAL_TIMEOUT_S = 60.0

# Headers that must not be forwarded verbatim between hops.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        # Recomputed by the upstream client from the actual body/target.
        "content-length",
        "host",
        # Cayu owns this response-only diagnostic namespace. Never allow an
        # upstream origin or sandbox request to forge one of these values.
        CAYU_EGRESS_ERROR_HEADER.lower(),
    }
)

_PROCESS_CONTROL_SIGNALS = (KeyboardInterrupt, SystemExit, GeneratorExit)


def _contains_process_control_signal(error: BaseException) -> bool:
    if isinstance(error, _PROCESS_CONTROL_SIGNALS):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_contains_process_control_signal(child) for child in error.exceptions)
    return False


class CapturedRequest(BaseModel):
    """One outbound request captured outside the sandbox by the egress proxy."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    method: str
    host: str
    path: str
    protocol: EgressProtocol = "https"
    port: int = 443
    query: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes = b""

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).upper()

    @field_validator("host")
    @classmethod
    def normalize_host(cls, value: str, info) -> str:
        return normalize_egress_hostname(value, field_name=info.field_name)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if not value.startswith("/"):
            raise ValueError("`path` must start with '/'.")
        return value

    @field_validator("port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if type(value) is not int:
            raise TypeError("`port` must be an integer.")
        if value <= 0 or value > 65535:
            raise ValueError("`port` must be between 1 and 65535.")
        return value

    def policy_view(self) -> EgressRequest:
        return EgressRequest(
            method=self.method,
            host=self.host,
            path=self.path,
            query=self.query,
            body=self.body,
            content_type=_header_get(self.headers, "content-type"),
        )

    def url(self) -> str:
        suffix = f"?{self.query}" if self.query else ""
        authority = self.host if self.port == 443 else f"{self.host}:{self.port}"
        return f"{self.protocol}://{authority}{self.path}{suffix}"


class CapturedResponse(BaseModel):
    """The provider response returned to the sandbox after scrubbing."""

    model_config = ConfigDict(extra="forbid")

    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes = b""


@dataclass(frozen=True)
class EgressDecision:
    """Secret-free audit record of one broker decision.

    Intentionally has no field that could carry the real credential, so it is
    always safe to log or emit as an event.
    """

    allowed: bool
    status_code: int
    destination: str
    method: str
    path: str
    grant_id: str | None
    policy_name: str | None
    reason: str | None
    authorization_kind: Literal["virtual_credential", "credentialless"] = "virtual_credential"


@dataclass(frozen=True)
class EgressUpstreamLimits:
    """Cayu-owned limits that an upstream must enforce while reading."""

    max_response_bytes: int
    total_timeout_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_response_bytes",
            _bounded_response_bytes(self.max_response_bytes),
        )
        object.__setattr__(
            self,
            "total_timeout_s",
            _bounded_timeout(self.total_timeout_s),
        )


UpstreamOperationFactory = Callable[[], Awaitable[CapturedResponse]]
UpstreamCancellationOwner = Callable[[asyncio.Task[CapturedResponse]], Awaitable[None]]


class EgressUpstreamOperation:
    """One prepared upstream call with positive cancellation settlement.

    Construction is side-effect free. ``result()`` starts the operation once.
    When no cancellation owner is supplied, cancellation waits for natural
    completion rather than treating cancellation of an asyncio wrapper as
    proof that opaque delegated work stopped.
    """

    def __init__(
        self,
        operation_factory: UpstreamOperationFactory,
        *,
        cancel_and_wait: UpstreamCancellationOwner | None = None,
    ) -> None:
        if not callable(operation_factory):
            raise TypeError("operation_factory must be callable.")
        if cancel_and_wait is not None and not callable(cancel_and_wait):
            raise TypeError("cancel_and_wait must be callable or None.")
        self._operation_factory = operation_factory
        self._cancellation_owner = cancel_and_wait
        self._task: asyncio.Task[CapturedResponse] | None = None
        self._cancelled_before_start = False
        self._factory_started = False
        self._settlement_lock = asyncio.Lock()

    def _start_task(self) -> asyncio.Task[CapturedResponse] | None:
        task = self._task
        if task is None:
            if self._cancelled_before_start:
                return None

            async def invoke() -> CapturedResponse:
                if self._cancelled_before_start:
                    raise asyncio.CancelledError(
                        "Egress upstream operation was cancelled before dispatch."
                    )
                self._factory_started = True
                try:
                    return await self._operation_factory()
                except BaseException as error:
                    if _contains_process_control_signal(error):
                        raise _QuiescentUpstreamProcessControl(error) from None
                    raise

            task = asyncio.create_task(
                invoke(),
                name="cayu-egress-upstream-operation",
            )
            self._task = task
        return task

    async def result(self) -> CapturedResponse:
        """Return the provider result, preserving fatal signals for direct callers."""

        task = self._start_task()
        if task is None:
            raise asyncio.CancelledError("Egress upstream operation was cancelled before dispatch.")
        try:
            return await asyncio.shield(task)
        except _QuiescentUpstreamProcessControl as outcome:
            raise outcome.signal from None

    async def cancel_and_wait(self) -> None:
        """Request cancellation and return only after positive quiescence."""

        async with self._settlement_lock:
            task = self._task
            if task is None:
                self._cancelled_before_start = True
                return
            if not self._factory_started:
                self._cancelled_before_start = True
                task.cancel("Egress upstream operation was cancelled before dispatch.")
                outcome = await await_shielded_task_outcome(task)
                if outcome.timed_out:  # pragma: no cover - no timeout is supplied
                    raise RuntimeError("Upstream operation settlement timed out.")
                return
            if task.done():
                try:
                    task.result()
                except _QuiescentUpstreamProcessControl:
                    raise
                except BaseException as error:
                    if _contains_process_control_signal(error):
                        raise _QuiescentUpstreamProcessControl(error) from None
                return
            if self._cancellation_owner is not None:
                try:
                    await self._cancellation_owner(task)
                except BaseException as error:
                    task_error: BaseException | None = None
                    if task.done():
                        with contextlib.suppress(asyncio.CancelledError):
                            task_error = task.exception()
                    if error is task_error and isinstance(error, _QuiescentUpstreamProcessControl):
                        raise
                    if error is task_error and _contains_process_control_signal(error):
                        raise _QuiescentUpstreamProcessControl(error) from None
                    raise
            outcome = await await_shielded_task_outcome(task)
            if isinstance(outcome.error, _QuiescentUpstreamProcessControl):
                raise outcome.error
            if outcome.error is not None and _contains_process_control_signal(outcome.error):
                raise _QuiescentUpstreamProcessControl(outcome.error) from None
            if outcome.timed_out:  # pragma: no cover - no timeout is supplied
                raise RuntimeError("Upstream operation settlement timed out.")


@runtime_checkable
class EgressUpstream(Protocol):
    """Prepares a bounded provider call without dispatching it."""

    def prepare(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> EgressUpstreamOperation: ...


DestinationResolver = Callable[[str, int], Awaitable[Sequence[str]]]
AuthorizationKind = Literal["virtual_credential", "credentialless"]


@dataclass(frozen=True)
class _ResolvedUpstreamTarget:
    url: str
    host_header: str
    sni_hostname: str | None


@dataclass(frozen=True)
class _ForwardingAuthorization:
    grant_id: str | None
    policy_name: str
    authorization_kind: AuthorizationKind
    secrets: tuple[str, ...] = ()
    require_identity_encoding: bool = False


@dataclass
class _ActiveUpstreamOperation:
    operation: EgressUpstreamOperation
    result_task: asyncio.Task[CapturedResponse]
    settlement_task: asyncio.Task[None] | None = None


@dataclass
class _ActiveCredentialResolution:
    task: asyncio.Task[ResolvedSecret] | None = None
    started: bool = False


@dataclass(eq=False)
class _ConnectDestinationAdmission:
    host: str
    port: int
    revoked: threading.Event = field(default_factory=threading.Event)


class _UpstreamDestinationDeniedError(ValueError):
    pass


class _UpstreamDnsError(ValueError):
    pass


class _UpstreamTimeoutError(RuntimeError):
    pass


class _UpstreamResponseTooLargeError(RuntimeError):
    pass


class _UpstreamUnsupportedEncodingError(RuntimeError):
    pass


class _UpstreamCapacityError(RuntimeError):
    pass


class _QuiescentUpstreamProcessControl(BaseException):
    """Carries a fatal operation outcome after exact quiescence is proven."""

    def __init__(self, signal: BaseException) -> None:
        super().__init__("A quiescent upstream operation produced a process-control signal.")
        self.signal = signal


class _UnsettledUpstreamProcessControl(BaseException):
    """Carries a fatal settlement failure while upstream work may remain active."""

    def __init__(self, signal: BaseException) -> None:
        super().__init__("Upstream settlement produced a process-control signal.")
        self.signal = signal


async def _owned_upstream_result(operation: EgressUpstreamOperation) -> CapturedResponse:
    """Invoke the public result seam without leaking fatal signals through its task."""

    try:
        return await operation.result()
    except _QuiescentUpstreamProcessControl:
        raise
    except BaseException as error:
        if _contains_process_control_signal(error):
            raise _QuiescentUpstreamProcessControl(error) from None
        raise


async def _owned_upstream_settlement(operation: EgressUpstreamOperation) -> None:
    """Run extension settlement without leaking fatal signals through its task."""

    try:
        await operation.cancel_and_wait()
    except _QuiescentUpstreamProcessControl:
        raise
    except BaseException as error:
        if _contains_process_control_signal(error):
            raise _UnsettledUpstreamProcessControl(error) from None
        raise


class _CredentialResolutionCapacityError(RuntimeError):
    pass


class HttpxUpstream:
    """Default upstream that forwards requests to the provider over HTTPS."""

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        max_response_bytes: int = DEFAULT_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
        routes: Mapping[str, str] | None = None,
        destination_resolver: DestinationResolver | None = None,
    ) -> None:
        self._timeout_s = _bounded_timeout(timeout_s)
        self._max_response_bytes = _bounded_response_bytes(max_response_bytes)
        self._transport = transport
        self._routes = _validated_upstream_routes(routes or {})
        self._owns_destination_resolver = destination_resolver is None
        self._destination_resolver = destination_resolver or _resolve_destination_owned

    def prepare(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> EgressUpstreamOperation:
        if not isinstance(limits, EgressUpstreamLimits):
            raise TypeError("limits must be an EgressUpstreamLimits instance.")
        effective_limits = EgressUpstreamLimits(
            max_response_bytes=min(self._max_response_bytes, limits.max_response_bytes),
            total_timeout_s=min(self._timeout_s, limits.total_timeout_s),
        )
        return EgressUpstreamOperation(
            lambda: self._send(request, limits=effective_limits),
            cancel_and_wait=self._cancel_and_wait,
        )

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        """Direct convenience entrance using this upstream's configured limits."""

        operation = self.prepare(
            request,
            limits=EgressUpstreamLimits(
                max_response_bytes=self._max_response_bytes,
                total_timeout_s=self._timeout_s,
            ),
        )
        caller_cancellation: asyncio.CancelledError | None = None
        cancellation_cause: RuntimeError | None = None
        child_stopped = False
        response: CapturedResponse | None = None
        try:
            response = await operation.result()
        except asyncio.CancelledError as cancellation:
            current_task = asyncio.current_task()
            caller_cancelled = current_task is not None and current_task.cancelling() > 0
            settlement_task = asyncio.create_task(
                _owned_upstream_settlement(operation),
                name="cayu-httpx-upstream-direct-settlement",
            )
            outcome = await await_shielded_task_outcome(
                settlement_task,
                cancellation=cancellation if caller_cancelled else None,
            )
            if caller_cancelled and outcome.cancellation is not None:
                restore_task_cancellation_requests(
                    outcome.cancellation_requests_consumed,
                    cancellation=outcome.cancellation,
                )
            if outcome.error is not None:
                if isinstance(outcome.error, _QuiescentUpstreamProcessControl):
                    raise outcome.error.signal from cancellation
                if isinstance(outcome.error, _UnsettledUpstreamProcessControl):
                    raise outcome.error.signal from cancellation
                if _contains_process_control_signal(outcome.error):
                    raise outcome.error from cancellation
                if caller_cancelled:
                    caller_cancellation = cancellation
                    cancellation_cause = RuntimeError(
                        "Direct upstream cancellation settlement was incomplete."
                    )
                else:
                    child_stopped = True
            elif caller_cancelled:
                caller_cancellation = cancellation
            else:
                child_stopped = True
        if caller_cancellation is not None:
            if cancellation_cause is not None:
                raise caller_cancellation from cancellation_cause
            raise caller_cancellation
        if child_stopped:
            raise RuntimeError("Direct upstream operation stopped unexpectedly.")
        assert response is not None
        return response

    @staticmethod
    async def _cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
        task.cancel("Cayu egress upstream operation was cancelled.")
        await await_shielded_task_outcome(task)

    async def _send(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> CapturedResponse:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + limits.total_timeout_s
        try:
            target = await asyncio.wait_for(
                self._target(request),
                timeout=max(deadline - loop.time(), 0.0),
            )
        except ProhibitedResolvedAddressError as exc:
            raise _UpstreamDestinationDeniedError(
                "Upstream destination resolved to a prohibited address."
            ) from exc
        except InvalidResolvedAddressError as exc:
            raise _UpstreamDnsError("Upstream destination returned an invalid address.") from exc
        except TimeoutError as exc:
            raise _UpstreamTimeoutError("Upstream request timed out.") from exc

        remaining = max(deadline - loop.time(), 0.0)
        if remaining <= 0:
            raise _UpstreamTimeoutError("Upstream request timed out.")
        if self._transport is None:
            try:
                async with asyncio.timeout(remaining):
                    return await self._send_to_target(
                        request,
                        target=target,
                        limits=limits,
                        timeout_s=remaining,
                    )
            except (TimeoutError, httpx.TimeoutException) as exc:
                raise _UpstreamTimeoutError("Upstream request timed out.") from exc

        send_task = asyncio.create_task(
            self._send_to_target(
                request,
                target=target,
                limits=limits,
                timeout_s=remaining,
            ),
            name="cayu-httpx-injected-transport-send",
        )
        try:
            return await asyncio.wait_for(asyncio.shield(send_task), timeout=remaining)
        except asyncio.CancelledError as cancellation:
            outcome = await await_shielded_task_outcome(
                send_task,
                cancellation=cancellation,
            )
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
            raise cancellation
        except (TimeoutError, httpx.TimeoutException) as exc:
            await await_shielded_task_outcome(send_task)
            raise _UpstreamTimeoutError("Upstream request timed out.") from exc

    async def _send_to_target(
        self,
        request: CapturedRequest,
        *,
        target: _ResolvedUpstreamTarget,
        limits: EgressUpstreamLimits,
        timeout_s: float,
    ) -> CapturedResponse:
        headers = {
            key: value
            for key, value in _forwardable_headers(request.headers).items()
            if key.lower() != "accept-encoding"
        }
        headers["Host"] = target.host_header
        headers["Accept-Encoding"] = "identity"
        extensions = (
            {"sni_hostname": target.sni_hostname} if target.sni_hostname is not None else None
        )
        async with (
            httpx.AsyncClient(
                timeout=timeout_s,
                transport=self._transport,
                trust_env=False,
                follow_redirects=False,
            ) as client,
            client.stream(
                request.method,
                target.url,
                headers=headers,
                content=request.body or None,
                extensions=extensions,
            ) as response,
        ):
            content_encoding = response.headers.get("content-encoding")
            if content_encoding is not None and content_encoding.strip().lower() != "identity":
                raise _UpstreamUnsupportedEncodingError(
                    "Upstream ignored the required identity content encoding."
                )
            content_length = response.headers.get("content-length")
            if content_length is not None:
                with contextlib.suppress(ValueError):
                    if int(content_length) > limits.max_response_bytes:
                        raise _UpstreamResponseTooLargeError(
                            "Upstream response exceeded the configured byte limit."
                        )
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > limits.max_response_bytes:
                    raise _UpstreamResponseTooLargeError(
                        "Upstream response exceeded the configured byte limit."
                    )
                body.extend(chunk)
            return CapturedResponse(
                status_code=response.status_code,
                headers=_identity_response_headers(dict(response.headers)),
                body=bytes(body),
            )

    async def _target(self, request: CapturedRequest) -> _ResolvedUpstreamTarget:
        route = self._routes.get(request.host)
        origin = route or f"{request.protocol}://{request.host}:{request.port}"
        split = urlsplit(origin)
        host = split.hostname
        if host is None:  # pragma: no cover - constructors already validate origins
            raise ValueError("Upstream target has no hostname.")
        port = split.port or (443 if split.scheme == "https" else 80)
        try:
            if self._owns_destination_resolver:
                addresses = tuple(await self._destination_resolver(host, port))
            else:

                async def resolve_destination() -> Sequence[str]:
                    return await self._destination_resolver(host, port)

                resolver_task = asyncio.create_task(
                    resolve_destination(),
                    name="cayu-egress-destination-resolution",
                )
                try:
                    addresses = tuple(await asyncio.shield(resolver_task))
                except asyncio.CancelledError as cancellation:
                    outcome = await await_shielded_task_outcome(
                        resolver_task,
                        cancellation=cancellation,
                    )
                    restore_task_cancellation_requests(
                        outcome.cancellation_requests_consumed,
                        cancellation=outcome.cancellation,
                    )
                    raise cancellation
        except OSError as exc:
            raise _UpstreamDnsError("Upstream destination resolution failed.") from exc
        if not addresses:
            raise _UpstreamDnsError("Upstream destination did not resolve to an address.")
        allow_private = route is not None
        normalized = tuple(
            _validated_resolved_address(address, allow_private=allow_private)
            for address in addresses
        )
        pinned_host = normalized[0]
        pinned_authority = _format_authority(pinned_host, port, split.scheme)
        host_header = _format_authority(host, port, split.scheme)
        suffix = f"?{request.query}" if request.query else ""
        return _ResolvedUpstreamTarget(
            url=f"{split.scheme}://{pinned_authority}{request.path}{suffix}",
            host_header=host_header,
            sni_hostname=host if split.scheme == "https" else None,
        )


def _bounded_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout_s must be a finite positive number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout_s must be a finite positive number.")
    return normalized


def _bounded_response_bytes(value: int) -> int:
    if type(value) is not int:
        raise TypeError("max_response_bytes must be an integer.")
    if value <= 0 or value > MAX_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES:
        raise ValueError(
            f"max_response_bytes must be between 1 and {MAX_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES}."
        )
    return value


def _format_authority(host: str, port: int, scheme: str) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    return rendered_host if port == default_port else f"{rendered_host}:{port}"


def _validated_upstream_routes(routes: Mapping[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for logical_host, target in routes.items():
        host = normalize_egress_hostname(
            logical_host,
            field_name="HttpxUpstream route host",
        )
        split = urlsplit(target)
        if (
            split.scheme not in {"http", "https"}
            or split.hostname is None
            or split.username is not None
            or split.password is not None
            or split.path not in {"", "/"}
            or split.query
            or split.fragment
        ):
            raise ValueError(
                "HttpxUpstream route target must be an absolute HTTP(S) origin without "
                "credentials, path, query, or fragment."
            )
        try:
            port = split.port
        except ValueError as exc:
            raise ValueError("HttpxUpstream route target has an invalid port.") from exc
        if port is not None and port <= 0:
            raise ValueError("HttpxUpstream route target has an invalid port.")
        if host in validated:
            raise ValueError(f"HttpxUpstream route host {host!r} is duplicated.")
        validated[host] = target.rstrip("/")
    return validated


class TransparentEgressBroker:
    """Authorizes, resolves, rewrites, forwards, and scrubs outbound requests.

    The broker is the *only* place a ``SecretRef`` is resolved to a real value.
    The real value is written into the upstream request and nowhere else — never
    into a decision record, a returned response's diagnostics, or an exception.
    """

    def __init__(
        self,
        *,
        registry: VirtualCredentialRegistry,
        resolver: SecretResolver | None = None,
        policies: Mapping[str, EgressPolicy],
        approved_destinations: Sequence[ApprovedEgressDestination] = (),
        upstream: EgressUpstream | None = None,
        audit: Callable[[EgressDecision], None] | None = None,
        require_test_mode_credentials: bool = True,
        browser_max_response_bytes: int | None = None,
        max_active_upstream_operations: int = DEFAULT_EGRESS_MAX_ACTIVE_UPSTREAM_OPERATIONS,
    ) -> None:
        if resolver is not None:
            validate_secret_resolver(resolver, "resolver")
        if type(max_active_upstream_operations) is not int:
            raise TypeError("max_active_upstream_operations must be an integer.")
        if not 1 <= max_active_upstream_operations <= MAX_EGRESS_MAX_ACTIVE_UPSTREAM_OPERATIONS:
            raise ValueError(
                "max_active_upstream_operations must be between 1 and "
                f"{MAX_EGRESS_MAX_ACTIVE_UPSTREAM_OPERATIONS}."
            )
        self._registry = registry
        self._resolver = resolver
        self._policies = dict(policies)
        self._approved_destinations = _approved_destination_map(approved_destinations)
        self._credentialless_authority_active = True
        self._credentialless_active_requests = 0
        self._credentialless_idle = asyncio.Event()
        self._credentialless_idle.set()
        self._active_connect_admissions: set[_ConnectDestinationAdmission] = set()
        self._connect_admissions_idle = asyncio.Event()
        self._connect_admissions_idle.set()
        self._revoking_grant_ids: set[str] = set()
        self._upstream = upstream or HttpxUpstream()
        self._audit = audit
        self._require_test_mode = require_test_mode_credentials
        self._max_active_upstream_operations = max_active_upstream_operations
        self._active_upstream_operations: dict[object, _ActiveUpstreamOperation | None] = {}
        self._upstream_operations_idle = asyncio.Event()
        self._upstream_operations_idle.set()
        self._active_credential_resolutions: dict[
            object,
            _ActiveCredentialResolution | None,
        ] = {}
        self._credential_resolutions_idle = asyncio.Event()
        self._credential_resolutions_idle.set()
        self._browser_max_response_bytes = _bounded_response_bytes(
            DEFAULT_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES
            if browser_max_response_bytes is None
            else browser_max_response_bytes
        )

    @property
    def registry(self) -> VirtualCredentialRegistry:
        """The credential registry, for session-close revocation by adapters."""
        return self._registry

    @property
    def has_credentialless_destinations(self) -> bool:
        """Whether the broker requires an isolated, independently authenticated transport."""

        return bool(self._approved_destinations)

    async def authorize_connect_destination(self, *, host: str, port: int) -> bool:
        """Return positive coarse authority before the proxy mints a leaf certificate.

        This entrance deliberately authorizes only the HTTPS destination. It does
        not acquire a credential lease, resolve a secret, or replace the complete
        method/path authorization performed by :meth:`handle_request` after TLS.
        Keeping the lookup on the broker's event loop also avoids reading the
        mutable grant registry from a proxy worker thread.
        """

        return self._authorized_connect_host(host=host, port=port) is not None

    async def begin_connect_destination_admission(
        self,
        *,
        host: str,
        port: int,
    ) -> _ConnectDestinationAdmission | None:
        """Lease coarse CONNECT authority through leaf generation and success."""

        normalized_host = self._authorized_connect_host(host=host, port=port)
        if normalized_host is None:
            return None
        admission = _ConnectDestinationAdmission(host=normalized_host, port=port)
        self._active_connect_admissions.add(admission)
        self._connect_admissions_idle.clear()
        return admission

    async def connect_destination_admission_is_active(
        self,
        admission: _ConnectDestinationAdmission,
    ) -> bool:
        """Revalidate one broker-owned CONNECT admission before success."""

        return (
            admission in self._active_connect_admissions
            and not admission.revoked.is_set()
            and self._authorized_connect_host(host=admission.host, port=admission.port)
            == admission.host
        )

    async def end_connect_destination_admission(
        self,
        admission: _ConnectDestinationAdmission,
    ) -> None:
        """Release one CONNECT admission after success or denial is written."""

        self._active_connect_admissions.discard(admission)
        if not self._active_connect_admissions:
            self._connect_admissions_idle.set()

    def _authorized_connect_host(self, *, host: str, port: int) -> str | None:
        if type(port) is not int or port != 443:
            return None
        try:
            normalized_host = normalize_egress_hostname(host, field_name="CONNECT host")
        except (TypeError, ValueError):
            return None
        destination = self._approved_destinations.get((normalized_host, "https", port))
        if (
            self._credentialless_authority_active
            and destination is not None
            and destination.policy_name in self._policies
        ):
            return normalized_host
        if any(
            grant.destination == normalized_host
            and grant.policy_name is not None
            and grant.policy_name in self._policies
            for grant in self._registry.active_grants()
        ):
            return normalized_host
        return None

    async def handle_request(self, request: CapturedRequest) -> CapturedResponse:
        if type(request) is not CapturedRequest:
            raise TypeError("request must be a CapturedRequest instance.")
        request = CapturedRequest(**request.model_dump(mode="python", warnings=False))
        presented = extract_presented_credential(request.headers)
        if presented is None:
            if self._approved_destinations:
                return await self._handle_credentialless(request)
            return self._deny(request, None, None, 401, "No credential presented to broker.")

        try:
            lease = self._registry.acquire(presented.value)
        except Exception:  # unknown / expired / revoked — never echo the value
            if self._credentialless_destination(
                request
            ) is not None and not uses_virtual_credential_namespace(presented.value):
                return await self._handle_credentialless(request)
            return self._deny(request, None, None, 403, "Virtual credential is not valid.")

        try:
            grant = lease.grant

            if (
                grant.destination != request.host
                or request.protocol != "https"
                or request.port != 443
            ):
                return self._deny(
                    request,
                    grant.grant_id,
                    grant.policy_name,
                    403,
                    "Destination not bound to grant.",
                    error_code="destination_denied",
                )

            policy = self._policies.get(grant.policy_name) if grant.policy_name else None
            if policy is None:
                return self._deny(
                    request,
                    grant.grant_id,
                    grant.policy_name,
                    403,
                    "No egress policy bound to grant.",
                )

            decision = policy.authorize(request.policy_view())
            if not decision.allowed:
                # Denied BEFORE any vault resolution — the real secret is never touched.
                return self._deny(
                    request,
                    grant.grant_id,
                    policy.name,
                    403,
                    decision.reason or "Denied by policy.",
                    error_code="destination_denied",
                )

            credential_kind = supported_credential_kind_descriptor(grant.credential_kind)
            if credential_kind is None:
                return self._deny(
                    request,
                    grant.grant_id,
                    policy.name,
                    403,
                    f"Unsupported credential kind {grant.credential_kind!r}.",
                )
            if not credential_kind.accepts(presented):
                return self._deny(
                    request,
                    grant.grant_id,
                    policy.name,
                    403,
                    "Virtual credential authentication scheme does not match its grant.",
                )

            # Resolve + rewrite in one guarded step; a failure here (bad vault, etc.)
            # is reported distinctly from an upstream failure and never leaks a value.
            try:
                if self._resolver is None:
                    raise RuntimeError("No credential resolver is configured.")
                resolved = await self._resolve_credential(
                    grant.secret,
                    grant_id=grant.grant_id,
                )
                real_secret = resolved.value.get_secret_value()
            except _CredentialResolutionCapacityError:
                return self._deny(
                    request,
                    grant.grant_id,
                    policy.name,
                    503,
                    "Credential resolution capacity is exhausted.",
                    error_code="credential_resolution_capacity_exhausted",
                )
            except Exception:
                return self._deny(
                    request, grant.grant_id, policy.name, 502, "Credential resolution failed."
                )

            try:
                lease.ensure_active()
            except VirtualCredentialError:
                return self._deny(
                    request, grant.grant_id, policy.name, 403, "Virtual credential is not valid."
                )

            # Test-mode-only guard: the key class is checked inside broker code (the
            # value never leaves) so a live key bound by mistake fails closed.
            if self._require_test_mode:
                prefixes = credential_kind.test_mode_real_secret_prefixes
                if prefixes and not real_secret.startswith(prefixes):
                    return self._deny(
                        request,
                        grant.grant_id,
                        policy.name,
                        403,
                        "Bound credential is not a test-mode key; refusing "
                        "(set require_test_mode_credentials=False to allow live keys).",
                    )

            rewritten = request.model_copy(
                update={"headers": credential_kind.rewrite_headers(real_secret, request.headers)}
            )

            return await self._forward_authorized(
                request=request,
                upstream_request=rewritten,
                authorization=_ForwardingAuthorization(
                    grant_id=grant.grant_id,
                    policy_name=policy.name,
                    authorization_kind="virtual_credential",
                    secrets=(real_secret,),
                    require_identity_encoding=isinstance(policy, BrowserEgressPolicy),
                ),
                ensure_authority=lease.ensure_active,
            )
        finally:
            lease.close()

    def _credentialless_destination(
        self,
        request: CapturedRequest,
    ) -> ApprovedEgressDestination | None:
        if not self._credentialless_authority_active:
            return None
        return self._approved_destinations.get(
            (request.host, request.protocol, request.port),
        )

    async def revoke_authority_and_wait(self, presented_values: Sequence[str]) -> int:
        """Revoke routes, settle dispatched upstream work, and drain requests."""

        self._credentialless_authority_active = False
        revoked_count, grant_ids = self._registry.revoke_values(presented_values)
        for admission in tuple(self._active_connect_admissions):
            admission.revoked.set()
        self._revoking_grant_ids.update(grant_ids)
        drain_grant_ids = tuple(self._revoking_grant_ids)
        await self.settle_active_operations()
        await asyncio.gather(
            self._registry.wait_for_inactive_grants(drain_grant_ids),
            self._credentialless_idle.wait(),
            self._connect_admissions_idle.wait(),
        )
        self._revoking_grant_ids.difference_update(drain_grant_ids)
        return revoked_count

    async def settle_active_operations(self) -> None:
        """Settle every dispatched credential resolution and upstream call."""

        outcomes = await asyncio.gather(
            self.settle_active_credential_resolutions(),
            self.settle_active_upstream_operations(),
            return_exceptions=True,
        )
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            raise BaseExceptionGroup("Egress operation settlement was incomplete.", failures)

    async def settle_active_credential_resolutions(self) -> None:
        """Retain cancellation-opaque resolution work until natural settlement."""

        resolutions = tuple(
            (token, owner)
            for token, owner in tuple(self._active_credential_resolutions.items())
            if owner is not None and owner.task is not None
        )
        failures: list[BaseException] = []
        for _token, owner in resolutions:
            task = owner.task
            assert task is not None
            if not owner.started:
                task.cancel("Credential resolution was revoked before dispatch.")
        for token, owner in resolutions:
            task = owner.task
            assert task is not None
            outcome = await await_shielded_task_outcome(task)
            if outcome.cancellation is not None:
                restore_task_cancellation_requests(
                    outcome.cancellation_requests_consumed,
                    cancellation=outcome.cancellation,
                )
            if outcome.error is not None and _contains_process_control_signal(outcome.error):
                failures.append(outcome.error)
            self._release_credential_resolution(token)
        if failures:
            raise BaseExceptionGroup(
                "Credential resolution settlement received a process-control signal.",
                failures,
            )
        await self._credential_resolutions_idle.wait()

    async def settle_active_upstream_operations(self) -> None:
        """Cancel dispatched upstream calls and retain failures for retry."""

        settlements = tuple(
            (token, owner, self._arm_upstream_settlement(owner))
            for token, owner in tuple(self._active_upstream_operations.items())
            if owner is not None
        )
        failures: list[BaseException] = []
        for token, owner, settlement_task in settlements:
            try:
                await self._await_upstream_settlement(
                    token,
                    owner,
                    settlement_task,
                )
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup("Egress upstream settlement was incomplete.", failures)
        await self._upstream_operations_idle.wait()

    def renew_authority(self, grants: Sequence[VirtualCredentialGrant]) -> None:
        """Install fresh virtual values and reopen unchanged credentialless routes."""

        if self._credentialless_authority_active:
            raise RuntimeError("Egress authority must be revoked before it can be renewed.")
        bound_values: list[str] = []
        try:
            for grant in grants:
                self._registry.bind(grant)
                bound_values.append(grant.presented_value)
        except BaseException:
            for value in bound_values:
                self._registry.revoke(value)
            raise
        self._credentialless_authority_active = True

    async def _handle_credentialless(self, request: CapturedRequest) -> CapturedResponse:
        destination = self._credentialless_destination(request)
        if destination is None:
            return self._deny(
                request,
                None,
                None,
                403,
                "Destination is not approved for credentialless egress.",
                authorization_kind="credentialless",
                error_code="destination_denied",
            )
        if not self._begin_credentialless_request():
            return self._deny(
                request,
                None,
                destination.policy_name,
                403,
                "Credentialless egress authority has been revoked.",
                authorization_kind="credentialless",
            )
        try:
            policy = self._policies.get(destination.policy_name)
            if policy is None:
                return self._deny(
                    request,
                    None,
                    destination.policy_name,
                    403,
                    "No egress policy bound to approved destination.",
                    authorization_kind="credentialless",
                )
            decision = policy.authorize(request.policy_view())
            if not decision.allowed:
                return self._deny(
                    request,
                    None,
                    policy.name,
                    403,
                    decision.reason or "Denied by policy.",
                    authorization_kind="credentialless",
                    error_code="destination_denied",
                )
            return await self._forward_authorized(
                request=request,
                upstream_request=request,
                authorization=_ForwardingAuthorization(
                    grant_id=None,
                    policy_name=policy.name,
                    authorization_kind="credentialless",
                    require_identity_encoding=isinstance(policy, BrowserEgressPolicy),
                ),
                ensure_authority=self._ensure_credentialless_authority,
            )
        finally:
            self._end_credentialless_request()

    async def _resolve_credential(
        self,
        secret: SecretRef,
        *,
        grant_id: str,
    ) -> ResolvedSecret:
        token = self._reserve_credential_resolution()
        if token is None:
            raise _CredentialResolutionCapacityError("Credential resolution capacity is exhausted.")
        try:
            resolver = self._resolver
            if resolver is None:  # pragma: no cover - checked by the caller
                raise RuntimeError("No credential resolver is configured.")

            owner = _ActiveCredentialResolution()

            async def resolve() -> ResolvedSecret:
                owner.started = True
                return await resolver.resolve(secret, scope={"grant_id": grant_id})

            task = asyncio.create_task(
                resolve(),
                name="cayu-egress-credential-resolution",
            )
            owner.task = task
            self._active_credential_resolutions[token] = owner
            child_cancellation_error: RuntimeError | None = None
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError as cancellation:
                current_task = asyncio.current_task()
                if current_task is None or current_task.cancelling() <= 0:
                    child_cancellation_error = RuntimeError(
                        "Credential resolution stopped without caller cancellation."
                    )
                else:
                    if not owner.started:
                        task.cancel("Credential resolution was cancelled before dispatch.")
                    outcome = await await_shielded_task_outcome(
                        task,
                        cancellation=cancellation,
                    )
                    if outcome.cancellation is not None:
                        restore_task_cancellation_requests(
                            outcome.cancellation_requests_consumed,
                            cancellation=outcome.cancellation,
                        )
                    if outcome.error is not None and _contains_process_control_signal(
                        outcome.error
                    ):
                        raise outcome.error from cancellation
                    raise cancellation
            if child_cancellation_error is not None:
                # Raise only after leaving the handler so the resolver-owned
                # cancellation and any sensitive diagnostic cannot become the
                # sanitized error's implicit ``__context__``.
                raise child_cancellation_error
            raise RuntimeError("Credential resolution ended without an outcome.")
        finally:
            self._release_credential_resolution(token)

    def _reserve_credential_resolution(self) -> object | None:
        if len(self._active_credential_resolutions) >= _MAX_ACTIVE_CREDENTIAL_RESOLUTIONS:
            return None
        token = object()
        self._active_credential_resolutions[token] = None
        self._credential_resolutions_idle.clear()
        return token

    def _release_credential_resolution(self, token: object) -> None:
        self._active_credential_resolutions.pop(token, None)
        if not self._active_credential_resolutions:
            self._credential_resolutions_idle.set()

    async def _forward_authorized(
        self,
        *,
        request: CapturedRequest,
        upstream_request: CapturedRequest,
        authorization: _ForwardingAuthorization,
        ensure_authority: Callable[[], None] | None = None,
    ) -> CapturedResponse:
        if ensure_authority is not None:
            try:
                ensure_authority()
            except VirtualCredentialError:
                return self._authority_revoked(request, authorization)
        if authorization.require_identity_encoding:
            upstream_request = upstream_request.model_copy(
                update={
                    "headers": {
                        **{
                            key: value
                            for key, value in upstream_request.headers.items()
                            if key.lower() != "accept-encoding"
                        },
                        "Accept-Encoding": "identity",
                    }
                }
            )
        effective_response_limit = (
            min(
                MAX_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES,
                self._browser_max_response_bytes,
            )
            if authorization.require_identity_encoding
            else MAX_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES
        )
        try:
            response = await self._run_upstream_operation(
                upstream_request,
                max_response_bytes=effective_response_limit,
            )
        except _UpstreamCapacityError:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                503,
                "Upstream operation capacity is exhausted.",
                authorization_kind=authorization.authorization_kind,
                error_code="upstream_capacity_exhausted",
            )
        except _UpstreamDestinationDeniedError:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                403,
                "Upstream destination is not publicly routable.",
                authorization_kind=authorization.authorization_kind,
                error_code="destination_denied",
            )
        except _UpstreamDnsError:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                502,
                "Upstream destination resolution failed.",
                authorization_kind=authorization.authorization_kind,
                error_code="dns_failure",
            )
        except _UpstreamTimeoutError:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                504,
                "Upstream request timed out.",
                authorization_kind=authorization.authorization_kind,
                error_code="timeout",
            )
        except _UpstreamResponseTooLargeError:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                502,
                "Upstream response exceeded the configured byte limit.",
                authorization_kind=authorization.authorization_kind,
                error_code="oversized_response",
            )
        except _UpstreamUnsupportedEncodingError:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                502,
                "Upstream ignored the required identity content encoding.",
                authorization_kind=authorization.authorization_kind,
                error_code="unsupported_content",
            )
        except Exception:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                502,
                "Upstream request failed.",
                authorization_kind=authorization.authorization_kind,
                error_code="fetch_failed",
            )
        if ensure_authority is not None:
            try:
                ensure_authority()
            except VirtualCredentialError:
                return self._authority_revoked(request, authorization)
        if authorization.require_identity_encoding:
            content_encoding = _header_get(response.headers, "content-encoding")
            if content_encoding is not None and content_encoding.strip().lower() != "identity":
                return self._deny(
                    request,
                    authorization.grant_id,
                    authorization.policy_name,
                    502,
                    "Upstream ignored the required identity content encoding.",
                    authorization_kind=authorization.authorization_kind,
                    error_code="unsupported_content",
                )
            if len(response.body) > self._browser_max_response_bytes:
                return self._deny(
                    request,
                    authorization.grant_id,
                    authorization.policy_name,
                    502,
                    "Upstream response exceeded the browser byte limit.",
                    authorization_kind=authorization.authorization_kind,
                    error_code="oversized_response",
                )
        elif len(response.body) > effective_response_limit:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                502,
                "Upstream response exceeded the proxy byte limit.",
                authorization_kind=authorization.authorization_kind,
                error_code="oversized_response",
            )
        response_limit = (
            self._browser_max_response_bytes
            if authorization.require_identity_encoding
            else effective_response_limit
        )
        try:
            scrubbed_response = _scrub_response(
                response,
                secrets=authorization.secrets,
                max_body_bytes=response_limit,
            )
        except _UpstreamResponseTooLargeError:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                502,
                "Redacted upstream response exceeded the proxy byte limit.",
                authorization_kind=authorization.authorization_kind,
                error_code="oversized_response",
            )
        self._record(
            EgressDecision(
                allowed=True,
                status_code=response.status_code,
                destination=request.host,
                method=request.method,
                path=request.path,
                grant_id=authorization.grant_id,
                policy_name=authorization.policy_name,
                reason=None,
                authorization_kind=authorization.authorization_kind,
            )
        )
        return scrubbed_response

    async def _run_upstream_operation(
        self,
        request: CapturedRequest,
        *,
        max_response_bytes: int,
    ) -> CapturedResponse:
        token = self._reserve_upstream_operation()
        if token is None:
            raise _UpstreamCapacityError("Upstream operation capacity is exhausted.")
        release_owner = True
        caller_cancellation: asyncio.CancelledError | None = None
        cancellation_cause: RuntimeError | None = None
        child_stopped = False
        response: CapturedResponse | None = None
        try:
            operation = self._upstream.prepare(
                request,
                limits=EgressUpstreamLimits(
                    max_response_bytes=max_response_bytes,
                    total_timeout_s=DEFAULT_EGRESS_UPSTREAM_TOTAL_TIMEOUT_S,
                ),
            )
            if not isinstance(operation, EgressUpstreamOperation):
                raise TypeError("Egress upstream returned an invalid prepared operation.")
            result_task = asyncio.create_task(
                _owned_upstream_result(operation),
                name="cayu-egress-upstream-result",
            )
            owner = _ActiveUpstreamOperation(
                operation=operation,
                result_task=result_task,
            )
            self._active_upstream_operations[token] = owner
            try:
                response = await asyncio.shield(result_task)
            except _QuiescentUpstreamProcessControl as outcome:
                raise outcome.signal from None
            except asyncio.CancelledError as cancellation:
                current_task = asyncio.current_task()
                if current_task is None or current_task.cancelling() <= 0:
                    await self._settle_upstream_owner(token, owner)
                    child_stopped = True
                else:
                    try:
                        await self._settle_upstream_owner(
                            token,
                            owner,
                            cancellation=cancellation,
                        )
                    except BaseException as settlement_error:
                        release_owner = False
                        if _contains_process_control_signal(settlement_error):
                            raise settlement_error from cancellation
                        cancellation_cause = RuntimeError(
                            "Egress upstream cancellation settlement was incomplete."
                        )
                    caller_cancellation = cancellation
        finally:
            if release_owner:
                self._release_upstream_operation(token)
        if caller_cancellation is not None:
            if cancellation_cause is not None:
                raise caller_cancellation from cancellation_cause
            raise caller_cancellation
        if child_stopped:
            raise RuntimeError("Egress upstream operation stopped unexpectedly.")
        assert response is not None
        return response

    def _reserve_upstream_operation(self) -> object | None:
        if len(self._active_upstream_operations) >= self._max_active_upstream_operations:
            return None
        token = object()
        self._active_upstream_operations[token] = None
        self._upstream_operations_idle.clear()
        return token

    def _release_upstream_operation(self, token: object) -> None:
        self._active_upstream_operations.pop(token, None)
        if not self._active_upstream_operations:
            self._upstream_operations_idle.set()

    async def _settle_upstream_owner(
        self,
        token: object,
        owner: _ActiveUpstreamOperation,
        *,
        cancellation: asyncio.CancelledError | None = None,
    ) -> None:
        settlement_task = self._arm_upstream_settlement(owner)
        await self._await_upstream_settlement(
            token,
            owner,
            settlement_task,
            cancellation=cancellation,
        )

    @staticmethod
    def _arm_upstream_settlement(
        owner: _ActiveUpstreamOperation,
    ) -> asyncio.Task[None]:
        settlement_task = owner.settlement_task
        retry_settlement = settlement_task is None
        if settlement_task is not None and settlement_task.done():
            try:
                retry_settlement = settlement_task.exception() is not None
            except asyncio.CancelledError:
                retry_settlement = True
        if retry_settlement:

            async def settle() -> None:
                quiescent_process_control: _QuiescentUpstreamProcessControl | None = None
                try:
                    await _owned_upstream_settlement(owner.operation)
                except _QuiescentUpstreamProcessControl as error:
                    quiescent_process_control = error
                result_outcome = await await_shielded_task_outcome(owner.result_task)
                if quiescent_process_control is not None:
                    raise quiescent_process_control from None
                if isinstance(result_outcome.error, _QuiescentUpstreamProcessControl):
                    raise result_outcome.error
                if result_outcome.error is not None and _contains_process_control_signal(
                    result_outcome.error
                ):
                    raise _QuiescentUpstreamProcessControl(result_outcome.error) from None

            settlement_task = asyncio.create_task(
                settle(),
                name="cayu-egress-upstream-settlement",
            )
            owner.settlement_task = settlement_task
        assert settlement_task is not None
        return settlement_task

    async def _await_upstream_settlement(
        self,
        token: object,
        owner: _ActiveUpstreamOperation,
        settlement_task: asyncio.Task[None],
        *,
        cancellation: asyncio.CancelledError | None = None,
    ) -> None:
        if owner.settlement_task is not settlement_task:
            raise RuntimeError("Egress upstream settlement ownership changed.")
        outcome = await await_shielded_task_outcome(
            settlement_task,
            cancellation=cancellation,
        )
        if outcome.cancellation is not None:
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
        if outcome.error is not None:
            if isinstance(outcome.error, _QuiescentUpstreamProcessControl):
                self._release_upstream_operation(token)
                if cancellation is not None:
                    raise outcome.error.signal from cancellation
                raise outcome.error.signal
            if isinstance(outcome.error, _UnsettledUpstreamProcessControl):
                if cancellation is not None:
                    raise outcome.error.signal from cancellation
                raise outcome.error.signal
            if _contains_process_control_signal(outcome.error):
                if cancellation is not None:
                    raise outcome.error from cancellation
                raise outcome.error
            raise RuntimeError("Egress upstream operation did not settle.") from None
        if outcome.timed_out:
            raise RuntimeError("Egress upstream operation settlement timed out.")
        self._release_upstream_operation(token)

    def _authority_revoked(
        self,
        request: CapturedRequest,
        authorization: _ForwardingAuthorization,
    ) -> CapturedResponse:
        reason = (
            "Virtual credential is not valid."
            if authorization.authorization_kind == "virtual_credential"
            else "Credentialless egress authority has been revoked."
        )
        return self._deny(
            request,
            authorization.grant_id,
            authorization.policy_name,
            403,
            reason,
            authorization_kind=authorization.authorization_kind,
        )

    def _begin_credentialless_request(self) -> bool:
        if not self._credentialless_authority_active:
            return False
        self._credentialless_active_requests += 1
        self._credentialless_idle.clear()
        return True

    def _ensure_credentialless_authority(self) -> None:
        if not self._credentialless_authority_active:
            raise VirtualCredentialError("Credentialless egress authority has been revoked.")

    def _end_credentialless_request(self) -> None:
        self._credentialless_active_requests -= 1
        if self._credentialless_active_requests == 0:
            self._credentialless_idle.set()

    def _deny(
        self,
        request: CapturedRequest,
        grant_id: str | None,
        policy_name: str | None,
        status_code: int,
        reason: str,
        *,
        authorization_kind: Literal["virtual_credential", "credentialless"] = (
            "virtual_credential"
        ),
        error_code: str = "request_denied",
    ) -> CapturedResponse:
        self._record(
            EgressDecision(
                allowed=False,
                status_code=status_code,
                destination=request.host,
                method=request.method,
                path=request.path,
                grant_id=grant_id,
                policy_name=policy_name,
                reason=reason,
                authorization_kind=authorization_kind,
            )
        )
        body = json.dumps({"error": {"message": reason}}).encode()
        return CapturedResponse(
            status_code=status_code,
            headers={
                "Content-Type": "application/json",
                CAYU_EGRESS_ERROR_HEADER: error_code,
            },
            body=body,
        )

    def _record(self, decision: EgressDecision) -> None:
        if self._audit is None:
            return
        # Auditing is best-effort: a failing sink must never discard an
        # already-fetched provider response or turn a success into an error.
        with contextlib.suppress(Exception):
            self._audit(decision)


def _approved_destination_map(
    destinations: Sequence[ApprovedEgressDestination],
) -> dict[tuple[str, str, int], ApprovedEgressDestination]:
    return {
        destination.authority: destination
        for destination in validate_approved_destinations(destinations)
    }


def _header_get(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def _forwardable_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in _HOP_BY_HOP}


def _identity_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    # The response passed the identity-encoding gate above. Do not forward even
    # a redundant identity marker as authority for a downstream transformation.
    return {
        key: value
        for key, value in _forwardable_headers(headers).items()
        if key.lower() != "content-encoding"
    }


def _scrub_response(
    response: CapturedResponse,
    *,
    secrets: tuple[str, ...] = (),
    max_body_bytes: int,
) -> CapturedResponse:
    headers = _forwardable_headers(response.headers)
    if not secrets:
        return response.model_copy(update={"headers": headers})

    redactor = SecretRedactor(secrets)
    redacted_headers = {key: redactor.redact_text(value) for key, value in headers.items()}
    redacted_body = response.body
    replacement = REDACTED_SECRET.encode()
    for secret in secrets:
        encoded_secret = secret.encode()
        if len(replacement) > len(encoded_secret):
            projected_size = len(redacted_body) + redacted_body.count(encoded_secret) * (
                len(replacement) - len(encoded_secret)
            )
            if projected_size > max_body_bytes:
                raise _UpstreamResponseTooLargeError(
                    "Redaction would expand the response beyond its byte limit."
                )
        redacted_body = redacted_body.replace(encoded_secret, replacement)
    return response.model_copy(update={"headers": redacted_headers, "body": redacted_body})
