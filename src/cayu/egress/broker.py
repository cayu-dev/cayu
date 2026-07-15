from __future__ import annotations

import asyncio
import base64
import contextlib
import ipaddress
import json
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import require_clean_nonblank
from cayu.egress.credential_kinds import (
    supported_credential_kind_descriptor,
    uses_virtual_credential_namespace,
)
from cayu.egress.destinations import (
    ApprovedEgressDestination,
    EgressProtocol,
    validate_approved_destinations,
)
from cayu.egress.errors import VirtualCredentialError
from cayu.egress.grants import VirtualCredentialRegistry
from cayu.egress.policy import EgressPolicy, EgressRequest
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretResolver, validate_secret_resolver

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
    }
)


class CapturedRequest(BaseModel):
    """One outbound request captured outside the sandbox by the egress proxy."""

    model_config = ConfigDict(extra="forbid")

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
        return require_clean_nonblank(value, info.field_name).lower()

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


@runtime_checkable
class EgressUpstream(Protocol):
    """Forwards a fully-rewritten request to the real provider."""

    async def send(self, request: CapturedRequest) -> CapturedResponse: ...


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


class HttpxUpstream:
    """Default upstream that forwards requests to the provider over HTTPS."""

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        routes: Mapping[str, str] | None = None,
        destination_resolver: DestinationResolver | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._transport = transport
        self._routes = _validated_upstream_routes(routes or {})
        self._destination_resolver = destination_resolver or _resolve_destination

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        target = await self._target(request)
        headers = _forwardable_headers(request.headers)
        headers["Host"] = target.host_header
        extensions = (
            {"sni_hostname": target.sni_hostname} if target.sni_hostname is not None else None
        )
        async with httpx.AsyncClient(
            timeout=self._timeout_s,
            transport=self._transport,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            response = await client.request(
                request.method,
                target.url,
                headers=headers,
                content=request.body or None,
                extensions=extensions,
            )
        return CapturedResponse(
            status_code=response.status_code,
            headers=_decoded_response_headers(dict(response.headers)),
            body=response.content,
        )

    async def _target(self, request: CapturedRequest) -> _ResolvedUpstreamTarget:
        route = self._routes.get(request.host)
        origin = route or f"{request.protocol}://{request.host}:{request.port}"
        split = urlsplit(origin)
        host = split.hostname
        if host is None:  # pragma: no cover - constructors already validate origins
            raise ValueError("Upstream target has no hostname.")
        port = split.port or (443 if split.scheme == "https" else 80)
        addresses = tuple(await self._destination_resolver(host, port))
        if not addresses:
            raise ValueError("Upstream destination did not resolve to an address.")
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


async def _resolve_destination(host: str, port: int) -> tuple[str, ...]:
    records = await asyncio.get_running_loop().getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
    )
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


def _validated_resolved_address(address: str, *, allow_private: bool) -> str:
    resolved = ipaddress.ip_address(address)
    if (
        resolved.is_loopback
        or resolved.is_link_local
        or resolved.is_multicast
        or resolved.is_reserved
        or resolved.is_unspecified
        or (not allow_private and not resolved.is_global)
    ):
        raise ValueError("Upstream destination resolved to a prohibited address.")
    return resolved.compressed


def _format_authority(host: str, port: int, scheme: str) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    return rendered_host if port == default_port else f"{rendered_host}:{port}"


def _validated_upstream_routes(routes: Mapping[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for logical_host, target in routes.items():
        host = require_clean_nonblank(logical_host, "HttpxUpstream route host").lower()
        if any(character in host for character in "/:@") or any(
            character.isspace() for character in host
        ):
            raise ValueError("HttpxUpstream route host must be a bare hostname.")
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
    ) -> None:
        if resolver is not None:
            validate_secret_resolver(resolver, "resolver")
        self._registry = registry
        self._resolver = resolver
        self._policies = dict(policies)
        self._approved_destinations = _approved_destination_map(approved_destinations)
        self._credentialless_authority_active = True
        self._credentialless_active_requests = 0
        self._credentialless_idle = asyncio.Event()
        self._credentialless_idle.set()
        self._upstream = upstream or HttpxUpstream()
        self._audit = audit
        self._require_test_mode = require_test_mode_credentials

    @property
    def registry(self) -> VirtualCredentialRegistry:
        """The credential registry, for session-close revocation by adapters."""
        return self._registry

    @property
    def has_credentialless_destinations(self) -> bool:
        """Whether the broker requires an isolated, independently authenticated transport."""

        return bool(self._approved_destinations)

    async def handle_request(self, request: CapturedRequest) -> CapturedResponse:
        presented = _extract_presented_credential(request.headers)
        if presented is None:
            if self._approved_destinations:
                return await self._handle_credentialless(request)
            return self._deny(request, None, None, 401, "No credential presented to broker.")

        try:
            lease = self._registry.acquire(presented)
        except Exception:  # unknown / expired / revoked — never echo the value
            if self._credentialless_destination(
                request
            ) is not None and not uses_virtual_credential_namespace(presented):
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

            # Resolve + rewrite in one guarded step; a failure here (bad vault, etc.)
            # is reported distinctly from an upstream failure and never leaks a value.
            try:
                if self._resolver is None:
                    raise RuntimeError("No credential resolver is configured.")
                resolved = await self._resolver.resolve(
                    grant.secret, scope={"grant_id": grant.grant_id}
                )
                real_secret = resolved.value.get_secret_value()
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
        """Disable all routes, revoke virtual values, and drain active requests."""

        self._credentialless_authority_active = False
        virtual_drain = asyncio.create_task(self._registry.revoke_values_and_wait(presented_values))
        await asyncio.gather(virtual_drain, self._credentialless_idle.wait())
        return virtual_drain.result()

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
                )
            return await self._forward_authorized(
                request=request,
                upstream_request=request,
                authorization=_ForwardingAuthorization(
                    grant_id=None,
                    policy_name=policy.name,
                    authorization_kind="credentialless",
                ),
                ensure_authority=self._ensure_credentialless_authority,
            )
        finally:
            self._end_credentialless_request()

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
        try:
            response = await self._upstream.send(upstream_request)
        except Exception:
            return self._deny(
                request,
                authorization.grant_id,
                authorization.policy_name,
                502,
                "Upstream request failed.",
                authorization_kind=authorization.authorization_kind,
            )
        if ensure_authority is not None:
            try:
                ensure_authority()
            except VirtualCredentialError:
                return self._authority_revoked(request, authorization)
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
        return _scrub_response(response, secrets=authorization.secrets)

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
            headers={"Content-Type": "application/json"},
            body=body,
        )

    def _record(self, decision: EgressDecision) -> None:
        if self._audit is None:
            return
        # Auditing is best-effort: a failing sink must never discard an
        # already-fetched provider response or turn a success into an error.
        with contextlib.suppress(Exception):
            self._audit(decision)


def _extract_presented_credential(headers: Mapping[str, str]) -> str | None:
    value = _header_get(headers, "authorization")
    if value is None:
        return None
    value = value.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip() or None
    if value.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(value[6:].strip()).decode("utf-8", "replace")
        except (ValueError, UnicodeDecodeError):
            return None
        # Stripe-style basic auth carries the key as the username.
        return decoded.split(":", 1)[0] or None
    return value or None


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


def _decoded_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    # httpx returns decoded response bytes but keeps the provider's compression
    # metadata. Forwarding stale Content-Encoding makes SDK clients decode twice.
    return {
        key: value
        for key, value in _forwardable_headers(headers).items()
        if key.lower() != "content-encoding"
    }


def _scrub_response(
    response: CapturedResponse, *, secrets: tuple[str, ...] = ()
) -> CapturedResponse:
    headers = _forwardable_headers(response.headers)
    if not secrets:
        return response.model_copy(update={"headers": headers})

    redactor = SecretRedactor(secrets)
    redacted_headers = {key: redactor.redact_text(value) for key, value in headers.items()}
    redacted_body = response.body
    for secret in secrets:
        redacted_body = redacted_body.replace(secret.encode(), REDACTED_SECRET.encode())
    return response.model_copy(update={"headers": redacted_headers, "body": redacted_body})
