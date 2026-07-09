from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import require_clean_nonblank
from cayu.egress.credential_kinds import supported_credential_kind_descriptor
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
        return f"https://{self.host}{self.path}{suffix}"


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


@runtime_checkable
class EgressUpstream(Protocol):
    """Forwards a fully-rewritten request to the real provider."""

    async def send(self, request: CapturedRequest) -> CapturedResponse: ...


class HttpxUpstream:
    """Default upstream that forwards requests to the provider over HTTPS."""

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._transport = transport

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
            response = await client.request(
                request.method,
                request.url(),
                headers=_forwardable_headers(request.headers),
                content=request.body or None,
            )
        return CapturedResponse(
            status_code=response.status_code,
            headers=_decoded_response_headers(dict(response.headers)),
            body=response.content,
        )


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
        resolver: SecretResolver,
        policies: Mapping[str, EgressPolicy],
        upstream: EgressUpstream | None = None,
        audit: Callable[[EgressDecision], None] | None = None,
        require_test_mode_credentials: bool = True,
    ) -> None:
        validate_secret_resolver(resolver, "resolver")
        self._registry = registry
        self._resolver = resolver
        self._policies = dict(policies)
        self._upstream = upstream or HttpxUpstream()
        self._audit = audit
        self._require_test_mode = require_test_mode_credentials

    @property
    def registry(self) -> VirtualCredentialRegistry:
        """The credential registry, for session-close revocation by adapters."""
        return self._registry

    async def handle_request(self, request: CapturedRequest) -> CapturedResponse:
        presented = _extract_presented_credential(request.headers)
        if presented is None:
            return self._deny(request, None, None, 401, "No credential presented to broker.")

        try:
            lease = self._registry.acquire(presented)
        except Exception:  # unknown / expired / revoked — never echo the value
            return self._deny(request, None, None, 403, "Virtual credential is not valid.")

        try:
            grant = lease.grant

            if grant.destination != request.host:
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

            try:
                lease.ensure_active()
            except VirtualCredentialError:
                return self._deny(
                    request, grant.grant_id, policy.name, 403, "Virtual credential is not valid."
                )

            try:
                response = await self._upstream.send(rewritten)
            except Exception:
                # Sanitized failure: no secret in the message, body, or record.
                return self._deny(
                    request, grant.grant_id, policy.name, 502, "Upstream request failed."
                )

            try:
                lease.ensure_active()
            except VirtualCredentialError:
                return self._deny(
                    request, grant.grant_id, policy.name, 403, "Virtual credential is not valid."
                )

            self._record(
                EgressDecision(
                    allowed=True,
                    status_code=response.status_code,
                    destination=request.host,
                    method=request.method,
                    path=request.path,
                    grant_id=grant.grant_id,
                    policy_name=policy.name,
                    reason=None,
                )
            )
            return _scrub_response(response, secrets=(real_secret,))
        finally:
            lease.close()

    def _deny(
        self,
        request: CapturedRequest,
        grant_id: str | None,
        policy_name: str | None,
        status_code: int,
        reason: str,
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
