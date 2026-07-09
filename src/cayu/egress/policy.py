from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import require_clean_nonblank
from cayu.proxies import ProxyAuthorizationResult


class EgressRequest(BaseModel):
    """The policy's view of one captured outbound request.

    Deliberately excludes headers: a policy authorizes on destination, method,
    path, query, body metadata, and never needs to see the credential.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    host: str
    path: str
    query: str = ""
    body: bytes = b""
    content_type: str | None = None

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).upper()

    @field_validator("content_type")
    @classmethod
    def normalize_content_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.split(";", 1)[0].strip().lower() or None

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


class EgressPolicy(ABC):
    """Authorizes a captured request before any secret is resolved.

    ``authorize`` is pure and synchronous: it must reach a decision using only
    the request, so the broker can deny disallowed traffic *before* touching the
    vault. A denial always carries a reason.
    """

    #: Stable identifier recorded in audit events.
    name: str

    @abstractmethod
    def authorize(self, request: EgressRequest) -> ProxyAuthorizationResult:
        """Return whether the request may proceed to secret resolution."""


def _deny(reason: str, **metadata: Any) -> ProxyAuthorizationResult:
    return ProxyAuthorizationResult(allowed=False, reason=reason, metadata=metadata)


class HttpEgressPolicy(EgressPolicy):
    """Coarse HTTP egress policy for brokered credentials.

    This policy constrains credential use by host, method, and path. It does not
    infer provider-specific business semantics from request bodies or opaque
    provider object ids; applications and provider-scoped credentials remain
    responsible for business authorization.
    """

    def __init__(
        self,
        *,
        name: str,
        allowed_hosts: Iterable[str],
        allowed_endpoints: Iterable[tuple[str, str]],
        denied_prefixes: Iterable[str] = (),
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.allowed_hosts = _normalize_hosts(allowed_hosts)
        self.allowed_endpoints = _normalize_endpoints(allowed_endpoints)
        self.denied_prefixes = _normalize_prefixes(denied_prefixes)

    def authorize(self, request: EgressRequest) -> ProxyAuthorizationResult:
        if request.host not in self.allowed_hosts:
            return _deny(
                f"Destination {request.host!r} is not allowed by policy {self.name!r}.",
                policy=self.name,
            )

        for prefix in self.denied_prefixes:
            if _path_matches_prefix(request.path, prefix):
                return _deny(
                    f"Endpoint {request.path!r} is explicitly denied by policy {self.name!r}.",
                    policy=self.name,
                )

        if (request.method, request.path) not in self.allowed_endpoints:
            return _deny(
                f"{request.method} {request.path} is not in the allowlist "
                f"for policy {self.name!r}.",
                policy=self.name,
            )

        return ProxyAuthorizationResult(allowed=True, metadata={"policy": self.name})


def _normalize_hosts(hosts: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(require_clean_nonblank(host, "allowed_hosts").lower() for host in hosts)
    if not normalized:
        raise ValueError("HttpEgressPolicy requires at least one allowed host.")
    return normalized


def _normalize_endpoints(endpoints: Iterable[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, Sequence) or len(endpoint) != 2:
            raise TypeError("allowed_endpoints entries must be (method, path) pairs.")
        method = require_clean_nonblank(endpoint[0], "allowed endpoint method").upper()
        path = require_clean_nonblank(endpoint[1], "allowed endpoint path")
        if not path.startswith("/"):
            raise ValueError("allowed endpoint paths must start with '/'.")
        normalized.append((method, path))
    if not normalized:
        raise ValueError("HttpEgressPolicy requires at least one allowed endpoint.")
    return frozenset(normalized)


def _normalize_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for prefix in prefixes:
        value = require_clean_nonblank(prefix, "denied_prefixes")
        if not value.startswith("/"):
            raise ValueError("denied prefixes must start with '/'.")
        normalized.append(value.rstrip("/") or "/")
    return tuple(normalized)


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if prefix == "/":
        return path.startswith("/")
    return path == prefix or path.startswith(prefix + "/")
