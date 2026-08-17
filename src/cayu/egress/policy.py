from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import require_clean_nonblank
from cayu.egress.destinations import normalize_egress_hostname
from cayu.proxies import ProxyAuthorizationResult


class EgressRequest(BaseModel):
    """The policy's view of one captured outbound request.

    Deliberately excludes headers: a policy authorizes on destination, method,
    path, query, body metadata, and never needs to see the credential.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

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
        return normalize_egress_hostname(value, field_name=info.field_name)

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


class BrowserEgressPolicy(EgressPolicy):
    """Credentialless GET/HEAD policy for explicitly admitted web origins.

    Browser page loads request paths that are not practical to enumerate one by
    one (documents, scripts, stylesheets, images, and fonts). This policy keeps
    destination authority application-owned while allowing only read-only HTTP
    methods beneath configured path prefixes. It deliberately does not accept
    model-provided hosts, methods, headers, or policy overrides. The broker
    also requires identity content encoding for this policy so decoded browser
    input can be bounded before buffering.
    """

    def __init__(
        self,
        *,
        name: str,
        allowed_hosts: Iterable[str],
        allowed_path_prefixes: Iterable[str] = ("/",),
        denied_prefixes: Iterable[str] = (),
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.allowed_hosts = _normalize_hosts(
            allowed_hosts,
            policy_name="BrowserEgressPolicy",
        )
        self.allowed_path_prefixes = _normalize_browser_policy_prefixes(
            allowed_path_prefixes,
            field_name="allowed_path_prefixes",
            policy_name="BrowserEgressPolicy",
        )
        self.denied_prefixes = _normalize_browser_policy_prefixes(
            denied_prefixes,
            field_name="denied_prefixes",
            policy_name="BrowserEgressPolicy",
            allow_empty=True,
        )

    def authorize(self, request: EgressRequest) -> ProxyAuthorizationResult:
        if request.host not in self.allowed_hosts:
            return _deny(
                f"Destination {request.host!r} is not allowed by policy {self.name!r}.",
                policy=self.name,
            )
        if request.method not in {"GET", "HEAD"}:
            return _deny(
                f"Method {request.method!r} is not allowed by policy {self.name!r}.",
                policy=self.name,
            )
        if request.body:
            return _deny(
                f"Request bodies are not allowed by policy {self.name!r}.",
                policy=self.name,
            )
        try:
            policy_path = _require_unambiguous_browser_path(
                request.path,
                field_name="request path",
            )
        except ValueError:
            return _deny(
                "The request path has ambiguous or unsafe encoding.",
                policy=self.name,
            )
        if any(_path_matches_prefix(policy_path, prefix) for prefix in self.denied_prefixes):
            return _deny(
                f"Endpoint {policy_path!r} is explicitly denied by policy {self.name!r}.",
                policy=self.name,
            )
        if not any(
            _path_matches_prefix(policy_path, prefix) for prefix in self.allowed_path_prefixes
        ):
            return _deny(
                f"Endpoint {policy_path!r} is outside the allowed path prefixes "
                f"for policy {self.name!r}.",
                policy=self.name,
            )
        return ProxyAuthorizationResult(allowed=True, metadata={"policy": self.name})


def _normalize_hosts(
    hosts: Iterable[str],
    *,
    policy_name: str = "HttpEgressPolicy",
) -> frozenset[str]:
    normalized = frozenset(
        normalize_egress_hostname(host, field_name="allowed_hosts") for host in hosts
    )
    if not normalized:
        raise ValueError(f"{policy_name} requires at least one allowed host.")
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
    validated = _normalize_path_prefixes(
        prefixes,
        field_name="denied_prefixes",
        policy_name="HttpEgressPolicy",
        allow_empty=True,
    )
    return tuple(dict.fromkeys(value.rstrip("/") or "/" for value in validated))


def _normalize_path_prefixes(
    prefixes: Iterable[str],
    *,
    field_name: str,
    policy_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for prefix in prefixes:
        value = require_clean_nonblank(prefix, field_name)
        if not value.startswith("/"):
            raise ValueError(f"{field_name.replace('_', ' ')} must start with '/'.")
        normalized.append(value)
    if not normalized and not allow_empty:
        raise ValueError(f"{policy_name} requires at least one allowed path prefix.")
    return tuple(dict.fromkeys(normalized))


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if prefix == "/":
        return path.startswith("/")
    return path == prefix or path.startswith(prefix + "/")


def _normalize_browser_policy_prefixes(
    prefixes: Iterable[str],
    *,
    field_name: str,
    policy_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = _normalize_path_prefixes(
        prefixes,
        field_name=field_name,
        policy_name=policy_name,
        allow_empty=allow_empty,
    )
    canonical: list[str] = []
    for prefix in normalized:
        decoded = _require_unambiguous_browser_path(prefix, field_name=field_name)
        if decoded != prefix:
            raise ValueError(f"{field_name} entries must use decoded canonical paths.")
        canonical.append(decoded.rstrip("/") or "/")
    return tuple(dict.fromkeys(canonical))


def _require_unambiguous_browser_path(path: str, *, field_name: str) -> str:
    """Decode one path exactly once and reject alternate routing spellings."""

    try:
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} contains invalid Unicode.") from exc
    index = 0
    while index < len(path):
        if path[index] != "%":
            index += 1
            continue
        if index + 2 >= len(path):
            raise ValueError(f"{field_name} contains an incomplete percent escape.")
        try:
            int(path[index + 1 : index + 3], 16)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains an invalid percent escape.") from exc
        index += 3
    try:
        decoded = unquote(path, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field_name} contains invalid UTF-8 percent encoding.") from exc
    if (
        "%" in decoded
        or ";" in decoded
        or "\\" in decoded
        or "//" in decoded
        or any(ord(character) < 32 or ord(character) == 127 for character in decoded)
    ):
        raise ValueError(f"{field_name} contains ambiguous path characters.")
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        raise ValueError(f"{field_name} contains a dot path segment.")
    return decoded
