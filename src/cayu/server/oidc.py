"""OpenID Connect (OIDC) bearer-token authentication for Cayu servers.

``OidcBearerAuth`` verifies JWT access or ID tokens that a client already
obtained from an identity provider such as Amazon Cognito, Auth0, Okta,
Microsoft Entra ID, Google, or Workday. It is an ordinary server auth
dependency (request in, ``AuthContext`` out), so it works anywhere
``ServerConfig.protected()``, ``AuthenticatedAccess``, or
``[tool.cayu.serve].auth`` accept one. ``product_dependency()`` adapts the same
verifier for ``AuthenticatedProductAccess``.

Browser sign-in (authorization-code redirects, callback handling, refresh
tokens, and session cookies) is not provided. Clients send
``Authorization: Bearer <token>`` themselves, or an identity-aware proxy in front
of the server does.

Verification uses PyJWT from the optional ``cayu[oidc]`` extra. Signing keys
come from the issuer's JWKS, discovered from
``{issuer}/.well-known/openid-configuration`` unless ``jwks_url`` is given, and
are cached by ``OidcSigningKeys``, which can also be shared between verifiers.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import logging
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import Any, Final, Self
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request
from pydantic import ValidationError

from cayu._validation import copy_bounded_durable_json_value
from cayu.server.auth import (
    AuthContext,
    _quote_http_string,
    _require_basic_auth_realm,
)
from cayu.server.service import ProductPrincipal

logger = logging.getLogger(__name__)

SUPPORTED_OIDC_ALGORITHMS: Final[tuple[str, ...]] = (
    "RS256",
    "RS384",
    "RS512",
    "PS256",
    "PS384",
    "PS512",
    "ES256",
    "ES384",
    "ES512",
    "EdDSA",
)
"""Asymmetric JWS algorithms ``OidcBearerAuth`` accepts. ``none`` and HMAC never are."""

OIDC_ISSUER_VARIABLE: Final = "CAYU_OIDC_ISSUER"
OIDC_AUDIENCE_VARIABLE: Final = "CAYU_OIDC_AUDIENCE"

DEFAULT_MAX_TOKEN_BYTES: Final = 16 * 1024
_MAX_TOKEN_BYTES_LIMIT: Final = 64 * 1024
_MIN_TOKEN_BYTES_LIMIT: Final = 256
DEFAULT_LEEWAY_SECONDS: Final = 60
_MAX_LEEWAY_SECONDS: Final = 300
DEFAULT_JWKS_CACHE_SECONDS: Final = 600
_MIN_JWKS_CACHE_SECONDS: Final = 60
_MAX_JWKS_CACHE_SECONDS: Final = 24 * 60 * 60
DEFAULT_JWKS_REFETCH_INTERVAL_SECONDS: Final = 30
_MAX_JWKS_REFETCH_INTERVAL_SECONDS: Final = 3600
_FAILED_FETCH_RETRY_SECONDS: Final = 5.0
# While the issuer cannot be reached, the last authoritative key set stays
# usable for this long after it expires. An issuer that answers always replaces
# the cached set, so this only bridges outages.
DEFAULT_MAX_STALE_KEY_SECONDS: Final = 60 * 60
_MAX_STALE_KEY_SECONDS_LIMIT: Final = 24 * 60 * 60
DEFAULT_MAX_JWKS_KEYS: Final = 64
_MAX_JWKS_KEYS_LIMIT: Final = 1024
DEFAULT_HTTP_TIMEOUT_SECONDS: Final = 10.0
_MAX_HTTP_TIMEOUT_SECONDS: Final = 60.0
_MAX_DOCUMENT_BYTES: Final = 256 * 1024
_MAX_KEY_ID_CHARS: Final = 256
_MAX_CLAIM_NAME_CHARS: Final = 256
_MAX_CONTEXT_CLAIMS_BYTES: Final = 16 * 1024
_MAX_CONTEXT_CLAIMS_NODES: Final = 512
_MAX_CONTEXT_CLAIMS_NESTING: Final = 8
_MIN_RSA_KEY_BITS: Final = 2048

_COMPACT_JWS: Final = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", re.ASCII)
_SCOPE_TOKEN: Final = re.compile(r"[\x21\x23-\x5B\x5D-\x7E]+", re.ASCII)
_EC_CURVES: Final = {"ES256": "P-256", "ES384": "P-384", "ES512": "P-521"}
_OKP_CURVES: Final = frozenset({"Ed25519", "Ed448"})
_PRIVATE_JWK_MEMBERS: Final = frozenset({"d", "p", "q", "dp", "dq", "qi", "oth", "k"})
_DEFAULT_CONTEXT_CLAIMS: Final = (
    "iss",
    "sub",
    "aud",
    "azp",
    "client_id",
    "scope",
    "scp",
    "exp",
    "iat",
)

ClaimPath = str | Sequence[str]
"""A top-level claim name, or a sequence of names addressing a nested claim.

Dots are not path separators, so namespaced claims such as
``"https://example.com/tenant"`` or ``"custom:tenant_id"`` work as plain names.
"""

ClaimsMapper = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class OidcTokenError(Exception):
    """A bearer token was rejected, or its signing keys are unavailable.

    The message is a fixed category and never contains token contents.
    ``error`` is the RFC 6750 error code (``invalid_token`` or
    ``insufficient_scope``), or ``None`` when verification is temporarily
    unavailable (``status_code`` 503).
    """

    def __init__(
        self,
        reason: str,
        *,
        error: str | None = "invalid_token",
        status_code: int = 401,
        scopes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.error = error
        self.status_code = status_code
        self.scopes = scopes


class _FetchError(Exception):
    pass


_MISSING: Final = object()


def _require_pyjwt() -> Any:
    try:
        import jwt
        from jwt.algorithms import has_crypto
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] != "jwt":
            raise
        raise RuntimeError(
            'OIDC bearer authentication requires PyJWT. Install it with `pip install "cayu[oidc]"`.'
        ) from exc
    if not has_crypto:
        raise RuntimeError(
            "OIDC bearer authentication requires PyJWT with cryptography support. "
            'Install it with `pip install "cayu[oidc]"`.'
        )
    return jwt


@dataclass(slots=True)
class _KeySet:
    by_kid: dict[str, tuple[dict[str, Any], ...]]
    fetched_at: float
    expires_at: float
    resolved: dict[tuple[str, str], Any] = field(default_factory=dict)

    def key_for(self, kid: str, algorithm: str, jwt: Any) -> Any | None:
        cache_key = (kid, algorithm)
        if cache_key in self.resolved:
            return self.resolved[cache_key]
        resolved = None
        for entry in self.by_kid.get(kid, ()):
            declared = entry.get("alg")
            if declared is not None and declared != algorithm:
                continue
            if not _key_type_matches(entry, algorithm):
                continue
            try:
                candidate = jwt.PyJWK(entry, algorithm=algorithm).key
            except (jwt.PyJWTError, ValueError, TypeError, KeyError):
                continue
            if entry["kty"] == "RSA" and getattr(candidate, "key_size", 0) < _MIN_RSA_KEY_BITS:
                continue
            resolved = candidate
            break
        self.resolved[cache_key] = resolved
        return resolved


class OidcSigningKeys:
    """Discover, fetch, and cache one issuer's JWKS signing keys.

    Keys are fetched lazily with ``httpx`` under a timeout and cached for the
    JWKS response's ``Cache-Control: max-age`` (bounded to 60 seconds through
    24 hours; ``cache_seconds`` when absent). An unknown ``kid`` triggers at most
    one refetch per ``refetch_interval_seconds``, concurrent refreshes share one
    request, and a failed fetch is retried no more than every 5 seconds.

    Every authoritative JWKS response (HTTP 200 with a JSON ``keys`` array within
    the size and ``max_keys`` bounds) replaces the cached keys, even when it holds
    no usable signing keys, so a key the issuer withdraws stops verifying tokens
    at the next refresh, and such tokens get 401 ``invalid_token``. Only a failure
    to get an authoritative response (network error, timeout, non-200 status, or
    a malformed, compressed, or oversized document) keeps the previous keys, and
    then for at most ``max_stale_seconds`` (default one hour, 0 disables it)
    after they expire. A longer window rides out longer issuer outages; a shorter
    one limits how long keys stay trusted while the issuer cannot be reached.
    When no key set is usable, verification returns 503.

    Issuer and JWKS URLs must use HTTPS. ``allow_insecure_loopback=True`` also
    admits ``http://`` URLs on a loopback host, for tests and local development.
    Pass ``http_client`` to route fetches through your own ``httpx.AsyncClient``
    (for example an egress proxy); Cayu does not close it.
    """

    def __init__(
        self,
        issuer: str,
        *,
        jwks_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        allow_insecure_loopback: bool = False,
        cache_seconds: float = DEFAULT_JWKS_CACHE_SECONDS,
        refetch_interval_seconds: float = DEFAULT_JWKS_REFETCH_INTERVAL_SECONDS,
        http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        max_keys: int = DEFAULT_MAX_JWKS_KEYS,
        max_stale_seconds: float = DEFAULT_MAX_STALE_KEY_SECONDS,
    ) -> None:
        if type(allow_insecure_loopback) is not bool:
            raise ValueError("`allow_insecure_loopback` must be a boolean.")
        self.allow_insecure_loopback = allow_insecure_loopback
        self.issuer = _require_endpoint_url(
            issuer,
            "issuer",
            allow_insecure_loopback=allow_insecure_loopback,
            is_issuer=True,
        )
        self.jwks_url = (
            None
            if jwks_url is None
            else _require_endpoint_url(
                jwks_url,
                "jwks_url",
                allow_insecure_loopback=allow_insecure_loopback,
            )
        )
        if http_client is not None and not isinstance(http_client, httpx.AsyncClient):
            raise ValueError("`http_client` must be an httpx.AsyncClient.")
        self._http_client = http_client
        self.cache_seconds = _bounded_number(
            cache_seconds,
            "cache_seconds",
            minimum=_MIN_JWKS_CACHE_SECONDS,
            maximum=_MAX_JWKS_CACHE_SECONDS,
        )
        self.refetch_interval_seconds = _bounded_number(
            refetch_interval_seconds,
            "refetch_interval_seconds",
            minimum=1,
            maximum=_MAX_JWKS_REFETCH_INTERVAL_SECONDS,
        )
        self.http_timeout_seconds = _bounded_number(
            http_timeout_seconds,
            "http_timeout_seconds",
            minimum=0.1,
            maximum=_MAX_HTTP_TIMEOUT_SECONDS,
        )
        if type(max_keys) is not int or not 1 <= max_keys <= _MAX_JWKS_KEYS_LIMIT:
            raise ValueError(f"`max_keys` must be an integer from 1 to {_MAX_JWKS_KEYS_LIMIT}.")
        self.max_keys = max_keys
        self.max_stale_seconds = _bounded_number(
            max_stale_seconds,
            "max_stale_seconds",
            minimum=0,
            maximum=_MAX_STALE_KEY_SECONDS_LIMIT,
        )
        self._resolved_jwks_url = self.jwks_url
        self._key_set: _KeySet | None = None
        self._lock = asyncio.Lock()
        self._generation = 0
        self._last_attempt: float | None = None
        self._last_attempt_failed = False
        self._last_unknown_kid_fetch: float | None = None
        self._monotonic: Callable[[], float] = time.monotonic

    def __repr__(self) -> str:
        return f"OidcSigningKeys(issuer={self.issuer!r})"

    @property
    def discovery_url(self) -> str:
        return f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"

    async def signing_key(self, kid: str, algorithm: str) -> Any:
        """Return the verification key for ``kid`` and ``algorithm``.

        Raises ``OidcTokenError`` (401) when no matching key exists after any
        allowed refetch, including when the issuer's current JWKS has no usable
        keys at all, and ``OidcTokenError`` (503) when no authoritative key set
        could be fetched and no earlier one is within ``max_stale_seconds`` of
        its expiry.
        """

        jwt = _require_pyjwt()
        key_set = self._key_set
        if key_set is None or self._monotonic() >= key_set.expires_at:
            await self._refresh(self._generation, unknown_kid=False)
        elif kid not in key_set.by_kid:
            await self._refresh(self._generation, unknown_kid=True)
        key_set = self._key_set
        if key_set is None or self._monotonic() >= key_set.expires_at + self.max_stale_seconds:
            raise OidcTokenError(
                "signing keys are unavailable",
                error=None,
                status_code=503,
            )
        if kid not in key_set.by_kid:
            raise OidcTokenError("unknown signing key")
        key = key_set.key_for(kid, algorithm, jwt)
        if key is None:
            raise OidcTokenError("signing key does not match the token algorithm")
        return key

    async def _refresh(self, observed_generation: int, *, unknown_kid: bool) -> None:
        async with self._lock:
            if self._generation != observed_generation:
                # Another request refreshed (or tried to) while this one waited.
                return
            now = self._monotonic()
            if (
                self._last_attempt is not None
                and self._last_attempt_failed
                and now - self._last_attempt < _FAILED_FETCH_RETRY_SECONDS
            ):
                return
            if unknown_kid:
                if (
                    self._last_unknown_kid_fetch is not None
                    and now - self._last_unknown_kid_fetch < self.refetch_interval_seconds
                ):
                    return
                self._last_unknown_kid_fetch = now
            self._last_attempt = now
            try:
                key_set = await self._fetch(now)
            except Exception as exc:
                # Any failure backs off the same way, so an unexpected error cannot
                # turn every request into another fetch.
                self._last_attempt_failed = True
                self._generation += 1
                logger.warning(
                    "Could not refresh OIDC signing keys for issuer %s: %s",
                    self.issuer,
                    _fetch_failure(exc),
                )
                return
            # An authoritative response replaces the cached keys, even with none.
            self._last_attempt_failed = False
            self._key_set = key_set
            self._generation += 1
            if not key_set.by_kid:
                logger.warning(
                    "The JWKS for OIDC issuer %s has no usable signing keys; "
                    "bearer tokens are rejected until it publishes one.",
                    self.issuer,
                )

    async def _fetch(self, now: float) -> _KeySet:
        async with AsyncExitStack() as stack:
            client = self._http_client
            if client is None:
                client = await stack.enter_async_context(httpx.AsyncClient(follow_redirects=False))
            async with asyncio.timeout(self.http_timeout_seconds):
                jwks_url = self._resolved_jwks_url
                if jwks_url is None:
                    document, _ = await self._get_json(client, self.discovery_url)
                    if type(document) is not dict or document.get("issuer") != self.issuer:
                        raise _FetchError("discovery document issuer does not match")
                    try:
                        jwks_url = _require_endpoint_url(
                            document.get("jwks_uri"),
                            "jwks_uri",
                            allow_insecure_loopback=self.allow_insecure_loopback,
                        )
                    except ValueError as exc:
                        raise _FetchError("discovery document has no usable jwks_uri") from exc
                    self._resolved_jwks_url = jwks_url
                document, cache_control = await self._get_json(client, jwks_url)
        by_kid = _parse_jwks(document, max_keys=self.max_keys)
        ttl = _cache_ttl(cache_control, default=self.cache_seconds)
        return _KeySet(by_kid=by_kid, fetched_at=now, expires_at=now + ttl)

    async def _get_json(self, client: httpx.AsyncClient, url: str) -> tuple[Any, str | None]:
        timeout = httpx.Timeout(self.http_timeout_seconds)
        # Ask for an uncompressed body and read the raw stream, so the size cap
        # bounds what is held in memory rather than what was sent over the wire.
        async with client.stream(
            "GET",
            url,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            timeout=timeout,
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise _FetchError(f"HTTP {response.status_code} from {url}")
            encoding = response.headers.get("content-encoding", "").strip().lower()
            if encoding not in {"", "identity"}:
                raise _FetchError(f"response from {url} is compressed")
            declared = response.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > _MAX_DOCUMENT_BYTES:
                raise _FetchError(f"response from {url} is too large")
            # A response built in memory (for example by httpx.MockTransport) arrives
            # already read; with identity encoding its buffered content is the raw body.
            chunks = response.aiter_bytes() if response.is_stream_consumed else response.aiter_raw()
            body = bytearray()
            async for chunk in chunks:
                if len(chunk) > _MAX_DOCUMENT_BYTES - len(body):
                    raise _FetchError(f"response from {url} is too large")
                body.extend(chunk)
            cache_control = response.headers.get("cache-control")
        try:
            return json.loads(bytes(body)), cache_control
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise _FetchError(f"response from {url} is not JSON") from exc


class OidcBearerAuth:
    """Verify OIDC/JWT bearer tokens as a Cayu server auth dependency.

    Construct it once per process (it caches signing keys) and pass it wherever
    a server auth dependency is accepted::

        auth = OidcBearerAuth(
            issuer="https://example.okta.com/oauth2/default",
            audience="api://cayu",
        )
        config = ServerConfig.protected(auth)

    Each request must carry ``Authorization: Bearer <JWT>``. The token must be
    signed by a key in the issuer's JWKS with one of ``algorithms`` (asymmetric
    only), and must carry ``iss`` equal to ``issuer``, an audience in
    ``audience``, and an unexpired ``exp``. ``nbf`` and ``iat`` are checked when
    present. All time checks allow ``leeway_seconds`` of clock skew.

    ``subject_claim`` becomes ``AuthContext.subject``. ``tenant_claim``, when
    set, is required and becomes ``AuthContext.tenant`` (actor provenance only;
    it does not scope Cayu data). ``required_claims`` lists claim names that must
    be present, or maps claim names to the exact value they must have (a
    list-valued claim must contain it). ``required_scopes`` must all appear in
    the token's ``scope`` or ``scp`` claim, otherwise the request gets 403
    ``insufficient_scope``. ``audience_claim`` names the claim checked against
    ``audience``; use ``"client_id"`` for Amazon Cognito access tokens, which
    have no ``aud``.

    ``AuthContext.claims`` holds the token's standard ``iss``, ``sub``,
    ``aud``, ``azp``, ``client_id``, ``scope``, ``scp``, ``exp``, and ``iat``
    values. Pass ``claims_mapper`` to choose them yourself: it receives a copy of
    the verified claims and returns a JSON object of at most 16 KiB.

    Rejections return 401 with ``WWW-Authenticate: Bearer`` (and
    ``error="invalid_token"`` when a token was presented). Error bodies and logs
    never include the token. A token whose key the issuer's current JWKS no
    longer lists gets 401. When no signing keys can be fetched and no earlier key
    set is within its stale window, requests get 503 rather than 401.

    This verifies tokens that clients already hold. Browser sign-in flows
    (authorization-code redirects, sessions, refresh) are not provided; put an
    identity-aware proxy or your own login service in front for those.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str | Sequence[str],
        jwks_url: str | None = None,
        algorithms: Sequence[str] = SUPPORTED_OIDC_ALGORITHMS,
        leeway_seconds: float = DEFAULT_LEEWAY_SECONDS,
        subject_claim: ClaimPath = "sub",
        tenant_claim: ClaimPath | None = None,
        audience_claim: str = "aud",
        required_claims: Sequence[str] | Mapping[str, str | int | bool] = (),
        required_scopes: Sequence[str] = (),
        realm: str = "Cayu",
        claims_mapper: ClaimsMapper | None = None,
        max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
        signing_keys: OidcSigningKeys | None = None,
        http_client: httpx.AsyncClient | None = None,
        allow_insecure_loopback: bool = False,
    ) -> None:
        self._jwt = _require_pyjwt()
        if signing_keys is None:
            signing_keys = OidcSigningKeys(
                issuer,
                jwks_url=jwks_url,
                http_client=http_client,
                allow_insecure_loopback=allow_insecure_loopback,
            )
        else:
            if not isinstance(signing_keys, OidcSigningKeys):
                raise ValueError("`signing_keys` must be an OidcSigningKeys instance.")
            if jwks_url is not None or http_client is not None or allow_insecure_loopback:
                raise ValueError(
                    "Configure jwks_url, http_client, and allow_insecure_loopback on the "
                    "shared OidcSigningKeys instead of OidcBearerAuth."
                )
            if signing_keys.issuer != issuer:
                raise ValueError("`signing_keys` belongs to a different issuer.")
        self.signing_keys = signing_keys
        self.issuer = signing_keys.issuer
        self.audience = _require_audience(audience)
        self.algorithms = _require_algorithms(algorithms)
        self.leeway_seconds = _bounded_number(
            leeway_seconds,
            "leeway_seconds",
            minimum=0,
            maximum=_MAX_LEEWAY_SECONDS,
        )
        self.subject_claim = _require_claim_path(subject_claim, "subject_claim")
        self.tenant_claim = (
            None if tenant_claim is None else _require_claim_path(tenant_claim, "tenant_claim")
        )
        self.audience_claim = _require_claim_name(audience_claim, "audience_claim")
        self.required_claims = _require_required_claims(required_claims)
        self.required_scopes = _require_scopes(required_scopes)
        self.realm = _require_basic_auth_realm(realm)
        if claims_mapper is not None and not callable(claims_mapper):
            raise ValueError("`claims_mapper` must be callable.")
        self.claims_mapper = claims_mapper
        if (
            type(max_token_bytes) is not int
            or not _MIN_TOKEN_BYTES_LIMIT <= max_token_bytes <= _MAX_TOKEN_BYTES_LIMIT
        ):
            raise ValueError(
                f"`max_token_bytes` must be an integer from {_MIN_TOKEN_BYTES_LIMIT} "
                f"to {_MAX_TOKEN_BYTES_LIMIT}."
            )
        self.max_token_bytes = max_token_bytes

    @classmethod
    def from_environment(
        cls,
        issuer_variable: str = OIDC_ISSUER_VARIABLE,
        audience_variable: str = OIDC_AUDIENCE_VARIABLE,
        *,
        environ: Mapping[str, str] | None = None,
        **options: Any,
    ) -> Self:
        """Build OIDC bearer authentication from two environment variables.

        The issuer variable holds the exact issuer URL; the audience variable
        holds one audience or a comma-separated list. Both are read once, when
        this method is called. A variable that is unset, empty, or
        whitespace-only raises ``ValueError`` naming it (never its value); there
        is no fallback to open access. Other keyword arguments go to the
        constructor. ``environ`` replaces ``os.environ`` as the source.
        """

        names = (
            _require_variable_name(issuer_variable, "issuer_variable"),
            _require_variable_name(audience_variable, "audience_variable"),
        )
        if names[0] == names[1]:
            raise ValueError("issuer_variable and audience_variable must differ.")
        source: Mapping[str, str] = os.environ if environ is None else environ
        values = [source.get(name) for name in names]
        missing = [
            name
            for name, value in zip(names, values, strict=True)
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            verb = "is" if len(missing) == 1 else "are"
            raise ValueError(
                f"OIDC bearer authentication is not configured: {' and '.join(missing)} "
                f"{verb} unset or empty. Set {names[0]} to the issuer URL and {names[1]} "
                "to the expected audience before starting the server; it does not fall "
                "back to open access."
            )
        issuer = str(values[0]).strip()
        audiences = [part.strip() for part in str(values[1]).split(",") if part.strip()]
        if not audiences:
            raise ValueError(f"{names[1]} does not name an audience.")
        return cls(
            issuer=issuer,
            audience=audiences[0] if len(audiences) == 1 else audiences,
            **options,
        )

    def __repr__(self) -> str:
        return f"OidcBearerAuth(issuer={self.issuer!r}, audience={list(self.audience)!r})"

    async def __call__(self, request: Request) -> AuthContext:
        claims = await self.verify_request(request)
        try:
            return self.auth_context(claims)
        except OidcTokenError as exc:
            raise self._http_error(exc) from None

    async def verify_request(self, request: Request) -> dict[str, Any]:
        """Verify the request's bearer token and return its claims.

        Raises ``HTTPException`` (401, 403, or 503) with a ``WWW-Authenticate``
        challenge. Use it to build your own dependency on the same verifier.
        """

        token = self._bearer_token(request)
        try:
            return await self.verify_token(token)
        except OidcTokenError as exc:
            raise self._http_error(exc) from None

    async def verify_token(self, token: str) -> dict[str, Any]:
        """Verify one compact JWT and return its claims, or raise ``OidcTokenError``."""

        jwt = self._jwt
        if (
            type(token) is not str
            or len(token) > self.max_token_bytes
            or _COMPACT_JWS.fullmatch(token) is None
        ):
            raise self._reject("malformed token")
        header = _unverified_header(token)
        if header is None:
            raise self._reject("malformed token header")
        algorithm = header.get("alg")
        if type(algorithm) is not str or algorithm not in self.algorithms:
            raise self._reject("token algorithm is not allowed")
        if "crit" in header or "b64" in header:
            raise self._reject("token uses unsupported header extensions")
        kid = header.get("kid")
        if type(kid) is not str or not kid or len(kid) > _MAX_KEY_ID_CHARS:
            raise self._reject("token has no usable key id")
        try:
            key = await self.signing_keys.signing_key(kid, algorithm)
        except OidcTokenError as exc:
            logger.debug("Rejected an OIDC bearer token: %s.", exc.reason)
            raise
        verify_aud = self.audience_claim == "aud"
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=[algorithm],
                audience=list(self.audience) if verify_aud else None,
                issuer=self.issuer,
                leeway=self.leeway_seconds,
                options={
                    "require": ["exp", "iss"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": verify_aud,
                },
            )
        except jwt.ExpiredSignatureError:
            raise self._reject("token has expired") from None
        except jwt.ImmatureSignatureError:
            raise self._reject("token is not yet valid") from None
        except jwt.InvalidIssuerError:
            raise self._reject("token issuer does not match") from None
        except jwt.InvalidAudienceError:
            raise self._reject("token audience does not match") from None
        except jwt.InvalidSignatureError:
            raise self._reject("token signature is invalid") from None
        except jwt.MissingRequiredClaimError:
            raise self._reject("token is missing a required claim") from None
        except (jwt.PyJWTError, TypeError, ValueError, OverflowError, RecursionError):
            raise self._reject("token is invalid") from None
        if type(claims) is not dict:
            raise self._reject("token payload is not an object")
        for name in ("exp", "nbf", "iat"):
            if name in claims and not _is_numeric_date(claims[name]):
                raise self._reject("token time claim is not a number")
        if not verify_aud and not _claim_matches(claims.get(self.audience_claim), self.audience):
            raise self._reject("token audience does not match")
        for path, expected in self.required_claims:
            if not _claim_matches(_claim(claims, path), expected):
                raise self._reject("token is missing a required claim value")
        if not _valid_identity(_claim(claims, self.subject_claim)):
            raise self._reject("token subject claim is missing or invalid")
        if self.tenant_claim is not None and not _valid_identity(_claim(claims, self.tenant_claim)):
            raise self._reject("token tenant claim is missing or invalid")
        if self.required_scopes:
            granted = _token_scopes(claims)
            if not set(self.required_scopes) <= granted:
                logger.debug("Rejected an OIDC bearer token: insufficient scope.")
                raise OidcTokenError(
                    "token lacks a required scope",
                    error="insufficient_scope",
                    status_code=403,
                    scopes=self.required_scopes,
                )
        return claims

    def auth_context(self, claims: Mapping[str, Any]) -> AuthContext:
        """Build the ``AuthContext`` for already-verified ``claims``.

        Raises ``OidcTokenError`` when the identity claims cannot form an
        ``AuthContext`` (for example a subject longer than 512 characters).
        """

        subject = _claim(claims, self.subject_claim)
        tenant = None if self.tenant_claim is None else _claim(claims, self.tenant_claim)
        if self.claims_mapper is None:
            selected = {name: claims[name] for name in _DEFAULT_CONTEXT_CLAIMS if name in claims}
            try:
                context_claims = _bounded_claims(selected)
            except ValueError:
                raise self._reject("token claims are too large") from None
        else:
            mapped = self.claims_mapper(copy.deepcopy(dict(claims)))
            try:
                context_claims = _bounded_claims(dict(mapped))
            except (TypeError, ValueError):
                raise RuntimeError(
                    "OIDC claims_mapper must return a JSON object of at most 16 KiB."
                ) from None
        try:
            return AuthContext(subject=subject, tenant=tenant, claims=context_claims)
        except ValidationError:
            raise self._reject("token identity claims are invalid") from None

    def product_dependency(
        self,
        *,
        tenant_claim: ClaimPath | None = None,
        subject_claim: ClaimPath | None = None,
    ) -> Callable[[Request], Awaitable[ProductPrincipal]]:
        """Return a product auth dependency for ``AuthenticatedProductAccess``.

        The tenant and subject come only from the verified token: ``tenant_claim``
        (or the verifier's own ``tenant_claim``) and ``subject_claim`` (or the
        verifier's). A token without a usable tenant claim is rejected with 401;
        there is no default tenant. Both values must be non-blank strings.
        """

        tenant_path = (
            _require_claim_path(tenant_claim, "tenant_claim")
            if tenant_claim is not None
            else self.tenant_claim
        )
        if tenant_path is None:
            raise ValueError(
                "Product access needs a tenant claim. Pass tenant_claim= here or configure "
                "tenant_claim on OidcBearerAuth."
            )
        subject_path = (
            _require_claim_path(subject_claim, "subject_claim")
            if subject_claim is not None
            else self.subject_claim
        )

        async def authenticate(request: Request) -> ProductPrincipal:
            claims = await self.verify_request(request)
            tenant = _claim(claims, tenant_path)
            subject = _claim(claims, subject_path)
            if not _valid_identity(tenant) or not _valid_identity(subject):
                raise self._http_error(self._reject("token tenant or subject claim is missing"))
            try:
                return ProductPrincipal(tenant_id=tenant, subject_id=subject)
            except ValidationError:
                raise self._http_error(
                    self._reject("token tenant or subject claim is invalid")
                ) from None

        return authenticate

    def _bearer_token(self, request: Request) -> str:
        authorization = request.headers.get("authorization")
        if not authorization:
            raise self._challenge()
        # "Bearer " plus the largest token we would accept.
        if len(authorization) > self.max_token_bytes + 7:
            raise self._http_error(self._reject("authorization header is too large"))
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise self._challenge()
        return token

    def _reject(self, reason: str) -> OidcTokenError:
        logger.debug("Rejected an OIDC bearer token: %s.", reason)
        return OidcTokenError(reason)

    def _challenge(self) -> HTTPException:
        return HTTPException(
            status_code=401,
            detail="Bearer token required.",
            headers={"WWW-Authenticate": f'Bearer realm="{_quote_http_string(self.realm)}"'},
        )

    def _http_error(self, exc: OidcTokenError) -> HTTPException:
        realm = f'Bearer realm="{_quote_http_string(self.realm)}"'
        if exc.status_code == 503:
            return HTTPException(
                status_code=503,
                detail="Bearer token verification is temporarily unavailable.",
                headers={"Retry-After": str(int(_FAILED_FETCH_RETRY_SECONDS))},
            )
        if exc.error == "insufficient_scope":
            scope = " ".join(exc.scopes)
            return HTTPException(
                status_code=403,
                detail="The bearer token lacks a required scope.",
                headers={
                    "WWW-Authenticate": (
                        f'{realm}, error="insufficient_scope", scope="{_quote_http_string(scope)}"'
                    )
                },
            )
        return HTTPException(
            status_code=401,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": f'{realm}, error="invalid_token"'},
        )


def _bounded_claims(value: dict[str, Any]) -> dict[str, Any]:
    copied = copy_bounded_durable_json_value(
        value,
        "claims",
        max_bytes=_MAX_CONTEXT_CLAIMS_BYTES,
        max_nodes=_MAX_CONTEXT_CLAIMS_NODES,
        max_nesting=_MAX_CONTEXT_CLAIMS_NESTING,
        canonical_numbers=False,
    )
    if type(copied) is not dict:
        raise ValueError("claims must be an object.")
    return copied


def _unverified_header(token: str) -> dict[str, Any] | None:
    encoded = token.partition(".")[0]
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        header = json.loads(raw)
    except (binascii.Error, UnicodeDecodeError, ValueError, RecursionError):
        return None
    return header if type(header) is dict else None


def _claim(claims: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = claims
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _claim_matches(actual: Any, expected: Any) -> bool:
    if actual is _MISSING or actual is None:
        return False
    if expected is _MISSING:
        return True
    if isinstance(expected, tuple):
        values = actual if type(actual) is list else [actual]
        return any(type(value) is str and value in expected for value in values)
    if type(actual) is list:
        return any(type(item) is type(expected) and item == expected for item in actual)
    return type(actual) is type(expected) and actual == expected


def _valid_identity(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _is_numeric_date(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _token_scopes(claims: Mapping[str, Any]) -> set[str]:
    scopes: set[str] = set()
    for name in ("scope", "scp"):
        value = claims.get(name)
        if type(value) is str:
            scopes.update(value.split())
        elif type(value) is list:
            scopes.update(item for item in value if type(item) is str)
    return scopes


def _key_type_matches(entry: Mapping[str, Any], algorithm: str) -> bool:
    kty = entry.get("kty")
    if algorithm.startswith(("RS", "PS")):
        return kty == "RSA"
    if algorithm in _EC_CURVES:
        return kty == "EC" and entry.get("crv") == _EC_CURVES[algorithm]
    if algorithm == "EdDSA":
        return kty == "OKP" and entry.get("crv") in _OKP_CURVES
    return False


def _parse_jwks(document: Any, *, max_keys: int) -> dict[str, tuple[dict[str, Any], ...]]:
    if type(document) is not dict or type(document.get("keys")) is not list:
        raise _FetchError("JWKS document has no keys array")
    entries = document["keys"]
    if len(entries) > max_keys:
        raise _FetchError(f"JWKS document has more than {max_keys} keys")
    by_kid: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if type(entry) is not dict:
            continue
        kid = entry.get("kid")
        if type(kid) is not str or not kid or len(kid) > _MAX_KEY_ID_CHARS:
            continue
        if entry.get("kty") not in {"RSA", "EC", "OKP"}:
            continue
        use = entry.get("use")
        if use is not None and use != "sig":
            continue
        operations = entry.get("key_ops")
        if operations is not None and (type(operations) is not list or "verify" not in operations):
            continue
        if _PRIVATE_JWK_MEMBERS.intersection(entry):
            continue
        by_kid.setdefault(kid, []).append(dict(entry))
    # An empty result is still authoritative: the issuer trusts no signing keys.
    return {kid: tuple(keys) for kid, keys in by_kid.items()}


def _cache_ttl(cache_control: str | None, *, default: float) -> float:
    if not cache_control:
        return default
    directives = [directive.strip().lower() for directive in cache_control.split(",")]
    if any(directive in {"no-store", "no-cache"} for directive in directives):
        return _MIN_JWKS_CACHE_SECONDS
    for directive in directives:
        name, _, value = directive.partition("=")
        if name.strip() == "max-age" and value.strip().isdigit():
            return float(
                min(max(int(value.strip()), _MIN_JWKS_CACHE_SECONDS), _MAX_JWKS_CACHE_SECONDS)
            )
    return default


def _fetch_failure(exc: BaseException) -> str:
    if isinstance(exc, _FetchError):
        return str(exc)
    if isinstance(exc, TimeoutError | httpx.TimeoutException):
        return "timed out"
    return type(exc).__name__


def _is_loopback_hostname(host: str | None) -> bool:
    if not host:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _require_endpoint_url(
    value: Any,
    field_name: str,
    *,
    allow_insecure_loopback: bool,
    is_issuer: bool = False,
) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"`{field_name}` must be a non-blank URL.")
    if any(ord(character) < 0x21 or ord(character) == 0x7F for character in value):
        raise ValueError(f"`{field_name}` must not contain whitespace or control characters.")
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
    except ValueError:
        raise ValueError(f"`{field_name}` is not a valid URL.") from None
    if parts.username is not None or parts.password is not None or parts.fragment:
        raise ValueError(f"`{field_name}` must not contain credentials or a fragment.")
    if is_issuer and parts.query:
        raise ValueError(f"`{field_name}` must not contain a query.")
    if parts.scheme == "https" and hostname:
        return value
    if parts.scheme == "http" and allow_insecure_loopback and _is_loopback_hostname(hostname):
        return value
    raise ValueError(
        f"`{field_name}` must be an https:// URL. Plain http:// is accepted only for a "
        "loopback host with allow_insecure_loopback=True."
    )


def _require_audience(value: str | Sequence[str]) -> tuple[str, ...]:
    audiences = (value,) if isinstance(value, str) else tuple(value)
    if not audiences:
        raise ValueError("`audience` must name at least one audience.")
    for audience in audiences:
        if type(audience) is not str or not audience.strip() or audience != audience.strip():
            raise ValueError("`audience` values must be non-blank strings.")
    return audiences


def _require_algorithms(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    algorithms = tuple(value)
    if not algorithms:
        raise ValueError("`algorithms` must name at least one algorithm.")
    for algorithm in algorithms:
        if algorithm not in SUPPORTED_OIDC_ALGORITHMS:
            raise ValueError(
                "`algorithms` accepts only asymmetric JWS algorithms: "
                f"{', '.join(SUPPORTED_OIDC_ALGORITHMS)}. 'none' and HMAC (HS*) "
                "algorithms are never accepted."
            )
    return tuple(dict.fromkeys(algorithms))


def _require_claim_name(value: Any, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > _MAX_CLAIM_NAME_CHARS
    ):
        raise ValueError(f"`{field_name}` must be a non-blank claim name.")
    return value


def _require_claim_path(value: ClaimPath, field_name: str) -> tuple[str, ...]:
    path = (value,) if isinstance(value, str) else tuple(value)
    if not path or len(path) > _MAX_CONTEXT_CLAIMS_NESTING:
        raise ValueError(f"`{field_name}` must be a claim name or a short sequence of names.")
    return tuple(_require_claim_name(part, field_name) for part in path)


def _require_required_claims(
    value: Sequence[str] | Mapping[str, str | int | bool],
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    if isinstance(value, str):
        raise ValueError("`required_claims` must be a sequence of claim names or a mapping.")
    if isinstance(value, Mapping):
        required = []
        for name, expected in value.items():
            if type(expected) not in (str, int, bool):
                raise ValueError("`required_claims` values must be strings, integers, or booleans.")
            required.append(((_require_claim_name(name, "required_claims"),), expected))
        return tuple(required)
    return tuple(((_require_claim_name(name, "required_claims"),), _MISSING) for name in value)


def _require_scopes(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError("`required_scopes` must be a sequence of scope names.")
    scopes = tuple(value)
    for scope in scopes:
        if type(scope) is not str or _SCOPE_TOKEN.fullmatch(scope) is None:
            raise ValueError("`required_scopes` values must be RFC 6749 scope tokens.")
    return tuple(dict.fromkeys(scopes))


def _require_variable_name(value: str, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "=" in value:
        raise ValueError(f"{field_name} must be a non-empty environment variable name.")
    return value


def _bounded_number(value: Any, field_name: str, *, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"`{field_name}` must be a finite number.")
    if not minimum <= value <= maximum:
        raise ValueError(f"`{field_name}` must be between {minimum} and {maximum}.")
    return float(value)
