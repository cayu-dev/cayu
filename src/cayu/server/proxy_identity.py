"""Trust a signed identity that a sign-in proxy adds to each request.

A sign-in proxy in front of the server (an identity-aware router or gateway)
signs people in itself, then forwards each of their requests with one header,
``X-Cayu-Identity`` by default, holding a short-lived JWT about the person.
``ProxyIdentityAuth`` verifies that token against the proxy's published keys
and returns an ``AuthContext``: ``subject`` from ``sub``, ``tenant`` from
``org``, and the person's ``role``, ``name`` and ``email`` in ``claims``. The
wire format is the ``cayu.proxy-identity`` v1 contract in
``docs/runtime-contracts.md``.

When the header is present it must verify, or the request gets 401: an invalid
identity never falls through to another kind of authentication. When it is
absent, the ``fallback`` dependency decides, so machine callers and an
operator login keep working beside the proxy.

On the control plane, ``ControlPlaneRoles`` decides which routes each role may
use. ``cayu.server.environment_auth`` builds a ``ProxyIdentityAuth`` with the
operator login as its fallback when the ``CAYU_PROXY_IDENTITY_*`` variables are
set. Verification uses PyJWT and ``OidcSigningKeys`` from ``cayu.server.oidc``.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Self
from urllib.parse import urlsplit

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import (
    HTTPConnection,  # noqa: TC002 - Auth annotations are inspected at runtime.
)

from cayu.server._browser_client import is_browser_client_endpoint
from cayu.server._decision_only import has_decision_overrides
from cayu.server.auth import (
    AuthConfigurationError,
    AuthContext,
    AuthDependency,
    resolve_auth_context,
)
from cayu.server.oidc import (
    DEFAULT_MAX_TOKEN_BYTES,
    OidcBearerAuth,
    OidcSigningKeys,
    OidcTokenError,
    _bounded_claims,
    _bounded_number,
    _is_numeric_date,
    _require_endpoint_url,
    _unverified_header,
)
from cayu.server.service import ProductPrincipal
from cayu.server.static import DashboardStaticFiles

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_LIFETIME_SECONDS",
    "DEFAULT_PROXY_IDENTITY_HEADER",
    "PROXY_IDENTITY_AUDIENCE_VARIABLE",
    "PROXY_IDENTITY_HEADER_VARIABLE",
    "PROXY_IDENTITY_ISSUER_VARIABLE",
    "PROXY_IDENTITY_JWKS_URL_VARIABLE",
    "PROXY_IDENTITY_ROLES",
    "PROXY_IDENTITY_TOKEN_TYPE",
    "ControlPlaneRoles",
    "ProxyIdentityAuth",
    "proxy_identity_configured",
]

PROXY_IDENTITY_ISSUER_VARIABLE: Final = "CAYU_PROXY_IDENTITY_ISSUER"
PROXY_IDENTITY_AUDIENCE_VARIABLE: Final = "CAYU_PROXY_IDENTITY_AUDIENCE"
PROXY_IDENTITY_JWKS_URL_VARIABLE: Final = "CAYU_PROXY_IDENTITY_JWKS_URL"
PROXY_IDENTITY_HEADER_VARIABLE: Final = "CAYU_PROXY_IDENTITY_HEADER"
_REQUIRED_VARIABLES: Final = (
    PROXY_IDENTITY_ISSUER_VARIABLE,
    PROXY_IDENTITY_AUDIENCE_VARIABLE,
    PROXY_IDENTITY_JWKS_URL_VARIABLE,
)

DEFAULT_PROXY_IDENTITY_HEADER: Final = "X-Cayu-Identity"
#: The JWS header ``typ`` a proxy identity carries, so no other token from the
#: same keys (a sign-in code, an API token) is accepted in its place.
PROXY_IDENTITY_TOKEN_TYPE: Final = "cayu-proxy-identity+jwt"
PROXY_IDENTITY_ROLES: Final = ("admin", "developer", "member")
DEFAULT_PROXY_IDENTITY_ALGORITHMS: Final = ("ES256",)
#: The longest a proxy identity may live, from ``iat`` to ``exp``.
DEFAULT_MAX_LIFETIME_SECONDS: Final = 120
_MAX_LIFETIME_SECONDS_LIMIT: Final = 3600
DEFAULT_PROXY_IDENTITY_LEEWAY_SECONDS: Final = 30
_MAX_JTI_CHARS: Final = 256
_PROXY_IDENTITY_AUTH: Final = "proxy-identity"
_CONTEXT_CLAIMS: Final = (
    "iss",
    "aud",
    "sub",
    "org",
    "name",
    "email",
    "origin",
    "iat",
    "exp",
    "jti",
)
_OPTIONAL_TEXT_CLAIMS: Final = ("name", "email")
_HEADER_TOKEN: Final = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", re.ASCII)
#: Headers a proxy identity can't be read from: they carry other credentials or
#: describe the request itself.
_RESERVED_HEADERS: Final = frozenset(
    {"authorization", "cookie", "host", "content-length", "content-type", "proxy-authorization"}
)
_SAFE_METHODS: Final = frozenset({"GET", "HEAD", "OPTIONS"})
#: ``Sec-Fetch-Site`` values a browser sends for a request the page's own origin
#: made, or that a person made by hand (a typed URL, a bookmark).
_SAME_ORIGIN_FETCH_SITES: Final = frozenset({"same-origin", "none"})

#: Control-plane routes the read-and-decide roles may use, relative to the API
#: path and matched against the whole route template: reads and queries, and
#: the human decisions (approve, deny and answer approvals and questions, settle
#: inbox items and knowledge proposals). Every entry names a route
#: ``create_server`` registers; tests check that none is left without one.
_READ_AND_DECIDE_ROUTES: Final = frozenset(
    {
        ("GET", route)
        for route in (
            "/app",
            "/agents",
            "/tools/usage",
            "/tools/calls",
            "/agents/{agent_name}",
            "/artifacts",
            "/artifacts/{artifact_id}",
            "/artifacts/{artifact_id}/content",
            "/causal-budgets/{causal_budget_id}/usage",
            "/contract",
            "/evals/suite-views",
            "/evals/suite-views/{suite_key}",
            "/evals/run-views/{run_key}",
            "/environments",
            "/environments/{environment_name}",
            "/health",
            "/human-attention/{attention_id}",
            "/inbox",
            "/inbox/counts",
            "/inbox/{item_id}",
            "/knowledge",
            "/knowledge/pending",
            "/knowledge/pending/{entry_id}",
            "/pending-actions",
            "/sessions",
            "/sessions/overview/counts",
            "/sessions/{session_id}",
            "/sessions/{session_id}/events",
            "/sessions/{session_id}/trace",
            "/sessions/{session_id}/events/stream",
            "/sessions/{session_id}/human-review",
            "/sessions/{session_id}/interactions",
            "/sessions/{session_id}/interactions/{interaction_id}",
            "/sessions/{session_id}/messages",
            "/sessions/{session_id}/state",
            "/sessions/{session_id}/summary",
            "/sessions/{session_id}/task-replay",
            "/sessions/{session_id}/tool-view",
            "/sessions/{session_id}/transcript",
            "/sessions/{session_id}/usage",
            "/task-types",
            "/tasks",
            "/tasks/{task_id}",
            "/tasks/{task_id}/replay",
            "/tasks/{task_id}/schedule/events",
            "/tasks/{task_id}/dependents",
        )
    }
    | {
        ("POST", route)
        for route in (
            # Decisions.
            "/inbox/{item_id}/settle",
            "/knowledge/{entry_id}/approve",
            "/knowledge/{entry_id}/reject",
            "/tool-approvals/resolve",
            "/user-input/resolve",
            # Reads that take their query as a body.
            "/causal-budgets/{causal_budget_id}/cost",
            "/causal-budgets/{causal_budget_id}/summary",
            "/operations/snapshot",
            "/sessions/summary",
            "/sessions/{session_id}/cost",
            "/sessions/{session_id}/topology",
            "/supervision-events/query",
            "/usage/rollup",
        )
    }
)


def proxy_identity_configured(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the ``CAYU_PROXY_IDENTITY_*`` variables are set.

    ``True`` when the issuer, audience and JWKS URL are all set, ``False`` when
    none of them (nor the header name) is. Any other combination raises
    ``AuthConfigurationError`` naming the missing variables, so a half-configured
    server fails at startup instead of quietly ignoring the proxy.
    """

    source: Mapping[str, str] = os.environ if environ is None else environ
    present = [name for name in _REQUIRED_VARIABLES if _value(source, name)]
    if not present:
        if _value(source, PROXY_IDENTITY_HEADER_VARIABLE):
            raise AuthConfigurationError(
                f"{PROXY_IDENTITY_HEADER_VARIABLE} is set but proxy identity is not "
                f"configured: set {', '.join(_REQUIRED_VARIABLES)} as well."
            )
        return False
    missing = [name for name in _REQUIRED_VARIABLES if name not in present]
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        raise AuthConfigurationError(
            f"Proxy identity is partly configured: {' and '.join(missing)} {verb} unset "
            f"or empty. Set all of {', '.join(_REQUIRED_VARIABLES)}, or none of them."
        )
    return True


class ControlPlaneRoles:
    """Which control-plane routes each role may use.

    Roles in ``full_access`` use every route. Roles in ``read_and_decide`` may
    use the read and query routes, the human decisions (approve, deny and
    answer approvals and questions, settle inbox items and knowledge proposals),
    and the Developer dashboard and browser client served beside the API; every
    other control-plane route answers 403. Any other role, or a caller whose
    ``AuthContext.claims`` has no ``role_claim``, gets 403 everywhere.

    Routes are matched on their template (``request.scope["route"].path``)
    relative to the API path, so ``/sessions/{session_id}`` is one entry for
    every session. ``create_server`` and ``mount_cayu`` call ``add_api_path``
    on the rule an auth dependency carries as ``control_plane_roles``; a host
    that includes ``create_router`` itself calls it too. Until an API path is
    known, read-and-decide roles are refused everywhere. WebSocket routes are
    full-access only.
    """

    def __init__(
        self,
        *,
        full_access: Sequence[str] = ("admin", "developer"),
        read_and_decide: Sequence[str] = ("member",),
        role_claim: str = "role",
    ) -> None:
        self.full_access = _require_roles(full_access, "full_access", allow_empty=False)
        self.read_and_decide = _require_roles(read_and_decide, "read_and_decide", allow_empty=True)
        if set(self.full_access) & set(self.read_and_decide):
            raise ValueError("`full_access` and `read_and_decide` must not share a role.")
        if type(role_claim) is not str or not role_claim or role_claim != role_claim.strip():
            raise ValueError("`role_claim` must be a non-blank claim name.")
        self.role_claim = role_claim
        self._api_paths: set[str] = set()

    def __repr__(self) -> str:
        return (
            f"ControlPlaneRoles(full_access={list(self.full_access)!r}, "
            f"read_and_decide={list(self.read_and_decide)!r})"
        )

    @property
    def api_paths(self) -> frozenset[str]:
        return frozenset(self._api_paths)

    def add_api_path(self, api_path: str) -> None:
        """Apply the route rule to the control-plane API served at ``api_path``."""

        if type(api_path) is not str or not api_path.startswith("/") or api_path == "/":
            raise ValueError("`api_path` must be an absolute URL path other than `/`.")
        self._api_paths.add(api_path.rstrip("/"))

    def permits(self, request: HTTPConnection, context: AuthContext) -> bool:
        """Whether ``context``'s role may use the route ``request`` matched."""

        role = context.claims.get(self.role_claim)
        if type(role) is not str:
            return False
        if role in self.full_access:
            return True
        return role in self.read_and_decide and self._read_and_decide_may(request)

    def require(self, request: HTTPConnection, context: AuthContext) -> None:
        """Raise 403 ``role_forbidden`` unless ``permits(request, context)``."""

        if not self.permits(request, context):
            raise _role_forbidden()

    def _read_and_decide_may(self, request: HTTPConnection) -> bool:
        scope = request.scope
        if scope.get("type") != "http" or not self._api_paths:
            return False
        method = scope.get("method")
        check = "GET" if method in {"GET", "HEAD"} else method
        template = getattr(scope.get("route"), "path", None)
        endpoint = scope.get("endpoint")
        if isinstance(endpoint, DashboardStaticFiles):
            # Static mounts retain the outer prefix in path/root_path. Their
            # configured base is local to the server that registered this API.
            template = endpoint._base_path
        if not isinstance(template, str):
            return False
        relative = self._api_relative(template)
        if relative is None:
            if isinstance(endpoint, DashboardStaticFiles):
                return check == "GET"
            browser_paths = {
                f"{api_path.rpartition('/')[0]}/{name}"
                for api_path in self._api_paths
                if api_path.rpartition("/")[2] == "api"
                for name in ("client.js", "client.d.ts")
            }
            return (
                check == "GET"
                and template in browser_paths
                and is_browser_client_endpoint(endpoint)
            )
        return (check, relative) in _READ_AND_DECIDE_ROUTES

    def _api_relative(self, path: str) -> str | None:
        for api_path in sorted(self._api_paths, key=len, reverse=True):
            if path == api_path:
                return ""
            if path.startswith(api_path + "/"):
                return path[len(api_path) :]
        return None


class ProxyIdentityAuth:
    """Server auth dependency trusting a sign-in proxy's signed identity header.

    Construct it once per process (it caches the proxy's keys) and use it
    wherever a server auth dependency is accepted::

        auth = ProxyIdentityAuth(
            issuer="https://sign-in.example.com",
            audience="my-agent",
            jwks_url="https://sign-in.example.com/jwks",
            fallback=BasicAuth.from_environment(),
        )
        config = ServerConfig.protected(auth)

    A request carrying ``header`` must hold exactly one compact JWT signed with
    one of ``algorithms`` (ES256 by default) by a key in the JWKS at
    ``jwks_url``, named by its ``kid``, with JWS header ``typ`` equal to
    ``token_type``. Its claims must carry ``iss`` equal to ``issuer``, an
    audience in ``audience``, ``sub``, ``org``, ``iat``, an unexpired ``exp`` at
    most ``max_lifetime_seconds`` after ``iat``, a ``jti``, and a
    ``role_claim`` in ``roles``; ``nbf`` is checked when present, and ``name``
    and ``email`` must be strings when present. Time checks allow
    ``leeway_seconds`` of clock skew. Anything else gets 401, whatever
    ``fallback`` would have said, and 503 while no signing keys can be fetched.

    A request without ``header`` goes to ``fallback``, an ordinary auth
    dependency such as ``BasicAuth``; without one it gets 401. The fallback's
    ``AuthContext`` gets ``role`` ``fallback_role`` (``admin`` by default; pass
    ``None`` to keep the fallback's own claim).

    The ``AuthContext`` for a proxy identity has ``subject`` = ``sub``,
    ``tenant`` = ``org`` (actor provenance only; it does not scope Cayu data),
    and ``claims`` with ``role``, ``name`` and ``email`` when sent, ``iss``,
    ``aud``, ``iat``, ``exp``, ``jti``, and ``auth`` = ``"proxy-identity"``.

    Called as a dependency, it also applies ``control_plane_roles`` (by default
    ``ControlPlaneRoles()``: ``admin`` and ``developer`` use every route,
    ``member`` reads and decides). ``authenticate`` returns who is calling
    without a route rule, and ``product_dependency`` adapts it for
    ``AuthenticatedProductAccess``.

    A browser write (any method but ``GET``, ``HEAD`` and ``OPTIONS``, and every
    WebSocket) carrying a proxy identity is refused with 403 when its
    ``Sec-Fetch-Site``, ``Origin`` or fallback ``Referer`` identifies another origin,
    because the proxy adds the identity to every request the person's browser sends. Pass
    ``same_origin_writes=False`` when the proxy enforces that itself.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str | Sequence[str],
        jwks_url: str | None = None,
        header: str = DEFAULT_PROXY_IDENTITY_HEADER,
        token_type: str = PROXY_IDENTITY_TOKEN_TYPE,
        algorithms: Sequence[str] = DEFAULT_PROXY_IDENTITY_ALGORITHMS,
        role_claim: str = "role",
        roles: Sequence[str] = PROXY_IDENTITY_ROLES,
        max_lifetime_seconds: float = DEFAULT_MAX_LIFETIME_SECONDS,
        leeway_seconds: float = DEFAULT_PROXY_IDENTITY_LEEWAY_SECONDS,
        fallback: AuthDependency | None = None,
        fallback_role: str | None = "admin",
        control_plane_roles: ControlPlaneRoles | None = None,
        same_origin_writes: bool = True,
        max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
        signing_keys: OidcSigningKeys | None = None,
        http_client: httpx.AsyncClient | None = None,
        allow_insecure_loopback: bool = False,
    ) -> None:
        _require_pyjwt_for_proxy_identity()
        if signing_keys is None:
            if jwks_url is None:
                raise ValueError("Pass `jwks_url`, or `signing_keys` that name it.")
            signing_keys = OidcSigningKeys(
                issuer,
                jwks_url=jwks_url,
                http_client=http_client,
                allow_insecure_loopback=allow_insecure_loopback,
            )
        elif not isinstance(signing_keys, OidcSigningKeys) or signing_keys.jwks_url is None:
            raise ValueError("`signing_keys` must be an OidcSigningKeys with a `jwks_url`.")
        elif jwks_url is not None or http_client is not None or allow_insecure_loopback:
            raise ValueError(
                "Configure jwks_url, http_client, and allow_insecure_loopback on the "
                "shared OidcSigningKeys instead of ProxyIdentityAuth."
            )
        self._verifier = OidcBearerAuth(
            issuer=issuer,
            audience=audience,
            algorithms=algorithms,
            leeway_seconds=leeway_seconds,
            tenant_claim="org",
            max_token_bytes=max_token_bytes,
            signing_keys=signing_keys,
        )
        self.signing_keys = signing_keys
        self.issuer = self._verifier.issuer
        self.audience = self._verifier.audience
        self.algorithms = self._verifier.algorithms
        self.header = _require_header_name(header)
        if type(token_type) is not str or not token_type or token_type != token_type.strip():
            raise ValueError("`token_type` must be a non-blank string.")
        self.token_type = token_type
        if type(role_claim) is not str or not role_claim or role_claim != role_claim.strip():
            raise ValueError("`role_claim` must be a non-blank claim name.")
        self.role_claim = role_claim
        self.roles = _require_roles(roles, "roles", allow_empty=False)
        self.max_lifetime_seconds = _bounded_number(
            max_lifetime_seconds,
            "max_lifetime_seconds",
            minimum=1,
            maximum=_MAX_LIFETIME_SECONDS_LIMIT,
        )
        if fallback is not None and not callable(fallback):
            raise ValueError("`fallback` must be a server auth dependency.")
        self.fallback = fallback
        if fallback_role is not None and (
            type(fallback_role) is not str or not fallback_role.strip()
        ):
            raise ValueError("`fallback_role` must be a non-blank string or None.")
        self.fallback_role = fallback_role
        if control_plane_roles is None:
            control_plane_roles = ControlPlaneRoles()
        elif not isinstance(control_plane_roles, ControlPlaneRoles):
            raise ValueError("`control_plane_roles` must be a ControlPlaneRoles.")
        self.control_plane_roles = control_plane_roles
        if type(same_origin_writes) is not bool:
            raise ValueError("`same_origin_writes` must be a boolean.")
        self.same_origin_writes = same_origin_writes

    @classmethod
    def from_environment(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        fallback: AuthDependency | None = None,
        **options: Any,
    ) -> Self:
        """Build proxy identity from the ``CAYU_PROXY_IDENTITY_*`` variables.

        ``CAYU_PROXY_IDENTITY_ISSUER``, ``CAYU_PROXY_IDENTITY_AUDIENCE`` and
        ``CAYU_PROXY_IDENTITY_JWKS_URL`` are required; the optional
        ``CAYU_PROXY_IDENTITY_HEADER`` names the header (``X-Cayu-Identity`` by
        default). They are read once, when this method is called. A variable
        that is missing or holds an unusable value raises
        ``AuthConfigurationError`` naming it, never its value. Other keyword
        arguments go to the constructor; ``environ`` replaces ``os.environ``.
        """

        source: Mapping[str, str] = os.environ if environ is None else environ
        if not proxy_identity_configured(source):
            raise AuthConfigurationError(
                f"Proxy identity is not configured: set {', '.join(_REQUIRED_VARIABLES)}."
            )
        issuer = _value(source, PROXY_IDENTITY_ISSUER_VARIABLE)
        audience = _value(source, PROXY_IDENTITY_AUDIENCE_VARIABLE)
        jwks_url = _value(source, PROXY_IDENTITY_JWKS_URL_VARIABLE)
        header = _value(source, PROXY_IDENTITY_HEADER_VARIABLE) or DEFAULT_PROXY_IDENTITY_HEADER
        allow_insecure_loopback = options.get("allow_insecure_loopback", False)
        if type(allow_insecure_loopback) is not bool:
            raise ValueError("`allow_insecure_loopback` must be a boolean.")
        for name, value, is_issuer in (
            (PROXY_IDENTITY_ISSUER_VARIABLE, issuer, True),
            (PROXY_IDENTITY_JWKS_URL_VARIABLE, jwks_url, False),
        ):
            try:
                _require_endpoint_url(
                    value,
                    "issuer" if is_issuer else "jwks_url",
                    allow_insecure_loopback=allow_insecure_loopback,
                    is_issuer=is_issuer,
                )
            except ValueError as exc:
                raise AuthConfigurationError(f"{name} cannot be used: {exc}") from None
        try:
            _require_header_name(header)
        except ValueError as exc:
            raise AuthConfigurationError(
                f"{PROXY_IDENTITY_HEADER_VARIABLE} cannot be used: {exc}"
            ) from None
        return cls(
            issuer=issuer,
            audience=audience,
            jwks_url=jwks_url,
            header=header,
            fallback=fallback,
            **options,
        )

    def __repr__(self) -> str:
        return (
            f"ProxyIdentityAuth(issuer={self.issuer!r}, audience={list(self.audience)!r}, "
            f"header={self.header!r})"
        )

    async def __call__(self, request: HTTPConnection) -> AuthContext:
        """Authenticate, then apply ``control_plane_roles`` to the matched route."""

        context = await self.authenticate(request)
        self.control_plane_roles.require(request, context)
        role = context.claims.get(self.control_plane_roles.role_claim)
        if role not in self.control_plane_roles.full_access and await has_decision_overrides(
            request
        ):
            raise _role_forbidden()
        return context

    async def authenticate(self, request: HTTPConnection) -> AuthContext:
        """Who is calling, with their ``role`` claim; no route rule is applied."""

        if not request.headers.getlist(self.header):
            return await self._fallback_context(request)
        return await self._proxy_context(request)

    async def verify_token(self, token: str) -> dict[str, Any]:
        """Verify one proxy identity token and return its claims.

        Raises ``OidcTokenError`` with a fixed reason that never contains token
        contents.
        """

        if type(token) is not str or len(token) > self._verifier.max_token_bytes:
            raise _reject("malformed token")
        header = _unverified_header(token)
        if header is None or header.get("typ") != self.token_type:
            raise _reject("token is not a proxy identity")
        claims = await self._verifier.verify_token(token)
        issued_at = claims.get("iat")
        # The verifier requires a numeric `exp` and checks `iat` is numeric when present.
        if not _is_numeric_date(issued_at):
            raise _reject("token has no issue time")
        if claims["exp"] - issued_at > self.max_lifetime_seconds:
            raise _reject("token lifetime is too long")
        jti = claims.get("jti")
        if type(jti) is not str or not jti.strip() or len(jti) > _MAX_JTI_CHARS:
            raise _reject("token has no usable id")
        if claims.get(self.role_claim) not in self.roles:
            raise _reject("token role is not accepted")
        for name in _OPTIONAL_TEXT_CLAIMS:
            if name in claims and type(claims[name]) is not str:
                raise _reject("token profile claim is invalid")
        if "origin" in claims and (
            type(claims["origin"]) is not str
            or _url_origin(claims["origin"], allow_path=False) is None
        ):
            raise _reject("token origin is invalid")
        return claims

    def auth_context(self, claims: Mapping[str, Any]) -> AuthContext:
        """Build the ``AuthContext`` for claims ``verify_token`` returned."""

        selected = {name: claims[name] for name in _CONTEXT_CLAIMS if name in claims}
        selected["role"] = claims[self.role_claim]
        selected["auth"] = _PROXY_IDENTITY_AUTH
        try:
            return AuthContext(
                subject=claims["sub"],
                tenant=claims["org"],
                claims=_bounded_claims(selected),
            )
        except (ValidationError, ValueError):
            raise _reject("token identity claims are invalid") from None

    def product_dependency(
        self, *, roles: Sequence[str] | None = None
    ) -> Callable[[HTTPConnection], Awaitable[ProductPrincipal]]:
        """Return a product auth dependency for people the proxy signed in.

        The Organization (``org``) is the tenant and the person (``sub``) the
        subject. Only a proxy identity is accepted, never ``fallback``: a request
        without the header gets 401, and a person whose role is not in ``roles``
        (default: every accepted role) gets 403.
        """

        allowed = self.roles if roles is None else _require_roles(roles, "roles", allow_empty=False)
        if not set(allowed) <= set(self.roles):
            raise ValueError(f"`roles` must name some of {', '.join(self.roles)}.")

        async def authenticate(request: HTTPConnection) -> ProductPrincipal:
            if not request.headers.getlist(self.header):
                raise _unauthorized()
            context = await self._proxy_context(request)
            if context.claims.get("role") not in allowed:
                raise _role_forbidden()
            try:
                return ProductPrincipal(tenant_id=context.tenant or "", subject_id=context.subject)
            except ValidationError:
                raise _invalid_identity() from None

        return authenticate

    async def _proxy_context(self, request: HTTPConnection) -> AuthContext:
        values = request.headers.getlist(self.header)
        if len(values) != 1:
            # A proxy sends exactly one; more means a client's header got through.
            raise _invalid_identity()
        try:
            claims = await self.verify_token(values[0].strip())
            context = self.auth_context(claims)
        except OidcTokenError as exc:
            raise _http_error(exc) from None
        # The proxy names the origin the browser sees, signed, because behind it the
        # Agent may see another scheme or host than the browser does.
        if self.same_origin_writes and _cross_origin_write(request, claims.get("origin")):
            raise HTTPException(status_code=403, detail="Cross-origin request refused.")
        return context

    async def _fallback_context(self, request: HTTPConnection) -> AuthContext:
        if self.fallback is None:
            raise _unauthorized()
        context = await resolve_auth_context(self.fallback, request)
        if self.fallback_role is None:
            return context
        return AuthContext(
            subject=context.subject,
            tenant=context.tenant,
            claims={**context.claims, "role": self.fallback_role},
        )


def _cross_origin_write(request: HTTPConnection, expected_origin: str | None = None) -> bool:
    """Refuse cross-origin browser writes, including browsers without Fetch Metadata.

    ``expected_origin`` is the origin the proxy says the browser sees; without one,
    the request's own URL is the target.
    """

    if request.scope.get("type") == "http" and request.scope.get("method") in _SAFE_METHODS:
        return False
    fetch_sites = request.headers.getlist("sec-fetch-site")
    if fetch_sites and (
        len(fetch_sites) != 1 or fetch_sites[0].strip().lower() not in _SAME_ORIGIN_FETCH_SITES
    ):
        return True
    # Origin is authoritative, including the opaque "null" origin. Older
    # browsers may omit Fetch Metadata; Referer is the next available signal.
    for header in ("origin", "referer"):
        values = request.headers.getlist(header)
        if values:
            if len(values) != 1:
                return True
            source = _url_origin(values[0], allow_path=header == "referer")
            target = _url_origin(
                expected_origin or str(request.url.replace(path="", query="", fragment="")),
                allow_path=False,
            )
            return source is None or target is None or source != target
    # Non-browser callers can legitimately omit all three browser headers.
    return False


def _url_origin(value: str, *, allow_path: bool) -> tuple[str, str, int] | None:
    if not value or "\\" in value or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        return None
    try:
        parts = urlsplit(value)
        scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
        if (
            scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or (not allow_path and (parts.path or parts.query or parts.fragment))
        ):
            return None
        port = parts.port if parts.port is not None else (443 if scheme == "https" else 80)
        return scheme, parts.hostname.lower(), port
    except ValueError:
        return None


def _require_pyjwt_for_proxy_identity() -> None:
    try:
        from jwt.algorithms import has_crypto
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] != "jwt":
            raise
        has_crypto = False
    if not has_crypto:
        raise RuntimeError(
            "Proxy identity requires PyJWT with cryptography support. Install it with "
            '`pip install "cayu[server]"`.'
        )


def _require_header_name(value: Any) -> str:
    if type(value) is not str or _HEADER_TOKEN.fullmatch(value) is None:
        raise ValueError("`header` must be an HTTP header name.")
    if value.lower() in _RESERVED_HEADERS:
        raise ValueError("`header` must not name a header that carries other credentials.")
    return value


def _require_roles(value: Sequence[str], field_name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"`{field_name}` must be a sequence of role names.")
    roles = tuple(value)
    if not roles and not allow_empty:
        raise ValueError(f"`{field_name}` must name at least one role.")
    for role in roles:
        if type(role) is not str or not role.strip() or role != role.strip():
            raise ValueError(f"`{field_name}` values must be non-blank strings.")
    return tuple(dict.fromkeys(roles))


def _value(source: Mapping[str, str], name: str) -> str:
    value = source.get(name)
    return value.strip() if isinstance(value, str) else ""


def _reject(reason: str) -> OidcTokenError:
    logger.debug("Rejected a proxy identity: %s.", reason)
    return OidcTokenError(reason)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Sign-in required.",
        headers={"WWW-Authenticate": 'ProxyIdentity realm="Cayu"'},
    )


def _invalid_identity() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid sign-in.",
        headers={"WWW-Authenticate": 'ProxyIdentity realm="Cayu", error="invalid_token"'},
    )


def _http_error(exc: OidcTokenError) -> HTTPException:
    if exc.status_code == 503:
        return HTTPException(
            status_code=503,
            detail="Sign-in verification is temporarily unavailable.",
            headers={"Retry-After": "5"},
        )
    return _invalid_identity()


def _role_forbidden() -> HTTPException:
    return HTTPException(
        status_code=403,
        detail={
            "code": "role_forbidden",
            "message": "Your role can't use this.",
        },
    )
