"""A sign-in proxy for local development and tests that signs identities as Cayu Cloud's does.

Cayu Cloud's Agent router signs people in and forwards each of their requests
with ``X-Cayu-Identity``, a short-lived ES256 token that ``ProxyIdentityAuth``
verifies. ``DevelopmentIdentityProxy`` mints the same tokens with a key that
lives only in this process, so an app takes its deployed sign-in path on a
laptop or in a test::

    proxy = DevelopmentIdentityProxy()
    auth = ProxyIdentityAuth(
        issuer=proxy.issuer, audience=proxy.audience, signing_keys=proxy.signing_keys()
    )
    client.get("/api/inbox", headers=proxy.headers("alice", role="member"))

``proxy.middleware(app, people=..., default=...)`` plays the router in front of
an ASGI app, and ``cayu serve --identity-proxy`` uses it.

``proxy.environment()`` names the proxy in the ``CAYU_PROXY_IDENTITY_*``
variables, with an issuer that can never resolve
(``https://identity-proxy.cayu.invalid``). ``ProxyIdentityAuth.from_environment``
trusts it only while ``proxy.register()`` has registered it in this process and
raises ``AuthConfigurationError`` otherwise, so copying the variables to a real
deployment fails at startup.
"""

from __future__ import annotations

import html
import threading
import time
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import parse_qsl, quote, unquote, urlsplit

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response

from cayu._development_people import (
    DEFAULT_DEVELOPMENT_ORG,
    DEFAULT_DEVELOPMENT_ROLE,
    DevelopmentPerson,
    parse_development_person,
)
from cayu.server.auth import _is_loopback_host as _is_loopback_hostname
from cayu.server.oidc import OidcSigningKeys
from cayu.server.proxy_identity import (
    DEFAULT_PROXY_IDENTITY_HEADER,
    DEVELOPMENT_IDENTITY_ISSUER,
    PROXY_IDENTITY_AUDIENCE_VARIABLE,
    PROXY_IDENTITY_HEADER_VARIABLE,
    PROXY_IDENTITY_ISSUER_VARIABLE,
    PROXY_IDENTITY_JWKS_URL_VARIABLE,
    PROXY_IDENTITY_TOKEN_TYPE,
    _cross_origin_write,
    _require_header_name,
)

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

__all__ = [
    "DEFAULT_DEVELOPMENT_AUDIENCE",
    "DEVELOPMENT_IDENTITY_ISSUER",
    "DEVELOPMENT_IDENTITY_JWKS_URL",
    "DEVELOPMENT_PERSON_COOKIE",
    "DEVELOPMENT_PERSON_HEADER",
    "DevelopmentIdentityMiddleware",
    "DevelopmentIdentityProxy",
    "DevelopmentPerson",
    "registered_development_identity",
]

DEVELOPMENT_IDENTITY_JWKS_URL: Final = f"{DEVELOPMENT_IDENTITY_ISSUER}/jwks"
DEFAULT_DEVELOPMENT_AUDIENCE: Final = "cayu-agent:local"
#: Names the person for one request: a subject, or ``sub=..&org=..&role=..&name=..&email=..``.
DEVELOPMENT_PERSON_HEADER: Final = "X-Cayu-Dev-Person"
#: The person ``/.cayu/sign-in`` picked, in the same form as the header.
DEVELOPMENT_PERSON_COOKIE: Final = "cayu-dev-person"
#: The cookie value ``/.cayu/sign-out`` sets: nobody, not even the default person.
_SIGNED_OUT: Final = "-"
_SIGN_IN_PATH: Final = "/.cayu/sign-in"
_SIGN_OUT_PATH: Final = "/.cayu/sign-out"
#: As Cayu Cloud's router signs them.
_TOKEN_LIFETIME_SECONDS: Final = 60
_MAX_FORM_BYTES: Final = 16 * 1024

_registered: DevelopmentIdentityProxy | None = None
_registered_lock = threading.Lock()


def registered_development_identity() -> DevelopmentIdentityProxy | None:
    """The proxy ``register()`` registered in this process, if any."""

    return _registered


class DevelopmentIdentityProxy:
    """Mint the proxy identities Cayu Cloud's router sends, with a key held in memory.

    Each instance makes a new ES256 key. Tokens carry ``iss`` (``issuer``),
    ``aud`` (``audience``), ``sub``, ``org``, ``role``, ``name`` and ``email``
    when given, ``origin`` when given, ``iat``, ``nbf``, ``exp`` (60 seconds
    later) and a ``jti``, with JWS header ``typ`` ``cayu-proxy-identity+jwt``.
    """

    def __init__(
        self,
        *,
        audience: str = DEFAULT_DEVELOPMENT_AUDIENCE,
        header: str = DEFAULT_PROXY_IDENTITY_HEADER,
    ) -> None:
        import jwt
        from cryptography.hazmat.primitives.asymmetric import ec
        from jwt.algorithms import ECAlgorithm

        if type(audience) is not str or not audience.strip() or audience != audience.strip():
            raise ValueError("`audience` must be a non-blank string.")
        self._jwt = jwt
        self.issuer: str = DEVELOPMENT_IDENTITY_ISSUER
        self.jwks_url: str = DEVELOPMENT_IDENTITY_JWKS_URL
        self.audience = audience
        self.header = _require_header_name(header)
        self._key = ec.generate_private_key(ec.SECP256R1())
        self._kid = uuid.uuid4().hex
        public = ECAlgorithm.to_jwk(self._key.public_key(), as_dict=True)
        self._jwks = {"keys": [{**public, "kid": self._kid, "use": "sig", "alg": "ES256"}]}

    def __repr__(self) -> str:
        return f"DevelopmentIdentityProxy(audience={self.audience!r})"

    def token(
        self,
        subject: str,
        *,
        org: str = DEFAULT_DEVELOPMENT_ORG,
        role: str = DEFAULT_DEVELOPMENT_ROLE,
        name: str | None = None,
        email: str | None = None,
        origin: str | None = None,
    ) -> str:
        """A signed proxy identity for this person."""

        person = DevelopmentPerson(subject=subject, org=org, role=role, name=name, email=email)
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": person.subject,
            "org": person.org,
            "role": person.role,
            "iat": now,
            "nbf": now,
            "exp": now + _TOKEN_LIFETIME_SECONDS,
            "jti": uuid.uuid4().hex,
        }
        if person.name is not None:
            claims["name"] = person.name
        if person.email is not None:
            claims["email"] = person.email
        if origin is not None:
            claims["origin"] = origin
        return self._jwt.encode(
            claims,
            self._key,
            algorithm="ES256",
            headers={"kid": self._kid, "typ": PROXY_IDENTITY_TOKEN_TYPE},
        )

    def headers(
        self,
        subject: str,
        *,
        org: str = DEFAULT_DEVELOPMENT_ORG,
        role: str = DEFAULT_DEVELOPMENT_ROLE,
        name: str | None = None,
        email: str | None = None,
        origin: str | None = None,
    ) -> dict[str, str]:
        """The request headers the router would add for this person."""

        return {
            self.header: self.token(
                subject, org=org, role=role, name=name, email=email, origin=origin
            )
        }

    def person_token(self, person: DevelopmentPerson, *, origin: str | None = None) -> str:
        return self.token(
            person.subject,
            org=person.org,
            role=person.role,
            name=person.name,
            email=person.email,
            origin=origin,
        )

    def jwks(self) -> dict[str, Any]:
        """The public JWKS document for this proxy's key."""

        return {"keys": [dict(key) for key in self._jwks["keys"]]}

    def environment(self) -> dict[str, str]:
        """The ``CAYU_PROXY_IDENTITY_*`` variables naming this proxy."""

        return {
            PROXY_IDENTITY_ISSUER_VARIABLE: self.issuer,
            PROXY_IDENTITY_AUDIENCE_VARIABLE: self.audience,
            PROXY_IDENTITY_JWKS_URL_VARIABLE: self.jwks_url,
            PROXY_IDENTITY_HEADER_VARIABLE: self.header,
        }

    def signing_keys(self) -> OidcSigningKeys:
        """``OidcSigningKeys`` that read this proxy's JWKS in memory, never over a network."""

        def serve(request: httpx.Request) -> httpx.Response:
            if str(request.url) == self.jwks_url:
                return httpx.Response(200, json=self.jwks())
            return httpx.Response(404)

        return OidcSigningKeys(
            self.issuer,
            jwks_url=self.jwks_url,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
        )

    def register(self) -> None:
        """Make this the proxy ``ProxyIdentityAuth.from_environment`` trusts in this process."""

        global _registered
        with _registered_lock:
            _registered = self

    def unregister(self) -> None:
        """Stop trusting this proxy, if it is the registered one."""

        global _registered
        with _registered_lock:
            if _registered is self:
                _registered = None

    def middleware(
        self,
        app: ASGIApp,
        *,
        people: Iterable[DevelopmentPerson] = (),
        default: DevelopmentPerson | str | None = None,
    ) -> DevelopmentIdentityMiddleware:
        """Put this proxy in front of ``app``, as Cayu Cloud's router is in front of an Agent."""

        return DevelopmentIdentityMiddleware(app, proxy=self, people=people, default=default)


class DevelopmentIdentityMiddleware:
    """ASGI middleware that signs each request in as a chosen person.

    On every HTTP and WebSocket request it removes a client-sent identity header,
    as Cayu Cloud's router does. When the ``Host`` header is a loopback name it
    then picks the person from the ``X-Cayu-Dev-Person`` header, the
    ``cayu-dev-person`` cookie, or ``default``, and adds a token whose ``origin``
    is the request's own origin. The header and cookie never reach ``app``.
    ``GET /.cayu/sign-in`` is a person picker that sets the cookie, and
    ``/.cayu/sign-out`` signs out, even from the default person.

    Add it with ``server.add_middleware(DevelopmentIdentityMiddleware, proxy=...)``
    or ``proxy.middleware(app)``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        proxy: DevelopmentIdentityProxy,
        people: Iterable[DevelopmentPerson] = (),
        default: DevelopmentPerson | str | None = None,
    ) -> None:
        self.app = app
        self.proxy = proxy
        self.people: dict[str, DevelopmentPerson] = {}
        for person in people:
            if not isinstance(person, DevelopmentPerson):
                raise ValueError("`people` must hold DevelopmentPerson values.")
            self.people[person.subject] = person
        if isinstance(default, str):
            if default not in self.people:
                raise ValueError(f"`default` {default!r} is not one of `people`.")
            default = self.people[default]
        elif default is not None and not isinstance(default, DevelopmentPerson):
            raise ValueError("`default` must be a DevelopmentPerson, a subject, or None.")
        self.default = default
        self._identity_header = proxy.header.lower().encode("latin-1")
        self._person_header = DEVELOPMENT_PERSON_HEADER.lower().encode("latin-1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = list(scope.get("headers") or ())
        host = _single(headers, b"host")
        loopback = host is not None and _is_loopback_host(host)
        person_values = [value for name, value in headers if name == self._person_header]
        cookie, others = _take_cookie(headers, DEVELOPMENT_PERSON_COOKIE)
        kept = [
            (name, value)
            for name, value in headers
            if name not in {self._identity_header, self._person_header, b"cookie"}
        ]
        if others:
            kept.append((b"cookie", others.encode("latin-1")))
        if (
            loopback
            and scope["type"] == "http"
            and scope.get("path")
            in {
                _SIGN_IN_PATH,
                _SIGN_OUT_PATH,
            }
        ):
            response = await self._page(scope, receive, cookie)
            await response(scope, receive, send)
            return
        if loopback:
            try:
                person = self._person(person_values, cookie)
            except ValueError as exc:
                if scope["type"] == "http":
                    await PlainTextResponse(f"{DEVELOPMENT_PERSON_HEADER}: {exc}", status_code=400)(
                        scope, receive, send
                    )
                    return
                person = None
            if person is not None:
                token = self.proxy.person_token(person, origin=_origin(scope, host or ""))
                kept.append((self._identity_header, token.encode("latin-1")))
        await self.app({**scope, "headers": kept}, receive, send)

    def _person(self, header_values: list[bytes], cookie: str | None) -> DevelopmentPerson | None:
        if header_values:
            if len(header_values) != 1:
                raise ValueError("send it once.")
            return parse_development_person(header_values[0].decode("latin-1"), self.people)
        if cookie == _SIGNED_OUT:
            return None
        if cookie:
            try:
                return parse_development_person(cookie, self.people)
            except ValueError:
                # A cookie that names nobody is ignored, as no cookie is.
                return self.default
        return self.default

    async def _page(self, scope: Scope, receive: Receive, cookie: str | None) -> Response:
        request = Request(scope, receive)
        method = scope.get("method")
        if scope["path"] == _SIGN_OUT_PATH:
            if method not in {"GET", "POST"}:
                return PlainTextResponse("Method not allowed.", status_code=405)
            response = RedirectResponse(
                _safe_next(request.query_params.get("next"), _SIGN_IN_PATH), status_code=303
            )
            _set_cookie(response, _SIGNED_OUT)
            return response
        if method == "GET":
            current = None
            if cookie and cookie != _SIGNED_OUT:
                try:
                    current = parse_development_person(cookie, self.people)
                except ValueError:
                    current = None
            elif cookie is None:
                current = self.default
            return HTMLResponse(
                _picker_page(
                    self.people.values(), current, _safe_next(request.query_params.get("next"))
                )
            )
        if method != "POST":
            return PlainTextResponse("Method not allowed.", status_code=405)
        if _cross_origin_write(request):
            return PlainTextResponse("Cross-origin request refused.", status_code=403)
        body = await _read_body(receive)
        if body is None:
            return PlainTextResponse("Form too large.", status_code=413)
        form = dict(parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True))
        try:
            chosen = form.get("person", "").strip()
            if chosen:
                person = parse_development_person(quote(chosen, safe=""), self.people)
            else:
                person = DevelopmentPerson(
                    subject=form.get("subject", "").strip(),
                    org=form.get("org", "").strip() or DEFAULT_DEVELOPMENT_ORG,
                    role=form.get("role", "").strip() or DEFAULT_DEVELOPMENT_ROLE,
                    name=form.get("name", "").strip() or None,
                    email=form.get("email", "").strip() or None,
                )
        except ValueError as exc:
            return PlainTextResponse(f"Can't sign in as that person: {exc}", status_code=400)
        response = RedirectResponse(_safe_next(form.get("next")), status_code=303)
        _set_cookie(response, person.encode())
        return response


def _single(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    values = [value for key, value in headers if key == name]
    return values[0].decode("latin-1") if len(values) == 1 else None


def _is_loopback_host(host: str) -> bool:
    """Whether a ``Host`` header names this machine, so no other site's page can use it."""

    try:
        return _is_loopback_hostname(urlsplit(f"//{host}").hostname)
    except ValueError:
        return False


def _origin(scope: Scope, host: str) -> str:
    scheme = scope.get("scheme") or "http"
    scheme = {"ws": "http", "wss": "https"}.get(scheme, scheme)
    return f"{scheme}://{host.lower()}"


def _take_cookie(headers: list[tuple[bytes, bytes]], name: str) -> tuple[str | None, str]:
    """The named cookie's value, and the Cookie header without it."""

    found: str | None = None
    others: list[str] = []
    for key, value in headers:
        if key != b"cookie":
            continue
        for part in value.decode("latin-1").split(";"):
            cookie_name, separator, cookie_value = part.strip().partition("=")
            if not cookie_name:
                continue
            if cookie_name == name and separator:
                found = unquote(cookie_value.strip().strip('"'))
            else:
                others.append(part.strip())
    return found, "; ".join(others)


def _set_cookie(response: Response, value: str) -> None:
    response.set_cookie(
        DEVELOPMENT_PERSON_COOKIE, quote(value, safe=""), path="/", httponly=True, samesite="lax"
    )


def _safe_next(value: str | None, fallback: str = "/") -> str:
    """A path on this site to go to next; anything else becomes ``fallback``."""

    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return fallback
    if any(ord(character) < 0x21 for character in value):
        return fallback
    return value


async def _read_body(receive: Receive) -> bytes | None:
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] != "http.request":
            return bytes(body)
        body.extend(message.get("body", b""))
        if len(body) > _MAX_FORM_BYTES:
            return None
        if not message.get("more_body", False):
            return bytes(body)


def _picker_page(
    people: Iterable[DevelopmentPerson], current: DevelopmentPerson | None, next_path: str
) -> str:
    escape = html.escape
    rows = "".join(
        '<form method="post"><input type="hidden" name="next" value="{next}">'
        '<button name="person" value="{subject}">{label}</button> <small>{detail}</small></form>'.format(
            next=escape(next_path),
            subject=escape(person.subject),
            label=escape(person.name or person.subject),
            detail=escape(
                f"{person.subject} · {person.role} in {person.org}"
                + (f" · {person.email}" if person.email else "")
            ),
        )
        for person in people
    )
    who = (
        f"Signed in as <strong>{escape(current.name or current.subject)}</strong> "
        f"({escape(current.role)} in {escape(current.org)}). "
        f'<a href="{_SIGN_OUT_PATH}">Sign out</a>'
        if current is not None
        else "Nobody is signed in."
    )
    roles = "".join(
        f"<option{' selected' if role == DEFAULT_DEVELOPMENT_ROLE else ''}>{role}</option>"
        for role in ("member", "admin", "developer")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Development sign-in</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:36rem;margin:2rem auto;padding:0 1rem}}
form{{margin:.5rem 0}}label{{display:block;margin:.25rem 0}}small{{color:#666}}</style></head>
<body><h1>Development sign-in</h1>
<p>{who}</p>
<p>This page is served by <code>cayu serve --identity-proxy</code> in place of Cayu Cloud's
sign-in. Requests carry a signed identity for the person you pick.</p>
{rows}
<h2>Someone else</h2>
<form method="post"><input type="hidden" name="next" value="{escape(next_path)}">
<label>Subject <input name="subject" required></label>
<label>Organization <input name="org" placeholder="{DEFAULT_DEVELOPMENT_ORG}"></label>
<label>Role <select name="role">{roles}</select></label>
<label>Name <input name="name"></label>
<label>Email <input name="email" type="email"></label>
<button>Sign in</button></form>
</body></html>
"""
