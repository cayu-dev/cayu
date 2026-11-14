"""Who is calling an Agent's own routes, for an Agent whose people are its operators.

An operator Agent's people are its own team: Cayu Cloud's operators when it is deployed, and
the person running ``cayu serve --dev`` on their machine. Its own routes (the generated web
app's ``/app`` routes, or any it writes by hand) sign them in with the same auth dependency as
Cayu's Developer dashboard, the ``[tool.cayu.serve].auth`` target, and give each request the
person behind it::

    from fastapi import APIRouter, Depends

    from cayu.server.operator_sign_in import (
        OperatorPerson,
        may_change_dependency,
        operator_person_dependency,
    )


    def build_router(app, *, development, auth=None):
        person = operator_person_dependency(auth, development=development)
        changer = may_change_dependency(person)
        router = APIRouter(prefix="/app")

        @router.get("/me")
        async def me(signed_in: OperatorPerson = Depends(person)):
            return {"subject": signed_in.subject, "role": signed_in.role}

        @router.post("/recheck")
        async def recheck(signed_in: OperatorPerson = Depends(changer)): ...

        return router

``cayu serve`` calls a ``[tool.cayu.serve] routers`` function with ``auth`` (``--auth`` over
``[tool.cayu.serve].auth``, or ``None`` under ``--dev``) and ``development``; an application
that builds its own server with ``mount_cayu`` passes the auth dependency it gives
``AuthenticatedAccess``, or ``auth=None, development=True`` beside
``OpenAccess(trusted_local_development=True)``.

The parts, each usable on its own:

- ``operator_context_dependency(auth, development=...)``: the verified ``AuthContext``. A
  ``ProxyIdentityAuth`` (Cayu Cloud's sign-in) is asked only who is calling, through
  ``authenticate``, so its Developer dashboard route rule doesn't refuse a ``member`` on the
  Agent's own routes. A write (any method but ``GET``, ``HEAD`` and ``OPTIONS``) from
  another site is refused with 403; ``ProxyIdentityAuth`` refuses one with its identity
  itself, and a browser's saved operator login has no such check of its own. With
  ``auth=None`` and ``development=True`` the caller is the ``LocalOperator``.
- ``OperatorRoles``: Cayu Cloud's role rule, the default. Every role (``admin``,
  ``developer``, ``member``) may use the routes and decide what waits for a person in Cayu's
  inbox; ``admin`` and ``developer`` may change things. A caller whose context has no role,
  the operator login without Cloud's identity header, is ``admin``: the auth target already
  gives it the whole Developer dashboard. Any other role gets 403, and so does a role the
  auth's ``control_plane_roles`` keeps off the Developer dashboard (a ``member`` under
  ``OPERATOR_ONLY_AUTH`` is the application's own user). Configure it, or pass any function
  from ``AuthContext`` to a person as ``rule``.
- ``operator_person_dependency(auth, development=..., rule=...)``: the two together, the
  person behind the request (an ``OperatorPerson`` by default).
- ``may_change_dependency(person)``: the same person, or 403 when their role may not change
  things.
- ``LocalOperator``: the auth dependency for the person running ``cayu serve --dev``, as
  ``dev:<user>`` with the ``admin`` role. Like Cayu's own routes under ``--dev`` it admits
  only a loopback client that addresses the server by a loopback name.
"""

from __future__ import annotations

import getpass
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, overload
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request

from cayu.server.auth import (
    AuthContext,
    AuthDependency,
    TrustedLocalAuth,
    resolve_auth_context,
)
from cayu.server.proxy_identity import PROXY_IDENTITY_ROLES, ControlPlaneRoles

__all__ = [
    "CHANGE_ROLES",
    "OPERATOR_ROLES",
    "LocalOperator",
    "OperatorPerson",
    "OperatorRoles",
    "may_change_dependency",
    "operator_context_dependency",
    "operator_person_dependency",
]

#: Cayu Cloud's operator roles. Each may use an operator Agent's routes and decide what
#: waits in Cayu's inbox.
OPERATOR_ROLES = PROXY_IDENTITY_ROLES
#: The roles that may change an Agent's state or settings, beyond deciding.
CHANGE_ROLES = ("admin", "developer")

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# What a browser sends for a request its page's own origin made, or one typed by hand.
_SAME_ORIGIN_FETCH_SITES = frozenset({"same-origin", "none"})
_PROXY_IDENTITY_AUTH = "proxy-identity"

PersonT = TypeVar("PersonT")


class _MayChange(Protocol):
    @property
    def may_change(self) -> bool: ...


ChangerT = TypeVar("ChangerT", bound=_MayChange)


@dataclass(frozen=True, slots=True)
class OperatorPerson:
    """The person behind a request to an Agent's own routes.

    The same fields as the generated web app's ``Person``. ``may_decide``: they may decide
    what waits for a person (approve, answer). ``may_change``: they may change the Agent's
    state or settings, beyond deciding. ``name`` and ``email`` are there when the sign-in
    supplied them, as Cayu Cloud's does.
    """

    subject: str
    role: str
    may_decide: bool
    may_change: bool
    name: str | None = None
    email: str | None = None


class OperatorRoles(Generic[PersonT]):
    """What each operator role may do on an Agent's own routes: Cayu Cloud's rule by default.

    Called with a verified ``AuthContext``, it returns the person, built by ``person`` (an
    ``OperatorPerson`` by default, or a class with the same keyword fields such as the
    generated ``Person``). The role is the context's ``role_claim``; a context without one
    has ``unnamed_role`` (``admin``, as the operator login alone has; ``None`` refuses it).
    A role outside ``roles`` gets 403. ``decide`` and ``change`` name the roles that may
    decide and change: pass ``decide=()`` when nobody decides in this app, as when Cayu
    settles an approval only where its protected review is shown.
    """

    @overload
    def __init__(
        self: OperatorRoles[OperatorPerson],
        *,
        roles: Iterable[str] = ...,
        decide: Iterable[str] | None = ...,
        change: Iterable[str] = ...,
        role_claim: str = ...,
        unnamed_role: str | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        roles: Iterable[str] = ...,
        decide: Iterable[str] | None = ...,
        change: Iterable[str] = ...,
        role_claim: str = ...,
        unnamed_role: str | None = ...,
        person: Callable[..., PersonT],
    ) -> None: ...

    def __init__(
        self,
        *,
        roles: Iterable[str] = OPERATOR_ROLES,
        decide: Iterable[str] | None = None,
        change: Iterable[str] = CHANGE_ROLES,
        role_claim: str = "role",
        unnamed_role: str | None = "admin",
        person: Callable[..., Any] = OperatorPerson,
    ) -> None:
        self.roles = _role_names(roles, "roles")
        if not self.roles:
            raise ValueError("`roles` must name at least one role.")
        self.decide = self.roles if decide is None else _role_names(decide, "decide")
        self.change = _role_names(change, "change")
        outside = sorted((set(self.decide) | set(self.change)) - set(self.roles))
        if outside:
            raise ValueError(f"`decide` and `change` name roles outside `roles`: {outside}.")
        if type(role_claim) is not str or not role_claim.strip():
            raise ValueError("`role_claim` must be a non-blank claim name.")
        self.role_claim = role_claim
        if unnamed_role is not None and unnamed_role not in self.roles:
            raise ValueError("`unnamed_role` must be one of `roles`, or None.")
        self.unnamed_role = unnamed_role
        if not callable(person):
            raise ValueError("`person` must build the person from keyword fields.")
        self.person = person

    def __repr__(self) -> str:
        return (
            f"OperatorRoles(roles={list(self.roles)!r}, decide={list(self.decide)!r}, "
            f"change={list(self.change)!r})"
        )

    def __call__(self, context: AuthContext) -> PersonT:
        role = context.claims.get(self.role_claim, self.unnamed_role)
        if type(role) is not str or role not in self.roles:
            raise HTTPException(status_code=403, detail="Your role can't use this Agent's pages.")
        return self.person(
            subject=context.subject,
            role=role,
            may_decide=role in self.decide,
            may_change=role in self.change,
            name=_profile_text(context.claims.get("name")),
            email=_profile_text(context.claims.get("email")),
        )


class LocalOperator:
    """The auth dependency for the person running ``cayu serve --dev``: ``dev:<user>`` (the
    operating system's user name), with the ``admin`` role.

    It admits only a loopback client that addresses the server by a loopback name, as Cayu's
    own routes do under ``--dev``, so another machine or a page that rebinds its own name to
    127.0.0.1 is refused with 403. ``kind`` replaces ``dev`` in the subject, for another local
    caller such as a command (``cli``).
    """

    def __init__(self, kind: str = "dev", *, role: str = "admin") -> None:
        if type(kind) is not str or not kind.strip():
            raise ValueError("`kind` must be a non-blank name.")
        if type(role) is not str or not role.strip():
            raise ValueError("`role` must be a non-blank role name.")
        self.subject = f"{kind}:{_user_name()}"
        self.role = role
        self._loopback = TrustedLocalAuth()

    def __repr__(self) -> str:
        return f"LocalOperator(subject={self.subject!r}, role={self.role!r})"

    async def __call__(self, request: Request) -> AuthContext:
        self._loopback(request)
        return AuthContext(subject=self.subject, claims={"role": self.role})


def operator_context_dependency(
    auth: AuthDependency | None, *, development: bool = False
) -> Callable[[Request], Awaitable[AuthContext]]:
    """The FastAPI dependency that returns who is calling an Agent's own routes.

    ``auth`` is a Cayu server auth dependency returning an ``AuthContext`` or a mapping of
    one: the ``[tool.cayu.serve].auth`` target, or what ``cayu serve`` passes a router
    function as ``auth``. One with an ``authenticate`` method (``ProxyIdentityAuth``) is asked
    only who is calling, without its Developer dashboard route rule. ``auth=None`` is the
    ``LocalOperator`` under ``development=True`` and refused otherwise, so a missing auth
    target never serves the routes open. A write from another site gets 403.
    """

    if auth is None:
        if not development:
            raise ValueError(
                "An Agent's own routes need an auth dependency outside development: pass "
                "the [tool.cayu.serve].auth target, or development=True for the person "
                "running `cayu serve --dev`."
            )
        auth = LocalOperator()
    elif not callable(auth):
        raise ValueError("`auth` must be a Cayu server auth dependency.")
    identify = getattr(auth, "authenticate", None)
    who: AuthDependency = identify if callable(identify) else auth

    async def operator_context(request: Request) -> AuthContext:
        context = await resolve_auth_context(who, request)
        # A proxy identity's own check knows the origin the browser sees, which can differ
        # from this server's behind the proxy; every other sign-in is checked here.
        if context.claims.get("auth") != _PROXY_IDENTITY_AUTH and _cross_site_write(request):
            raise HTTPException(status_code=403, detail="Cross-origin request refused.")
        return context

    return operator_context


@overload
def operator_person_dependency(
    auth: AuthDependency | None, *, development: bool = False, rule: None = None
) -> Callable[[Request], Awaitable[OperatorPerson]]: ...


@overload
def operator_person_dependency(
    auth: AuthDependency | None,
    *,
    development: bool = False,
    rule: Callable[[AuthContext], PersonT],
) -> Callable[[Request], Awaitable[PersonT]]: ...


def operator_person_dependency(
    auth: AuthDependency | None,
    *,
    development: bool = False,
    rule: Callable[[AuthContext], Any] | None = None,
) -> Callable[[Request], Awaitable[Any]]:
    """The FastAPI dependency that returns the person behind a request to an Agent's own
    routes: ``operator_context_dependency(auth, development=...)``, then ``rule``.

    ``rule`` turns the verified ``AuthContext`` into the person, or raises (an
    ``HTTPException`` 403 for a role it refuses); by default ``OperatorRoles()``, Cayu Cloud's
    rule, which returns an ``OperatorPerson``. When ``auth`` carries ``control_plane_roles``
    (a ``ProxyIdentityAuth``), the default admits only the roles it lets use the Developer
    dashboard: under ``OPERATOR_ONLY_AUTH`` a ``member`` is the application's own user, not
    an operator. FastAPI resolves it once per request, however many of a route's
    dependencies use it.
    """

    context = operator_context_dependency(auth, development=development)
    decide: Callable[[AuthContext], Any] = _default_rule(auth) if rule is None else rule
    if not callable(decide):
        raise ValueError("`rule` must turn an AuthContext into the person.")

    async def operator_person(request: Request) -> Any:
        return decide(await context(request))

    return operator_person


def may_change_dependency(
    person: Callable[[Request], Awaitable[ChangerT]],
) -> Callable[..., Awaitable[ChangerT]]:
    """Like ``person``, but refuses with 403 anyone whose role may not change things."""

    async def changer(found: ChangerT = Depends(person)) -> ChangerT:  # noqa: B008
        if not found.may_change:
            raise HTTPException(status_code=403, detail="Your role can't change this.")
        return found

    return changer


def _default_rule(auth: AuthDependency | None) -> OperatorRoles[OperatorPerson]:
    """Cayu Cloud's rule, limited to the roles ``auth`` lets use the Developer dashboard."""

    dashboard = getattr(auth, "control_plane_roles", None)
    if not isinstance(dashboard, ControlPlaneRoles):
        return OperatorRoles()
    admitted = {*dashboard.full_access, *dashboard.read_and_decide}
    roles = tuple(role for role in OPERATOR_ROLES if role in admitted)
    if not roles:
        raise ValueError(
            "The auth dependency's control_plane_roles admit none of Cayu Cloud's operator "
            "roles; pass `rule=` to say who may use these routes."
        )
    return OperatorRoles(
        roles=roles,
        change=tuple(role for role in CHANGE_ROLES if role in admitted),
        role_claim=dashboard.role_claim,
        unnamed_role="admin" if "admin" in roles else None,
    )


def _cross_site_write(request: Request) -> bool:
    """Whether a write came from another site, as far as the browser says.

    Fetch Metadata decides when the browser sends it. Otherwise an ``Origin`` must name this
    server's host: a browser without Fetch Metadata still sends one on a cross-origin write.
    Only the host is compared, because behind a proxy that ends TLS this server sees another
    scheme than the browser. A caller that sends neither, such as a script, isn't a browser
    whose saved sign-in another site could use.
    """

    if request.method in _SAFE_METHODS:
        return False
    sites = request.headers.getlist("sec-fetch-site")
    if sites:
        return len(sites) != 1 or sites[0].strip().lower() not in _SAME_ORIGIN_FETCH_SITES
    origins = request.headers.getlist("origin")
    if not origins:
        return False
    if len(origins) != 1:
        return True
    host = request.headers.get("host", "").strip().lower()
    return not host or urlsplit(origins[0].strip()).netloc.lower() != host


def _role_names(values: Iterable[str], field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"`{field}` must be a list of role names, not one string.")
    names = tuple(values)
    if any(type(name) is not str or not name.strip() for name in names):
        raise ValueError(f"`{field}` must hold non-blank role names.")
    return names


def _profile_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _user_name() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        # A container can run as a user id with no name.
        return "local"
