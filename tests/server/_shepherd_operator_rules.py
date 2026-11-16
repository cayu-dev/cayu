"""A real Agent's `integrations/http/auth.py`, the rule `cayu.server.operator_sign_in` replaces.

That Agent copied it from another Agent and cut it down; a third carries the same rule.
It is kept here unchanged but for the generated `Person`, defined below instead of imported,
so `test_operator_sign_in_shepherd.py` can show the part gives the same people the same access.
"""

from __future__ import annotations

import getpass
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request

from cayu.server import AuthContext

ROLES = ("admin", "developer", "member")
# The roles that may change things: queue a re-check, which may act on GitHub.
CHANGE_ROLES = frozenset({"admin", "developer"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_SAME_ORIGIN_FETCH_SITES = frozenset({"same-origin", "none"})


@dataclass(frozen=True, slots=True)
class Person:
    """The generated web app's `Person` (`integrations/http/generated/whoami.py`)."""

    subject: str
    role: str
    may_decide: bool
    may_change: bool
    name: str | None = None
    email: str | None = None


PersonDependency = Callable[[Request], Awaitable[Person]]


def person_from_context(context: AuthContext, *, may_decide: bool = True) -> Person:
    """The person a verified Cayu auth context names, with what their role allows.

    ``may_decide`` is False when the app can't decide inbox items itself: see
    `person_dependency`.
    """

    claims = context.claims
    role = claims.get("role", "admin")
    if role not in ROLES:
        raise HTTPException(status_code=403, detail="Your role can't use the shepherd's pages.")
    name, email = claims.get("name"), claims.get("email")
    return Person(
        subject=context.subject,
        role=role,
        may_decide=may_decide,
        may_change=role in CHANGE_ROLES,
        name=name if isinstance(name, str) and name else None,
        email=email if isinstance(email, str) and email else None,
    )


def person_dependency(auth: Any, *, decisions_here: bool = True) -> PersonDependency:
    """The FastAPI dependency that returns the `Person` behind a request.

    `auth` is a Cayu server auth dependency, returning an `AuthContext` or a mapping of
    one. A `ProxyIdentityAuth` is asked only who is calling (`authenticate`), so Cayu's
    route rule doesn't refuse members on the shepherd's routes; any other dependency is
    called as Cayu calls it.

    ``decisions_here`` is False when the app has a protected human review policy (a
    merge's arguments shown through `ToolArgumentReview`): Cayu then settles an approval
    only with the reference its protected review returns, which the web app can't show
    yet, so nobody decides in the app and the pages send them to the Developer dashboard.
    """

    authenticate = getattr(auth, "authenticate", None)
    who = authenticate if callable(authenticate) else auth

    async def person(request: Request) -> Person:
        found = who(request)
        if inspect.isawaitable(found):
            found = await found
        context = (
            found if isinstance(found, AuthContext) else AuthContext.model_validate(dict(found))
        )
        # ProxyIdentityAuth refuses a proxied cross-origin write itself; a browser's saved
        # operator login has no such check, so writes on it must come from this origin.
        if context.claims.get("auth") != "proxy-identity" and _cross_origin_write(request):
            raise HTTPException(status_code=403, detail="Cross-origin request refused.")
        return person_from_context(context, may_decide=decisions_here)

    return person


def change_dependency(person: PersonDependency) -> PersonDependency:
    """Like `person`, but refuses anyone whose role can't change things."""

    async def changer(found: Person = Depends(person)) -> Person:  # noqa: B008
        if not found.may_change:
            raise HTTPException(status_code=403, detail="Only an admin or a developer may do this.")
        return found

    return changer


def _cross_origin_write(request: Request) -> bool:
    if request.method in _SAFE_METHODS:
        return False
    sites = request.headers.getlist("sec-fetch-site")
    if sites:
        return len(sites) != 1 or sites[0].strip().lower() not in _SAME_ORIGIN_FETCH_SITES
    # Browsers without Fetch Metadata still send Origin on a cross-origin POST.
    origin = request.headers.get("origin")
    if origin is None:
        return False
    return urlsplit(origin).netloc.lower() != request.headers.get("host", "").lower()


class LocalOperator:
    """The auth dependency for `cayu serve --dev`: the person running the process, as
    `dev:<user>`, with the operator's role."""

    def __init__(self) -> None:
        self.subject = f"dev:{getpass.getuser()}"

    async def __call__(self, request: Any) -> AuthContext:
        return AuthContext(subject=self.subject, claims={"role": "admin"})
