"""A real Agent's route rules, run against `cayu.server.operator_sign_in`: same people, same access.

The shepherd (and another Agent, with the same rule) signs its `/app` routes in with its own copy
of a third Agent's `integrations/http/auth.py`, kept in `_shepherd_operator_rules.py`. Each route
below is served twice, once on the shepherd's dependencies and once on the part's, and every
request must get the same status and, when admitted, the same person. Only the wording of a
refusal may differ.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jwt")

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from tests.server import _shepherd_operator_rules as shepherd

from cayu.server import AuthContext, BasicAuth, ProxyIdentityAuth
from cayu.server.development_identity import DevelopmentIdentityProxy
from cayu.server.operator_sign_in import (
    OperatorRoles,
    may_change_dependency,
    operator_person_dependency,
)

_PROXY = DevelopmentIdentityProxy()
_LOGIN = BasicAuth(username="operator", password="secret")
_ORIGIN = "http://127.0.0.1"


def _cloud() -> ProxyIdentityAuth:
    """`OPERATOR_BASIC_AUTH` under Cayu Cloud's sign-in."""

    return ProxyIdentityAuth(
        issuer=_PROXY.issuer,
        audience=_PROXY.audience,
        signing_keys=_PROXY.signing_keys(),
        fallback=_LOGIN,
    )


async def _claims_from_header(request: Any) -> AuthContext:
    """A custom auth target that signs in whoever the test names, with any role or none."""

    subject = request.headers.get("x-test-subject")
    if subject is None:
        raise HTTPException(status_code=401, detail="Who are you?")
    claims: dict[str, Any] = {}
    for name in ("role", "name", "email"):
        if (value := request.headers.get(f"x-test-{name}")) is not None:
            claims[name] = value
    return AuthContext(subject=subject, claims=claims)


def _routes(person: Any, changer: Any) -> APIRouter:
    """The shapes of the shepherd's routes: a read, an inbox decision, and a change."""

    router = APIRouter()

    @router.get("/whoami")
    async def whoami(signed_in: Any = Depends(person)) -> dict[str, Any]:  # noqa: B008
        return asdict(signed_in)

    @router.post("/inbox/decide")
    async def decide(signed_in: Any = Depends(person)) -> dict[str, Any]:  # noqa: B008
        if not signed_in.may_decide:
            raise HTTPException(status_code=403, detail="Decide on the Developer dashboard.")
        return asdict(signed_in)

    @router.post("/pulls/recheck")
    async def recheck(signed_in: Any = Depends(changer)) -> dict[str, Any]:  # noqa: B008
        return asdict(signed_in)

    return router


def _server(auth: Any, *, development: bool, decisions_here: bool) -> FastAPI:
    server = FastAPI()
    theirs_auth = shepherd.LocalOperator() if development else auth
    theirs = shepherd.person_dependency(theirs_auth, decisions_here=decisions_here)
    server.include_router(_routes(theirs, shepherd.change_dependency(theirs)), prefix="/shepherd")
    rule = OperatorRoles(decide=None if decisions_here else (), person=shepherd.Person)
    ours = operator_person_dependency(
        None if development else auth, development=development, rule=rule
    )
    server.include_router(_routes(ours, may_change_dependency(ours)), prefix="/part")
    return server


def _basic(username: str = "operator", password: str = "secret") -> dict[str, str]:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"authorization": f"Basic {encoded}"}


def _test(subject: str, **claims: str) -> dict[str, str]:
    return {"x-test-subject": subject, **{f"x-test-{k}": v for k, v in claims.items()}}


_CROSS_SITE = {"sec-fetch-site": "cross-site"}
_SAME_ORIGIN = {"sec-fetch-site": "same-origin"}

_CLOUD_CALLERS: dict[str, dict[str, str]] = {
    "admin": _PROXY.headers("ada", role="admin", name="Ada", email="ada@example.com"),
    "developer": _PROXY.headers("dev", role="developer", name="Dev"),
    "member": _PROXY.headers("mel", role="member", email="mel@example.com"),
    "member, same-origin": {**_PROXY.headers("mel", role="member"), **_SAME_ORIGIN},
    "member, cross-site": {**_PROXY.headers("mel", role="member"), **_CROSS_SITE},
    "admin, token for another origin": _PROXY.headers(
        "ada", role="admin", origin="https://elsewhere.example"
    ),
    "invalid identity": {_PROXY.header: "not-a-token"},
    "operator login": _basic(),
    "operator login, same-origin": {**_basic(), **_SAME_ORIGIN},
    "operator login, cross-site": {**_basic(), **_CROSS_SITE},
    "operator login, Origin elsewhere": {**_basic(), "origin": "https://evil.example"},
    "operator login, Origin here": {**_basic(), "origin": _ORIGIN},
    "wrong password": _basic(password="nope"),
    "nobody": {},
}
_CUSTOM_CALLERS: dict[str, dict[str, str]] = {
    "admin": _test("ada", role="admin", name="Ada"),
    "member": _test("mel", role="member"),
    "no role": _test("op"),
    "role outside the list": _test("vic", role="viewer"),
    "empty name and email": _test("eve", role="developer", name="", email=""),
    "no role, cross-site": {**_test("op"), **_CROSS_SITE},
    "nobody": {},
}
_DEV_CALLERS: dict[str, dict[str, str]] = {
    "person running it": {},
    "same-origin": _SAME_ORIGIN,
    "cross-site": _CROSS_SITE,
    "Origin elsewhere": {"origin": "https://evil.example"},
}
_PATHS = (("GET", "/whoami"), ("POST", "/inbox/decide"), ("POST", "/pulls/recheck"))


def _cases() -> list[Any]:
    cases = []
    for setting, auth, callers in (
        ("cloud", _cloud, _CLOUD_CALLERS),
        ("operator login alone", lambda: _LOGIN, {**_CLOUD_CALLERS}),
        ("custom auth", lambda: _claims_from_header, _CUSTOM_CALLERS),
        ("cayu serve --dev", lambda: None, _DEV_CALLERS),
    ):
        for caller, headers in callers.items():
            for decisions_here in (True, False):
                cases.append(
                    pytest.param(
                        auth,
                        setting == "cayu serve --dev",
                        decisions_here,
                        headers,
                        id=f"{setting}-{caller}-{'decides' if decisions_here else 'no decisions'}",
                    )
                )
    return cases


@pytest.mark.parametrize(("auth", "development", "decisions_here", "headers"), _cases())
def test_the_part_gives_the_shepherds_people_the_same_access(
    auth: Any, development: bool, decisions_here: bool, headers: dict[str, str]
) -> None:
    server = _server(auth(), development=development, decisions_here=decisions_here)
    # A loopback client addressing the server by a loopback name, as `cayu serve --dev` is.
    with TestClient(server, base_url=_ORIGIN, client=("127.0.0.1", 50000)) as client:
        for method, path in _PATHS:
            theirs = client.request(method, f"/shepherd{path}", headers=headers)
            ours = client.request(method, f"/part{path}", headers=headers)
            assert ours.status_code == theirs.status_code, (method, path, ours.json())
            if theirs.status_code == 200:
                assert ours.json() == theirs.json()
            if theirs.status_code == 401:
                assert ours.headers.get("www-authenticate") == theirs.headers.get(
                    "www-authenticate"
                )


def test_the_comparison_covers_every_outcome() -> None:
    """The matrix above admits, refuses a role, refuses a change, a decision and a
    cross-site write, and asks for a sign-in; a matrix that only ever answered 200 would
    prove nothing."""

    seen: set[tuple[str, int]] = set()
    for param in _cases():
        auth, development, decisions_here, headers = param.values
        server = _server(auth(), development=development, decisions_here=decisions_here)
        with TestClient(server, base_url=_ORIGIN, client=("127.0.0.1", 50000)) as client:
            for method, path in _PATHS:
                answer = client.request(method, f"/part{path}", headers=headers)
                detail = answer.json().get("detail") if answer.status_code != 200 else None
                seen.add((str(detail), answer.status_code))
    assert {status for _, status in seen} == {200, 401, 403}
    assert {
        "Your role can't use this Agent's pages.",
        "Your role can't change this.",
        "Decide on the Developer dashboard.",
        "Cross-origin request refused.",
    } <= {detail for detail, _ in seen}
