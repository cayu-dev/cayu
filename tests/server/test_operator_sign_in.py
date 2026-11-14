"""`cayu.server.operator_sign_in`: who is calling an operator Agent's own routes."""

from __future__ import annotations

import base64
import getpass
from dataclasses import asdict, dataclass
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jwt")

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import (
    AuthContext,
    AuthenticatedAccess,
    BasicAuth,
    ControlPlaneRoles,
    OpenAccess,
    ProxyIdentityAuth,
    mount_cayu,
)
from cayu.server.auth import server_auth_dependency
from cayu.server.development_identity import DevelopmentIdentityProxy
from cayu.server.operator_sign_in import (
    LocalOperator,
    OperatorPerson,
    OperatorRoles,
    may_change_dependency,
    operator_context_dependency,
    operator_person_dependency,
    operator_roles,
)

_PROXY = DevelopmentIdentityProxy()
_LOGIN = BasicAuth(username="operator", password="secret")
_LOOPBACK = {"base_url": "http://127.0.0.1", "client": ("127.0.0.1", 50000)}


def _cloud() -> ProxyIdentityAuth:
    return ProxyIdentityAuth(
        issuer=_PROXY.issuer,
        audience=_PROXY.audience,
        signing_keys=_PROXY.signing_keys(),
        fallback=_LOGIN,
    )


def _basic(password: str = "secret") -> dict[str, str]:
    return {"authorization": "Basic " + base64.b64encode(f"operator:{password}".encode()).decode()}


def _app_routes(person: Any) -> APIRouter:
    router = APIRouter(prefix="/app")

    @router.get("/whoami")
    async def whoami(signed_in: Any = Depends(person)) -> dict[str, Any]:  # noqa: B008
        return asdict(signed_in)

    @router.post("/recheck")
    async def recheck(signed_in: Any = Depends(may_change_dependency(person))) -> dict[str, Any]:  # noqa: B008
        return asdict(signed_in)

    return router


def _client(person: Any, **options: Any) -> TestClient:
    server = FastAPI()
    server.include_router(_app_routes(person))
    return TestClient(server, **options)


def test_cloud_sign_in_gives_the_person_their_role_name_and_email() -> None:
    client = _client(operator_person_dependency(_cloud()))

    member = client.get(
        "/app/whoami",
        headers=_PROXY.headers("mel", role="member", name="Mel", email="mel@example.com"),
    )
    assert member.json() == {
        "subject": "mel",
        "role": "member",
        "may_decide": True,
        "may_change": False,
        "name": "Mel",
        "email": "mel@example.com",
    }
    for role in ("admin", "developer"):
        answer = client.get("/app/whoami", headers=_PROXY.headers("dev", role=role))
        assert answer.json() == {
            "subject": "dev",
            "role": role,
            "may_decide": True,
            "may_change": True,
            "name": None,
            "email": None,
        }
    # A member may read and decide, but not change.
    refused = client.post("/app/recheck", headers=_PROXY.headers("mel", role="member"))
    assert refused.status_code == 403
    assert refused.json() == {"detail": "Your role can't change this."}
    allowed = client.post("/app/recheck", headers=_PROXY.headers("dev", role="developer"))
    assert allowed.status_code == 200


def test_an_invalid_identity_is_never_the_operator_login() -> None:
    client = _client(operator_person_dependency(_cloud()))

    answer = client.get("/app/whoami", headers={**_basic(), _PROXY.header: "not-a-token"})

    assert answer.status_code == 401


def test_the_operator_login_without_clouds_header_is_admin() -> None:
    for auth in (_cloud(), _LOGIN):
        client = _client(operator_person_dependency(auth))

        assert client.get("/app/whoami", headers=_basic()).json() == {
            "subject": "operator",
            "role": "admin",
            "may_decide": True,
            "may_change": True,
            "name": None,
            "email": None,
        }
        assert client.post("/app/recheck", headers=_basic()).status_code == 200
        refused = client.get("/app/whoami", headers=_basic("wrong"))
        assert refused.status_code == 401
        assert refused.headers["www-authenticate"].startswith("Basic")
        assert client.get("/app/whoami").status_code == 401


def test_cayu_serve_dev_on_loopback_is_the_person_running_it_as_admin() -> None:
    client = _client(operator_person_dependency(None, development=True), **_LOOPBACK)

    assert client.get("/app/whoami").json() == {
        "subject": f"dev:{getpass.getuser()}",
        "role": "admin",
        "may_decide": True,
        "may_change": True,
        "name": None,
        "email": None,
    }
    assert client.post("/app/recheck").status_code == 200


@pytest.mark.parametrize(
    "options",
    [
        # Another machine.
        {"base_url": "http://127.0.0.1", "client": ("203.0.113.9", 50000)},
        # A page that rebinds its own name to 127.0.0.1.
        {"base_url": "http://attacker.example", "client": ("127.0.0.1", 50000)},
    ],
)
def test_the_local_operator_admits_only_loopback(options: dict[str, Any]) -> None:
    client = _client(operator_person_dependency(None, development=True), **options)

    assert client.get("/app/whoami").status_code == 403


def test_no_auth_outside_development_is_refused_when_the_routes_are_built() -> None:
    with pytest.raises(ValueError, match="need an auth dependency outside development"):
        operator_person_dependency(None)
    with pytest.raises(ValueError, match="server auth dependency"):
        operator_context_dependency("module:attribute")  # type: ignore[arg-type]


async def _named(request: Any) -> dict[str, Any]:
    """A custom auth target, returning a mapping as Cayu's routes accept."""

    claims = {"role": request.headers["x-role"]} if "x-role" in request.headers else {}
    return {"subject": "someone", "claims": claims}


@pytest.mark.parametrize("role", ["viewer", "", "ADMIN"])
def test_a_role_outside_the_list_is_refused(role: str) -> None:
    client = _client(operator_person_dependency(_named))

    answer = client.get("/app/whoami", headers={"x-role": role})

    assert answer.status_code == 403
    assert answer.json() == {"detail": "Your role can't use this Agent's pages."}


@pytest.mark.parametrize(
    "headers",
    [
        {"sec-fetch-site": "cross-site"},
        {"sec-fetch-site": "same-site"},
        {"sec-fetch-site": "same-origin, cross-site"},
        {"origin": "https://attacker.example"},
        {"origin": "null"},
    ],
)
@pytest.mark.parametrize("sign_in", ["operator login", "cloud", "local operator"])
def test_a_cross_site_post_is_refused(headers: dict[str, str], sign_in: str) -> None:
    if sign_in == "local operator":
        person, signed = operator_person_dependency(None, development=True), {}
    elif sign_in == "cloud":
        person, signed = operator_person_dependency(_cloud()), _PROXY.headers("a", role="admin")
    else:
        person, signed = operator_person_dependency(_LOGIN), _basic()
    client = _client(person, **_LOOPBACK)

    refused = client.post("/app/recheck", headers={**signed, **headers})

    assert refused.status_code == 403
    assert refused.json() == {"detail": "Cross-origin request refused."}
    # Reads stay open to a link from another site.
    assert client.get("/app/whoami", headers={**signed, **headers}).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"sec-fetch-site": "same-origin"},
        {"sec-fetch-site": "none"},
        {"origin": "http://127.0.0.1"},
        # A script, not a browser: nothing another site could borrow.
        {},
    ],
)
def test_a_same_origin_post_is_admitted(headers: dict[str, str]) -> None:
    client = _client(operator_person_dependency(_LOGIN), **_LOOPBACK)

    assert client.post("/app/recheck", headers={**_basic(), **headers}).status_code == 200


def test_the_origin_check_compares_the_host_not_the_scheme() -> None:
    """Behind a proxy that ends TLS the server sees http while the browser is on https."""

    client = _client(operator_person_dependency(_LOGIN), base_url="http://agent.example")

    answer = client.post("/app/recheck", headers={**_basic(), "origin": "https://agent.example"})

    assert answer.status_code == 200


def test_what_cayu_serve_gives_a_router_still_admits_cloud_members() -> None:
    """`cayu serve` passes a router function its auth wrapped as for Cayu's routes. That
    wrapper applies the Developer dashboard's route rule, which refuses a member outside
    `/api`; the part asks only who is calling."""

    wrapped = server_auth_dependency(_cloud())
    member = _PROXY.headers("mel", role="member")
    server = FastAPI()
    server.include_router(_app_routes(operator_person_dependency(wrapped)))

    @server.get("/raw/whoami")
    async def raw(context: AuthContext = Depends(wrapped)) -> dict[str, str]:  # noqa: B008
        return {"subject": context.subject}

    client = TestClient(server)
    assert client.get("/raw/whoami", headers=member).status_code == 403
    assert client.get("/app/whoami", headers=member).json()["role"] == "member"
    assert client.get("/app/whoami", headers=_basic()).json()["role"] == "admin"
    assert client.get("/app/whoami").status_code == 401


@pytest.mark.parametrize("wrapped", [False, True])
def test_people_the_developer_dashboard_refuses_are_not_operators(wrapped: bool) -> None:
    """Under `OPERATOR_ONLY_AUTH` a member is the application's own user, not an operator."""

    auth: Any = ProxyIdentityAuth(
        issuer=_PROXY.issuer,
        audience=_PROXY.audience,
        signing_keys=_PROXY.signing_keys(),
        fallback=_LOGIN,
        control_plane_roles=ControlPlaneRoles(read_and_decide=()),
    )
    if wrapped:
        auth = server_auth_dependency(auth)
    client = _client(operator_person_dependency(auth))

    member = client.get("/app/whoami", headers=_PROXY.headers("mel", role="member"))
    assert member.status_code == 403
    assert client.get("/app/whoami", headers=_PROXY.headers("dev", role="developer")).json()[
        "may_change"
    ]
    assert client.get("/app/whoami", headers=_basic()).json()["role"] == "admin"
    # An explicit rule says who is an operator, whatever the dashboard admits.
    everyone = _client(operator_person_dependency(auth, rule=OperatorRoles()))
    assert (
        everyone.get("/app/whoami", headers=_PROXY.headers("mel", role="member")).json()["role"]
        == "member"
    )


def test_operator_roles_keeps_the_dashboards_roles_with_another_decide_and_person() -> None:
    """The generated app passes `operator_roles` to change who decides and build its own
    person, and members the Developer dashboard refuses stay refused."""

    @dataclass(frozen=True)
    class Teammate:
        subject: str
        role: str
        may_decide: bool
        may_change: bool
        name: str | None = None
        email: str | None = None

    auth = ProxyIdentityAuth(
        issuer=_PROXY.issuer,
        audience=_PROXY.audience,
        signing_keys=_PROXY.signing_keys(),
        fallback=_LOGIN,
        control_plane_roles=ControlPlaneRoles(read_and_decide=()),
    )
    rule = operator_roles(auth, decide=(), person=Teammate)
    client = _client(operator_person_dependency(auth, rule=rule))

    assert (
        client.get("/app/whoami", headers=_PROXY.headers("mel", role="member")).status_code == 403
    )
    developer = client.get("/app/whoami", headers=_PROXY.headers("dev", role="developer")).json()
    assert (developer["role"], developer["may_decide"], developer["may_change"]) == (
        "developer",
        False,
        True,
    )
    # Without the dashboard's roles it is Cayu Cloud's rule, with the same changes.
    plain = operator_roles(None, decide=("admin",), person=Teammate)
    assert (plain.roles, plain.decide, plain.change) == (
        ("admin", "developer", "member"),
        ("admin",),
        ("admin", "developer"),
    )
    assert operator_roles(auth).person is OperatorPerson


def test_a_dashboard_rule_without_operator_roles_needs_an_explicit_rule() -> None:
    auth = ProxyIdentityAuth(
        issuer=_PROXY.issuer,
        audience=_PROXY.audience,
        signing_keys=_PROXY.signing_keys(),
        roles=("owner", "member"),
        control_plane_roles=ControlPlaneRoles(full_access=("owner",), read_and_decide=()),
    )

    with pytest.raises(ValueError, match="pass `rule=`"):
        operator_person_dependency(auth)


def test_the_role_rule_is_replaceable() -> None:
    @dataclass(frozen=True)
    class Teammate:
        subject: str
        role: str
        may_decide: bool
        may_change: bool
        name: str | None = None
        email: str | None = None

    nobody_decides = OperatorRoles(decide=(), person=Teammate)
    client = _client(operator_person_dependency(_cloud(), rule=nobody_decides))
    answer = client.get("/app/whoami", headers=_PROXY.headers("mel", role="member"))
    assert answer.json()["may_decide"] is False

    admins_only = OperatorRoles(roles=("admin",), change=("admin",), unnamed_role=None)
    client = _client(operator_person_dependency(_named, rule=admins_only))
    assert client.get("/app/whoami", headers={"x-role": "admin"}).status_code == 200
    assert client.get("/app/whoami", headers={"x-role": "member"}).status_code == 403
    assert client.get("/app/whoami").status_code == 403

    def everyone_reads(context: AuthContext) -> OperatorPerson:
        if context.subject == "blocked":
            raise HTTPException(status_code=403, detail="Not you.")
        return OperatorPerson(
            subject=context.subject, role="reader", may_decide=False, may_change=False
        )

    client = _client(operator_person_dependency(_named, rule=everyone_reads))
    assert client.get("/app/whoami", headers={"x-role": "anything"}).json()["role"] == "reader"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"roles": ()}, "at least one role"),
        ({"roles": "admin"}, "not one string"),
        ({"roles": ("admin", " ")}, "non-blank role names"),
        ({"change": ("owner",)}, "outside `roles`"),
        ({"decide": ("owner",)}, "outside `roles`"),
        ({"unnamed_role": "owner"}, "`unnamed_role` must be one of `roles`"),
        ({"role_claim": " "}, "`role_claim`"),
    ],
)
def test_operator_roles_refuse_a_rule_that_cant_work(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OperatorRoles(**options)


def test_mount_cayu_hosts_use_the_same_part() -> None:
    auth = _cloud()
    server = FastAPI()
    mount_cayu(server, CayuApp(), dashboard=False, access=AuthenticatedAccess(dependency=auth))
    server.include_router(_app_routes(operator_person_dependency(auth)))
    client = TestClient(server)

    member = _PROXY.headers("mel", role="member", name="Mel")
    assert client.get("/app/whoami", headers=member).json()["name"] == "Mel"
    assert client.get("/cayu/api/app", headers=member).json()["role"] == "member"
    assert client.get("/app/whoami", headers=_basic()).json()["role"] == "admin"


def test_mount_cayu_local_development_uses_the_local_operator() -> None:
    server = FastAPI()
    mount_cayu(
        server,
        CayuApp(),
        dashboard=False,
        access=OpenAccess(trusted_local_development=True),
    )
    server.include_router(_app_routes(operator_person_dependency(None, development=True)))

    with TestClient(server, **_LOOPBACK) as client:
        assert client.get("/app/whoami").json()["subject"] == f"dev:{getpass.getuser()}"


def test_the_local_operator_names_its_kind() -> None:
    assert LocalOperator("cli").subject == f"cli:{getpass.getuser()}"
    with pytest.raises(ValueError, match="kind"):
        LocalOperator(" ")
