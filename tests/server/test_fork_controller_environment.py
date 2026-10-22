"""A fork controller enrolled from the reserved ``CAYU_FORK_CONTROLLER_*`` variables."""

from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.core.test_session_fork_overlays import PostJournal

from cayu import AgentSpec, CayuApp
from cayu.fork_controllers import ForkController
from cayu.server import (
    AuthenticatedAccess,
    DashboardConfig,
    ServerApiConfig,
    ServerConfig,
    create_server,
    mount_cayu,
)
from cayu.server.auth import AuthConfigurationError, AuthContext
from cayu.server.fork_controller import (
    FORK_CONTROLLER_ENVIRONMENT_VARIABLES,
    ForkControllerBearerAuth,
    environment_fork_controller_router,
    fork_controller_from_environment,
)

AGENT = "finance-monthend"
TOKEN = "compass-" + "a" * 40
ROTATED = "compass-" + "b" * 40
OPERATOR = "operator-" + "c" * 40


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _environment(**overrides: str) -> dict[str, str]:
    environment = {
        "CAYU_FORK_CONTROLLER_NAME": "compass",
        "CAYU_FORK_CONTROLLER_AGENTS": AGENT,
        "CAYU_FORK_CONTROLLER_TOKEN_SHA256": f"{_digest(TOKEN)},{_digest(ROTATED)}",
    }
    environment.update(overrides)
    return {name: value for name, value in environment.items() if value != "<unset>"}


def _app() -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name=AGENT, model="fake-model"),
        tools=[PostJournal()],
    )
    app.register_agent(AgentSpec(name="unenrolled", model="fake-model"))
    return app


async def _operator(request) -> AuthContext:
    if request.headers.get("authorization") != f"Bearer {OPERATOR}":
        from fastapi import HTTPException

        raise HTTPException(status_code=401)
    return AuthContext(subject="operator")


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def environ(monkeypatch: pytest.MonkeyPatch):
    for name in FORK_CONTROLLER_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    def apply(values: dict[str, str]) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    return apply


def test_no_variables_enroll_nothing_and_mount_nothing(environ) -> None:
    assert fork_controller_from_environment({}) is None
    app = _app()
    host = FastAPI()
    mount_cayu(host, app, access=AuthenticatedAccess(dependency=_operator), dashboard=False)

    assert app._agent_registry.fork_controllers == ()
    response = TestClient(host).get("/cayu/api/fork-controller/enrollment", headers=_bearer(TOKEN))
    assert response.status_code == 404


def test_variables_build_the_controller_with_safe_defaults() -> None:
    configured = fork_controller_from_environment(_environment())

    assert configured is not None
    assert configured.controller == ForkController(
        name="compass", subjects=("fork-controller:compass",), agents=(AGENT,)
    )
    assert configured.controller.allow_model_runs is False
    assert configured.controller.host_tools == ()
    assert repr(configured) == "EnvironmentForkController(controller='compass')"
    assert _digest(TOKEN) not in repr(configured.authenticate)

    explicit = fork_controller_from_environment(
        _environment(
            CAYU_FORK_CONTROLLER_AGENTS=f" {AGENT} , unenrolled",
            CAYU_FORK_CONTROLLER_MODE="observe",
            CAYU_FORK_CONTROLLER_HOST_TOOLS="post_journal",
            CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS="true",
            CAYU_FORK_CONTROLLER_FORK_BUDGET_USD="2.50",
            CAYU_FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS="200000",
            CAYU_FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS="16000",
        )
    )
    assert explicit is not None
    assert explicit.controller.agents == (AGENT, "unenrolled")
    assert explicit.controller.host_tools == ("post_journal",)
    assert explicit.controller.allow_model_runs is True
    cap = explicit.controller.fork_budget_limit
    assert cap is not None and cap.max_estimated_cost == Decimal("2.50")
    assert cap.reservation is not None
    assert (cap.reservation.max_input_tokens, cap.reservation.max_output_tokens) == (
        200_000,
        16_000,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # Model runs never start without a cap.
        ({"CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS": "true"}, "requires fork_budget_limit"),
        (
            {
                "CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS": "true",
                "CAYU_FORK_CONTROLLER_FORK_BUDGET_USD": "2",
            },
            "needs all of",
        ),
        (
            {
                "CAYU_FORK_CONTROLLER_FORK_BUDGET_USD": "2",
                "CAYU_FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS": "-1",
                "CAYU_FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS": "16",
            },
            "whole number of tokens",
        ),
        (
            {
                "CAYU_FORK_CONTROLLER_FORK_BUDGET_USD": "lots",
                "CAYU_FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS": "100",
                "CAYU_FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS": "16",
            },
            "per-fork cap is invalid",
        ),
    ],
)
def test_model_runs_from_the_environment_need_a_complete_cap(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        fork_controller_from_environment(_environment(**overrides))


def test_a_total_from_the_environment_shares_the_per_fork_reservation() -> None:
    variables = {
        "CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS": "true",
        "CAYU_FORK_CONTROLLER_FORK_BUDGET_USD": "2.50",
        "CAYU_FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS": "200000",
        "CAYU_FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS": "16000",
    }
    without = fork_controller_from_environment(_environment(**variables))
    assert without is not None and without.controller.total_budget_limit is None
    configured = fork_controller_from_environment(
        _environment(**variables, CAYU_FORK_CONTROLLER_TOTAL_BUDGET_USD="20")
    )
    assert configured is not None
    total = configured.controller.total_budget_limit
    cap = configured.controller.fork_budget_limit
    assert total is not None and total.max_estimated_cost == Decimal("20")
    assert cap is not None and total.reservation == cap.reservation
    with pytest.raises(ValueError, match="needs the per-fork cap variables"):
        fork_controller_from_environment(_environment(CAYU_FORK_CONTROLLER_TOTAL_BUDGET_USD="20"))
    with pytest.raises(ValueError, match="controller total is invalid"):
        fork_controller_from_environment(
            _environment(**variables, CAYU_FORK_CONTROLLER_TOTAL_BUDGET_USD="lots")
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"CAYU_FORK_CONTROLLER_NAME": "<unset>"},
        {"CAYU_FORK_CONTROLLER_NAME": " "},
        {"CAYU_FORK_CONTROLLER_NAME": "a@fork-controller:b"},
        {"CAYU_FORK_CONTROLLER_AGENTS": "<unset>"},
        {"CAYU_FORK_CONTROLLER_AGENTS": ""},
        {"CAYU_FORK_CONTROLLER_AGENTS": f"{AGENT},,other"},
        {"CAYU_FORK_CONTROLLER_AGENTS": f"{AGENT},{AGENT}"},
        {"CAYU_FORK_CONTROLLER_TOKEN_SHA256": "<unset>"},
        {"CAYU_FORK_CONTROLLER_TOKEN_SHA256": ""},
        {"CAYU_FORK_CONTROLLER_TOKEN_SHA256": _digest(TOKEN).upper()},
        {"CAYU_FORK_CONTROLLER_TOKEN_SHA256": _digest(TOKEN)[:-1]},
        {"CAYU_FORK_CONTROLLER_TOKEN_SHA256": TOKEN},
        {"CAYU_FORK_CONTROLLER_TOKEN_SHA256": f"{_digest(TOKEN)},{_digest(TOKEN)}"},
        {"CAYU_FORK_CONTROLLER_TOKEN_SHA256": f"{_digest(TOKEN)},"},
        {"CAYU_FORK_CONTROLLER_MODE": "act"},
        {"CAYU_FORK_CONTROLLER_MODE": ""},
        {"CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS": "yes"},
        {"CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS": "True"},
        {"CAYU_FORK_CONTROLLER_HOST_TOOLS": "post_journal,"},
    ],
)
def test_malformed_configuration_fails_closed_without_echoing_digests(overrides) -> None:
    with pytest.raises(ValueError) as raised:
        fork_controller_from_environment(_environment(**overrides))

    message = str(raised.value)
    assert _digest(TOKEN) not in message
    assert _digest(TOKEN)[:16] not in message
    assert TOKEN not in message
    assert raised.value.__cause__ is None


def test_any_reserved_variable_without_a_name_fails_closed() -> None:
    for name in FORK_CONTROLLER_ENVIRONMENT_VARIABLES[1:]:
        with pytest.raises(ValueError, match="CAYU_FORK_CONTROLLER_NAME must be set"):
            fork_controller_from_environment({name: "x"})


def test_digest_auth_matches_the_token_auth() -> None:
    import asyncio

    from starlette.requests import Request

    def request(token: str) -> Request:
        return Request(
            {"type": "http", "headers": [(b"authorization", f"Bearer {token}".encode())]}
        )

    by_digest = ForkControllerBearerAuth.from_token_digests({_digest(TOKEN): "svc:compass"})
    by_token = ForkControllerBearerAuth({TOKEN: "svc:compass"})

    assert by_digest._entries == by_token._entries
    assert asyncio.run(by_digest(request(TOKEN))) == "svc:compass"
    assert asyncio.run(by_digest(request(ROTATED))) == ""
    for digests in ({}, {_digest(TOKEN): " svc"}, {"0" * 63: "svc"}, {"G" * 64: "svc"}):
        with pytest.raises(AuthConfigurationError):
            ForkControllerBearerAuth.from_token_digests(digests)


def test_mount_cayu_serves_the_environment_controller_with_its_own_bearer(environ) -> None:
    environ(_environment(CAYU_FORK_CONTROLLER_HOST_TOOLS="post_journal"))
    app = _app()
    host = FastAPI()
    mount_cayu(host, app, access=AuthenticatedAccess(dependency=_operator), dashboard=False)
    client = TestClient(host)
    path = "/cayu/api/fork-controller/enrollment"

    for token in (TOKEN, ROTATED):
        response = client.get(path, headers=_bearer(token))
        assert response.status_code == 200
        assert response.json() == {
            "controller": "compass",
            "subject": "fork-controller:compass",
            "mode": "observe",
            "agents": [AGENT],
            "host_tools": ["post_journal"],
            "served_tools": [],
            "allow_model_runs": False,
            "fork_budget_limit": None,
            "total_budget_limit": None,
        }
    assert client.get(path, headers=_bearer(OPERATOR)).status_code == 401
    assert client.get(path, headers=_bearer("x" * 48)).status_code == 401
    assert client.get(path).status_code == 401
    # The controller's token is not a Developer dashboard login.
    assert client.get("/cayu/api/sessions", headers=_bearer(TOKEN)).status_code == 401


def test_create_server_serves_it_under_the_api_path(environ) -> None:
    environ(_environment())
    server = create_server(
        _app(),
        config=ServerConfig.protected(_operator, dashboard=DashboardConfig(enabled=False)),
    )
    client = TestClient(server)

    response = client.get("/api/fork-controller/enrollment", headers=_bearer(TOKEN))
    assert response.status_code == 200
    assert response.json()["controller"] == "compass"


def test_a_disabled_api_serves_no_controller_routes(environ) -> None:
    environ(_environment())
    app = _app()
    server = create_server(
        app,
        config=ServerConfig.protected(
            _operator,
            api=ServerApiConfig(enabled=False),
            dashboard=DashboardConfig(enabled=False),
        ),
    )

    response = TestClient(server).get("/api/fork-controller/enrollment", headers=_bearer(TOKEN))
    assert response.status_code == 404
    assert app._agent_registry.fork_controllers == ()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"CAYU_FORK_CONTROLLER_AGENTS": "missing-agent"}, "cannot be enrolled"),
        ({"CAYU_FORK_CONTROLLER_HOST_TOOLS": "wire_money"}, "cannot be enrolled"),
    ],
)
def test_enrollment_errors_stop_the_mount(environ, overrides, message) -> None:
    environ(_environment(**overrides))
    app = _app()
    host = FastAPI()

    with pytest.raises(ValueError, match=message):
        mount_cayu(host, app, access=AuthenticatedAccess(dependency=_operator), dashboard=False)
    assert app._agent_registry.fork_controllers == ()
    assert len(host.router.routes) == len(FastAPI().router.routes)


def test_a_code_enrollment_conflict_fails_closed_and_an_identical_one_is_reused(environ) -> None:
    environ(_environment())
    app = _app()
    app.enroll_fork_controller(
        ForkController(name="compass", subjects=("svc:other",), agents=(AGENT,))
    )
    with pytest.raises(ValueError, match="already enrolled differently"):
        environment_fork_controller_router(app)

    same = _app()
    configured = fork_controller_from_environment(_environment())
    assert configured is not None
    same.enroll_fork_controller(configured.controller)
    assert environment_fork_controller_router(same) is not None
    # Mounting twice reuses the one enrollment.
    assert environment_fork_controller_router(same) is not None
    assert len(same._agent_registry.fork_controllers) == 1

    subject_clash = _app()
    subject_clash.enroll_fork_controller(
        ForkController(name="other", subjects=("fork-controller:compass",), agents=(AGENT,))
    )
    with pytest.raises(ValueError, match="cannot be enrolled"):
        environment_fork_controller_router(subject_clash)
