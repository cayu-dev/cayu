"""`cayu.server.operator_check`: an app's own check steps, served behind the operator login."""

from __future__ import annotations

import asyncio
import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cayu.server.auth import AuthConfigurationError, BasicAuth
from cayu.server.operator_check import CheckFailed, OperatorCheck

_LOGIN = {"CAYU_OPERATOR_USERNAME": "operator", "CAYU_OPERATOR_PASSWORD": "operator-password"}


def _basic(username: str, password: str) -> dict[str, str]:
    return {
        "Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    }


def _check() -> OperatorCheck:
    check = OperatorCheck()

    @check.step("ai_gateway")
    async def gateway() -> str:
        return "Answered in 12 ms"

    @check.step("workflow")
    def workflow() -> None:
        return None

    return check


@pytest.fixture
def operator_login(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _LOGIN.items():
        monkeypatch.setenv(name, value)


def _client(check: OperatorCheck, **options: object) -> TestClient:
    app = FastAPI()
    app.include_router(check.router(**options))  # type: ignore[arg-type]
    return TestClient(app)


def test_the_route_answers_every_step_to_the_operator_login(operator_login: None) -> None:
    client = _client(_check())

    response = client.post("/operator-check", headers=_basic("operator", "operator-password"))

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert (body["ok"], body["summary"]) == (True, "2 of 2 checks passed.")
    assert [(step["name"], step["ok"], step["detail"]) for step in body["checks"]] == [
        ("ai_gateway", True, "Answered in 12 ms"),
        ("workflow", True, None),
    ]
    assert all(type(step["duration_ms"]) is int for step in body["checks"])


def test_anyone_without_the_operator_login_is_refused_before_a_step_runs(
    operator_login: None,
) -> None:
    ran: list[str] = []
    check = OperatorCheck()
    check.add("workflow", lambda: ran.append("ran"))
    client = _client(check)

    for headers in (
        {},
        _basic("operator", "wrong"),
        {"X-Cayu-Identity": "a-signed-in-person"},
        {"Authorization": "Bearer operator-password"},
    ):
        assert client.post("/operator-check", headers=headers).status_code == 401
    assert ran == []


def test_a_failed_step_answers_503_and_the_next_still_runs(operator_login: None) -> None:
    check = OperatorCheck()

    @check.step("ai_gateway")
    def gateway() -> None:
        raise CheckFailed("The gateway answered 502.")

    @check.step("database")
    def database() -> None:
        raise RuntimeError("password=hunter2 in a connection string")

    check.add("workflow", lambda: "done")
    response = _client(check).post(
        "/operator-check", headers=_basic("operator", "operator-password")
    )

    assert response.status_code == 503
    body = response.json()
    assert (body["ok"], body["summary"]) == (False, "1 of 3 checks passed.")
    assert [(step["ok"], step["detail"]) for step in body["checks"]] == [
        (False, "The gateway answered 502."),
        # Only the type: an unexpected error's text can carry secrets, so it stays in the log.
        (False, "RuntimeError (see the app's log)."),
        (True, "done"),
    ]
    assert "hunter2" not in response.text


def test_path_method_and_auth_are_the_apps_choice() -> None:
    client = _client(
        _check(),
        path="/ops/deep-health",
        method="GET",
        auth=BasicAuth(username="ops", password="other-password"),
    )

    assert (
        client.get("/ops/deep-health", headers=_basic("ops", "other-password")).status_code == 200
    )
    assert (
        client.post("/ops/deep-health", headers=_basic("ops", "other-password")).status_code == 405
    )
    assert client.get("/operator-check", headers=_basic("ops", "other-password")).status_code == 404


def test_the_default_route_refuses_to_start_without_the_operator_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _LOGIN:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(AuthConfigurationError):
        _check().router()


def test_a_check_without_steps_does_not_pass() -> None:
    report = asyncio.run(OperatorCheck().run())

    assert (report.ok, report.summary) == (False, "No checks are registered.")


def test_step_names_are_unique_and_short() -> None:
    check = OperatorCheck()
    check.add("gateway", lambda: None)

    with pytest.raises(ValueError, match="already added"):
        check.add("gateway", lambda: None)
    for name in ("", " gateway", "x" * 101):
        with pytest.raises(ValueError, match="1 to 100 characters"):
            check.add(name, lambda: None)


def test_run_reports_without_a_route() -> None:
    report = asyncio.run(_check().run())

    assert report.ok
    assert report.to_dict()["summary"] == "2 of 2 checks passed."


def test_a_long_detail_is_shortened() -> None:
    check = OperatorCheck()
    check.add("verbose", lambda: "x" * 2000)

    (step,) = asyncio.run(check.run()).checks

    assert len(step.detail or "") == 500
    assert (step.detail or "").endswith("...")
