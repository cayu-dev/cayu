from __future__ import annotations

import importlib
import os
import secrets
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu._operator_credentials import (
    ENVIRONMENT_OPERATOR_AUTH_TARGET,
    OPERATOR_PASSWORD_VARIABLE,
    OPERATOR_USERNAME_VARIABLE,
)
from cayu.applications import CayuApp
from cayu.server import AuthConfigurationError, BasicAuth, ServerConfig, create_server

_ENVIRONMENT = {
    OPERATOR_USERNAME_VARIABLE: "operator",
    OPERATOR_PASSWORD_VARIABLE: "secret-password",
}


def _import_environment_auth():
    sys.modules.pop("cayu.server.environment_auth", None)
    try:
        return importlib.import_module("cayu.server.environment_auth")
    finally:
        sys.modules.pop("cayu.server.environment_auth", None)


def test_from_environment_builds_basic_auth_that_protects_the_control_plane() -> None:
    auth = BasicAuth.from_environment(environ=_ENVIRONMENT, tenant="tenant-a")

    assert type(auth) is BasicAuth
    assert auth.username == "operator"
    assert auth.tenant == "tenant-a"
    client = TestClient(create_server(CayuApp(), config=ServerConfig.protected(auth)))
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions", auth=("operator", "wrong")).status_code == 401
    assert client.get("/api/sessions", auth=("operator", "secret-password")).status_code == 200


def test_from_environment_compares_credentials_in_constant_time(monkeypatch) -> None:
    compared: list[tuple[bytes, bytes]] = []
    real_compare = secrets.compare_digest

    def recording_compare(left: bytes, right: bytes) -> bool:
        compared.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr("cayu.server.auth.secrets.compare_digest", recording_compare)
    client = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.protected(BasicAuth.from_environment(environ=_ENVIRONMENT)),
        )
    )

    assert client.get("/api/sessions", auth=("operator", "guess")).status_code == 401
    assert (b"guess", b"secret-password") in compared


def test_from_environment_reads_the_process_environment_by_default(monkeypatch) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    assert BasicAuth.from_environment().username == "operator"


@pytest.mark.parametrize(
    ("environment", "missing"),
    [
        ({}, f"{OPERATOR_USERNAME_VARIABLE} and {OPERATOR_PASSWORD_VARIABLE} are"),
        ({OPERATOR_USERNAME_VARIABLE: "operator"}, f"{OPERATOR_PASSWORD_VARIABLE} is"),
        (
            {OPERATOR_USERNAME_VARIABLE: "operator", OPERATOR_PASSWORD_VARIABLE: ""},
            f"{OPERATOR_PASSWORD_VARIABLE} is",
        ),
        (
            {OPERATOR_USERNAME_VARIABLE: "   ", OPERATOR_PASSWORD_VARIABLE: "secret-password"},
            f"{OPERATOR_USERNAME_VARIABLE} is",
        ),
    ],
)
def test_from_environment_fails_closed_with_actionable_guidance(
    environment: dict[str, str],
    missing: str,
) -> None:
    with pytest.raises(AuthConfigurationError) as raised:
        BasicAuth.from_environment(environ=environment)

    message = str(raised.value)
    assert isinstance(raised.value, ValueError)
    assert f"{missing} unset or empty" in message
    assert "does not fall back to open access" in message
    assert "`cayu serve --dev`" in message
    assert "`cayu cloud service credentials --application APP`" in message
    assert "secret-password" not in message


def test_from_environment_names_custom_variables_without_cloud_guidance() -> None:
    with pytest.raises(AuthConfigurationError) as raised:
        BasicAuth.from_environment("ADMIN_USER", "ADMIN_PASSWORD", environ={"ADMIN_USER": "a"})

    message = str(raised.value)
    assert "ADMIN_PASSWORD is unset or empty" in message
    assert "Set both ADMIN_USER and ADMIN_PASSWORD" in message
    assert "cayu cloud" not in message

    auth = BasicAuth.from_environment(
        "ADMIN_USER",
        "ADMIN_PASSWORD",
        environ={"ADMIN_USER": "admin", "ADMIN_PASSWORD": "pw"},
        realm="Admin",
    )
    assert (auth.username, auth.realm) == ("admin", "Admin")


@pytest.mark.parametrize(
    ("environment", "variable", "reason"),
    [
        (
            {OPERATOR_USERNAME_VARIABLE: "op:erator", OPERATOR_PASSWORD_VARIABLE: "pw-value"},
            OPERATOR_USERNAME_VARIABLE,
            "must not contain a colon",
        ),
        (
            {OPERATOR_USERNAME_VARIABLE: "operator", OPERATOR_PASSWORD_VARIABLE: " pw-value"},
            OPERATOR_PASSWORD_VARIABLE,
            "whitespace",
        ),
    ],
)
def test_from_environment_names_the_invalid_variable_without_its_value(
    environment: dict[str, str],
    variable: str,
    reason: str,
) -> None:
    with pytest.raises(AuthConfigurationError) as raised:
        BasicAuth.from_environment(environ=environment)

    message = str(raised.value)
    assert message.startswith(f"{variable} cannot be used as a Basic auth")
    assert reason in message
    assert "op:erator" not in message
    assert "pw-value" not in message


@pytest.mark.parametrize(
    ("username_variable", "password_variable"),
    [("", "PASSWORD"), ("USER", " PASSWORD"), ("SAME", "SAME")],
)
def test_from_environment_rejects_unusable_variable_names(
    username_variable: str,
    password_variable: str,
) -> None:
    with pytest.raises(ValueError, match="variable"):
        BasicAuth.from_environment(username_variable, password_variable, environ={})


def test_ready_made_target_reads_operator_credentials_when_imported(monkeypatch) -> None:
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    module_name, attribute = ENVIRONMENT_OPERATOR_AUTH_TARGET.split(":")

    assert module_name == "cayu.server.environment_auth"
    auth = getattr(_import_environment_auth(), attribute)
    assert type(auth) is BasicAuth
    assert auth.username == "operator"


def test_ready_made_target_refuses_to_import_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv(OPERATOR_USERNAME_VARIABLE, raising=False)
    monkeypatch.setenv(OPERATOR_PASSWORD_VARIABLE, "secret-password")

    with pytest.raises(AuthConfigurationError, match=f"{OPERATOR_USERNAME_VARIABLE} is unset"):
        _import_environment_auth()


def test_cayu_server_does_not_read_operator_credentials_on_import() -> None:
    script = "import sys, cayu.server; assert 'cayu.server.environment_auth' not in sys.modules"
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {OPERATOR_USERNAME_VARIABLE, OPERATOR_PASSWORD_VARIABLE}
    }

    subprocess.run([sys.executable, "-c", script], check=True, env=environment, timeout=60)
