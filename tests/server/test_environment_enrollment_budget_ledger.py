"""A fork controller enrolled from the environment on an app built from application stores.

The app passes only the stores' session and task stores to ``CayuApp``. Its
budget ledger is the durable one ``open_application_stores`` built in the same
database, so enrolling a controller that runs models needs no ledger code.
"""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.core.test_session_fork_overlays import PostJournal

from cayu import AgentSpec, CayuApp
from cayu.server import (
    AuthenticatedAccess,
    DashboardConfig,
    ServerConfig,
    create_server,
    mount_cayu,
)
from cayu.server.auth import AuthContext
from cayu.server.fork_controller import FORK_CONTROLLER_ENVIRONMENT_VARIABLES
from cayu.storage import PostgresBudgetLedger, SQLiteBudgetLedger
from cayu.storage.application import open_application_stores

AGENT = "finance-monthend"
TOKEN = "compass-" + "a" * 40

# The per-fork cap variables are read once cayu-tech/cayu#2554 lands; until
# then they are ignored, and ALLOW_MODEL_RUNS=true enrolls without a cap.
MODEL_RUNS_ENVIRONMENT = {
    "CAYU_FORK_CONTROLLER_NAME": "compass",
    "CAYU_FORK_CONTROLLER_AGENTS": AGENT,
    "CAYU_FORK_CONTROLLER_TOKEN_SHA256": hashlib.sha256(TOKEN.encode("ascii")).hexdigest(),
    "CAYU_FORK_CONTROLLER_HOST_TOOLS": "post_journal",
    "CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS": "true",
    "CAYU_FORK_CONTROLLER_FORK_BUDGET_USD": "2",
    "CAYU_FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS": "200000",
    "CAYU_FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS": "16000",
}


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        *FORK_CONTROLLER_ENVIRONMENT_VARIABLES,
        "CAYU_DATABASE_URL",
        "CAYU_DATABASE_DIRECT_URL",
        "CAYU_DATABASE_POOL_MAX",
        "CAYU_REQUIRE_POSTGRES",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in MODEL_RUNS_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


async def _operator(request) -> AuthContext:
    del request
    return AuthContext(subject="operator")


def _app(stores, **kwargs) -> CayuApp:
    app = CayuApp(
        session_store=stores.session_store,
        task_store=stores.task_store,
        enable_logging=False,
        **kwargs,
    )
    app.register_agent(AgentSpec(name=AGENT, model="fake-model"), tools=[PostJournal()])
    return app


def _closing(*resources):
    """A host lifespan that closes ``resources`` on the server's own event loop."""

    @asynccontextmanager
    async def lifespan(server):
        del server
        try:
            yield
        finally:
            for resource in resources:
                await resource.close()

    return lifespan


def _enrollment(client: TestClient, path: str) -> dict:
    response = client.get(path, headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(params=["sqlite", "postgres"])
def database_url(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str | None:
    if request.param == "sqlite":
        return None
    url = request.getfixturevalue("migrated_postgres_url")
    monkeypatch.setenv("CAYU_DATABASE_URL", url)
    return url


def test_create_server_enrolls_model_runs_on_the_stores_durable_ledger(
    database_url: str | None, tmp_path: Path
) -> None:
    stores = open_application_stores(database_url, sqlite_path=tmp_path / "cayu.db")
    app = _app(stores)
    expected = SQLiteBudgetLedger if database_url is None else PostgresBudgetLedger
    assert isinstance(app.budget_ledger, expected)
    assert app.budget_ledger is stores.budget_ledger
    server = create_server(
        app,
        config=ServerConfig.protected(_operator, dashboard=DashboardConfig(enabled=False)),
        fastapi_options={"lifespan": _closing(stores)},
    )
    with TestClient(server) as client:
        assert _enrollment(client, "/api/fork-controller/enrollment")["allow_model_runs"]


def test_mount_cayu_enrolls_model_runs_on_the_stores_durable_ledger(
    database_url: str | None, tmp_path: Path
) -> None:
    stores = open_application_stores(database_url, sqlite_path=tmp_path / "cayu.db")
    app = _app(stores)
    host = FastAPI(lifespan=_closing(stores))
    mount_cayu(host, app, access=AuthenticatedAccess(dependency=_operator), dashboard=False)
    with TestClient(host) as client:
        assert _enrollment(client, "/cayu/api/fork-controller/enrollment")["allow_model_runs"]
    assert app.budget_ledger is stores.budget_ledger


def test_an_app_supplied_ledger_still_wins(tmp_path: Path) -> None:
    stores = open_application_stores(None, sqlite_path=tmp_path / "cayu.db")
    own = SQLiteBudgetLedger(tmp_path / "own-budgets.db")
    app = _app(stores, budget_ledger=own)
    server = create_server(
        app,
        config=ServerConfig.protected(_operator, dashboard=DashboardConfig(enabled=False)),
        fastapi_options={"lifespan": _closing(own, stores)},
    )
    with TestClient(server) as client:
        assert _enrollment(client, "/api/fork-controller/enrollment")["allow_model_runs"]
    assert app.budget_ledger is own
