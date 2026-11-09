"""Generated service projects: who closes the stores ``build_service()`` opens.

``build_service()`` opens its stores and hands them to ``build_app()`` through
``owned_resources``, so shutting the service down closes them. Stores a caller
injects into ``build_service()`` stay open for that caller to close.
"""

from __future__ import annotations

import asyncio
import importlib
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cayu.cli.project import project_context
from cayu.cli.scaffold import project_files
from cayu.evals import ScriptedModelProvider
from cayu.server import ServiceMode
from cayu.storage import SQLiteProductOperationStore, SQLiteSessionStore, SQLiteTaskStore


@pytest.fixture(autouse=True)
def _local_database(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CAYU_DATABASE_URL", "CAYU_REQUIRE_POSTGRES", "CAYU_DATABASE_DIRECT_URL"):
        monkeypatch.delenv(name, raising=False)


def _service_project(root: Path) -> Path:
    for relative, content in project_files("store-service", preset="service").items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _open(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return False
    return True


def _serve_and_stop(service) -> None:
    # Entering and leaving the ASGI lifespan is the service's real shutdown path.
    with TestClient(service.asgi_app) as client:
        assert client.get("/health").json() == {"ok": True}


def test_service_shutdown_closes_the_stores_build_service_opened(tmp_path: Path) -> None:
    project = _service_project(tmp_path / "consumer")
    with project_context(project):
        service = importlib.import_module("service").build_service(
            mode=ServiceMode.DEVELOPMENT, provider=ScriptedModelProvider([])
        )
        app = service.cayu_app
        assert (project / "data" / "cayu.db").is_file()
        assert _open(app.session_store._connection)
        assert _open(service.product_store._connection)
        _serve_and_stop(service)

    outcome = app.shutdown_outcome
    assert outcome is not None and outcome.settled, outcome
    assert outcome.owned_resources == "released"
    assert not _open(app.session_store._connection)
    assert not _open(app.task_store._connection)
    assert not _open(service.product_store._connection)


def test_service_shutdown_leaves_injected_stores_open(tmp_path: Path) -> None:
    project = _service_project(tmp_path / "consumer")
    database = str(tmp_path / "injected.db")
    session_store = SQLiteSessionStore(database)
    task_store = SQLiteTaskStore(database)
    product_store = SQLiteProductOperationStore(database)
    try:
        with project_context(project):
            service = importlib.import_module("service").build_service(
                mode=ServiceMode.DEVELOPMENT,
                provider=ScriptedModelProvider([]),
                session_store=session_store,
                task_store=task_store,
                product_store=product_store,
            )
            assert service.cayu_app.session_store is session_store
            assert service.product_store is product_store
            _serve_and_stop(service)

        outcome = service.cayu_app.shutdown_outcome
        assert outcome is not None and outcome.settled, outcome
        assert _open(session_store._connection)
        assert _open(task_store._connection)
        assert _open(product_store._connection)
    finally:

        async def close() -> None:
            for store in (session_store, task_store, product_store):
                await store.close()

        asyncio.run(close())
