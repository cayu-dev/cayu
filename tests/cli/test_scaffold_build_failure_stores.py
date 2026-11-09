"""Generated factories close the stores they opened when building raises.

``build_app()`` (agent, coding) and ``build_service()`` (service) open stores and
hand them to the app. If a later step raises, no app or service exists to close
them, so the factory closes them before the error propagates. Resources a
caller hands to ``build_app()`` stay with that caller when it raises.
"""

from __future__ import annotations

import asyncio
import importlib
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest
from tests.coding_scaffold_fixtures import admit_fixture_image

from cayu.cli.project import project_context
from cayu.cli.scaffold import project_files
from cayu.evals import ScriptedModelProvider
from cayu.server import ServiceMode


@pytest.fixture(autouse=True)
def _local_database(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CAYU_DATABASE_URL", "CAYU_REQUIRE_POSTGRES", "CAYU_DATABASE_DIRECT_URL"):
        monkeypatch.delenv(name, raising=False)


class _Boom(RuntimeError):
    pass


def _raise(*args: object, **kwargs: object) -> None:
    raise _Boom("build step failed")


def _write_project(root: Path, files: dict[str, str]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _git_repository(path: Path) -> Path:
    path.mkdir()
    (path / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    for command in (
        ["init", "-q", "-b", "main"],
        ["add", "example.py"],
        [
            *("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"),
            *("commit", "-q", "-m", "baseline"),
        ],
    ):
        subprocess.run(["git", "-C", str(path), *command], check=True, capture_output=True)
    return path


def _capture(monkeypatch: pytest.MonkeyPatch, module: object, name: str) -> list[Any]:
    opened: list[Any] = []
    build = getattr(module, name)

    def capturing(*args: object, **kwargs: object) -> object:
        stores = build(*args, **kwargs)
        opened.append(stores)
        return stores

    monkeypatch.setattr(module, name, capturing)
    return opened


def _is_open(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return False
    return True


class _HandedOver:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_agent_build_app_closes_its_stores_and_leaves_handed_over_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAYU_MEMORY_EVIDENCE_KEY", "k" * 64)
    project = _write_project(tmp_path / "consumer", project_files("fail-agent"))
    handed_over = _HandedOver()
    with project_context(project):
        app_module = importlib.import_module("app")
        opened = _capture(monkeypatch, app_module, "build_stores")
        monkeypatch.setattr(app_module, "register_agents", _raise)
        with pytest.raises(_Boom):
            app_module.build_app(provider=ScriptedModelProvider([]), owned_resources=(handed_over,))
    [stores] = opened
    assert stores.configured is not None
    assert not _is_open(stores.session_store._connection)
    assert not handed_over.closed


def test_service_build_service_closes_its_stores_when_the_service_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _write_project(tmp_path / "consumer", project_files("fail-service", preset="service"))
    with project_context(project):
        service_module = importlib.import_module("service")
        opened = _capture(monkeypatch, service_module, "build_stores")
        monkeypatch.setattr(service_module, "create_agent_service", _raise)
        with pytest.raises(_Boom):
            service_module.build_service(
                mode=ServiceMode.DEVELOPMENT, provider=ScriptedModelProvider([])
            )
    [stores] = opened
    assert stores.configured is not None
    assert not _is_open(stores.session_store._connection)
    assert not _is_open(stores.product_store._connection)


@pytest.mark.parametrize("execution", [None, "docker"], ids=["local", "docker"])
def test_coding_build_app_closes_its_stores_inside_a_running_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution: str | None
) -> None:
    project = _write_project(
        tmp_path / "consumer",
        project_files(
            "fail-coder",
            preset="coding",
            execution=execution,
            coding_toolchain="python" if execution == "docker" else None,
        ),
    )
    source = _git_repository(tmp_path / "source")
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        monkeypatch.setattr(operations, "_verify_coding_dependencies", lambda root: None)
        if execution == "docker":
            admit_fixture_image(monkeypatch, operations)
        opened = _capture(monkeypatch, operations, "build_coding_stores")
        monkeypatch.setattr(operations, "register_coding_agents", _raise)
        build_app = importlib.import_module("app").build_app

        async def build() -> None:
            # Called from async code, as tests and services often do.
            build_app(provider=ScriptedModelProvider([]), workspace_root=source)

        with pytest.raises(_Boom):
            asyncio.run(build())
    [stores] = opened
    assert stores.configured is not None
    assert not _is_open(stores.session_store._connection)


def test_coding_product_factory_closes_the_app_when_the_front_door_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _write_project(
        tmp_path / "consumer",
        project_files(
            "fail-product", preset="coding", execution="docker", coding_toolchain="python"
        ),
    )
    source = _git_repository(tmp_path / "source")
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        monkeypatch.setattr(operations, "_verify_coding_dependencies", lambda root: None)
        admit_fixture_image(monkeypatch, operations)
        opened = _capture(monkeypatch, operations, "build_coding_stores")
        app_module = importlib.import_module("app")
        monkeypatch.setattr(app_module, "CodingProductApplication", _raise)
        with pytest.raises(_Boom):
            app_module.build_coding_product_application(
                provider=ScriptedModelProvider([]), workspace_root=source
            )
    [stores] = opened
    assert stores.configured is not None
    assert not _is_open(stores.session_store._connection)
