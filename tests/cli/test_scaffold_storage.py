"""Generated storage selection agrees with the CLI's store resolution."""

from __future__ import annotations

import asyncio
import importlib
import itertools
import os
from pathlib import Path

import pytest

from cayu import SQLiteKnowledgeStore, SQLiteSessionStore, SQLiteTaskStore
from cayu.cli import main
from cayu.cli.project import project_context
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_check import check_declared_scaffold_source
from cayu.cli.session import _open_read_only_store
from cayu.cli.store_targets import resolve_session_store_target
from cayu.storage.targets import PostgresRequiredError

_POSTGRES_URL = "postgresql://app:secret@127.0.0.1:1/app"
_LOCAL_DATABASE = {"agent": "data/cayu.db", "coding": ".cayu/runtime/cayu.db"}


def _generate(tmp_path: Path, preset: str, capsys: pytest.CaptureFixture[str]) -> Path:
    assert main(["new", "app", "--preset", preset, "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    return tmp_path / "app"


@pytest.fixture(autouse=True)
def _database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CAYU_DATABASE_URL", "CAYU_REQUIRE_POSTGRES", "CAYU_DATABASE_DIRECT_URL"):
        monkeypatch.delenv(name, raising=False)


async def _close(*stores: object) -> None:
    for store in stores:
        close = getattr(store, "close", None)
        if close is not None:
            await close()


def _app_selection(project: Path, preset: str) -> tuple[str, object]:
    """Build the stores the way the generated app factory does."""

    try:
        if preset == "coding":
            storage = importlib.import_module("configuration.coding_storage")
            stores = storage.build_coding_stores(project / ".cayu" / "runtime", None)
        else:
            stores = importlib.import_module("configuration.storage").build_stores()
    except Exception as exc:
        return ("error", type(exc).__name__)
    configured = stores.configured
    assert configured is not None
    session_store = configured.session_store
    try:
        if isinstance(session_store, SQLiteSessionStore):
            return ("sqlite", Path(session_store.path).resolve())
        return ("postgres", session_store._pool.conninfo)
    finally:
        asyncio.run(configured.close())


def _cli_selection(project: Path) -> tuple[str, object]:
    """Resolve and open the store the way `cayu session` does."""

    try:
        target = resolve_session_store_target(start=project)
        store = _open_read_only_store(target)
    except Exception as exc:
        return ("error", type(exc).__name__)
    asyncio.run(_close(store))
    if target.sqlite_path is not None:
        return ("sqlite", target.sqlite_path.resolve())
    return ("postgres", target.postgres_dsn)


def _session_store_sections(preset: str) -> dict[str, str | None]:
    local = _LOCAL_DATABASE[preset]
    return {
        "generated": f'backend = "sqlite"\npath = "{local}"',
        "absent": None,
        "other-sqlite": 'backend = "sqlite"\npath = "data/elsewhere.db"',
        "retired-postgres": 'backend = "postgres"\nenv = "CAYU_DATABASE_URL"',
        "other-postgres": 'backend = "postgres"\nenv = "OTHER_DATABASE_URL"',
    }


def _database_urls(tmp_path: Path) -> dict[str, str | None]:
    return {
        "unset": None,
        "postgres": _POSTGRES_URL,
        "sqlite": f"sqlite://{tmp_path / 'selected.db'}",
        "blank": "",
        "unsupported": "mysql://app:secret@db.example/app",
        "relative-sqlite": "sqlite:relative.db",
    }


@pytest.mark.parametrize("preset", ["agent", "coding"])
def test_app_factory_and_cli_resolver_select_the_same_store(
    preset: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _generate(tmp_path, preset, capsys)
    pyproject = project / "pyproject.toml"
    generated = pyproject.read_text(encoding="utf-8")
    generated_section = _session_store_sections(preset)["generated"]
    assert f"[tool.cayu.session_store]\n{generated_section}\n" in generated
    monkeypatch.setenv("OTHER_DATABASE_URL", "postgresql://other@127.0.0.1:1/other")

    agreed = drifted = 0
    with project_context(project):
        for (section_name, section), (url_name, url), required in itertools.product(
            _session_store_sections(preset).items(),
            _database_urls(tmp_path).items(),
            (False, True),
        ):
            replacement = "" if section is None else f"[tool.cayu.session_store]\n{section}\n"
            pyproject.write_text(
                generated.replace(f"[tool.cayu.session_store]\n{generated_section}\n", replacement),
                encoding="utf-8",
            )
            if url is None:
                monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
            else:
                monkeypatch.setenv("CAYU_DATABASE_URL", url)
            if required:
                monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")
            else:
                monkeypatch.delenv("CAYU_REQUIRE_POSTGRES", raising=False)
            # The app runs first: the read-only CLI opens a SQLite file only once
            # the app has created it.
            app = _app_selection(project, preset)
            cli = _cli_selection(project)
            case = (section_name, url_name, required)
            if url is not None or section_name == "generated":
                assert app == cli, case
                agreed += 1
            else:
                # Any other local section is scaffold drift, reported before it can
                # split the app from Cayu tooling.
                findings = check_declared_scaffold_source(project)
                assert any(
                    item.code == "SCAFFOLD_PLAN_DRIFT" and item.parameters["field"] == "storage"
                    for item in findings
                ), case
                drifted += 1
    assert (agreed, drifted) == (2 * (6 + 4 * 5), 2 * 4)


@pytest.mark.parametrize("preset", ["agent", "service", "coding"])
def test_generated_sqlite_stores_ignore_the_working_directory(
    preset: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _generate(tmp_path, preset, capsys)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    database = (project / _LOCAL_DATABASE["coding" if preset == "coding" else "agent"]).resolve()

    with project_context(project):
        coding = preset == "coding"
        if coding:
            composition = importlib.import_module("operations.coding")
            monkeypatch.setattr(composition, "_verify_coding_dependencies", lambda root: None)
        os.chdir(elsewhere)
        application = importlib.import_module("app").build_app(
            **({"workspace_root": project} if coding else {})
        )
        stores = [application.session_store, application.task_store]
        if application.knowledge_store is not None:
            stores.append(application.knowledge_store)
        try:
            assert isinstance(application.session_store, SQLiteSessionStore)
            assert isinstance(application.task_store, SQLiteTaskStore)
            if preset != "service":
                assert isinstance(application.knowledge_store, SQLiteKnowledgeStore)
            for store in stores:
                assert Path(store.path).resolve() == database
        finally:
            asyncio.run(_close(*stores))
    assert database.is_file()
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize("preset", ["agent", "service", "coding"])
def test_required_postgres_stops_a_generated_app_without_a_database_url(
    preset: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _generate(tmp_path, preset, capsys)
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")

    with project_context(project):
        if preset == "coding":
            composition = importlib.import_module("operations.coding")
            monkeypatch.setattr(composition, "_verify_coding_dependencies", lambda root: None)
        with pytest.raises(PostgresRequiredError, match="CAYU_DATABASE_URL"):
            importlib.import_module("app").build_app()
    assert not (project / "data" / "cayu.db").exists()
    assert not (project / ".cayu" / "runtime" / "cayu.db").exists()


@pytest.mark.parametrize("preset", ["agent", "service", "coding"])
@pytest.mark.parametrize("execution", ["none", "docker"])
def test_every_scaffold_depends_on_the_postgres_driver(preset: str, execution: str) -> None:
    if execution == "docker" and preset != "coding":
        pytest.skip("Docker execution is admitted only for the coding preset")
    files = project_files("app", preset=preset, execution=execution)
    assert '"cayu[postgres' in files["pyproject.toml"].split("\n[", 1)[0]


def test_generated_project_imports_the_postgres_driver(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _generate(tmp_path, "agent", capsys)
    monkeypatch.setenv("CAYU_DATABASE_URL", _POSTGRES_URL)

    with project_context(project):
        import psycopg
        import psycopg_pool

        stores = importlib.import_module("configuration.storage").build_stores()
        assert stores.configured is not None
        assert type(stores.configured.session_store._pool) is psycopg_pool.AsyncConnectionPool
        assert psycopg.AsyncConnection is not None
        asyncio.run(stores.configured.close())
