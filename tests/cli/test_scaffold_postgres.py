from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

from cayu import (
    PostgresKnowledgeStore,
    PostgresSessionStore,
    PostgresTaskStore,
    SessionQuery,
    TaskCreate,
    TaskQuery,
)
from cayu.cli import main
from cayu.cli.project import project_context
from cayu.storage.migrations import SchemaMode


@pytest.mark.parametrize("preset", ["agent", "service", "coding"])
def test_generated_project_runs_on_migrated_postgres_from_the_database_url(
    preset: str,
    postgres_dsn: str,
    postgres_url: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_name = f"postgres-{preset}"
    assert main(["new", project_name, "--preset", preset, "--dir", str(tmp_path), "--json"]) == 0
    capsys.readouterr()
    project = tmp_path / project_name
    monkeypatch.setenv("CAYU_DATABASE_URL", postgres_url)
    monkeypatch.setenv("CAYU_DATABASE_POOL_MAX", "3")
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")

    async def exercise() -> None:
        # Migrations are a deploy step; the application stores only validate.
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        with project_context(project):
            application = importlib.import_module("app").build_app()
            assert isinstance(application.session_store, PostgresSessionStore)
            assert isinstance(application.task_store, PostgresTaskStore)
            active_stores = [application.session_store, application.task_store]
            if preset != "service":
                assert isinstance(application.knowledge_store, PostgresKnowledgeStore)
                active_stores.append(application.knowledge_store)
            pool = application.session_store._pool
            assert all(store._pool is pool for store in active_stores)
            assert pool.max_size == 3
            try:
                for store in active_stores:
                    await store.ensure_schema()
                    assert store._schema_mode is SchemaMode.VALIDATE
                await application.task_store.create_task(
                    TaskCreate(task_id=f"{preset}-task", type="job")
                )
                listed = await application.task_store.list_tasks(TaskQuery(type="job"))
                assert f"{preset}-task" in {task.id for task in listed}
                await application.session_store.list_sessions(SessionQuery())
            finally:
                for store in active_stores:
                    await store.close()
                await pool.close()
        assert not (project / "data" / "cayu.db").exists()
        assert not (project / ".cayu" / "runtime" / "cayu.db").exists()

    asyncio.run(exercise())
