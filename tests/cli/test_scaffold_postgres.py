from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest
from tests.core.postgres_contention_support import drop_cayu_tables

from cayu import (
    PostgresKnowledgeStore,
    PostgresSessionStore,
    PostgresTaskStore,
)
from cayu.cli import main
from cayu.cli.project import project_context
from cayu.storage.migrations import SchemaMode


@pytest.mark.parametrize("preset", ["agent", "coding"])
def test_generated_postgres_profile_uses_real_postgres_for_every_active_store(
    preset: str,
    postgres_dsn: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_name = f"postgres-{preset}"
    assert (
        main(
            [
                "new",
                project_name,
                "--preset",
                preset,
                "--database",
                "postgres",
                "--dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    monkeypatch.setenv("CAYU_DATABASE_URL", postgres_dsn)

    async def exercise() -> None:
        await drop_cayu_tables(postgres_dsn)
        creator = PostgresSessionStore(
            postgres_dsn,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        project = tmp_path / project_name
        with project_context(project):
            application = importlib.import_module("app").build_app()
            assert isinstance(application.session_store, PostgresSessionStore)
            assert isinstance(application.task_store, PostgresTaskStore)
            active_stores = [application.session_store, application.task_store]
            if preset == "coding":
                assert isinstance(application.knowledge_store, PostgresKnowledgeStore)
                active_stores.append(application.knowledge_store)
            else:
                assert application.knowledge_store is None

            try:
                for store in active_stores:
                    await store.ensure_schema()
            finally:
                for store in active_stores:
                    await store.close()

        await drop_cayu_tables(postgres_dsn)

    asyncio.run(exercise())
