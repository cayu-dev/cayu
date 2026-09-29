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


def test_generated_service_settles_product_operations_on_postgres(
    postgres_dsn: str,
    postgres_url: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from cayu import ModelStreamEvent, PostgresProductOperationStore, ScriptedModelProvider
    from cayu.server import ServiceMode

    assert main(["new", "pgservice", "--preset", "service", "--dir", str(tmp_path), "--json"]) == 0
    capsys.readouterr()
    project = tmp_path / "pgservice"
    assert not (project / "product_store.py").exists()
    monkeypatch.setenv("CAYU_DATABASE_URL", postgres_url)
    monkeypatch.setenv("CAYU_DATABASE_POOL_MAX", "3")
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")

    async def create_schema() -> None:
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

    asyncio.run(create_schema())
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("postgres answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    with project_context(project):
        service = importlib.import_module("service").build_service(
            mode=ServiceMode.DEVELOPMENT, provider=provider
        )
        product_store = service.product_store
        session_store = service.cayu_app.session_store
        assert isinstance(product_store, PostgresProductOperationStore)
        assert product_store._pool is session_store._pool
        assert product_store._schema_mode is SchemaMode.VALIDATE
        headers = {
            "Idempotency-Key": "postgres-operation",
            "X-Cayu-Dev-Tenant": "tenant-a",
            "X-Cayu-Dev-Subject": "alice",
        }

        async def exercise() -> None:
            transport = httpx.ASGITransport(app=service.asgi_app)
            try:
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://service"
                ) as client:
                    created = await client.post(
                        "/api/operations", headers=headers, json={"request": "work"}
                    )
                    assert created.status_code == 201, created.text
                    assert created.json()["status"] == "completed"
                    assert created.json()["result"] == "postgres answer"
                    public_id = created.json()["id"]
                    read = await client.get(f"/api/operations/{public_id}", headers=headers)
                    assert read.json() == created.json()
                stored = await product_store.find(tenant_id="tenant-a", public_id=public_id)
                assert stored is not None and stored.status == "completed"
            finally:
                pool = session_store._pool
                for store in (product_store, service.cayu_app.task_store, session_store):
                    await store.close()
                await pool.close()

        asyncio.run(exercise())
    assert not (project / "data").exists() or not any((project / "data").glob("*.db"))
