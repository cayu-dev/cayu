from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest

import cayu.cli.doctor as doctor_cli
from cayu import (
    Event,
    RunRequest,
    SQLiteSessionStore,
    SQLiteTaskStore,
)
from cayu.cli import main
from cayu.runtime.sessions import SessionIdentity
from cayu.storage._diagnostic_inspection import diagnostic_store_inspection
from cayu.support_bundles import (
    SupportBundleOutcome,
    encode_support_bundle,
    validate_support_bundle_archive,
)


def _write_project(root: Path, body: str) -> None:
    (root / "project.py").write_text(body, encoding="utf-8")
    sys.modules.pop("project", None)


def _report_document(bundle: Path) -> dict:
    payload = bundle.read_bytes()
    validate_support_bundle_archive(payload)
    with zipfile.ZipFile(bundle) as archive:
        return json.loads(archive.read("report.json"))


def _file_snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {item.name: (item.read_bytes(), item.stat().st_mtime_ns) for item in directory.iterdir()}


def test_doctor_factory_created_sqlite_stores_are_read_only_and_unchanged(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data = tmp_path / "data"
    database = data / "runtime.sqlite"

    async def initialize() -> None:
        session_store = SQLiteSessionStore(database)
        task_store = SQLiteTaskStore(database)
        await task_store.close()
        await session_store.close()

    asyncio.run(initialize())
    before = _file_snapshot(data)
    _write_project(
        tmp_path,
        """from concurrent.futures import ThreadPoolExecutor
from cayu import CayuApp, SQLiteSessionStore, SQLiteTaskStore


def build_app():
    with ThreadPoolExecutor(max_workers=2) as executor:
        session_store = executor.submit(
            SQLiteSessionStore, "data/runtime.sqlite"
        ).result()
        task_store = executor.submit(SQLiteTaskStore, "data/runtime.sqlite").result()
    return CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "support.zip"

    assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 1

    assert json.loads(capsys.readouterr().out)["outcome"] == "partial"
    assert _file_snapshot(data) == before
    document = _report_document(bundle)
    stores = next(item for item in document["collectors"] if item["name"] == "stores")
    durability = {item["role"]: item["durability"] for item in stores["evidence"]["stores"]}
    readiness = {item["role"]: item["schema_readiness"] for item in stores["evidence"]["stores"]}
    assert durability["session"] == "durable"
    assert durability["task"] == "durable"
    assert readiness["session"] == "validated_compatible"
    assert readiness["task"] == "validated_compatible"


def test_doctor_boots_process_private_in_memory_sqlite_stores(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp, SQLiteSessionStore, SQLiteTaskStore


def build_app():
    return CayuApp(
        session_store=SQLiteSessionStore(":memory:"),
        task_store=SQLiteTaskStore(":memory:"),
        enable_logging=False,
    )
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "support.zip"

    assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 1

    assert json.loads(capsys.readouterr().out)["outcome"] == "partial"
    document = _report_document(bundle)
    collectors = {item["name"]: item for item in document["collectors"]}
    stores = collectors["stores"]["evidence"]["stores"]
    durability = {item["role"]: item["durability"] for item in stores}
    readiness = {item["role"]: item["schema_readiness"] for item in stores}
    assert durability["session"] == "development"
    assert durability["task"] == "development"
    assert readiness["session"] == "not_applicable"
    assert readiness["task"] == "not_applicable"
    assert collectors["sessions"]["evidence"]["snapshot"]["total_count"] == "0"
    assert collectors["tasks"]["evidence"]["snapshot"]["total_count"] == "0"
    assert not tuple(tmp_path.glob("*.sqlite*"))


def test_doctor_factory_created_sqlite_store_does_not_create_missing_database(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp, SQLiteSessionStore


def build_app():
    return CayuApp(
        session_store=SQLiteSessionStore("data/runtime.sqlite"),
        enable_logging=False,
    )
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "support.zip"

    assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 1

    assert json.loads(capsys.readouterr().out)["outcome"] == "partial"
    assert not (tmp_path / "data").exists()
    document = _report_document(bundle)
    collectors = {item["name"]: item for item in document["collectors"]}
    assert collectors["sessions"]["reason_code"] == "store_source_not_available"
    stores = collectors["stores"]["evidence"]["stores"]
    session = next(item for item in stores if item["role"] == "session")
    assert session["durability"] == "durable"
    assert session["schema_readiness"] == "unavailable"


def test_missing_sqlite_diagnostic_shadow_is_query_only_and_absence_guarded(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "runtime.sqlite"

    with diagnostic_store_inspection() as inspection:
        store = SQLiteSessionStore(database)
        try:
            assert store._diagnostic_source_missing is True
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                store._connection.execute("CREATE TABLE forbidden (value TEXT)")
            database.parent.mkdir(parents=True)
            database.write_bytes(b"appeared-during-diagnostic-inspection")
            with pytest.raises(ExceptionGroup, match="changed during collection"):
                inspection.verify()
        finally:
            asyncio.run(store.close())


def test_doctor_reads_committed_live_wal_without_mutating_it(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    database = tmp_path / "data" / "runtime.sqlite"

    async def seed() -> SQLiteSessionStore:
        store = SQLiteSessionStore(database)
        store._connection.execute("PRAGMA wal_autocheckpoint = 0")
        await store.create(
            RunRequest(
                session_id="live-wal-session",
                agent_name="assistant",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        return store

    writer = asyncio.run(seed())
    try:
        assert Path(f"{database}-wal").exists()
        immutable = sqlite3.connect(
            f"file:{database.resolve()}?mode=ro&immutable=1",
            uri=True,
        )
        try:
            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                immutable.execute("SELECT COUNT(*) FROM cayu_sessions").fetchone()
        finally:
            immutable.close()
        main_before = database.read_bytes()
        wal_before = Path(f"{database}-wal").read_bytes()
        _write_project(
            tmp_path,
            """from cayu import CayuApp, SQLiteSessionStore


def build_app():
    return CayuApp(
        session_store=SQLiteSessionStore("data/runtime.sqlite"),
        enable_logging=False,
    )
""",
        )
        monkeypatch.chdir(tmp_path)
        bundle = tmp_path / "support.zip"

        assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 1

        assert json.loads(capsys.readouterr().out)["outcome"] == "partial"
        document = _report_document(bundle)
        sessions = next(item for item in document["collectors"] if item["name"] == "sessions")
        assert sessions["evidence"]["snapshot"]["total_count"] == "1"
        assert database.read_bytes() == main_before
        assert Path(f"{database}-wal").read_bytes() == wal_before

        async def prove_writer_remains_usable() -> None:
            await writer.append_event(
                "live-wal-session",
                Event(
                    type="custom.writer.still_usable",
                    session_id="live-wal-session",
                    payload={},
                ),
            )

        asyncio.run(prove_writer_remains_usable())
    finally:
        asyncio.run(writer.close())


def test_static_sqlite_snapshot_change_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "runtime.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(database)
        await store.close()

    asyncio.run(initialize())
    _write_project(
        tmp_path,
        """from cayu import CayuApp, SQLiteSessionStore


def build_app():
    return CayuApp(
        session_store=SQLiteSessionStore("runtime.sqlite"),
        enable_logging=False,
    )
""",
    )
    monkeypatch.chdir(tmp_path)
    original_collect = doctor_cli.collect_support_bundle

    async def mutate_during_collection(context, collectors):
        connection = sqlite3.connect(database)
        try:
            connection.execute("CREATE TABLE application_change (id INTEGER)")
            connection.commit()
        finally:
            connection.close()
        return await original_collect(context, collectors)

    monkeypatch.setattr(doctor_cli, "collect_support_bundle", mutate_during_collection)

    report = doctor_cli._collect_project_report(target="project:build_app", sessions=())
    validated = validate_support_bundle_archive(encode_support_bundle(report))

    assert validated.outcome is SupportBundleOutcome.BOOT_FAILED
    assert validated.collectors[-1].name == "store_inspection"
    assert validated.collectors[-1].reason_code == "store_inspection_changed"


def test_boot_failure_and_cleanup_failure_preserve_both_typed_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_project(
        tmp_path,
        """def build_app():
    raise RuntimeError("boot-secret-canary")
""",
    )
    monkeypatch.chdir(tmp_path)

    def fail_cleanup(_context) -> None:
        raise RuntimeError("cleanup-secret-canary")

    monkeypatch.setattr(doctor_cli, "close_project_control_plane_context", fail_cleanup)

    report = doctor_cli._collect_project_report(target="project:build_app", sessions=())
    validated = validate_support_bundle_archive(encode_support_bundle(report))

    assert validated.outcome is SupportBundleOutcome.BOOT_FAILED
    assert [item.name for item in validated.collectors] == [
        "bootstrap",
        "control_plane_cleanup",
    ]
    assert [item.reason_code for item in validated.collectors] == [
        "application_boot_failed",
        "control_plane_cleanup_failed",
    ]
    assert "secret-canary" not in validated.model_dump_json()


def test_doctor_factory_created_postgres_store_is_transaction_read_only(
    postgres_dsn: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from uuid import uuid4

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    schema_name = f"doctor_{uuid4().hex}"
    scoped_dsn = make_conninfo(
        postgres_dsn,
        options=f"-c search_path={schema_name}",
    )

    async def create_schema_and_seed() -> tuple[int, int]:
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            await connection.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
            )
            await connection.commit()
        store = PostgresSessionStore(scoped_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.create(
                RunRequest(
                    session_id="postgres-doctor-session",
                    agent_name="assistant",
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
        finally:
            await store.close()
        async with await psycopg.AsyncConnection.connect(scoped_dsn) as connection:
            session_count = (
                await (await connection.execute("SELECT COUNT(*) FROM cayu_sessions")).fetchone()
            )[0]
            migration_count = (
                await (
                    await connection.execute("SELECT COUNT(*) FROM cayu_schema_migrations")
                ).fetchone()
            )[0]
        return session_count, migration_count

    async def read_counts() -> tuple[int, int]:
        async with await psycopg.AsyncConnection.connect(scoped_dsn) as connection:
            session_count = (
                await (await connection.execute("SELECT COUNT(*) FROM cayu_sessions")).fetchone()
            )[0]
            migration_count = (
                await (
                    await connection.execute("SELECT COUNT(*) FROM cayu_schema_migrations")
                ).fetchone()
            )[0]
        return session_count, migration_count

    async def drop_schema() -> None:
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
            )
            await connection.commit()

    before = asyncio.run(create_schema_and_seed())
    try:
        _write_project(
            tmp_path,
            f"""from cayu import CayuApp, PostgresSessionStore


def build_app():
    return CayuApp(
        session_store=PostgresSessionStore({scoped_dsn!r}),
        enable_logging=False,
    )
""",
        )
        monkeypatch.chdir(tmp_path)
        bundle = tmp_path / "support.zip"

        assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 1

        assert json.loads(capsys.readouterr().out)["outcome"] == "partial"
        assert asyncio.run(read_counts()) == before
        document = _report_document(bundle)
        sessions = next(item for item in document["collectors"] if item["name"] == "sessions")
        assert sessions["evidence"]["snapshot"]["total_count"] == "1"
        stores = next(item for item in document["collectors"] if item["name"] == "stores")
        session_descriptor = next(
            item for item in stores["evidence"]["stores"] if item["role"] == "session"
        )
        assert session_descriptor["durability"] == "durable"
        assert session_descriptor["schema_readiness"] == "validated_compatible"
    finally:
        asyncio.run(drop_schema())
