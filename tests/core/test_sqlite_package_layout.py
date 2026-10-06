"""SQLite backend imports preserve compatibility and independent ownership."""

import asyncio
import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


@pytest.mark.parametrize("module_name", ("cayu", "cayu.storage", "cayu.storage.tasks_sqlite"))
def test_sqlite_task_store_operates_without_loading_session_adapter(module_name, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import importlib
import sys

from cayu.tasks.base import TaskCreate
from cayu.tasks.topology import TaskTopologyQuery

store_type = importlib.import_module(sys.argv[1]).SQLiteTaskStore

async def exercise():
    store = store_type(sys.argv[2])
    try:
        task = await store.create_task(
            TaskCreate(task_id="task", type="work", session_id="session")
        )
        result = await store.query_task_topology(
            TaskTopologyQuery(linked_session_ids=("session",))
        )
        assert result.session_branches[0].tasks[0].id == task.id
    finally:
        await store.close()
    reopened = store_type(sys.argv[2])
    try:
        assert (await reopened.load_task("task")).session_id == "session"
    finally:
        await reopened.close()

asyncio.run(exercise())
assert "cayu.storage.sqlite" not in sys.modules
assert "cayu.storage.postgres" not in sys.modules
""",
            module_name,
            str(tmp_path / "tasks.sqlite"),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("missing_index", [False, True])
def test_sqlite_knowledge_schema_validates_without_migration_or_store_imports(
    tmp_path, sqlite_resources, missing_index
):
    from cayu.storage import _sqlite_connection, _sqlite_support

    async def prepare():
        async with sqlite_resources as resources:
            path = tmp_path / "knowledge.sqlite"
            connection = resources.own(_sqlite_connection.connect(path), kind="connection")
            _sqlite_support.initialize_schema(connection)
            if missing_index:
                connection.execute("DROP INDEX idx_cayu_knowledge_revisions_status")
                connection.commit()
            return path

    path = asyncio.run(prepare())
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from cayu.storage import _sqlite_knowledge_schema as knowledge_schema

with closing(sqlite3.connect(Path(sys.argv[1]).as_uri() + "?mode=ro", uri=True)) as connection:
    try:
        knowledge_schema._validate_revision_42_knowledge_schema(
            connection, require_payload_bytes=True
        )
        knowledge_schema._validate_revision_43_knowledge_schema(connection, relation_aware=True)
        knowledge_schema._validate_revision_44_knowledge_schema(connection)
        knowledge_schema._validate_revision_60_knowledge_schema(connection)
        knowledge_schema._validate_revision_63_knowledge_schema(connection)
        knowledge_schema._validate_revision_67_knowledge_schema(connection)
        knowledge_schema._validate_revision_75_knowledge_activation_schema(connection)
        knowledge_schema._validate_revision_77_knowledge_maintenance_governance_schema(connection)
        knowledge_schema._validate_revision_78_knowledge_semantic_watch_schema(connection)
        knowledge_schema._validate_knowledge_publication_access_snapshot_column(connection)
    except RuntimeError as error:
        assert sys.argv[2] == "True", str(error)
        assert "idx_cayu_knowledge_revisions_status" in str(error), str(error)
    else:
        assert sys.argv[2] == "False", "missing index was accepted"
    assert not connection.in_transaction

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_connection",
    "cayu.storage.migrations",
    "cayu.sessions.base",
    "cayu.tasks.base",
    "cayu.storage.knowledge_sqlite",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
):
    assert name not in sys.modules, name
""",
            str(path),
            str(missing_index),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_task_store_preserves_public_and_legacy_identity():
    from cayu.storage import sqlite, tasks_sqlite

    canonical = tasks_sqlite.SQLiteTaskStore
    assert canonical.__module__ == "cayu.storage.tasks_sqlite"
    for name in ("cayu", "cayu.storage", "cayu.storage.sqlite"):
        assert importlib.import_module(name).SQLiteTaskStore is canonical
    assert pickle.loads(b"ccayu.storage.sqlite\nSQLiteTaskStore\n.") is canonical
    assert pickle.loads(pickle.dumps(canonical)) is canonical
    for name in (
        "_SQLITE_TASK_MIN_REQUIRED_REVISION",
        "_like_contains_pattern",
        "_sqlite_task_terminalization_receipt",
        "_sqlite_interrupted_task_handoff_receipt",
        "_validate_task_positive_int",
    ):
        assert getattr(sqlite, name) is getattr(tasks_sqlite, name)


def test_sqlite_backends_share_one_connection_ownership_implementation():
    from cayu.storage import (
        _sqlite_connection,
        collaboration_sqlite,
        evals_sqlite,
        sqlite,
        tasks_sqlite,
    )

    for module in (sqlite, tasks_sqlite, collaboration_sqlite, evals_sqlite):
        assert module._run_off_thread_with_connection_ownership is (
            _sqlite_connection._run_off_thread_with_connection_ownership
        )


def test_sqlite_task_store_keeps_support_bundle_schema_readiness(tmp_path):
    from cayu import CayuApp
    from cayu.runtime.checks import check_manifest
    from cayu.storage.tasks_sqlite import SQLiteTaskStore
    from cayu.support_bundles import (
        CollectorDisposition,
        StoreSummaryEvidence,
        SupportBundleContext,
        builtin_support_collectors,
        collect_support_bundle,
    )

    async def collect():
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            manifest = app.describe()
            context = SupportBundleContext(
                app=app,
                manifest=manifest,
                check_report=check_manifest(manifest),
                service_manifest=None,
                project_id=None,
                application_release_id=f"manifest-{manifest.fingerprint}",
                eval_backend=None,
                eval_source=None,
            )
            collector = next(item for item in builtin_support_collectors() if item.name == "stores")
            report = await collect_support_bundle(context, (collector,))
            result = report.collectors[0]
            assert result.disposition is CollectorDisposition.COLLECTED
            assert isinstance(result.evidence, StoreSummaryEvidence)
            descriptor = next(item for item in result.evidence.stores if item.role == "task")
            assert descriptor.schema_readiness == "validated_compatible"
        finally:
            await store.close()

    asyncio.run(collect())


def test_sqlite_records_import_without_schema_or_store_adapters():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

from cayu.storage import _sqlite_records

assert _sqlite_records.json_dumps({"text": "café"}) == '{"text":"café"}'
for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
):
    assert name not in sys.modules, name
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_connection_setup_works_without_schema_or_store_adapters():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from contextlib import closing
from pathlib import Path

from cayu.storage import _sqlite_connection

for name in ("cayu.sessions.base", "cayu.tasks.base", "cayu.storage._sqlite_functions"):
    assert name not in sys.modules, name

with closing(_sqlite_connection.connect(Path(":memory:"))) as connection:
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert connection.execute("SELECT cayu_is_clean_nonblank_text('valid')").fetchone()[0] == 1

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
):
    assert name not in sys.modules, name
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_transactions_work_without_schema_or_store_adapters():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sqlite3
import sys
from contextlib import closing

from cayu.storage import _sqlite_connection

with closing(sqlite3.connect(":memory:")) as connection:
    with _sqlite_connection._transaction(connection):
        connection.execute("CREATE TABLE records (value TEXT)")
        connection.execute("INSERT INTO records VALUES ('committed')")
    failure = RuntimeError("roll back this write")
    try:
        with _sqlite_connection._transaction(connection):
            connection.execute("INSERT INTO records VALUES ('rolled back')")
            raise failure
    except RuntimeError as caught:
        assert caught is failure
    else:
        raise AssertionError("transaction failure was suppressed")
    assert not connection.in_transaction
    with _sqlite_connection._transaction(connection, begin_immediate=False):
        assert connection.execute("SELECT value FROM records").fetchall() == [("committed",)]

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_functions",
    "cayu.sessions.base",
    "cayu.tasks.base",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
):
    assert name not in sys.modules, name
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
