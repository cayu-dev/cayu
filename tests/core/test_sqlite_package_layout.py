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

from cayu.tasks.creation import TaskCreate
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


@pytest.mark.parametrize(
    ("damage", "expected_error"),
    [
        (None, None),
        (
            "DROP TABLE cayu_sessions",
            "session-instance authority columns are missing",
        ),
        (
            "PRAGMA foreign_keys=OFF; DROP TABLE cayu_sessions; "
            "CREATE TABLE cayu_sessions(instance_id TEXT); "
            "INSERT INTO cayu_sessions(instance_id) VALUES(NULL)",
            "session-instance authority is incomplete",
        ),
        (
            "ALTER TABLE cayu_sessions RENAME COLUMN invocation_json TO old_invocation_json",
            "invocation-provenance contract",
        ),
        ("DROP TABLE cayu_targeted_tool_grants", "targeted-grant durability contract"),
        (
            "DROP INDEX idx_cayu_targeted_tool_grant_uses_grant",
            "targeted-grant contention contract",
        ),
        ("DROP TABLE cayu_deferred_interaction_inputs", "deferred interaction input schema"),
        (
            "ALTER TABLE cayu_session_message_queue RENAME COLUMN message_json TO old_message_json",
            "typed queued-message contract",
        ),
        (
            "ALTER TABLE cayu_session_message_queue RENAME COLUMN terminal_json TO old_terminal_json",
            "session-message lifecycle columns",
        ),
        (
            "DROP INDEX idx_cayu_events_queue_acceptance; "
            "CREATE INDEX idx_cayu_events_queue_acceptance "
            "ON cayu_events(session_id, json_extract(payload_json, '$.queue_id')) "
            "WHERE event_type='session.message.started'",
            "queue acceptance lookup index",
        ),
        (
            "DROP TABLE cayu_child_session_lifecycle_candidates",
            "bounded child-lifecycle projection",
        ),
        (
            "DROP TRIGGER cayu_index_child_lifecycle_event_insert; "
            "CREATE TRIGGER cayu_index_child_lifecycle_event_insert "
            "AFTER INSERT ON cayu_events BEGIN SELECT 1; END",
            "bounded child-lifecycle projection",
        ),
        (
            "DROP INDEX idx_cayu_child_lifecycle_candidates_page; "
            "CREATE INDEX idx_cayu_child_lifecycle_candidates_page "
            "ON cayu_child_session_lifecycle_candidates "
            "(parent_session_id, sort_at, priority, child_session_id)",
            "bounded child-lifecycle projection",
        ),
    ],
    ids=[
        "current",
        "missing-session-instance",
        "null-session-instance",
        "missing-invocation-column",
        "missing-targeted-grants",
        "missing-grant-use-index",
        "missing-deferred-inputs",
        "missing-queued-message-column",
        "missing-message-lifecycle-column",
        "wrong-queue-acceptance-predicate",
        "missing-child-candidates",
        "weakened-child-trigger",
        "wrong-child-index-order",
    ],
)
def test_sqlite_session_schema_validates_without_migration_or_store_imports(
    tmp_path, sqlite_resources, damage, expected_error
):
    from cayu.storage import _sqlite_connection, _sqlite_support

    async def prepare():
        async with sqlite_resources as resources:
            path = tmp_path / "session-schema.sqlite"
            connection = resources.own(_sqlite_connection.connect(path), kind="connection")
            _sqlite_support.initialize_schema(connection)
            if damage is not None:
                connection.executescript(damage)
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

from cayu.storage import _sqlite_session_schema as session_schema

with closing(sqlite3.connect(Path(sys.argv[1]).as_uri() + "?mode=ro", uri=True)) as connection:
    try:
        session_schema._validate_session_instance_schema(connection)
        session_schema._validate_session_invocation_column(connection)
        session_schema._validate_targeted_tool_grant_schema(connection)
        session_schema._validate_revision_sixty_two_payload_schema(connection)
        session_schema._validate_session_message_queue_typed_message_column(connection)
        session_schema._validate_session_message_lifecycle_columns(connection)
        session_schema._validate_revision_79_child_lifecycle_schema(connection)
    except RuntimeError as error:
        assert sys.argv[2] and sys.argv[2] in str(error), str(error)
    else:
        assert not sys.argv[2], "incompatible schema was accepted"
    assert not connection.in_transaction

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_connection",
    "cayu.storage.migrations",
    "cayu.sessions.base",
    "cayu.tasks.base",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
    "cayu.runtime.task_worker",
):
    assert name not in sys.modules, name
""",
            str(path),
            expected_error or "",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("damage", "expected_error"),
    [
        (None, None),
        (
            "DROP TABLE cayu_task_terminalization_receipts",
            "terminalization receipt table",
        ),
        (
            "ALTER TABLE cayu_tasks RENAME COLUMN invocation_json TO old_invocation_json",
            "invocation-provenance contract",
        ),
        ("DROP TABLE cayu_task_retry_settlements", "retry-series schema"),
        (
            "DROP TABLE cayu_task_retry_reconciliation_rejections",
            "retry-reconciliation schema",
        ),
        (
            "DROP TABLE cayu_task_interrupted_handoff_receipts",
            "interrupted-task handoff storage",
        ),
        (
            "DROP INDEX idx_cayu_tasks_interrupted_handoff_recovery",
            "interrupted-task handoff storage",
        ),
        (
            "DROP INDEX idx_cayu_tasks_interrupted_handoff_generation",
            "task handoff generation or bounded continuation index",
        ),
        (
            "DROP TABLE cayu_task_interrupted_continuation_claims",
            "task handoff generation or bounded continuation index",
        ),
        (
            "DROP INDEX idx_cayu_tasks_interrupted_handoff_continuation; "
            "CREATE INDEX idx_cayu_tasks_interrupted_handoff_continuation "
            "ON cayu_tasks(status, created_at, id) WHERE worker_id IS NULL",
            "task handoff generation or bounded continuation index",
        ),
    ],
    ids=[
        "current",
        "missing-terminal-receipts",
        "missing-invocation-column",
        "missing-retry-settlements",
        "missing-retry-rejections",
        "missing-handoff-receipts",
        "missing-handoff-recovery-index",
        "missing-handoff-generation-index",
        "missing-continuation-claims",
        "weak-continuation-predicate",
    ],
)
def test_sqlite_task_schema_validates_without_migration_or_store_imports(
    tmp_path, sqlite_resources, damage, expected_error
):
    from cayu.storage import _sqlite_connection, _sqlite_support

    async def prepare():
        async with sqlite_resources as resources:
            path = tmp_path / "task-schema.sqlite"
            connection = resources.own(_sqlite_connection.connect(path), kind="connection")
            _sqlite_support.initialize_schema(connection)
            if damage is not None:
                connection.executescript(damage)
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

from cayu.storage import _sqlite_task_schema as task_schema

with closing(sqlite3.connect(Path(sys.argv[1]).as_uri() + "?mode=ro", uri=True)) as connection:
    try:
        task_schema._validate_task_terminalization_receipt_table(connection)
        task_schema._validate_task_invocation_column(connection)
        task_schema._validate_task_retry_series_schema(connection)
        task_schema._validate_task_retry_reconciliation_schema(connection)
        task_schema._validate_interrupted_task_handoff_schema(connection)
        task_schema._validate_interrupted_handoff_generation_column(connection)
    except RuntimeError as error:
        assert sys.argv[2] and sys.argv[2] in str(error), str(error)
    else:
        assert not sys.argv[2], "incompatible schema was accepted"
    assert not connection.in_transaction

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_connection",
    "cayu.storage.migrations",
    "cayu.sessions.base",
    "cayu.tasks.base",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
    "cayu.runtime.task_worker",
):
    assert name not in sys.modules, name
""",
            str(path),
            expected_error or "",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("revision", "require_profiles", "damage", "expected_error"),
    [
        (49, False, None, None),
        (49, True, None, "revision-58 verified-work contract"),
        (None, True, None, None),
        (
            None,
            True,
            "DROP TABLE cayu_work_contracts",
            "cayu_work_contracts",
        ),
        (
            None,
            True,
            "DROP TABLE cayu_completion_verifier_profiles",
            "cayu_completion_verifier_profiles",
        ),
        (
            None,
            True,
            "DROP INDEX idx_cayu_work_attempt_claim_current",
            "work-attempt admission schema",
        ),
        (
            None,
            True,
            "DROP TABLE cayu_work_attempt_preparation_holds",
            "work-attempt preparation hold schema",
        ),
        (
            None,
            True,
            "DROP TABLE cayu_work_attempt_lifecycle_receipts",
            "work-attempt lifecycle schema",
        ),
    ],
    ids=[
        "revision-49",
        "revision-49-requires-profiles",
        "current",
        "missing-contracts",
        "missing-profiles",
        "missing-claim-index",
        "missing-preparation-holds",
        "missing-lifecycle-receipts",
    ],
)
def test_sqlite_verified_work_schema_validates_without_migration_or_store_imports(
    tmp_path, sqlite_resources, revision, require_profiles, damage, expected_error
):
    from cayu.storage import _sqlite_connection, _sqlite_support, migrations

    async def prepare():
        async with sqlite_resources as resources:
            path = tmp_path / "verified-work.sqlite"
            connection = resources.own(_sqlite_connection.connect(path), kind="connection")
            if revision is None:
                _sqlite_support.initialize_schema(connection)
            else:
                connection.execute(_sqlite_support._MIGRATIONS_TABLE_DDL)
                _sqlite_support._apply_baseline(connection)
                for pending in migrations.pending(migrations.BASELINE_REVISION):
                    if pending.revision > revision:
                        break
                    _sqlite_support._apply_revision(connection, pending)
            if damage is not None:
                connection.execute(damage)
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

from cayu.storage import _sqlite_verified_work_schema as verified_work_schema

with closing(sqlite3.connect(Path(sys.argv[1]).as_uri() + "?mode=ro", uri=True)) as connection:
    revision = int(connection.execute("PRAGMA user_version").fetchone()[0])
    try:
        verified_work_schema._validate_verified_work_schema(
            connection, require_verifier_profiles=sys.argv[2] == "True"
        )
        if revision >= 61:
            verified_work_schema._validate_work_attempt_admission_schema(connection)
        if revision >= 84:
            verified_work_schema._validate_work_attempt_lifecycle_schema(connection)
    except RuntimeError as error:
        assert sys.argv[3] and sys.argv[3] in str(error), str(error)
    else:
        assert not sys.argv[3], "incompatible schema was accepted"
    assert not connection.in_transaction

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_connection",
    "cayu.storage.migrations",
    "cayu.sessions.base",
    "cayu.tasks.base",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
    "cayu.verification.verified_task_worker",
):
    assert name not in sys.modules, name
""",
            str(path),
            str(require_profiles),
            expected_error or "",
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


@pytest.mark.parametrize(
    "missing_table",
    [
        None,
        "cayu_agent_work_context_publications",
        "cayu_agent_recall_delivery_claims",
        "cayu_agent_recall_subscription_claims",
    ],
)
def test_sqlite_work_context_schema_validates_without_migration_or_store_imports(
    tmp_path, sqlite_resources, missing_table
):
    from cayu.storage import _sqlite_connection, _sqlite_support

    async def prepare():
        async with sqlite_resources as resources:
            path = tmp_path / "work-context.sqlite"
            connection = resources.own(_sqlite_connection.connect(path), kind="connection")
            _sqlite_support.initialize_schema(connection)
            if missing_table is not None:
                connection.execute(f"DROP TABLE {missing_table}")
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

from cayu.storage import _sqlite_work_context_schema as work_context_schema

with closing(sqlite3.connect(Path(sys.argv[1]).as_uri() + "?mode=ro", uri=True)) as connection:
    try:
        work_context_schema._validate_revision_69_work_context_schema(connection)
        work_context_schema._validate_revision_71_recall_delivery_schema(
            connection, require_processing_schema_version=True
        )
        work_context_schema._validate_revision_73_recall_subscription_schema(connection)
    except RuntimeError as error:
        assert sys.argv[2] and sys.argv[2] in str(error), str(error)
    else:
        assert not sys.argv[2], "missing table was accepted"
    assert not connection.in_transaction

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_connection",
    "cayu.storage.migrations",
    "cayu.sessions.base",
    "cayu.tasks.base",
    "cayu.storage.work_context_sqlite",
    "cayu.storage.sqlite",
    "cayu.storage.postgres",
):
    assert name not in sys.modules, name
""",
            str(path),
            missing_table or "",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "missing_table",
    [
        None,
        "cayu_eval_baselines",
        "cayu_eval_cases",
        "cayu_eval_runs",
        "cayu_eval_scenarios",
        "cayu_eval_authored_suites",
        "cayu_eval_judge_calibrations",
        "cayu_eval_run_trial_checkpoints",
    ],
)
def test_sqlite_eval_schema_validates_without_migration_or_store_imports(
    tmp_path, sqlite_resources, missing_table
):
    from cayu.storage import _sqlite_connection, _sqlite_support

    async def prepare():
        async with sqlite_resources as resources:
            path = tmp_path / "eval-schema.sqlite"
            connection = resources.own(_sqlite_connection.connect(path), kind="connection")
            _sqlite_support.initialize_schema(connection)
            if missing_table is not None:
                connection.execute(f"DROP TABLE {missing_table}")
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

from cayu.storage import _sqlite_eval_schema as eval_schema

with closing(sqlite3.connect(Path(sys.argv[1]).as_uri() + "?mode=ro", uri=True)) as connection:
    try:
        eval_schema._validate_eval_result_baseline_schema(connection)
        eval_schema._validate_captured_eval_case_schema(connection)
        eval_schema._validate_eval_run_invocation_column(connection)
        eval_schema._validate_eval_scenario_schema(connection)
        eval_schema._validate_eval_run_scenario_progress_column(connection)
        eval_schema._validate_eval_authored_suite_schema(connection)
        eval_schema._validate_eval_judge_calibration_schema(connection)
        eval_schema._validate_eval_run_max_concurrency_schema(connection)
        eval_schema._validate_eval_run_trial_checkpoint_schema(connection)
    except RuntimeError as error:
        assert sys.argv[2] and sys.argv[2] in str(error), str(error)
    else:
        assert not sys.argv[2], "missing table was accepted"
    assert not connection.in_transaction

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_connection",
    "cayu.storage.migrations",
    "cayu.evals.store",
    "cayu.evals.execution",
    "cayu.storage.evals_sqlite",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
):
    assert name not in sys.modules, name
""",
            str(path),
            missing_table or "",
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
