"""SQLite domain validators work without migration or concrete-store imports."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


def _prepare_database(tmp_path, sqlite_resources, damage, *, revision=None, schema_sql=None):
    from cayu.storage import _sqlite_connection, _sqlite_support

    async def prepare():
        async with sqlite_resources as resources:
            path = tmp_path / "validation.sqlite"
            connection = resources.own(_sqlite_connection.connect(path), kind="connection")
            if schema_sql is not None:
                connection.executescript(schema_sql)
            elif revision is None:
                _sqlite_support.initialize_schema(connection)
            else:
                from cayu.storage import _sqlite_schema_history, migrations

                connection.execute(_sqlite_schema_history._MIGRATIONS_TABLE_DDL)
                _sqlite_support._apply_baseline(connection)
                for pending in migrations.pending(migrations.BASELINE_REVISION):
                    if pending.revision > revision:
                        break
                    _sqlite_support._apply_revision(connection, pending)
            if damage is not None:
                connection.executescript(damage)
                connection.commit()
            return path

    return asyncio.run(prepare())


def _validate_in_fresh_process(
    path, owner, validator, expected_error, *, projection=None, **arguments
):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

owner = importlib.import_module("cayu.storage." + sys.argv[2])
with closing(sqlite3.connect(Path(sys.argv[1]).as_uri() + "?mode=ro", uri=True)) as connection:
    projection = json.loads(sys.argv[6])
    if projection is not None:
        connection.create_function("cayu_transcript_search_document", 1, lambda _: projection)
    try:
        getattr(owner, sys.argv[3])(connection, **json.loads(sys.argv[5]))
    except RuntimeError as error:
        assert sys.argv[4] and sys.argv[4] in str(error), str(error)
    else:
        assert not sys.argv[4], "incompatible schema was accepted"
    assert not connection.in_transaction

for name in (
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_schema_history",
    "cayu.storage._sqlite_connection",
    "cayu.storage.migrations",
    "cayu.sessions.base",
    "cayu.tasks.base",
    "cayu.tasks.memory",
    "cayu.sessions.memory",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
    "cayu.runtime.task_worker",
):
    assert name not in sys.modules, name
""",
            str(path),
            owner,
            validator,
            expected_error or "",
            json.dumps(arguments),
            json.dumps(projection),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "DROP TABLE cayu_recall_receipts",
        "DROP TABLE cayu_context_exposures",
        "DROP TABLE cayu_recall_item_exposures",
        "ALTER TABLE cayu_recall_receipts RENAME COLUMN receipt_json TO old_receipt_json",
        "DROP INDEX idx_cayu_recall_receipts_session_page",
        "CREATE UNIQUE INDEX extra_recall_identity ON cayu_recall_receipts(receipt_id)",
        "DROP INDEX idx_cayu_context_exposures_session_page; "
        "CREATE INDEX idx_cayu_context_exposures_session_page "
        "ON cayu_context_exposures(session_id COLLATE NOCASE, created_at, exposure_id)",
        "DROP INDEX idx_cayu_recall_item_exposures_receipt; "
        "CREATE INDEX idx_cayu_recall_item_exposures_receipt "
        "ON cayu_recall_item_exposures(receipt_id, exposure_id, ordinal) WHERE ordinal > 0",
    ],
    ids=[
        "current",
        "missing-receipts",
        "missing-context-exposures",
        "missing-item-exposures",
        "wrong-receipt-column",
        "missing-page-index",
        "unexpected-unique-index",
        "wrong-index-collation",
        "partial-receipt-index",
    ],
)
def test_memory_evidence_schema_validates_independently(tmp_path, sqlite_resources, damage):
    path = _prepare_database(tmp_path, sqlite_resources, damage)
    _validate_in_fresh_process(
        path,
        "_sqlite_memory_evidence_schema",
        "_validate_memory_evidence_schema",
        "revision-51 memory evidence contract" if damage else None,
    )


@pytest.mark.parametrize(
    ("damage", "version", "projection", "expected_error"),
    [
        (None, None, "x76697369626c65", None),
        ("DROP TABLE cayu_transcript_messages", None, "x76697369626c65", "document column"),
        (
            "DROP TABLE cayu_transcript_search_configuration",
            None,
            "x76697369626c65",
            "tokenizer configuration is missing or malformed",
        ),
        (
            "DELETE FROM cayu_transcript_search_configuration",
            None,
            "x76697369626c65",
            "tokenizer identity conflicts",
        ),
        (None, "different-tokenizer", "x76697369626c65", "tokenizer identity conflicts"),
        (
            "UPDATE cayu_transcript_search_configuration SET tokenizer_version='different-tokenizer'",
            "different-tokenizer",
            "x76697369626c65",
            None,
        ),
        (
            "DROP TABLE cayu_transcript_messages_fts",
            None,
            "x76697369626c65",
            "schema is incomplete",
        ),
        (
            "DROP TRIGGER cayu_transcript_messages_fts_delete",
            None,
            "x76697369626c65",
            "schema is incomplete",
        ),
        (
            "DROP TRIGGER cayu_transcript_messages_fts_insert; "
            "CREATE TRIGGER cayu_transcript_messages_fts_insert "
            "AFTER INSERT ON cayu_transcript_messages BEGIN SELECT 1; END",
            None,
            "x76697369626c65",
            "maintenance triggers conflict",
        ),
        (
            "DROP TRIGGER cayu_transcript_messages_search_document_update; "
            "CREATE TRIGGER cayu_transcript_messages_search_document_update "
            "BEFORE UPDATE ON cayu_transcript_messages BEGIN SELECT 1; END",
            None,
            "x76697369626c65",
            "maintenance triggers conflict",
        ),
        (None, None, "visible hidden", "narrative-only boundary"),
    ],
    ids=[
        "current",
        "missing-transcript",
        "missing-configuration",
        "missing-tokenizer-row",
        "mismatched-expected-tokenizer",
        "supplied-tokenizer-identity",
        "missing-search-index",
        "missing-delete-trigger",
        "weakened-insert-trigger",
        "weakened-document-guard",
        "invalid-projection",
    ],
)
def test_transcript_schema_validates_independently(
    tmp_path, sqlite_resources, damage, version, projection, expected_error
):
    from cayu.sessions.base import TRANSCRIPT_SEARCH_TOKENIZER_VERSION

    path = _prepare_database(tmp_path, sqlite_resources, damage)
    _validate_in_fresh_process(
        path,
        "_sqlite_transcript_schema",
        "_validate_revision_46_transcript_search_schema",
        expected_error,
        expected_tokenizer_version=version or TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
        projection=projection,
    )


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "DROP TABLE cayu_task_session_closure_claims",
        "DROP TABLE cayu_session_closure_progress",
        "DROP TRIGGER cayu_task_closure_insert_guard",
        "DROP TRIGGER cayu_task_closure_update_guard",
        "DROP TRIGGER cayu_task_closure_insert_guard; "
        "CREATE TRIGGER cayu_task_closure_insert_guard "
        "BEFORE INSERT ON cayu_tasks BEGIN SELECT 1; END",
        "PRAGMA writable_schema=ON; "
        "UPDATE sqlite_master SET sql=replace(sql, '384000', '384001') "
        "WHERE type='table' AND name='cayu_session_closure_progress'; "
        "PRAGMA writable_schema=OFF",
    ],
    ids=[
        "current",
        "missing-task-claims",
        "missing-progress",
        "missing-insert-guard",
        "missing-update-guard",
        "weakened-insert-guard",
        "wrong-progress-limit",
    ],
)
def test_closure_schema_validates_independently(tmp_path, sqlite_resources, damage):
    path = _prepare_database(tmp_path, sqlite_resources, damage)
    _validate_in_fresh_process(
        path,
        "_sqlite_closure_schema",
        "_validate_revision_88_closure_schema",
        "closure schema is missing or conflicts" if damage else None,
    )


def test_schema_statement_iteration_preserves_trigger_bodies_and_literals():
    import sqlite3
    from contextlib import closing

    from cayu.storage import _sqlite_catalog

    script = """
        CREATE TABLE source(value TEXT);
        CREATE TABLE audit(value TEXT);
        CREATE TRIGGER record_value AFTER INSERT ON source
        BEGIN
            INSERT INTO audit VALUES ('literal; -- still text');
            INSERT INTO audit VALUES (NEW.value);
        END;
    """
    statements = list(_sqlite_catalog._iter_statements(script))
    assert len(statements) == 3
    with closing(sqlite3.connect(":memory:")) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.execute("INSERT INTO source VALUES ('retained')")
        assert connection.execute("SELECT value FROM audit ORDER BY rowid").fetchall() == [
            ("literal; -- still text",),
            ("retained",),
        ]


def test_schema_statement_iteration_rejects_incomplete_tail_after_complete_statement():
    from cayu.storage import _sqlite_catalog

    statements = _sqlite_catalog._iter_statements(
        "CREATE TABLE complete(value TEXT);\nCREATE TABLE incomplete("
    )
    assert next(statements) == "CREATE TABLE complete(value TEXT)"
    with pytest.raises(ValueError, match="migration DDL ended with an incomplete statement"):
        next(statements)


@pytest.mark.parametrize("script", ["", " \n\t"])
def test_schema_statement_iteration_accepts_empty_script(script):
    from cayu.storage import _sqlite_catalog

    assert list(_sqlite_catalog._iter_statements(script)) == []


def test_schema_definition_comparison_preserves_structural_tokens():
    from cayu.storage import _sqlite_catalog

    normalize = _sqlite_catalog._normalize_sqlite_schema_definition
    canonical = (
        "CREATE TABLE IF NOT EXISTS sample(value TEXT COLLATE BINARY CHECK(length(value)<4))"
    )
    formatted = (
        'create table "sample" ( [value] text collate binary check ( length (`value`) < 4 ) )'
    )
    assert normalize(canonical) == normalize(formatted)
    for changed in (
        canonical.replace("TEXT", "INTEGER"),
        canonical.replace("BINARY", "NOCASE"),
        canonical.replace("<4", "<5"),
    ):
        assert normalize(canonical) != normalize(changed)


@pytest.mark.parametrize(
    "owner,validator,revision,index_name,required_argument",
    [
        (
            "_sqlite_session_schema",
            "_validate_revision_17_indexes",
            17,
            "idx_cayu_events_pending_action_lookup",
            "require_all",
        ),
        (
            "_sqlite_session_schema",
            "_validate_pending_action_scope_indexes",
            23,
            "idx_cayu_events_pending_action_round_scope",
            "require_all",
        ),
        (
            "_sqlite_session_schema",
            "_validate_workflow_replay_indexes",
            29,
            "idx_cayu_events_workflow_step_replay",
            "require_all",
        ),
        (
            "_sqlite_budget_schema",
            "_validate_reservation_event_index",
            23,
            "idx_cayu_events_budget_reservation_identity",
            "require",
        ),
    ],
    ids=["pending-action", "pending-action-scope", "workflow-replay", "reservation-event"],
)
@pytest.mark.parametrize("required", [False, True], ids=["optional", "required"])
@pytest.mark.parametrize("damage", ["current", "missing", "conflicting", "incomplete-contract"])
def test_index_schema_validates_independently(
    tmp_path,
    sqlite_resources,
    owner,
    validator,
    revision,
    index_name,
    required_argument,
    required,
    damage,
):
    from cayu.storage import _sqlite_schema_history

    corruption = None
    expected_error = None
    if damage in {"missing", "conflicting"}:
        corruption = f"DROP INDEX {index_name};"
        if damage == "conflicting":
            corruption += f"CREATE INDEX {index_name} ON cayu_events(event_id);"
            expected_error = "conflicts"
        elif required:
            expected_error = "missing"
    revision_sql = _sqlite_schema_history._MIGRATION_STEPS[revision]
    if damage == "incomplete-contract":
        revision_sql = ""
        expected_error = "incomplete"
    path = _prepare_database(tmp_path, sqlite_resources, corruption)
    _validate_in_fresh_process(
        path,
        owner,
        validator,
        expected_error,
        revision_sql=revision_sql,
        **{required_argument: required},
    )


@pytest.mark.parametrize("damage", ["current", "missing", "wrong-order"])
def test_reservation_inventory_schema_validates_independently(tmp_path, sqlite_resources, damage):
    from cayu.storage import _sqlite_schema_history

    corruption = None
    if damage != "current":
        corruption = "DROP INDEX idx_cayu_budget_reservations_session_identity;"
        if damage == "wrong-order":
            corruption += (
                "CREATE INDEX idx_cayu_budget_reservations_session_identity "
                "ON cayu_budget_reservations(reservation_id, session_id);"
            )
    path = _prepare_database(tmp_path, sqlite_resources, corruption)
    _validate_in_fresh_process(
        path,
        "_sqlite_budget_schema",
        "_validate_reservation_inventory_index",
        None if damage == "current" else "inventory index",
        revision_sql=_sqlite_schema_history._MIGRATION_STEPS[109],
    )


@pytest.mark.parametrize(
    "damage,require,verify_event_ownership,error",
    [
        ("current", True, True, None),
        ("missing", False, False, None),
        ("missing", True, False, "missing"),
        ("wrong-column", False, False, "conflicts"),
        ("unregistered-event", True, False, None),
        ("unregistered-event", True, True, "disagree"),
        ("matching-event", True, True, None),
        ("wrong-publication", True, True, "disagree"),
    ],
    ids=[
        "current",
        "optional-missing",
        "required-missing",
        "wrong-column",
        "schema-only",
        "unregistered-event",
        "matching-event",
        "wrong-publication",
    ],
)
def test_reservation_registry_validates_independently(
    tmp_path,
    sqlite_resources,
    damage,
    require,
    verify_event_ownership,
    error,
):
    from cayu.storage import _sqlite_catalog, _sqlite_schema_history

    registry_sql = next(
        statement
        for statement in _sqlite_catalog._iter_statements(_sqlite_schema_history._BASELINE_DDL)
        if statement.startswith("CREATE TABLE IF NOT EXISTS cayu_budget_reservation_identities")
    )
    # Only the event projection read by this check is needed; no session runtime
    # or public-alias trigger participates in reservation ownership validation.
    schema_sql = (
        registry_sql + "; CREATE TABLE cayu_events ("
        "session_id TEXT, event_id TEXT, event_type TEXT, timestamp TEXT, payload_json TEXT);"
    )
    corruption = None
    if damage == "missing":
        corruption = "DROP TABLE cayu_budget_reservation_identities"
    elif damage == "wrong-column":
        corruption = (
            "ALTER TABLE cayu_budget_reservation_identities "
            "RENAME COLUMN publication_id TO unexpected_id"
        )
    elif damage in {"unregistered-event", "matching-event", "wrong-publication"}:
        corruption = (
            "INSERT INTO cayu_events(session_id, event_id, event_type, timestamp, payload_json) "
            "VALUES ('session', 'event', 'budget.reserved', '2026-01-01T00:00:00Z', "
            '\'{"reservation_id":"reservation"}\');'
        )
        if damage != "unregistered-event":
            publication = "event" if damage == "matching-event" else "another-event"
            corruption += (
                "INSERT INTO cayu_budget_reservation_identities "
                "(reservation_id, publication_session_id, publication_id, published) "
                f"VALUES ('reservation', 'session', '{publication}', 1);"
            )
    path = _prepare_database(tmp_path, sqlite_resources, corruption, schema_sql=schema_sql)
    _validate_in_fresh_process(
        path,
        "_sqlite_budget_schema",
        "_validate_reservation_identity_registry",
        error,
        require=require,
        verify_event_ownership=verify_event_ownership,
    )


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "DROP TABLE cayu_producer_cleanup_receipts",
        "DROP TABLE cayu_producer_cleanup_retirements",
        "DROP INDEX idx_cayu_producer_cleanup_namespace; "
        "CREATE INDEX idx_cayu_producer_cleanup_namespace "
        "ON cayu_producer_cleanup_receipts(generation, namespace_key, operation_key)",
    ],
    ids=["current", "missing-receipts", "missing-retirement-fence", "wrong-index-order"],
)
def test_producer_cleanup_schema_validates_independently(tmp_path, sqlite_resources, damage):
    from cayu.storage import _sqlite_schema_history

    path = _prepare_database(tmp_path, sqlite_resources, damage)
    _validate_in_fresh_process(
        path,
        "_sqlite_producer_schema",
        "_validate_producer_cleanup_receipts",
        "missing or conflicting" if damage else None,
        revision_sql=_sqlite_schema_history._MIGRATION_STEPS[110],
    )


@pytest.mark.parametrize(
    "damage,error",
    [
        (None, None),
        ("DROP TABLE cayu_local_execution_attempts", "storage conflicts"),
        (
            "DROP INDEX idx_cayu_local_execution_attempts_recovery; "
            "CREATE INDEX idx_cayu_local_execution_attempts_recovery "
            "ON cayu_local_execution_attempts(phase, retry_admissible, updated_at, attempt_id)",
            "conflicts",
        ),
    ],
    ids=["current", "missing-attempts", "wrong-recovery-order"],
)
def test_local_execution_schema_validates_independently(
    tmp_path,
    sqlite_resources,
    damage,
    error,
):
    path = _prepare_database(tmp_path, sqlite_resources, damage)
    _validate_in_fresh_process(
        path,
        "_sqlite_task_schema",
        "_validate_local_execution_attempt_schema",
        error,
    )


@pytest.mark.parametrize(
    "damage,error",
    [
        (None, None),
        (
            "ALTER TABLE cayu_knowledge_chunks RENAME COLUMN fts_rowid TO invalid_key",
            "stable FTS key",
        ),
        ("DROP INDEX idx_cayu_knowledge_chunks_entry_index", "index is missing"),
        (
            "DROP TABLE cayu_knowledge_chunks_fts; "
            "CREATE VIRTUAL TABLE cayu_knowledge_chunks_fts USING fts5(text)",
            "FTS does not match",
        ),
    ],
    ids=["revision-37", "wrong-key", "missing-index", "wrong-fts-definition"],
)
def test_historical_knowledge_schema_validates_independently(
    tmp_path,
    sqlite_resources,
    damage,
    error,
):
    path = _prepare_database(tmp_path, sqlite_resources, damage, revision=37)
    _validate_in_fresh_process(
        path,
        "_sqlite_knowledge_schema",
        "_validate_revision_37_knowledge_fts_schema",
        error,
    )


def test_reconciliation_supplies_current_revision_sql(tmp_path, sqlite_resources, monkeypatch):
    from cayu.storage import _sqlite_connection, _sqlite_schema_history, _sqlite_support, migrations

    async def exercise():
        async with sqlite_resources as resources:
            connection = resources.own(
                _sqlite_connection.connect(tmp_path / "contract.sqlite"), kind="connection"
            )
            _sqlite_support.initialize_schema(connection)
            original = _sqlite_schema_history._MIGRATION_STEPS[109]
            with monkeypatch.context() as contract:
                contract.setitem(
                    _sqlite_schema_history._MIGRATION_STEPS,
                    109,
                    "CREATE INDEX idx_cayu_budget_reservations_session_identity "
                    "ON cayu_budget_reservations(reservation_id, session_id);",
                )
                with pytest.raises(RuntimeError, match="inventory index"):
                    _sqlite_support.reconcile_schema(connection, migrations.SchemaMode.VALIDATE)
            assert _sqlite_schema_history._MIGRATION_STEPS[109] == original
            _sqlite_support.reconcile_schema(connection, migrations.SchemaMode.VALIDATE)
            assert not connection.in_transaction

    asyncio.run(exercise())
