"""SQLite domain validators work without migration or concrete-store imports."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


def _prepare_database(tmp_path, sqlite_resources, damage):
    from cayu.storage import _sqlite_connection, _sqlite_support

    async def prepare():
        async with sqlite_resources as resources:
            path = tmp_path / "validation.sqlite"
            connection = resources.own(_sqlite_connection.connect(path), kind="connection")
            _sqlite_support.initialize_schema(connection)
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
