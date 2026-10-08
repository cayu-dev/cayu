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


def _validate_in_fresh_process(path, owner, validator, expected_error, **arguments):
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
