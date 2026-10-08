"""Canonical SQLite closure DDL and validation, independent of migration history."""

from __future__ import annotations

import re
import sqlite3

from cayu.storage import _sqlite_catalog as sqlite_catalog

SQLITE_CLOSURE_DDL = """
        CREATE TABLE IF NOT EXISTS cayu_task_session_closure_claims (
            session_id TEXT COLLATE BINARY PRIMARY KEY,
            plan_id TEXT COLLATE BINARY NOT NULL CHECK (
                length(plan_id) = 64 AND plan_id NOT GLOB '*[^0-9a-f]*'
            ),
            claim_json TEXT NOT NULL CHECK (
                json_valid(claim_json) AND json_type(claim_json) = 'object'
                AND length(CAST(claim_json AS BLOB)) BETWEEN 1 AND 16777216
            )
        );
        CREATE TRIGGER IF NOT EXISTS cayu_task_closure_insert_guard
        BEFORE INSERT ON cayu_tasks
        WHEN EXISTS (
            SELECT 1 FROM cayu_task_session_closure_claims WHERE session_id = NEW.session_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'Task session is owned by closure.');
        END;
        CREATE TRIGGER IF NOT EXISTS cayu_task_closure_update_guard
        BEFORE UPDATE ON cayu_tasks
        WHEN EXISTS (
            SELECT 1 FROM cayu_task_session_closure_claims
            WHERE session_id IN (OLD.session_id, NEW.session_id)
        )
        BEGIN
            SELECT RAISE(ABORT, 'Task session is owned by closure.');
        END;
        CREATE TABLE IF NOT EXISTS cayu_session_closure_progress (
            root_session_id TEXT COLLATE BINARY NOT NULL,
            plan_id TEXT COLLATE BINARY NOT NULL CHECK (
                length(plan_id) = 64 AND plan_id NOT GLOB '*[^0-9a-f]*'
            ),
            progress_json TEXT NOT NULL CHECK (
                json_valid(progress_json) AND json_type(progress_json) = 'object'
                AND length(CAST(progress_json AS BLOB)) BETWEEN 1 AND 384000
            ),
            PRIMARY KEY (root_session_id, plan_id)
        );
    """


def _validate_revision_88_closure_schema(connection: sqlite3.Connection) -> None:
    for statement in sqlite_catalog._iter_statements(SQLITE_CLOSURE_DDL):
        match = re.match(r"CREATE (TABLE|TRIGGER) IF NOT EXISTS (\w+)", statement)
        if match is None:
            raise RuntimeError("Unrecognized closure schema definition.")
        object_type, name = match.groups()
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type.lower(), name),
        ).fetchone()
        if row is None or sqlite_catalog._normalize_sqlite_schema_definition(row[0]) != (
            sqlite_catalog._normalize_sqlite_schema_definition(statement)
        ):
            raise RuntimeError("SQLite closure schema is missing or conflicts with its contract.")
