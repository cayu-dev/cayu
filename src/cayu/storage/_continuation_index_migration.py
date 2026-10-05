"""Upgrade continuation indexes without rewriting committed native receipts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

from cayu.sessions._session_continuation import ContinuationRecord
from cayu.sessions._session_continuation_store import (
    MAX_RETAINED_TICKETS,
    ROOT_KEY,
    ContinuationRoot,
    digest,
    require_history,
    service_receipt_epochs,
)

_PAGE_SIZE = 64


def _keys(checkpoint: dict[str, Any]) -> tuple[str, ...]:
    root = checkpoint[ROOT_KEY]
    if type(root) is not dict or type(root.get("schema_version")) is not int:
        raise ValueError("Continuation migration requires a versioned index.")
    if root["schema_version"] == 3:
        ContinuationRoot.model_validate(root)
        return ()
    if root["schema_version"] != 2:
        raise ValueError("Unsupported continuation index migration version.")
    entries = root.get("entries")
    if type(entries) is not list or len(entries) > MAX_RETAINED_TICKETS:
        raise ValueError("Continuation migration index exceeds its bound.")
    keys = tuple(entry["ticket_key"] for entry in entries)
    if any(type(key) is not str for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Continuation migration index has invalid identities.")
    return keys


def upgrade_continuation_index(
    checkpoint: dict[str, Any],
    records: dict[str, Any],
    *,
    session_id: str,
    session_instance_id: str,
) -> dict[str, Any]:
    """Derive purpose only from the exact record already committed by each index."""
    _keys(checkpoint)
    root = checkpoint[ROOT_KEY]
    if root["schema_version"] == 3:
        return checkpoint
    entries = []
    for entry in root["entries"]:
        key = entry["ticket_key"]
        raw = records.get(key)
        if raw is None or "purpose" in entry or digest(raw) != entry["record_sha256"]:
            raise ValueError(
                f"Continuation migration lacks exact indexed receiving evidence for ticket {key!r}."
            )
        try:
            record = ContinuationRecord.model_validate(raw)
            require_history(record)
        except ValueError as error:
            # Keep record content out of the message; the cause retains the details.
            raise ValueError(
                f"Continuation migration record for ticket {key!r} is invalid."
            ) from error
        if (
            record.namespace.model_dump(mode="json") != root["namespace"]
            or record.ticket.session_id != session_id
            or record.ticket.session_instance_id != session_instance_id
            or record.ticket.state != entry["state"]
            or record.ticket.writer_generation != entry["originating_writer_generation"]
            or service_receipt_epochs(record) != tuple(entry.get("service_receipt_epochs", ()))
        ):
            raise ValueError(
                f"Continuation migration receiving identity conflicts for ticket {key!r}."
            )
        entries.append({**entry, "purpose": record.ticket.purpose})
    upgraded = ContinuationRoot.model_validate({**root, "schema_version": 3, "entries": entries})
    if (
        upgraded.namespace.session_id != session_id
        or upgraded.namespace.session_instance_id != session_instance_id
    ):
        raise ValueError("Continuation migration namespace belongs to another session.")
    return {**checkpoint, ROOT_KEY: upgraded.model_dump(mode="json")}


@contextmanager
def _naming_session(session_id: str):
    # One unreadable index rolls back the whole revision, so name the session to fix.
    try:
        yield
    except ValueError as error:
        raise ValueError(
            f"Continuation index migration failed for session {session_id!r}: {error}"
        ) from error


def migrate_sqlite_continuation_indexes(connection) -> None:
    """Called inside revision 115's transaction; any conflict rolls back the revision."""
    after = ""
    while True:
        rows = connection.execute(
            "SELECT c.session_id,s.instance_id,c.state_json FROM cayu_checkpoints AS c "
            "JOIN cayu_sessions AS s ON s.id=c.session_id "
            "WHERE c.session_id>? AND json_type(c.state_json,'$.session_continuations') IS NOT NULL "
            "ORDER BY c.session_id LIMIT ?",
            (after, _PAGE_SIZE),
        ).fetchall()
        if not rows:
            return
        for session_id, instance_id, encoded in rows:
            checkpoint = json.loads(encoded)
            records = {}
            with _naming_session(session_id):
                for key in _keys(checkpoint):
                    row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id=? AND idempotency_key=?",
                        (session_id, key),
                    ).fetchone()
                    if row is not None:
                        records[key] = json.loads(row[0])
                updated = upgrade_continuation_index(
                    checkpoint, records, session_id=session_id, session_instance_id=instance_id
                )
            if updated != checkpoint:
                connection.execute(
                    "UPDATE cayu_checkpoints SET state_json=? WHERE session_id=?",
                    (json.dumps(updated, ensure_ascii=False, separators=(",", ":")), session_id),
                )
        after = rows[-1][0]


async def migrate_postgres_continuation_indexes(cursor) -> None:
    """Use the same authenticated upgrade under the revision's schema writer lock.

    Rows are read without locks, so the revision holds row locks only on checkpoints
    it rewrites. Each rewrite compares the read state, and a concurrent change rolls
    back the revision instead of being overwritten.
    """
    from psycopg.types.json import Jsonb

    after = ""
    while True:
        await cursor.execute(
            "SELECT c.session_id,s.instance_id,c.state FROM cayu_checkpoints AS c "
            "JOIN cayu_sessions AS s ON s.id=c.session_id "
            "WHERE c.session_id>%s AND c.state ? 'session_continuations' "
            "ORDER BY c.session_id LIMIT %s",
            (after, _PAGE_SIZE),
        )
        rows = await cursor.fetchall()
        if not rows:
            return
        for session_id, instance_id, checkpoint in rows:
            records = {}
            with _naming_session(session_id):
                for key in _keys(checkpoint):
                    await cursor.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id=%s AND idempotency_key=%s",
                        (session_id, key),
                    )
                    row = await cursor.fetchone()
                    if row is not None:
                        records[key] = row[0]
                updated = upgrade_continuation_index(
                    checkpoint, records, session_id=session_id, session_instance_id=instance_id
                )
            if updated != checkpoint:
                await cursor.execute(
                    "UPDATE cayu_checkpoints SET state=%s WHERE session_id=%s AND state=%s",
                    (Jsonb(updated), session_id, Jsonb(checkpoint)),
                )
                if cursor.rowcount != 1:
                    raise ValueError(
                        f"Continuation index for session {session_id!r} changed during "
                        "migration; retry the migration."
                    )
        after = rows[-1][0]
