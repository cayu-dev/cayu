"""Private relational record mapping; all SQL identifiers are schema-owned."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from typing import Any, Literal, cast

from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    CollaborationContractError,
    ContractValue,
    snapshot_input,
)
from cayu.collaboration._history_references import history_references
from cayu.collaboration._permits import PermitSnapshot
from cayu.collaboration._request_receipts import record_operation
from cayu.collaboration.base import Key, Table
from cayu.collaboration.participants import ParticipantEvent
from cayu.collaboration.requests import RequestSnapshot
from cayu.storage._collaboration_schema import EXTRA_COLUMNS, KEYS
from cayu.vaults.redaction import SecretRedactor


class _SQLRepository:
    def __init__(self, connection: Any, scope: str, *, postgres: bool) -> None:
        self.connection = connection
        self.scope = scope
        self.postgres = postgres

    async def now_ms(self) -> int:
        if not self.postgres:
            return time.time_ns() // 1_000_000
        rows = await self._rows(
            await self._execute(
                "SELECT floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint"
            )
        )
        value = rows[0][0]
        if type(value) is not int or not 1 <= value <= 2**53 - 1:
            raise CollaborationContractError("Collaboration owner time is unavailable.")
        return value

    async def scan_due_requests(self, *, after: int, now_ms: int, limit: int) -> list[object]:
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), participant_id, position, state, next_due_at_ms, namespace, generation, caller_key "
                "FROM cayu_collaboration_requests WHERE scope=? AND state='open' "
                "AND next_due_at_ms<=? AND position>? ORDER BY position LIMIT ?",
                (self.scope, now_ms, after, limit),
            )
        )
        result = []
        for row in rows:
            value = self._decode(row[0])
            self._require_request_projection(value, tuple(row[1:5]), tuple(row[5:]))
            result.append(value)
        return result

    async def scan_request_events(self, *, after: int, limit: int) -> list[object]:
        rows = await self._rows(
            await self._execute(
                f"SELECT sequence, substr(document, 1, {MAX_ENVELOPE_BYTES + 1}) "
                "FROM cayu_collaboration_request_events "
                "WHERE scope=? AND sequence>? ORDER BY sequence LIMIT ?",
                (self.scope, after, limit),
            )
        )
        result = []
        for sequence, document in rows:
            value = self._decode(document)
            if not isinstance(value, dict):
                raise CollaborationContractError("Request event is not an object.")
            if cast("dict[str, Any]", value)["sequence"] != sequence:
                raise CollaborationContractError("Request event index contradicts its record.")
            result.append(value)
        return result

    async def _execute(self, sql: str, args: tuple = ()):
        if self.postgres:
            return await self.connection.execute(sql.replace("?", "%s"), args)
        return self.connection.execute(sql, args)

    async def _rows(self, cursor):
        return await cursor.fetchall() if self.postgres else cursor.fetchall()

    @staticmethod
    def _decode(document: object) -> object:
        if type(document) is not str or len(document.encode("utf-8")) > MAX_ENVELOPE_BYTES:
            raise CollaborationContractError("Collaboration record exceeds its envelope bound.")
        return json.loads(document)

    def _where(self, table: Table, key: Key) -> tuple[str, tuple]:
        columns = KEYS[table]
        if len(columns) != len(key):
            raise ValueError("Invalid collaboration record key.")
        return " AND ".join(f"{name} = ?" for name in ("scope", *columns)), (self.scope, *key)

    async def get(self, table: Table, key: Key) -> object | None:
        where, args = self._where(table, key)
        extra = EXTRA_COLUMNS.get(table, ())
        projection = "" if not extra else ", " + ", ".join(extra)
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}){projection} FROM cayu_collaboration_{table} WHERE {where}",
                args,
            )
        )
        if not rows:
            return None
        value = self._decode(rows[0][0])
        if table == "permits":
            self._require_permit_projection(value, tuple(rows[0][1:]), key)
        elif table == "requests":
            self._require_request_projection(value, tuple(rows[0][1:]), key)
        return value

    def _require_request_projection(self, value: object, projection: tuple, key: Key) -> None:
        matches = False
        if isinstance(value, dict):
            document = cast("dict[str, Any]", value)
            with suppress(TypeError, KeyError):
                expected = document["receipt"]["expected"]
                operation = expected["operation"]
                matches = (
                    projection
                    == (
                        expected["intent"]["selection"]["recipient"]["reference"]["participant_id"],
                        document["receipt"]["event"]["sequence"],
                        document["state"],
                        document["next_due_at_ms"],
                    )
                    and operation["application_scope"] == self.scope
                    and key
                    == (
                        operation["namespace_incarnation"],
                        operation["generation"],
                        operation["caller_key"],
                    )
                )
        if not matches:
            raise CollaborationContractError("Request lookup index contradicts its record.")

    def _require_permit_projection(self, value: object, projection: tuple, key: Key) -> None:
        matches = False
        if isinstance(value, dict):
            document = cast("dict[str, Any]", value)
            with suppress(TypeError, KeyError):
                operation = document["expected"]["operation"]
                matches = (
                    projection
                    == (
                        document["expected"]["intent"]["request"]["participant"]["participant_id"],
                        document["position"],
                        document["state"],
                    )
                    and operation["application_scope"] == self.scope
                    and key
                    == (
                        operation["namespace_incarnation"],
                        operation["generation"],
                        operation["caller_key"],
                    )
                )
        if not matches:
            raise CollaborationContractError("Permit lookup index contradicts its record.")

    async def put(self, table: Table, key: Key, value: ContractValue, *, insert: bool) -> None:
        extra = EXTRA_COLUMNS.get(table, ())
        columns = ("scope", *KEYS[table], *extra)
        extra_values: tuple = ()
        if table == "permits":
            if not isinstance(value, PermitSnapshot):
                raise CollaborationContractError("Permit record requires its typed projection.")
            extra_values = (
                value.expected.intent.request.participant.participant_id,
                value.position,
                value.state,
            )
        elif table == "requests":
            if not isinstance(value, RequestSnapshot):
                raise CollaborationContractError("Request record requires its typed projection.")
            extra_values = (
                value.receipt.expected.intent.selection.recipient.reference.participant_id,
                value.receipt.event.sequence,
                value.state,
                value.next_due_at_ms,
            )
        document = json.dumps(
            snapshot_input(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        sql = f"INSERT INTO cayu_collaboration_{table} ({', '.join(columns)}, document) VALUES ({', '.join('?' for _ in (*columns, 'document'))})"
        if not insert:
            sql += f" ON CONFLICT ({', '.join(('scope', *KEYS[table]))}) DO UPDATE SET document = excluded.document"
            sql += "".join(f", {column} = excluded.{column}" for column in extra)
        await self._execute(sql, (self.scope, *key, *extra_values, document))
        if table == "operations":
            for family, participant_id, revision in history_references(value):
                await self._execute(
                    "INSERT INTO cayu_collaboration_history_uses (scope, family, participant_id, revision, namespace, generation, caller_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (self.scope, family, participant_id, revision, *key),
                )
        if isinstance(value, ParticipantEvent):
            for ref in value.participants:
                await self._execute(
                    "INSERT INTO cayu_collaboration_event_participants (scope, sequence, participant_id) VALUES (?, ?, ?)",
                    (self.scope, value.sequence, ref.participant_id),
                )

    async def delete(self, table: Table, key: Key) -> None:
        if table == "operations":
            await self._execute(
                "DELETE FROM cayu_collaboration_history_uses WHERE scope=? AND namespace=? AND generation=? AND caller_key=?",
                (self.scope, *key),
            )
        if table == "events":
            await self._execute(
                "DELETE FROM cayu_collaboration_event_participants WHERE scope=? AND sequence=?",
                (self.scope, *key),
            )
        where, args = self._where(table, key)
        await self._execute(f"DELETE FROM cayu_collaboration_{table} WHERE {where}", args)

    async def history_in_use(self, family: str, participant_id: str, revision: int) -> bool:
        rows = await self._rows(
            await self._execute(
                "SELECT 1 FROM cayu_collaboration_history_uses WHERE scope=? AND family=? AND participant_id=? AND revision=? LIMIT 1",
                (self.scope, family, participant_id, revision),
            )
        )
        return bool(rows)

    async def scan_operations(self, namespace: str, generation: int, *, limit: int) -> list[object]:
        rows = await self._rows(
            await self._execute(
                f"SELECT caller_key, substr(document, 1, {MAX_ENVELOPE_BYTES + 1}) FROM cayu_collaboration_operations WHERE scope=? AND namespace=? AND generation=? ORDER BY caller_key LIMIT ?",
                (self.scope, namespace, generation, limit),
            )
        )
        records = []
        for caller_key, document in rows:
            value = self._decode(document)
            if not isinstance(value, dict):
                raise CollaborationContractError("Operation index record is malformed.")
            value = cast("dict[str, Any]", value)
            operation = record_operation(value, redactor=SecretRedactor())
            if (
                operation.application_scope,
                operation.namespace_incarnation,
                operation.generation,
                operation.caller_key,
            ) != (self.scope, namespace, generation, caller_key):
                raise CollaborationContractError("Operation index contradicts its authority.")
            records.append(value)
        return records

    async def scan(
        self,
        table: Literal["participants", "events"],
        *,
        after: str | int,
        limit: int,
        allowed: tuple[str, ...] | None,
    ) -> list[object]:
        column = "participant_id" if table == "participants" else "sequence"
        sql = f"SELECT e.{column}, substr(e.document, 1, {MAX_ENVELOPE_BYTES + 1}) FROM cayu_collaboration_{table} e WHERE e.scope = ? AND e.{column} > ?"
        args: tuple = (self.scope, after)
        if allowed is not None:
            if not allowed:
                return []
            placeholders = ", ".join("?" for _ in allowed)
            if table == "participants":
                sql += f" AND e.participant_id IN ({placeholders})"
            else:
                sql += " AND EXISTS (SELECT 1 FROM cayu_collaboration_event_participants p WHERE p.scope=e.scope AND p.sequence=e.sequence)"
                sql += f" AND NOT EXISTS (SELECT 1 FROM cayu_collaboration_event_participants p WHERE p.scope=e.scope AND p.sequence=e.sequence AND p.participant_id NOT IN ({placeholders}))"
            args += allowed
        sql += f" ORDER BY e.{column} LIMIT ?"
        result = []
        for key, document in await self._rows(await self._execute(sql, (*args, limit))):
            value = self._decode(document)
            if not isinstance(value, dict):
                raise CollaborationContractError("Collaboration record is not an object.")
            value = cast("dict[str, Any]", value)
            actual = (
                value["reference"]["participant_id"]
                if table == "participants"
                else value["sequence"]
            )
            if key != actual:
                raise ValueError("Collaboration query index contradicts its record.")
            result.append(value)
        return result

    async def scan_permits(
        self, participant_id: str, *, after: int, limit: int, pending_only: bool
    ) -> list[object]:
        sql = f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), participant_id, position, state, namespace, generation, caller_key FROM cayu_collaboration_permits WHERE scope = ? AND participant_id = ? AND position > ?"
        if pending_only:
            sql += " AND state = 'pending'"
        sql += " ORDER BY position LIMIT ?"
        values = []
        for row in await self._rows(
            await self._execute(sql, (self.scope, participant_id, after, limit))
        ):
            value = self._decode(row[0])
            self._require_permit_projection(value, tuple(row[1:4]), tuple(row[4:]))
            values.append(value)
        return values
