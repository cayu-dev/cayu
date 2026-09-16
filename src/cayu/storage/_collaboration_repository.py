"""Private relational record mapping; all SQL identifiers are schema-owned."""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    CollaborationContractError,
    ContractValue,
    snapshot_input,
)
from cayu.collaboration.base import Key, Table
from cayu.collaboration.participants import ParticipantEvent
from cayu.storage._collaboration_schema import KEYS


class _SQLRepository:
    def __init__(self, connection: Any, scope: str, *, postgres: bool) -> None:
        self.connection = connection
        self.scope = scope
        self.postgres = postgres

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
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}) FROM cayu_collaboration_{table} WHERE {where}",
                args,
            )
        )
        return None if not rows else self._decode(rows[0][0])

    async def put(self, table: Table, key: Key, value: ContractValue, *, insert: bool) -> None:
        columns = ("scope", *KEYS[table])
        document = json.dumps(
            snapshot_input(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        sql = f"INSERT INTO cayu_collaboration_{table} ({', '.join(columns)}, document) VALUES ({', '.join('?' for _ in (*columns, 'document'))})"
        if not insert:
            sql += f" ON CONFLICT ({', '.join(columns)}) DO UPDATE SET document = excluded.document"
        await self._execute(sql, (self.scope, *key, document))
        if isinstance(value, ParticipantEvent):
            for ref in value.participants:
                await self._execute(
                    "INSERT INTO cayu_collaboration_event_participants (scope, sequence, participant_id) VALUES (?, ?, ?)",
                    (self.scope, value.sequence, ref.participant_id),
                )

    async def delete(self, table: Table, key: Key) -> None:
        where, args = self._where(table, key)
        await self._execute(f"DELETE FROM cayu_collaboration_{table} WHERE {where}", args)

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
