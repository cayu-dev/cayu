"""Private relational record mapping; all SQL identifiers are schema-owned."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from typing import Any, Literal, cast

from cayu.collaboration._clarification_records import (
    CLARIFICATION_RECORD_FAMILIES,
    clarification_record_projection,
    due_cursor_key,
    prepare_due_scan,
    prepare_lineage_scan,
    prepare_question_scan,
)
from cayu.collaboration._clarification_state import ClarificationQuestionState
from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    CollaborationContractError,
    ContractValue,
    snapshot_input,
)
from cayu.collaboration._history_references import history_references
from cayu.collaboration._permits import PermitSnapshot
from cayu.collaboration._planning_records import (
    PENDING_PLANNING_STATES,
    PLANNING_RECORD_FAMILIES,
    RequestPlanningCursor,
    RequestPlanningRecord,
    RequestPlanningStageRecord,
    planning_cursor_key,
    planning_record_projection,
    prepare_planning_scan,
)
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

    async def _planning_rows(self, where: str, args: tuple, order: str, limit: int) -> list[object]:
        columns = EXTRA_COLUMNS["request_plans"]
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), {', '.join(columns)}, "
                "namespace, generation, caller_key FROM cayu_collaboration_request_plans "
                f"WHERE scope=? AND {where} ORDER BY {order} LIMIT ?",
                (self.scope, *args, limit),
            )
        )
        result = []
        for row in rows:
            value = self._decode(row[0])
            record, projection = planning_record_projection(
                "request_plans", value, scope=self.scope, key=tuple(row[1 + len(columns) :])
            )
            assert isinstance(record, RequestPlanningRecord)
            if tuple(row[1 : 1 + len(columns)]) != projection:
                raise CollaborationContractError("Planning scan index contradicts its record.")
            result.append(value)
        return result

    async def scan_request_plans(self, request, *, limit):
        from cayu.collaboration.requests import RequestRef

        request = prepare_planning_scan(request, scope=self.scope, limit=limit, kind="request")
        assert isinstance(request, RequestRef)
        rows = await self._planning_rows(
            "request_id=? AND request_incarnation=?",
            (request.request_id, request.incarnation),
            "planning_generation",
            limit,
        )
        for raw in rows:
            from cayu.collaboration._preparation import prepare_contract

            record = prepare_contract(RequestPlanningRecord, raw, redactor=SecretRedactor())
            if record.receipt.command.expected.intent.selection.reference != request:
                raise CollaborationContractError("Planning request owner contradicts its index.")
        return rows

    async def _request_plan_stages(self, operation, *, limit, native):
        operation = prepare_planning_scan(
            operation, scope=self.scope, limit=limit, kind="native_stage" if native else "stages"
        )
        prefix = "native" if native else "plan"
        columns = EXTRA_COLUMNS["request_plan_stages"]
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), {', '.join(columns)}, "
                "namespace, generation, caller_key FROM cayu_collaboration_request_plan_stages "
                f"WHERE scope=? AND {prefix}_namespace=? AND {prefix}_generation=? "
                f"AND {prefix}_key=? ORDER BY ordinal LIMIT ?",
                (
                    self.scope,
                    operation.namespace_incarnation,
                    operation.generation,
                    operation.caller_key,
                    2 if native else limit,
                ),
            )
        )
        if native and len(rows) > 1:
            raise CollaborationContractError("Native operation has conflicting planning stages.")
        result = []
        for row in rows:
            value = self._decode(row[0])
            record, projection = planning_record_projection(
                "request_plan_stages", value, scope=self.scope, key=tuple(row[1 + len(columns) :])
            )
            assert isinstance(record, RequestPlanningStageRecord)
            target = record.intent.command.operation if native else record.intent.plan
            if tuple(row[1 : 1 + len(columns)]) != projection or target != operation:
                raise CollaborationContractError("Planning stage index contradicts its record.")
            result.append(value)
        return result

    async def scan_request_plan_stages(self, plan, *, limit):
        return await self._request_plan_stages(plan, limit=limit, native=False)

    async def find_request_plan_stage(self, native_operation):
        rows = await self._request_plan_stages(native_operation, limit=1, native=True)
        return rows[0] if rows else None

    async def scan_pending_request_plans(self, *, after, limit):
        cursor = prepare_planning_scan(after, scope=self.scope, limit=limit, kind="pending")
        assert cursor is None or isinstance(cursor, RequestPlanningCursor)
        where = (
            f"(state IN ({', '.join('?' for _ in PENDING_PLANNING_STATES)}) OR pending_stages>0)"
        )
        args = tuple(PENDING_PLANNING_STATES)
        if cursor is not None:
            where += " AND (next_due_at_ms, namespace, generation, caller_key) > (?, ?, ?, ?)"
            args = (*args, *planning_cursor_key(cursor))
        return await self._planning_rows(
            where, args, "next_due_at_ms, namespace, generation, caller_key", limit
        )

    async def scan_clarification_questions(self, request, *, limit: int) -> list[object]:
        request = prepare_question_scan(request, limit)
        if request.owner.application_scope != self.scope:
            raise CollaborationContractError("Clarification scan belongs to another scope.")
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), "
                "request_id, request_incarnation, participant_id, state, next_due_at_ms, "
                "lineage_namespace, lineage_generation, lineage_key, "
                "namespace, generation, caller_key FROM cayu_collaboration_clarification_questions "
                "WHERE scope=? AND request_id=? AND request_incarnation=? "
                "ORDER BY namespace, generation, caller_key LIMIT ?",
                (self.scope, request.request_id, request.incarnation, limit),
            )
        )
        result = []
        for row in rows:
            value = self._decode(row[0])
            record, projection = clarification_record_projection(
                "clarification_questions", value, scope=self.scope, key=tuple(row[9:])
            )
            assert isinstance(record, ClarificationQuestionState)
            if projection != tuple(row[1:9]) or record.question.request != request:
                raise CollaborationContractError("Clarification scan index contradicts its record.")
            result.append(value)
        return result

    async def scan_clarification_lineage_questions(self, lineage, *, limit):
        lineage = prepare_lineage_scan(lineage, limit, scope=self.scope)
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), "
                "request_id, request_incarnation, participant_id, state, next_due_at_ms, "
                "lineage_namespace, lineage_generation, lineage_key, namespace, generation, caller_key "
                "FROM cayu_collaboration_clarification_questions "
                "WHERE scope=? AND lineage_namespace=? AND lineage_generation=? AND lineage_key=? "
                "ORDER BY namespace, generation, caller_key LIMIT ?",
                (
                    self.scope,
                    lineage.namespace_incarnation,
                    lineage.generation,
                    lineage.caller_key,
                    limit,
                ),
            )
        )
        result = []
        for row in rows:
            value = self._decode(row[0])
            record, projection = clarification_record_projection(
                "clarification_questions", value, scope=self.scope, key=tuple(row[9:])
            )
            assert isinstance(record, ClarificationQuestionState)
            if projection != tuple(row[1:9]) or record.question.lineage != lineage:
                raise CollaborationContractError("Lineage index contradicts its question.")
            result.append(value)
        return result

    async def _execute(self, sql: str, args: tuple = ()):
        if self.postgres:
            return await self.connection.execute(sql.replace("?", "%s"), args)
        return self.connection.execute(sql, args)

    async def scan_pending_clarification_services(self, *, after, limit: int) -> list[object]:
        return await self._scan_pending_clarification_records(
            "clarification_services", after=after, limit=limit
        )

    async def scan_clarification_request_handoffs(
        self, request, *, family, limit, pending_only=False
    ):
        from cayu.collaboration._clarification_records import handoff_request, prepare_handoff_scan

        request, schema = prepare_handoff_scan(
            family, request, limit, scope=self.scope, pending_only=pending_only
        )
        state_filter = "AND state='pending' " if pending_only else ""
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), "
                "participant_id, state, next_due_at_ms, request_id, request_incarnation, "
                "namespace, generation, caller_key "
                f"FROM cayu_collaboration_{family} WHERE scope=? AND request_id=? AND request_incarnation=? "
                + state_filter
                + "ORDER BY namespace, generation, caller_key LIMIT ?",
                (self.scope, request.request_id, request.incarnation, limit),
            )
        )
        result = []
        for row in rows:
            value = self._decode(row[0])
            record, projection = clarification_record_projection(
                family, value, scope=self.scope, key=tuple(row[6:])
            )
            assert isinstance(record, schema)
            if projection != tuple(row[1:6]) or handoff_request(record) != request:
                raise CollaborationContractError("Handoff request index contradicts its record.")
            if pending_only and record.state != "pending":
                raise CollaborationContractError("Handoff pending index contradicts its record.")
            result.append(value)
        return result

    async def scan_pending_clarification_deliveries(self, *, after, limit: int) -> list[object]:
        return await self._scan_pending_clarification_records(
            "clarification_deliveries", after=after, limit=limit
        )

    async def _scan_pending_clarification_records(
        self,
        family: Literal["clarification_services", "clarification_deliveries"],
        *,
        after,
        limit: int,
    ) -> list[object]:
        cursor = prepare_due_scan(scope=self.scope, after=after, now_ms=1, limit=limit)
        continuation = ""
        args = (self.scope,)
        if cursor is not None:
            continuation = " AND (next_due_at_ms, namespace, generation, caller_key) > (?, ?, ?, ?)"
            args = (*args, *due_cursor_key(cursor))
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), "
                "participant_id, state, next_due_at_ms, request_id, request_incarnation, "
                "namespace, generation, caller_key "
                f"FROM cayu_collaboration_{family} WHERE scope=? AND state='pending'"
                + continuation
                + " ORDER BY next_due_at_ms, namespace, generation, caller_key LIMIT ?",
                (*args, limit),
            )
        )
        values = []
        for row in rows:
            value = self._decode(row[0])
            _, projection = clarification_record_projection(
                family, value, scope=self.scope, key=tuple(row[6:])
            )
            if projection != tuple(row[1:6]):
                raise CollaborationContractError(
                    "Clarification responsibility index contradicts its record."
                )
            values.append(value)
        return values

    async def scan_due_clarifications(self, *, after, now_ms: int, limit: int) -> list[object]:
        cursor = prepare_due_scan(scope=self.scope, after=after, now_ms=now_ms, limit=limit)
        continuation = ""
        args = (self.scope, now_ms)
        if cursor is not None:
            continuation = " AND (next_due_at_ms, namespace, generation, caller_key) > (?, ?, ?, ?)"
            args = (*args, *due_cursor_key(cursor))
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(document, 1, {MAX_ENVELOPE_BYTES + 1}), "
                "request_id, request_incarnation, participant_id, state, next_due_at_ms, "
                "lineage_namespace, lineage_generation, lineage_key, "
                "namespace, generation, caller_key FROM cayu_collaboration_clarification_questions "
                "WHERE scope=? AND state='open' AND next_due_at_ms<=?"
                + continuation
                + " ORDER BY next_due_at_ms, namespace, generation, caller_key LIMIT ?",
                (*args, limit),
            )
        )
        values = []
        for row in rows:
            value = self._decode(row[0])
            _, projection = clarification_record_projection(
                "clarification_questions", value, scope=self.scope, key=tuple(row[9:])
            )
            if projection != tuple(row[1:9]):
                raise CollaborationContractError("Clarification due index contradicts its record.")
            values.append(value)
        return values

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
        if table in PLANNING_RECORD_FAMILIES:
            _, expected_projection = planning_record_projection(
                table, value, scope=self.scope, key=key
            )
            if tuple(rows[0][1:]) != expected_projection:
                raise CollaborationContractError("Planning secondary index contradicts its record.")
        elif table == "permits":
            self._require_permit_projection(value, tuple(rows[0][1:]), key)
        elif table == "requests":
            self._require_request_projection(value, tuple(rows[0][1:]), key)
        elif table in CLARIFICATION_RECORD_FAMILIES:
            _, projection = clarification_record_projection(table, value, scope=self.scope, key=key)
            if tuple(rows[0][1:]) != projection:
                raise CollaborationContractError(
                    "Clarification secondary index contradicts its record."
                )
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
        if table in PLANNING_RECORD_FAMILIES:
            value, extra_values = planning_record_projection(
                table, value, scope=self.scope, key=key
            )
        elif table in CLARIFICATION_RECORD_FAMILIES:
            value, extra_values = clarification_record_projection(
                table, value, scope=self.scope, key=key
            )
        elif table == "permits":
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
            from cayu.collaboration.waits import WaitSnapshot

            if isinstance(value, WaitSnapshot):
                from cayu.collaboration._wait_discovery import wait_projection

                _, projection = wait_projection(value, scope=self.scope, key=key)
                await self._execute(
                    "INSERT INTO cayu_collaboration_wait_discovery "
                    "(scope, namespace, generation, caller_key, state, delivery) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (scope, namespace, generation, caller_key) DO UPDATE "
                    "SET state=excluded.state, delivery=excluded.delivery",
                    (self.scope, *key, *projection),
                )
            if not insert:
                # Replacing an owner record and its exact history projection is
                # one transaction. Re-inserting the old pins either conflicts or
                # accumulates stale references across lifecycle transitions.
                await self._execute(
                    "DELETE FROM cayu_collaboration_history_uses WHERE scope=? AND namespace=? AND generation=? AND caller_key=?",
                    (self.scope, *key),
                )
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
                "DELETE FROM cayu_collaboration_wait_discovery "
                "WHERE scope=? AND namespace=? AND generation=? AND caller_key=?",
                (self.scope, *key),
            )
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

    async def scan_waits(self, *, namespace, after, limit):
        from cayu.collaboration._wait_discovery import wait_projection

        # Match the native Python cursor comparison on every database locale.
        # The predicate and ordering must use the same indexable expression.
        key = 'i.caller_key COLLATE "C"' if self.postgres else "i.caller_key COLLATE BINARY"
        where = ""
        args: tuple = (self.scope, namespace)
        if after is not None:
            where = f" AND (i.generation, {key}) > (?, ?)"
            args += after
        rows = await self._rows(
            await self._execute(
                f"SELECT substr(o.document, 1, {MAX_ENVELOPE_BYTES + 1}), "
                "i.namespace, i.generation, i.caller_key, i.state, i.delivery "
                "FROM cayu_collaboration_wait_discovery i "
                "LEFT JOIN cayu_collaboration_operations o ON "
                "(o.scope, o.namespace, o.generation, o.caller_key) = "
                "(i.scope, i.namespace, i.generation, i.caller_key) "
                f"WHERE i.scope=? AND i.namespace=?{where} "
                f"ORDER BY i.generation, {key} LIMIT ?",
                (*args, limit),
            )
        )
        result = []
        for document, namespace, generation, key, state, delivery in rows:
            if document is None:
                raise CollaborationContractError("Wait discovery lost its source record.")
            record, projection = wait_projection(
                self._decode(document), scope=self.scope, key=(namespace, generation, key)
            )
            if projection != (state, delivery):
                raise CollaborationContractError("Wait discovery contradicts its source state.")
            result.append(snapshot_input(record))
        return result

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
