"""In-process collaboration identity storage with atomic owned publication."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, cast

from cayu.collaboration._clarification_records import (
    CLARIFICATION_RECORD_FAMILIES,
    clarification_record_projection,
    due_cursor_key,
    prepare_due_scan,
    prepare_lineage_scan,
    prepare_question_scan,
)
from cayu.collaboration._clarification_state import ClarificationQuestionState
from cayu.collaboration._contracts import CollaborationContractError, ContractValue, snapshot_input
from cayu.collaboration._history_references import history_references
from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration.base import CollaborationStore, Key, Table
from cayu.collaboration.participants import CollaborationUnavailable


class _MemoryRepository:
    def __init__(self, rows: dict[tuple[Table, Key], object], *, scope: str) -> None:
        self.rows = rows
        self.scope = scope

    async def now_ms(self) -> int:
        return time.time_ns() // 1_000_000

    async def get(self, table: Table, key: Key) -> object | None:
        value = deepcopy(self.rows.get((table, key)))
        if value is not None and table in CLARIFICATION_RECORD_FAMILIES:
            record, _ = clarification_record_projection(table, value, scope=self.scope, key=key)
            return snapshot_input(record)
        return value

    async def put(self, table: Table, key: Key, value: ContractValue, *, insert: bool) -> None:
        if insert and (table, key) in self.rows:
            raise CollaborationUnavailable("Collaboration unique record already exists.")
        if table in CLARIFICATION_RECORD_FAMILIES:
            value, _ = clarification_record_projection(table, value, scope=self.scope, key=key)
        self.rows[table, key] = snapshot_input(value)
        if table == "operations":
            for history in history_references(value):
                self.rows["history_uses", (*history, *key)] = True

    async def delete(self, table: Table, key: Key) -> None:
        if table == "operations":
            for entry in tuple(self.rows):
                if entry[0] == "history_uses" and entry[1][3:] == key:
                    del self.rows[entry]
        self.rows.pop((table, key), None)

    async def history_in_use(self, family, participant_id, revision):
        return any(
            table == "history_uses" and key[:3] == (family, participant_id, revision)
            for table, key in self.rows
        )

    async def scan_operations(self, namespace, generation, *, limit):
        records = [
            (key[2], value)
            for (table, key), value in self.rows.items()
            if table == "operations" and key[:2] == (namespace, generation)
        ]
        records.sort(key=lambda item: item[0])
        return [deepcopy(value) for _, value in records[:limit]]

    async def scan(self, table, *, after, limit, allowed):
        rows: list[tuple[Any, dict[str, Any]]] = []
        for (family, key), document in sorted(self.rows.items(), key=lambda item: str(item[0])):
            if family != table or key[0] <= after:
                continue
            assert isinstance(document, dict)
            document = cast("dict[str, Any]", document)
            if allowed is not None:
                ids = (
                    (key[0],)
                    if table == "participants"
                    else tuple(p["participant_id"] for p in document["participants"])
                )
                if not ids or any(identity not in allowed for identity in ids):
                    continue
            rows.append((key[0], document))
        rows.sort(key=lambda item: item[0])
        return [deepcopy(document) for _, document in rows[:limit]]

    async def scan_permits(self, participant_id, *, after, limit, pending_only):
        values = []
        for (family, _), raw in self.rows.items():
            if family != "permits":
                continue
            assert isinstance(raw, dict)
            value = cast("dict[str, Any]", raw)
            if (
                value["expected"]["intent"]["request"]["participant"]["participant_id"]
                == participant_id
                and value["position"] > after
                and (not pending_only or value["state"] == "pending")
            ):
                values.append(value)
        values.sort(key=lambda value: value["position"])
        return deepcopy(values[:limit])

    async def scan_due_requests(self, *, after, now_ms, limit):
        values = []
        for (family, _), raw in self.rows.items():
            if family != "requests":
                continue
            assert isinstance(raw, dict)
            value = cast("dict[str, Any]", raw)
            if (
                value["state"] == "open"
                and value["receipt"]["event"]["sequence"] > after
                and value["next_due_at_ms"] <= now_ms
            ):
                values.append(value)
        values.sort(key=lambda item: item["receipt"]["event"]["sequence"])
        return deepcopy(values[:limit])

    async def scan_request_events(self, *, after, limit):
        values = [
            (key[0], value)
            for (family, key), value in self.rows.items()
            if family == "request_events" and key[0] > after
        ]
        values.sort(key=lambda item: item[0])
        return deepcopy([value for _, value in values[:limit]])

    async def scan_clarification_questions(self, request, *, limit):
        request = prepare_question_scan(request, limit)
        if request.owner.application_scope != self.scope:
            raise CollaborationContractError("Clarification scan belongs to another scope.")
        result = []
        for (family, key), value in sorted(self.rows.items()):
            if family != "clarification_questions":
                continue
            record, _ = clarification_record_projection(family, value, scope=self.scope, key=key)
            assert isinstance(record, ClarificationQuestionState)
            candidate = record.question.request
            if (candidate.request_id, candidate.incarnation) == (
                request.request_id,
                request.incarnation,
            ):
                if candidate != request:
                    raise CollaborationContractError(
                        "Clarification scan index contradicts its record."
                    )
                result.append(snapshot_input(record))
                if len(result) == limit:
                    break
        return result

    async def scan_pending_clarification_services(self, *, after, limit):
        return await self._scan_pending_clarification_records(
            "clarification_services", after=after, limit=limit
        )

    async def scan_clarification_lineage_questions(self, lineage, *, limit):
        lineage = prepare_lineage_scan(lineage, limit, scope=self.scope)
        result = []
        for (family, key), value in sorted(self.rows.items()):
            if family != "clarification_questions":
                continue
            record, _ = clarification_record_projection(family, value, scope=self.scope, key=key)
            assert isinstance(record, ClarificationQuestionState)
            if record.question.lineage == lineage:
                result.append(snapshot_input(record))
                if len(result) == limit:
                    break
        return result

    async def scan_clarification_request_handoffs(
        self, request, *, family, limit, pending_only=False
    ):
        from cayu.collaboration._clarification_records import handoff_request, prepare_handoff_scan

        request, schema = prepare_handoff_scan(
            family, request, limit, scope=self.scope, pending_only=pending_only
        )
        result = []
        for (stored_family, key), value in sorted(self.rows.items()):
            if stored_family != family:
                continue
            record, _ = clarification_record_projection(family, value, scope=self.scope, key=key)
            assert isinstance(record, schema)
            candidate = handoff_request(record)
            if (candidate.request_id, candidate.incarnation) != (
                request.request_id,
                request.incarnation,
            ):
                continue
            if candidate != request:
                raise CollaborationContractError("Handoff request index contradicts its record.")
            if pending_only and record.state != "pending":
                continue
            result.append(snapshot_input(record))
            if len(result) == limit:
                break
        return result

    async def scan_pending_clarification_deliveries(self, *, after, limit):
        return await self._scan_pending_clarification_records(
            "clarification_deliveries", after=after, limit=limit
        )

    async def _scan_pending_clarification_records(self, selected_family, *, after, limit):
        from cayu.collaboration._clarification_deliveries import ClarificationDeliveryRecord
        from cayu.collaboration._clarification_services import ClarificationServiceRecord

        cursor = prepare_due_scan(scope=self.scope, after=after, now_ms=1, limit=limit)
        values = []
        for (family, key), value in self.rows.items():
            if family != selected_family:
                continue
            record, _ = clarification_record_projection(family, value, scope=self.scope, key=key)
            if isinstance(record, ClarificationServiceRecord):
                operation = record.dispatch.intent.operation
                deadline = record.dispatch.intent.question.deadline_at_ms
            else:
                assert isinstance(record, ClarificationDeliveryRecord)
                operation = record.intent.operation
                deadline = record.intent.append.attempt_key.deadline_at_ms
            order = (
                deadline,
                operation.namespace_incarnation,
                operation.generation,
                operation.caller_key,
            )
            if record.state == "pending" and (cursor is None or order > due_cursor_key(cursor)):
                values.append((order, record))
        values.sort(key=lambda item: item[0])
        return [snapshot_input(record) for _, record in values[:limit]]

    async def scan_due_clarifications(self, *, after, now_ms, limit):
        cursor = prepare_due_scan(scope=self.scope, after=after, now_ms=now_ms, limit=limit)
        values = []
        for (family, key), value in self.rows.items():
            if family != "clarification_questions":
                continue
            record, _ = clarification_record_projection(family, value, scope=self.scope, key=key)
            assert isinstance(record, ClarificationQuestionState)
            operation = record.question.operation
            order = (
                record.question.deadline_at_ms,
                operation.namespace_incarnation,
                operation.generation,
                operation.caller_key,
            )
            if (
                record.state == "open"
                and record.question.deadline_at_ms <= now_ms
                and (cursor is None or order > due_cursor_key(cursor))
            ):
                values.append((order, record))
        values.sort(key=lambda item: item[0])
        return [snapshot_input(record) for _, record in values[:limit]]


class InMemoryCollaborationStore(CollaborationStore):
    request_contract_version = 2

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._scopes: dict[str, dict[tuple[Table, Key], object]] = {}
        self._owners = _MutationOwners()

    @asynccontextmanager
    async def _transaction(self, scope: str, *, write: bool):
        async with self._lock:
            if self._owners.closed and asyncio.current_task() not in self._owners.pending:
                raise CollaborationUnavailable("Collaboration store is closing.")
            rows = deepcopy(self._scopes.get(scope, {}))
            yield _MemoryRepository(rows, scope=scope)
            if write:
                self._scopes[scope] = rows

    async def close(self) -> None:
        await self._owners.drain()
