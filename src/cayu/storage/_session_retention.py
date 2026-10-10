"""Session-store retention, shared by the SQLite and Postgres session stores.

Planning and every apply transaction use the same protection evaluator, so a
dry run lists exactly what an apply with the same inputs and store state acts
on. A parent/fork lineage is one retention unit: it is pruned only when every
session in it is selectable and unprotected, and each lineage is re-read,
re-checked and changed in its own write transaction together with its audit
entries. The store's write lock is released between lineages.

Planning reads the session graph and the large reference tables from a
non-blocking snapshot (a SQLite reader connection, or an ordinary Postgres
read), then evaluates protections in bounded batches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Collection, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cayu.storage import _retention_sql as audit
from cayu.storage._retention_sql import (
    PostgresRetentionSql,
    RetentionSql,
    SQLiteRetentionSql,
    chunks,
    marks,
)
from cayu.storage.retention import (
    MAX_RETENTION_ITEMS_PER_RUN,
    RetentionAuditEntry,
    RetentionAuditRecord,
    RetentionDisposition,
    RetentionItem,
    RetentionMode,
    RetentionPhase,
    RetentionProgress,
    RetentionProgressCallback,
    RetentionProtection,
    RetentionReport,
    SessionRetentionPolicy,
    bounded_retention_detail,
    copy_protected_ids,
    retention_audit_summary,
)

if TYPE_CHECKING:
    from cayu.sessions.records import Session

STORE_KIND = "sessions"
#: Lineages evaluated per planning batch before the store is released again.
PLAN_BATCH_SESSIONS = 128
_DELTA_EVENT_TYPES = ("model.text.delta", "model.thinking.delta")
_TOOL_OUTPUT_EVENT_TYPES = ("tool.call.completed", "tool.call.failed")
_TERMINAL_TASK_STATUSES = ("completed", "failed", "cancelled", "dependency_skipped")
_TERMINAL_CLARIFICATION_QUESTION_STATES = (
    "answered",
    "superseded",
    "cancelled",
    "expired",
    "request_terminal",
)
_TERMINAL_REQUEST_STATES = ("answered", "failed", "declined", "cancelled", "expired")
_SESSION_EXPORT_PREFIX = "session-export:"
_SNAPSHOT_TABLES = audit.SNAPSHOT_TABLES
_EVAL_TABLES = (
    "cayu_eval_runs",
    "cayu_eval_results",
    "cayu_eval_result_records",
    "cayu_eval_run_trial_checkpoints",
)


ExternalProtections = dict[str, frozenset[RetentionProtection]]


def external_protections(
    protected_ids: Collection[str],
    references: Mapping[RetentionProtection, Collection[str]] | None,
) -> ExternalProtections:
    """Merge caller-named ids and other stores' references into per-id reasons."""

    merged: dict[str, set[RetentionProtection]] = {}
    for identifier in copy_protected_ids(protected_ids, "protected_ids"):
        merged.setdefault(identifier, set()).add(RetentionProtection.CALLER_PROTECTED)
    for protection, identifiers in (references or {}).items():
        reason = RetentionProtection(protection)
        for identifier in copy_protected_ids(identifiers, "references"):
            merged.setdefault(identifier, set()).add(reason)
    return {identifier: frozenset(reasons) for identifier, reasons in merged.items()}


# -- backends ---------------------------------------------------------------------


class SessionRetentionBackend(ABC):
    """The store-specific seams the shared session retention core needs."""

    def __init__(self, store: Any) -> None:
        self.store = store

    @property
    def read_only(self) -> bool:
        return bool(getattr(self.store, "_read_only", False))

    @abstractmethod
    async def now(self, sql: RetentionSql) -> datetime: ...

    @abstractmethod
    def snapshot(self) -> Any:
        """A non-blocking read snapshot (async context manager -> RetentionSql)."""

    @abstractmethod
    def check_batch(self) -> Any:
        """A read transaction in which the store's own guards can run."""

    @abstractmethod
    def write(self) -> Any:
        """A write transaction holding the store's write lock; commits on success."""

    @abstractmethod
    async def lock_sessions(self, sql: RetentionSql, session_ids: Iterable[str]) -> None: ...

    @abstractmethod
    async def load_session(self, sql: RetentionSql, session_id: str) -> Session | None: ...

    @abstractmethod
    async def has_closure_owner(self, sql: RetentionSql, session_id: str) -> bool: ...

    @abstractmethod
    async def require_store_guards(self, sql: RetentionSql, session: Session) -> None:
        """Run the store's deletion admission and erasure quiescence guards."""

    @abstractmethod
    async def delete(self, sql: RetentionSql, session_id: str) -> bool: ...

    # Column names differ between the backends' schemas.
    event_payload: str = "payload_json"
    transcript_message: str = "message_json"
    checkpoint_state: str = "state_json"
    operation_record: str = "record_json"
    eval_sources: tuple[tuple[str, str], ...] = (
        ("cayu_eval_runs", "scenario_progress_json"),
        ("cayu_eval_results", "result_json"),
        ("cayu_eval_result_records", "captured_result_json"),
        ("cayu_eval_run_trial_checkpoints", "checkpoint_json"),
    )


class SQLiteSessionRetentionBackend(SessionRetentionBackend):
    async def now(self, sql: RetentionSql) -> datetime:
        return self.store._ownership_clock()

    @asynccontextmanager
    async def snapshot(self) -> AsyncIterator[RetentionSql]:
        reader = await self.store._available_readers.get()
        try:
            lock, connection = reader
            async with lock:
                connection.execute("BEGIN")
                try:
                    yield SQLiteRetentionSql(connection)
                finally:
                    connection.rollback()
        finally:
            self.store._available_readers.put_nowait(reader)

    @asynccontextmanager
    async def check_batch(self) -> AsyncIterator[RetentionSql]:
        async with self.store._lock:
            connection = self.store._connection
            connection.execute("BEGIN")
            try:
                yield SQLiteRetentionSql(connection)
            finally:
                connection.rollback()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[RetentionSql]:
        store = self.store
        await store._participant_creation_lock.acquire()
        try:
            async with store._lock:
                connection = store._connection
                connection.execute("BEGIN IMMEDIATE")
                try:
                    yield SQLiteRetentionSql(connection)
                except BaseException:
                    connection.rollback()
                    raise
                connection.commit()
        finally:
            store._participant_creation_lock.release()

    async def lock_sessions(self, sql: RetentionSql, session_ids: Iterable[str]) -> None:
        return None  # BEGIN IMMEDIATE already excludes every other writer.

    async def load_session(self, sql: RetentionSql, session_id: str) -> Session | None:
        return self.store._load_unlocked(session_id)

    async def has_closure_owner(self, sql: RetentionSql, session_id: str) -> bool:
        assert isinstance(sql, SQLiteRetentionSql)
        return bool(
            self.store._closure_lineage_owners_unlocked((session_id,), connection=sql.connection)
        )

    async def require_store_guards(self, sql: RetentionSql, session: Session) -> None:
        self.store._require_session_deletion_admission_unlocked(session)
        self.store._require_session_erasure_quiescence_unlocked(session)

    async def delete(self, sql: RetentionSql, session_id: str) -> bool:
        return self.store._delete_session_in_transaction_unlocked(session_id)


class PostgresSessionRetentionBackend(SessionRetentionBackend):
    event_payload = "payload"
    transcript_message = "message"
    checkpoint_state = "state"
    operation_record = "record"
    eval_sources = (
        ("cayu_eval_runs", "scenario_progress_json"),
        ("cayu_eval_results", "result"),
        ("cayu_eval_result_records", "captured_result"),
        ("cayu_eval_run_trial_checkpoints", "checkpoint_json"),
    )

    async def now(self, sql: RetentionSql) -> datetime:
        assert isinstance(sql, PostgresRetentionSql)
        return await self.store._session_store_now(sql.cursor)

    @asynccontextmanager
    async def _transaction(self, *, commit: bool) -> AsyncIterator[RetentionSql]:
        await self.store._ensure_ready()
        async with self.store._connection() as connection:
            try:
                async with connection.cursor() as cursor:
                    yield PostgresRetentionSql(cursor)
            except BaseException:
                await connection.rollback()
                raise
            if commit:
                await connection.commit()
            else:
                await connection.rollback()

    def snapshot(self) -> Any:
        return self._transaction(commit=False)

    def check_batch(self) -> Any:
        return self._transaction(commit=False)

    def write(self) -> Any:
        return self._transaction(commit=True)

    async def lock_sessions(self, sql: RetentionSql, session_ids: Iterable[str]) -> None:
        assert isinstance(sql, PostgresRetentionSql)
        ordered = sorted(set(session_ids))
        # Same lock order as delete_session: lineage, view pins, then rows.
        await self.store._lock_closure_lineage(sql.cursor)
        for session_id in ordered:
            await sql.run(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (f"context-view-session:{session_id}",),
            )
        for chunk in chunks(ordered):
            await sql.all(
                f"SELECT id FROM cayu_sessions WHERE id IN ({marks(chunk)}) "
                'ORDER BY id COLLATE "C" FOR UPDATE',
                chunk,
            )

    async def load_session(self, sql: RetentionSql, session_id: str) -> Session | None:
        assert isinstance(sql, PostgresRetentionSql)
        return await self.store._load(sql.cursor, session_id)

    async def has_closure_owner(self, sql: RetentionSql, session_id: str) -> bool:
        assert isinstance(sql, PostgresRetentionSql)
        return bool(await self.store._closure_lineage_owners(sql.cursor, (session_id,)))

    async def require_store_guards(self, sql: RetentionSql, session: Session) -> None:
        assert isinstance(sql, PostgresRetentionSql)
        await self.store._require_session_deletion_admission(sql.cursor, session)
        await self.store._require_session_erasure_quiescence(sql.cursor, session)

    async def delete(self, sql: RetentionSql, session_id: str) -> bool:
        assert isinstance(sql, PostgresRetentionSql)
        return await self.store._delete_session_in_transaction(sql.cursor, session_id)


# -- records ----------------------------------------------------------------------


@dataclass(frozen=True)
class _SessionRow:
    id: str
    parent_session_id: str | None
    status: str
    updated_at: datetime


@dataclass
class _References:
    """Reference sets read from large evidence tables in the current transaction."""

    eval_session_ids: frozenset[str]
    knowledge_session_ids: frozenset[str]
    knowledge_event_ids: tuple[str, ...]
    snapshot_text: str


@dataclass
class _Selected:
    row: _SessionRow
    component: frozenset[str]
    counts: dict[str, int]
    bytes: int


@dataclass
class _Plan:
    cutoff: datetime
    selected: list[_Selected] = field(default_factory=list)
    protected: list[RetentionItem] = field(default_factory=list)
    deferred: int = 0


@dataclass
class _CompactionPlan:
    delete_sequences: tuple[int, ...]
    delete_bytes: int
    #: (sequence, new payload, new event document or None, bytes saved)
    rewrites: tuple[tuple[int, Any, Any, int], ...]
    rewrite_bytes: int

    @property
    def counts(self) -> dict[str, int]:
        return {
            "delta_events_removed": len(self.delete_sequences),
            "tool_outputs_compacted": len(self.rewrites),
        }

    @property
    def bytes(self) -> int:
        return self.delete_bytes + self.rewrite_bytes


class _Blocked(Exception):
    def __init__(self, items: list[RetentionItem]) -> None:
        super().__init__("retention lineage blocked")
        self.items = items


# -- entry points -----------------------------------------------------------------


async def apply_retention_policy(
    backend: SessionRetentionBackend,
    policy: SessionRetentionPolicy,
    *,
    protected_session_ids: Collection[str] = (),
    references: Mapping[RetentionProtection, Collection[str]] | None = None,
    progress: RetentionProgressCallback | None = None,
) -> RetentionReport:
    if type(policy) is not SessionRetentionPolicy:
        raise TypeError("policy must be a SessionRetentionPolicy.")
    caller_protected = external_protections(protected_session_ids, references)
    if not policy.dry_run and backend.read_only:
        raise PermissionError("Retention apply requires a writable session store.")
    async with backend.snapshot() as sql:
        if not policy.dry_run:
            await audit.require_audit_tables(sql)
        started_at = await backend.now(sql)
        rows = await _session_rows(sql)
        stored = await _read_references(sql, backend)
    plan = await _plan(backend, policy, caller_protected, stored, rows, now=started_at)
    if progress is not None:
        await progress(
            RetentionProgress(
                store_kind=STORE_KIND,
                phase=RetentionPhase.PLANNED,
                dry_run=policy.dry_run,
                planned_items=len(plan.selected),
                protected=_bounded(plan.protected),
            )
        )
    if policy.dry_run:
        async with backend.snapshot() as sql:
            completed_at = await backend.now(sql)
        return RetentionReport(
            store_kind=STORE_KIND,
            mode=policy.mode,
            dry_run=True,
            policy=policy.policy_document(),
            started_at=started_at,
            completed_at=completed_at,
            cutoff=plan.cutoff,
            items=tuple(
                _item(selected, RetentionDisposition.SELECTED) for selected in plan.selected
            ),
            protected=_bounded(plan.protected),
            protected_truncated=len(plan.protected) > MAX_RETENTION_ITEMS_PER_RUN,
            deferred_count=plan.deferred,
        )
    audit_id = uuid4().hex
    async with backend.write() as sql:
        await audit.begin_audit(
            sql,
            audit_id=audit_id,
            store_kind=STORE_KIND,
            mode=policy.mode,
            started_at=started_at,
            policy=policy.policy_document(),
        )
    applied: list[RetentionItem] = []
    protected = list(plan.protected)
    for index, group in enumerate(_component_groups(plan.selected)):
        batch_applied: list[RetentionItem] = []
        batch_protected: list[RetentionItem] = []
        try:
            async with backend.write() as sql:
                stored = await _read_references(sql, backend, lock=True)
                batch_applied = await _apply_group(
                    backend, sql, policy, caller_protected, stored, plan.cutoff, group, audit_id
                )
        except _Blocked as blocked:
            batch_protected = blocked.items
        applied.extend(batch_applied)
        protected.extend(batch_protected)
        if progress is not None:
            await progress(
                RetentionProgress(
                    store_kind=STORE_KIND,
                    phase=RetentionPhase.BATCH,
                    dry_run=False,
                    audit_id=audit_id,
                    batch_index=index,
                    planned_items=len(plan.selected),
                    applied=tuple(batch_applied),
                    protected=tuple(batch_protected),
                )
            )
        await asyncio.sleep(0)  # Let queued store work run between lineages.
    async with backend.write() as sql:
        completed_at = await backend.now(sql)
        report = RetentionReport(
            store_kind=STORE_KIND,
            mode=policy.mode,
            dry_run=False,
            policy=policy.policy_document(),
            started_at=started_at,
            completed_at=completed_at,
            cutoff=plan.cutoff,
            audit_id=audit_id,
            items=tuple(applied),
            protected=_bounded(protected),
            protected_truncated=len(protected) > MAX_RETENTION_ITEMS_PER_RUN,
            deferred_count=plan.deferred,
        )
        await audit.complete_audit(
            sql,
            audit_id=audit_id,
            completed_at=completed_at,
            summary=retention_audit_summary(report),
        )
    return report


async def list_retention_audits(
    backend: SessionRetentionBackend,
    *,
    limit: int,
    item_id: str | None,
    store_kind: str | None,
) -> tuple[RetentionAuditRecord, ...]:
    async with backend.snapshot() as sql:
        return await audit.list_audits(sql, limit=limit, store_kind=store_kind, item_id=item_id)


async def load_retention_audit(
    backend: SessionRetentionBackend, audit_id: str
) -> RetentionAuditRecord | None:
    async with backend.snapshot() as sql:
        return await audit.load_audit(sql, audit_id)


async def inspect_session_protections(
    backend: SessionRetentionBackend,
    session_ids: Collection[str],
    *,
    mode: RetentionMode,
    store_guards: bool,
) -> dict[str, tuple[RetentionProtection, ...]]:
    """Direct protections for named sessions, without lineage or age selection.

    Missing sessions are omitted. Session status is not reported here; callers
    that act on sessions check it themselves.
    """

    wanted = sorted(copy_protected_ids(session_ids, "session_ids"))
    async with backend.snapshot() as sql:
        references = await _read_references(sql, backend)
    result: dict[str, tuple[RetentionProtection, ...]] = {}
    for batch in chunks(wanted, PLAN_BATCH_SESSIONS):
        async with backend.check_batch() as sql:
            for chunk in chunks(batch):
                for row in await sql.all(
                    "SELECT id, parent_session_id, status, updated_at FROM cayu_sessions "
                    f"WHERE id IN ({marks(chunk)})",
                    chunk,
                ):
                    reasons, _detail = await _session_protections(
                        backend,
                        sql,
                        mode,
                        {},
                        references,
                        _session_row(sql, row),
                        store_guards=store_guards,
                    )
                    result[row[0]] = reasons
        await asyncio.sleep(0)
    return result


async def session_artifact_references(backend: SessionRetentionBackend) -> frozenset[str]:
    """Artifact ids that transcripts, session operations or knowledge evidence name."""

    found: set[str] = set()
    async with backend.snapshot() as sql:
        for table, column in (
            ("cayu_transcript_messages", backend.transcript_message),
            ("cayu_session_operations", backend.operation_record),
        ):
            found |= await audit.json_reference_atoms(sql, table, column, key_suffix="artifact_id")
        if await sql.existing_tables(("cayu_knowledge_evidence",)):
            found.update(
                row[0]
                for row in await sql.all(
                    "SELECT source_id FROM cayu_knowledge_evidence "
                    "WHERE source_type = 'artifact' AND disposition <> 'detached' "
                    "AND source_id IS NOT NULL"
                )
            )
    return frozenset(found)


async def begin_retention_audit(
    backend: SessionRetentionBackend,
    *,
    store_kind: str,
    mode: RetentionMode,
    policy: dict[str, Any],
    started_at: datetime,
) -> str:
    audit_id = uuid4().hex
    async with backend.write() as sql:
        await audit.require_audit_tables(sql)
        await audit.begin_audit(
            sql,
            audit_id=audit_id,
            store_kind=store_kind,
            mode=mode,
            started_at=started_at,
            policy=policy,
        )
    return audit_id


async def record_retention_entry(
    backend: SessionRetentionBackend, audit_id: str, entry: RetentionAuditEntry
) -> None:
    async with backend.write() as sql:
        await audit.record_audit_entry(
            sql,
            audit_id=audit_id,
            item_id=entry.item_id,
            mode=entry.action,
            counts=entry.counts,
            size=entry.bytes,
            recorded_at=entry.recorded_at,
        )


async def complete_retention_audit(
    backend: SessionRetentionBackend,
    audit_id: str,
    *,
    completed_at: datetime,
    summary: dict[str, Any],
) -> None:
    async with backend.write() as sql:
        await audit.complete_audit(
            sql, audit_id=audit_id, completed_at=completed_at, summary=summary
        )


# -- planning -----------------------------------------------------------------------


def _bounded(items: list[RetentionItem]) -> tuple[RetentionItem, ...]:
    return tuple(items[:MAX_RETENTION_ITEMS_PER_RUN])


def _item(selected: _Selected, disposition: RetentionDisposition) -> RetentionItem:
    return RetentionItem(
        item_id=selected.row.id,
        status=selected.row.status,
        last_updated_at=selected.row.updated_at,
        disposition=disposition,
        counts=selected.counts,
        bytes=selected.bytes,
    )


def _protected_item(
    row: _SessionRow,
    protections: Iterable[RetentionProtection],
    detail: str | None = None,
) -> RetentionItem:
    return RetentionItem(
        item_id=row.id,
        status=row.status,
        last_updated_at=row.updated_at,
        disposition=RetentionDisposition.PROTECTED,
        protections=tuple(sorted(set(protections), key=lambda value: value.value)),
        detail=bounded_retention_detail(detail),
    )


def _session_row(sql: RetentionSql, row: tuple[Any, ...]) -> _SessionRow:
    return _SessionRow(
        id=row[0],
        parent_session_id=row[1],
        status=row[2],
        updated_at=sql.read_timestamp(row[3]),
    )


async def _session_rows(sql: RetentionSql) -> dict[str, _SessionRow]:
    rows = await sql.all("SELECT id, parent_session_id, status, updated_at FROM cayu_sessions")
    return {row[0]: _session_row(sql, row) for row in rows}


def _components(rows: dict[str, _SessionRow]) -> list[frozenset[str]]:
    parent: dict[str, str] = {session_id: session_id for session_id in rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for row in rows.values():
        if row.parent_session_id is not None and row.parent_session_id in rows:
            left, right = find(row.id), find(row.parent_session_id)
            if left != right:
                parent[max(left, right)] = min(left, right)
    groups: dict[str, set[str]] = {}
    for session_id in rows:
        groups.setdefault(find(session_id), set()).add(session_id)
    return [frozenset(members) for members in groups.values()]


def _post_order(rows: dict[str, _SessionRow], members: Iterable[str]) -> list[str]:
    """Children before parents: deeper sessions first, then oldest first."""

    members = set(members)
    depth: dict[str, int] = {}
    for session_id in members:
        level, current, seen = 0, rows[session_id].parent_session_id, {session_id}
        while current is not None and current in members and current not in seen:
            seen.add(current)
            level += 1
            current = rows[current].parent_session_id
        depth[session_id] = level
    return sorted(members, key=lambda value: (-depth[value], rows[value].updated_at, value))


def _is_candidate(row: _SessionRow, policy: SessionRetentionPolicy, cutoff: datetime) -> bool:
    return row.status in {status.value for status in policy.statuses} and row.updated_at <= cutoff


def _plan_batches(
    components: list[frozenset[str]], candidates: set[str]
) -> Iterable[list[frozenset[str]]]:
    batch: list[frozenset[str]] = []
    size = 0
    for component in components:
        batch.append(component)
        size += len(component & candidates)
        if size >= PLAN_BATCH_SESSIONS:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


async def _plan(
    backend: SessionRetentionBackend,
    policy: SessionRetentionPolicy,
    caller_protected: ExternalProtections,
    references: _References,
    rows: dict[str, _SessionRow],
    *,
    now: datetime,
) -> _Plan:
    cutoff = now - policy.older_than
    plan = _Plan(cutoff=cutoff)
    candidates = {row.id for row in rows.values() if _is_candidate(row, policy, cutoff)}
    if not candidates:
        return plan
    components = [component for component in _components(rows) if component & candidates]
    components.sort(
        key=lambda component: (
            max(rows[member].updated_at for member in component),
            min(component),
        )
    )
    selected_bytes = 0
    exhausted = False
    for batch in _plan_batches(components, candidates):
        if exhausted:
            plan.deferred += sum(len(component & candidates) for component in batch)
            continue
        async with backend.check_batch() as sql:
            for component in batch:
                members = sorted(component & candidates)
                if exhausted:
                    plan.deferred += len(members)
                    continue
                protections = {
                    member: await _session_protections(
                        backend, sql, policy, caller_protected, references, rows[member]
                    )
                    for member in members
                }
                if component - candidates or any(reasons for reasons, _ in protections.values()):
                    for member in members:
                        reasons, detail = protections[member]
                        plan.protected.append(
                            _protected_item(
                                rows[member], reasons or (RetentionProtection.LINEAGE,), detail
                            )
                        )
                    continue
                ordered = _post_order(rows, component)
                for index, member in enumerate(ordered):
                    if len(plan.selected) >= policy.max_items or (
                        policy.max_bytes is not None and selected_bytes >= policy.max_bytes
                    ):
                        exhausted = True
                        plan.deferred += len(ordered) - index
                        break
                    counts, size = await _item_size(backend, sql, policy, member)
                    if policy.mode is RetentionMode.COMPACT and size == 0:
                        continue  # Already compact: nothing to remove or audit.
                    plan.selected.append(
                        _Selected(row=rows[member], component=component, counts=counts, bytes=size)
                    )
                    selected_bytes += size
        await asyncio.sleep(0)
    return plan


def _component_groups(selected: list[_Selected]) -> list[list[_Selected]]:
    groups: list[list[_Selected]] = []
    for item in selected:
        if groups and groups[-1][0].component == item.component:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


# -- apply ----------------------------------------------------------------------------


async def _apply_group(
    backend: SessionRetentionBackend,
    sql: RetentionSql,
    policy: SessionRetentionPolicy,
    caller_protected: ExternalProtections,
    references: _References,
    cutoff: datetime,
    group: list[_Selected],
    audit_id: str,
) -> list[RetentionItem]:
    """Re-check and change one planned lineage inside its write transaction.

    Raises :class:`_Blocked` (and so rolls the transaction back) when the
    lineage must be kept.
    """

    component = group[0].component
    await backend.lock_sessions(sql, component)
    blocked = await _revalidate_group(
        backend, sql, policy, caller_protected, references, cutoff, group
    )
    if blocked:
        raise _Blocked(blocked)
    recorded_at = await backend.now(sql)
    items: list[RetentionItem] = []
    for selected in group:
        session_id = selected.row.id
        if policy.mode is RetentionMode.DELETE:
            counts, size = await _deletion_size(backend, sql, session_id)
            try:
                deleted = await backend.delete(sql, session_id)
            except ValueError as guard:
                raise _Blocked(
                    [
                        _protected_item(
                            member.row,
                            (RetentionProtection.ERASURE_GUARD,),
                            str(guard) if member.row.id == session_id else None,
                        )
                        for member in group
                    ]
                ) from None
            if not deleted:
                raise _Blocked(_changed(group, session_id, "session no longer exists"))
        else:
            compaction = await _compaction_plan(
                backend, sql, session_id, policy.compact_tool_output_min_bytes
            )
            if not await _execute_compaction(backend, sql, session_id, compaction):
                raise _Blocked(_changed(group, session_id, "events changed during compaction"))
            counts, size = compaction.counts, compaction.bytes
        await audit.record_audit_entry(
            sql,
            audit_id=audit_id,
            item_id=session_id,
            mode=policy.mode,
            counts=counts,
            size=size,
            recorded_at=recorded_at,
        )
        items.append(
            RetentionItem(
                item_id=session_id,
                status=selected.row.status,
                last_updated_at=selected.row.updated_at,
                disposition=RetentionDisposition.APPLIED,
                counts=counts,
                bytes=size,
            )
        )
    return items


def _changed(group: list[_Selected], session_id: str, detail: str) -> list[RetentionItem]:
    return [
        _protected_item(
            member.row,
            (RetentionProtection.CHANGED,),
            detail if member.row.id == session_id else None,
        )
        for member in group
    ]


async def _revalidate_group(
    backend: SessionRetentionBackend,
    sql: RetentionSql,
    policy: SessionRetentionPolicy,
    caller_protected: ExternalProtections,
    references: _References,
    cutoff: datetime,
    group: list[_Selected],
) -> list[RetentionItem]:
    component = group[0].component
    current = await _lineage(sql, component)
    rows: dict[str, _SessionRow] = {}
    for chunk in chunks(sorted(current | component)):
        for row in await sql.all(
            "SELECT id, parent_session_id, status, updated_at FROM cayu_sessions "
            f"WHERE id IN ({marks(chunk)})",
            chunk,
        ):
            rows[row[0]] = _session_row(sql, row)
    if current != component or set(rows) != component:
        return [
            _protected_item(
                selected.row,
                (RetentionProtection.CHANGED,),
                "the session's lineage changed after planning",
            )
            for selected in group
        ]
    member_reasons: dict[str, tuple[tuple[RetentionProtection, ...], str | None]] = {}
    lineage_blocked = False
    for member in sorted(component):
        row = rows[member]
        if not _is_candidate(row, policy, cutoff):
            lineage_blocked = True
            continue
        member_reasons[member] = await _session_protections(
            backend, sql, policy, caller_protected, references, row
        )
        if member_reasons[member][0]:
            lineage_blocked = True
    for selected in group:
        if rows[selected.row.id].updated_at != selected.row.updated_at:
            lineage_blocked = True
    if not lineage_blocked:
        return []
    blocked: list[RetentionItem] = []
    for selected in group:
        row = rows[selected.row.id]
        if not _is_candidate(row, policy, cutoff) or row.updated_at != selected.row.updated_at:
            blocked.append(
                _protected_item(row, (RetentionProtection.CHANGED,), "updated after planning")
            )
            continue
        reasons, detail = member_reasons[row.id]
        blocked.append(_protected_item(row, reasons or (RetentionProtection.LINEAGE,), detail))
    return blocked


async def _lineage(sql: RetentionSql, seeds: Iterable[str]) -> frozenset[str]:
    """Return every session connected to ``seeds`` through parent edges."""

    found: set[str] = set()
    frontier = set(seeds)
    while frontier:
        found |= frontier
        batch = sorted(frontier)
        frontier = set()
        for chunk in chunks(batch):
            for parent_id, child_id in await sql.all(
                f"SELECT parent_session_id, id FROM cayu_sessions WHERE id IN ({marks(chunk)}) "
                f"OR parent_session_id IN ({marks(chunk)})",
                (*chunk, *chunk),
            ):
                for value in (parent_id, child_id):
                    if value is not None and value not in found:
                        frontier.add(value)
    existing: set[str] = set()
    for chunk in chunks(sorted(found)):
        existing.update(
            row[0]
            for row in await sql.all(
                f"SELECT id FROM cayu_sessions WHERE id IN ({marks(chunk)})", chunk
            )
        )
    return frozenset(existing)


# -- protections ------------------------------------------------------------------------


async def _session_protections(
    backend: SessionRetentionBackend,
    sql: RetentionSql,
    policy: SessionRetentionPolicy | RetentionMode,
    caller_protected: ExternalProtections,
    references: _References,
    row: _SessionRow,
    *,
    store_guards: bool = True,
) -> tuple[tuple[RetentionProtection, ...], str | None]:
    mode = policy if isinstance(policy, RetentionMode) else policy.mode
    session_id = row.id
    reasons: set[RetentionProtection] = set()
    detail: str | None = None
    now = await backend.now(sql)
    reasons.update(caller_protected.get(session_id, ()))
    if await sql.exists(
        "SELECT 1 FROM cayu_tasks WHERE session_id = ? AND status NOT IN "
        f"({marks(_TERMINAL_TASK_STATUSES)}) LIMIT 1",
        (session_id, *_TERMINAL_TASK_STATUSES),
    ):
        reasons.add(RetentionProtection.LIVE_TASK)
    if await _execution_lease_is_live(sql, session_id, now):
        reasons.add(RetentionProtection.EXECUTION_LEASE)
    if await sql.exists(
        "SELECT 1 FROM cayu_checkpoints WHERE session_id = ? AND (pending_action_flags <> 0 "
        f"OR {sql.false('pending_action_metrics_ready')}) LIMIT 1",
        (session_id,),
    ):
        reasons.add(RetentionProtection.PENDING_ACTION)
    if await _has_pending_clarification(sql, session_id):
        reasons.add(RetentionProtection.PENDING_CLARIFICATION)
    if await sql.exists(
        "SELECT 1 FROM cayu_context_view_selections s "
        "JOIN cayu_context_views v ON v.view_id = s.view_id "
        "WHERE v.source_session_id = ? "
        "AND s.state IN ('selected', 'adopted', 'transferred') "
        "AND (s.state <> 'selected' OR s.expires_at_ms > ?) LIMIT 1",
        (session_id, int(now.timestamp() * 1000)),
    ):
        reasons.add(RetentionProtection.CHECKPOINT_DEPENDENCY)
    if references.snapshot_text and session_id in references.snapshot_text:
        reasons.add(RetentionProtection.SNAPSHOT_PIN)
    if session_id in references.eval_session_ids:
        reasons.add(RetentionProtection.EVAL_REFERENCE)
    if session_id in references.knowledge_session_ids or await _has_knowledge_event_evidence(
        sql, session_id, references.knowledge_event_ids
    ):
        reasons.add(RetentionProtection.KNOWLEDGE_EVIDENCE)
    if await sql.exists(
        "SELECT 1 FROM cayu_product_operations WHERE session_id = ? AND status = 'pending' LIMIT 1",
        (session_id,),
    ):
        reasons.add(RetentionProtection.PRODUCT_OPERATION)
    if await _has_event_delivery_backlog(sql, session_id):
        reasons.add(RetentionProtection.EVENT_DELIVERY_BACKLOG)
    if mode is RetentionMode.COMPACT and await sql.exists(
        "SELECT 1 FROM cayu_session_operations WHERE session_id = ? "
        "AND substr(idempotency_key, 1, ?) = ? LIMIT 1",
        (session_id, len(_SESSION_EXPORT_PREFIX), _SESSION_EXPORT_PREFIX),
    ):
        reasons.add(RetentionProtection.SESSION_EXPORT)
    if await backend.has_closure_owner(sql, session_id) or await sql.exists(
        "SELECT 1 FROM cayu_task_session_closure_claims WHERE session_id = ? LIMIT 1",
        (session_id,),
    ):
        reasons.add(RetentionProtection.CLOSURE_IN_PROGRESS)
    if not reasons and store_guards:
        session = await backend.load_session(sql, session_id)
        if session is None:
            return (RetentionProtection.CHANGED,), "session no longer exists"
        try:
            await backend.require_store_guards(sql, session)
        except ValueError as guard:
            reasons.add(RetentionProtection.ERASURE_GUARD)
            detail = str(guard)
    return tuple(sorted(reasons, key=lambda value: value.value)), detail


async def _execution_lease_is_live(sql: RetentionSql, session_id: str, now: datetime) -> bool:
    from cayu.sessions.execution import _ExecutionOwner

    row = await sql.one(
        "SELECT o.owner_json, s.instance_id, s.run_epoch FROM cayu_session_execution_owners o "
        "JOIN cayu_sessions s ON s.id = o.session_id WHERE o.session_id = ?",
        (session_id,),
    )
    if row is None:
        return False
    try:
        owner = _ExecutionOwner.model_validate(sql.read_json(row[0]))
    except ValueError:
        return True  # An unreadable lease cannot be proven expired.
    return (
        not owner.released
        and owner.session_instance_id == row[1]
        and owner.run_epoch == row[2]
        and owner.lease_expires_at > now
    )


async def _has_pending_clarification(sql: RetentionSql, session_id: str) -> bool:
    return await sql.exists(
        "SELECT 1 FROM cayu_participant_session_bindings b WHERE b.session_id = ? AND ("
        "EXISTS (SELECT 1 FROM cayu_collaboration_clarification_questions q "
        "WHERE q.participant_id = b.participant_id "
        f"AND q.state NOT IN ({marks(_TERMINAL_CLARIFICATION_QUESTION_STATES)})) "
        "OR EXISTS (SELECT 1 FROM cayu_collaboration_clarification_deliveries d "
        "WHERE d.participant_id = b.participant_id AND d.state <> 'settled') "
        "OR EXISTS (SELECT 1 FROM cayu_collaboration_clarification_services c "
        "WHERE c.participant_id = b.participant_id AND c.state <> 'settled') "
        "OR EXISTS (SELECT 1 FROM cayu_collaboration_requests r "
        "WHERE r.participant_id = b.participant_id "
        f"AND r.state NOT IN ({marks(_TERMINAL_REQUEST_STATES)}))"
        ") LIMIT 1",
        (session_id, *_TERMINAL_CLARIFICATION_QUESTION_STATES, *_TERMINAL_REQUEST_STATES),
    )


async def _has_event_delivery_backlog(sql: RetentionSql, session_id: str) -> bool:
    cursor = await sql.one(
        "SELECT MIN(CASE WHEN pending_event_sequence IS NOT NULL "
        "AND pending_event_sequence < cursor_sequence THEN pending_event_sequence "
        "ELSE cursor_sequence END) FROM cayu_event_watcher_state"
    )
    if (
        cursor is not None
        and cursor[0] is not None
        and await sql.exists(
            "SELECT 1 FROM cayu_events WHERE session_id = ? AND sequence > ? LIMIT 1",
            (session_id, cursor[0]),
        )
    ):
        return True
    if await sql.exists(
        "SELECT 1 FROM cayu_event_watcher_dead_letters d "
        "JOIN cayu_events e ON e.sequence = d.event_sequence "
        "WHERE d.resolved_at IS NULL AND e.session_id = ? LIMIT 1",
        (session_id,),
    ):
        return True
    return await sql.exists(
        "SELECT 1 FROM cayu_persisted_event_side_effects "
        "WHERE session_id = ? AND status <> 'delivered' LIMIT 1",
        (session_id,),
    )


async def _has_knowledge_event_evidence(
    sql: RetentionSql, session_id: str, event_ids: tuple[str, ...]
) -> bool:
    for chunk in chunks(event_ids):
        if await sql.exists(
            f"SELECT 1 FROM cayu_events WHERE session_id = ? AND event_id IN ({marks(chunk)}) "
            "LIMIT 1",
            (session_id, *chunk),
        ):
            return True
    return False


async def _eval_session_atoms(sql: RetentionSql, table: str, column: str) -> set[str]:
    return await audit.json_reference_atoms(
        sql, table, column, key_suffix="session_id", id_parent="session"
    )


async def _read_references(
    sql: RetentionSql,
    backend: SessionRetentionBackend,
    *,
    lock: bool = False,
) -> _References:
    # Counts and document sizes are not mutation revisions: an update can
    # introduce a new reference without changing either. Re-read under the
    # apply transaction, including between lineages.
    present = await sql.existing_tables(
        (*_EVAL_TABLES, "cayu_knowledge_evidence", *_SNAPSHOT_TABLES)
    )
    if lock and sql.postgres and present:
        # Reference publishers need not participate in a retention-specific
        # advisory lock. SHARE excludes their writes until this batch commits.
        await sql.run("LOCK TABLE " + ", ".join(sorted(present)) + " IN SHARE MODE")
    eval_ids: set[str] = set()
    for table, column in backend.eval_sources:
        if table in present:
            eval_ids |= await _eval_session_atoms(sql, table, column)
    knowledge_ids: set[str] = set()
    knowledge_events: set[str] = set()
    if "cayu_knowledge_evidence" in present:
        for source_type, source_id, source_uri in await sql.all(
            "SELECT source_type, source_id, source_uri FROM cayu_knowledge_evidence "
            "WHERE disposition <> 'detached' AND (source_type IN ('session', 'tool', "
            "'session_event') OR substr(source_uri, 1, 16) = 'cayu://sessions/')"
        ):
            if source_type == "session_event" and source_id is not None:
                knowledge_events.add(source_id)
            elif source_type in {"session", "tool"} and source_id is not None:
                knowledge_ids.add(source_id)
            if isinstance(source_uri, str) and source_uri.startswith("cayu://sessions/"):
                knowledge_ids.add(source_uri.removeprefix("cayu://sessions/").split("/", 1)[0])
    snapshot_text = (
        await audit.snapshot_pin_text(sql) if set(audit.SNAPSHOT_TABLES) <= present else ""
    )
    return _References(
        eval_session_ids=frozenset(eval_ids),
        knowledge_session_ids=frozenset(knowledge_ids),
        knowledge_event_ids=tuple(sorted(knowledge_events)),
        snapshot_text=snapshot_text,
    )


# -- sizes and mutations -----------------------------------------------------------------


async def _item_size(
    backend: SessionRetentionBackend,
    sql: RetentionSql,
    policy: SessionRetentionPolicy,
    session_id: str,
) -> tuple[dict[str, int], int]:
    if policy.mode is RetentionMode.DELETE:
        return await _deletion_size(backend, sql, session_id)
    plan = await _compaction_plan(backend, sql, session_id, policy.compact_tool_output_min_bytes)
    return plan.counts, plan.bytes


async def _deletion_size(
    backend: SessionRetentionBackend, sql: RetentionSql, session_id: str
) -> tuple[dict[str, int], int]:
    event_bytes = sql.size(backend.event_payload)
    if sql.postgres:
        event_bytes = f"{event_bytes} + {sql.size('event')}"
    events = await sql.one(
        f"SELECT COUNT(*), COALESCE(SUM({event_bytes}), 0) FROM cayu_events WHERE session_id = ?",
        (session_id,),
    )
    transcript = await sql.one(
        f"SELECT COUNT(*), COALESCE(SUM({sql.size(backend.transcript_message)} "
        f"+ {sql.size('transcript_search_document')}), 0) "
        "FROM cayu_transcript_messages WHERE session_id = ?",
        (session_id,),
    )
    checkpoint = await sql.one(
        f"SELECT COALESCE(SUM({sql.size(backend.checkpoint_state)}), 0) "
        "FROM cayu_checkpoints WHERE session_id = ?",
        (session_id,),
    )
    operations = await sql.one(
        f"SELECT COUNT(*), COALESCE(SUM({sql.size(backend.operation_record)}), 0) "
        "FROM cayu_session_operations WHERE session_id = ?",
        (session_id,),
    )
    assert events and transcript and checkpoint and operations
    counts = {
        "sessions_removed": 1,
        "events_removed": int(events[0]),
        "transcript_messages_removed": int(transcript[0]),
        "session_operations_removed": int(operations[0]),
    }
    size = int(events[1]) + int(transcript[1]) + int(checkpoint[0]) + int(operations[1])
    return counts, size


# Compaction never touches an event that an interaction pointer or an
# undelivered side effect still references. Pending-action lookup metadata lives
# in separate columns, so rewriting a stored tool-output body leaves it intact,
# but such events are never deleted.
_COMPACTABLE_EVENT_FILTER = (
    "session_id = ? "
    "AND NOT EXISTS (SELECT 1 FROM cayu_interaction_latest_events AS latest "
    "WHERE latest.session_id = cayu_events.session_id "
    "AND latest.latest_event_sequence = cayu_events.sequence) "
    "AND NOT EXISTS (SELECT 1 FROM cayu_persisted_event_side_effects AS delivery "
    "WHERE delivery.session_id = cayu_events.session_id "
    "AND delivery.event_id = cayu_events.event_id AND delivery.status <> 'delivered')"
)


def compacted_tool_output_marker(content: str) -> str:
    """The text that replaces a compacted tool-output body in a stored event."""

    encoded = content.encode("utf-8")
    return (
        f"[cayu storage retention removed {len(encoded)} bytes of tool output; "
        f"sha256 {hashlib.sha256(encoded).hexdigest()}]"
    )


def _compact_document(document: Any) -> tuple[Any, int] | None:
    """Replace ``result.content`` in a payload document; return it and bytes saved."""

    if type(document) is not dict:
        return None
    result = document.get("result")
    if type(result) is not dict or type(result.get("content")) is not str:
        return None
    content = result["content"]
    marker = compacted_tool_output_marker(content)
    compacted = dict(document)
    compacted["result"] = {**result, "content": marker}
    saved = len(json.dumps(content, ensure_ascii=False).encode("utf-8")) - len(
        json.dumps(marker, ensure_ascii=False).encode("utf-8")
    )
    return compacted, saved


async def _compaction_plan(
    backend: SessionRetentionBackend, sql: RetentionSql, session_id: str, min_bytes: int
) -> _CompactionPlan:
    deltas = await sql.all(
        f"SELECT sequence, {sql.size(backend.event_payload)} FROM cayu_events "
        f"WHERE {_COMPACTABLE_EVENT_FILTER} AND pending_action_lookup_key IS NULL "
        "AND event_type IN (?, ?) ORDER BY sequence",
        (session_id, *_DELTA_EVENT_TYPES),
    )
    if sql.postgres:
        delta_extra = await sql.all(
            f"SELECT COALESCE(SUM({sql.size('event')}), 0) FROM cayu_events "
            f"WHERE {_COMPACTABLE_EVENT_FILTER} AND pending_action_lookup_key IS NULL "
            "AND event_type IN (?, ?)",
            (session_id, *_DELTA_EVENT_TYPES),
        )
        delete_bytes = sum(int(row[1]) for row in deltas) + int(delta_extra[0][0])
        tool_rows = await sql.all(
            "SELECT sequence, payload, event FROM cayu_events "
            f"WHERE {_COMPACTABLE_EVENT_FILTER} AND event_type IN (?, ?) "
            "AND jsonb_typeof(payload->'result'->'content') = 'string' "
            "AND octet_length(payload->'result'->>'content') >= ? ORDER BY sequence",
            (session_id, *_TOOL_OUTPUT_EVENT_TYPES, min_bytes),
        )
    else:
        delete_bytes = sum(int(row[1]) for row in deltas)
        tool_rows = [
            (row[0], json.loads(row[1]), None)
            for row in await sql.all(
                "SELECT sequence, payload_json FROM cayu_events "
                f"WHERE {_COMPACTABLE_EVENT_FILTER} AND event_type IN (?, ?) "
                "AND json_type(payload_json, '$.result.content') = 'text' "
                "AND length(CAST(json_extract(payload_json, '$.result.content') AS BLOB)) >= ? "
                "ORDER BY sequence",
                (session_id, *_TOOL_OUTPUT_EVENT_TYPES, min_bytes),
            )
        ]
    rewrites: list[tuple[int, Any, Any, int]] = []
    for sequence, payload, event in tool_rows:
        compacted = _compact_document(sql.read_json(payload))
        if compacted is None:
            continue
        new_payload, saved = compacted
        new_event = None
        if event is not None:
            event_document = sql.read_json(event)
            if type(event_document) is dict and "payload" in event_document:
                compacted_event = _compact_document(event_document["payload"])
                if compacted_event is not None:
                    new_event = {**event_document, "payload": compacted_event[0]}
                    saved += compacted_event[1]
        if saved <= 0:
            continue
        rewrites.append((int(sequence), new_payload, new_event, saved))
    return _CompactionPlan(
        delete_sequences=tuple(int(row[0]) for row in deltas),
        delete_bytes=delete_bytes,
        rewrites=tuple(rewrites),
        rewrite_bytes=sum(rewrite[3] for rewrite in rewrites),
    )


async def _execute_compaction(
    backend: SessionRetentionBackend,
    sql: RetentionSql,
    session_id: str,
    plan: _CompactionPlan,
) -> bool:
    """Apply a compaction plan; ``False`` when the events changed underneath it."""

    for chunk in chunks(plan.delete_sequences):
        deleted = await sql.run(
            f"DELETE FROM cayu_events WHERE session_id = ? AND sequence IN ({marks(chunk)})",
            (session_id, *chunk),
        )
        if deleted != len(chunk):
            return False
    for sequence, payload, event, _saved in plan.rewrites:
        if sql.postgres:
            updated = await sql.run(
                "UPDATE cayu_events SET payload = ?, event = COALESCE(?, event) "
                "WHERE session_id = ? AND sequence = ?",
                (
                    sql.json(payload),
                    None if event is None else sql.json(event),
                    session_id,
                    sequence,
                ),
            )
        else:
            updated = await sql.run(
                f"UPDATE cayu_events SET {backend.event_payload} = ? "
                "WHERE session_id = ? AND sequence = ?",
                (sql.json(payload), session_id, sequence),
            )
        if updated != 1:
            return False
    return True


__all__ = [
    "PostgresSessionRetentionBackend",
    "SQLiteSessionRetentionBackend",
    "SessionRetentionBackend",
    "apply_retention_policy",
    "begin_retention_audit",
    "compacted_tool_output_marker",
    "complete_retention_audit",
    "inspect_session_protections",
    "list_retention_audits",
    "load_retention_audit",
    "record_retention_entry",
    "session_artifact_references",
]
