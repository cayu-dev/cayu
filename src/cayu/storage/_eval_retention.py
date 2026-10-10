"""Eval-store retention, shared by the SQLite and Postgres eval stores.

A retention item is one terminal eval run. Deleting it removes the run, its
result, its fresh result record and its trial checkpoints, in foreign-key
order, inside one write transaction that first re-reads the run and re-checks
every protection. The store's write lock is released between runs.

Captured result records, corpora, suites, cases, scenarios, authored suites,
baselines, baseline history and judge calibrations are never removed.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Collection, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from cayu.storage import _retention_sql as audit
from cayu.storage._retention_sql import PostgresRetentionSql, RetentionSql, SQLiteRetentionSql
from cayu.storage._session_retention import ExternalProtections, external_protections
from cayu.storage.retention import (
    MAX_RETENTION_ITEMS_PER_RUN,
    EvalRetentionPolicy,
    RetentionDisposition,
    RetentionItem,
    RetentionPhase,
    RetentionProgress,
    RetentionProgressCallback,
    RetentionProtection,
    RetentionReport,
    bounded_retention_detail,
    retention_audit_summary,
)

STORE_KIND = "evals"
_DIGEST = re.compile(r"[0-9a-f]{64}")


class EvalRetentionBackend(ABC):
    def __init__(self, store: Any) -> None:
        self.store = store

    @property
    def read_only(self) -> bool:
        return bool(getattr(self.store, "_read_only", False))

    @abstractmethod
    async def now(self, sql: RetentionSql) -> datetime: ...

    @abstractmethod
    def snapshot(self) -> Any: ...

    @abstractmethod
    def write(self) -> Any: ...

    @abstractmethod
    async def lock_run(self, sql: RetentionSql, run_id: str) -> None: ...

    result_document: str = "result_json"

    def invocation_flag(self, sql: RetentionSql, path: tuple[str, ...]) -> str:
        if sql.postgres:
            keys = ",".join(path)
            return f"(invocation_json::jsonb) #>> '{{{keys}}}'"
        return f"json_extract(invocation_json, '$.{'.'.join(path)}')"


class SQLiteEvalRetentionBackend(EvalRetentionBackend):
    async def now(self, sql: RetentionSql) -> datetime:
        return datetime.now(UTC)

    @asynccontextmanager
    async def snapshot(self) -> AsyncIterator[RetentionSql]:
        async with self.store._read_lock:
            connection = self.store._read_connection
            connection.execute("BEGIN")
            try:
                yield SQLiteRetentionSql(connection)
            finally:
                connection.rollback()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[RetentionSql]:
        async with self.store._lock:
            connection = self.store._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield SQLiteRetentionSql(connection)
            except BaseException:
                connection.rollback()
                raise
            connection.commit()

    async def lock_run(self, sql: RetentionSql, run_id: str) -> None:
        return None  # BEGIN IMMEDIATE already excludes every other writer.


class PostgresEvalRetentionBackend(EvalRetentionBackend):
    result_document = "result"

    async def now(self, sql: RetentionSql) -> datetime:
        row = await sql.one("SELECT clock_timestamp()")
        assert row is not None
        return sql.read_timestamp(row[0])

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

    def write(self) -> Any:
        return self._transaction(commit=True)

    async def lock_run(self, sql: RetentionSql, run_id: str) -> None:
        await sql.all("SELECT run_id FROM cayu_eval_runs WHERE run_id = ? FOR UPDATE", (run_id,))


@dataclass(frozen=True)
class _RunRow:
    run_id: str
    status: str
    finished_at: datetime
    result_revision: str | None


@dataclass
class _Plan:
    cutoff: datetime
    selected: list[tuple[_RunRow, dict[str, int], int]] = field(default_factory=list)
    protected: list[RetentionItem] = field(default_factory=list)
    deferred: int = 0


class _Blocked(Exception):
    def __init__(self, item: RetentionItem) -> None:
        super().__init__("eval retention item blocked")
        self.item = item


def _protected_item(row: _RunRow, reasons: Collection[RetentionProtection], detail=None):
    return RetentionItem(
        item_id=row.run_id,
        status=row.status,
        last_updated_at=row.finished_at,
        disposition=RetentionDisposition.PROTECTED,
        protections=tuple(sorted(set(reasons), key=lambda value: value.value)),
        detail=bounded_retention_detail(detail),
    )


async def _terminal_runs(
    sql: RetentionSql, statuses: Collection[str], run_id: str | None = None
) -> list[_RunRow]:
    ordered = sorted(statuses)
    filters = f"status IN ({audit.marks(ordered)}) AND finished_at IS NOT NULL"
    parameters: list[Any] = list(ordered)
    if run_id is not None:
        filters += " AND run_id = ?"
        parameters.append(run_id)
    rows = await sql.all(
        f"SELECT run_id, status, finished_at, result_revision FROM cayu_eval_runs WHERE {filters}",
        parameters,
    )
    return [
        _RunRow(
            run_id=row[0],
            status=row[1],
            finished_at=sql.read_timestamp(row[2]),
            result_revision=row[3],
        )
        for row in rows
    ]


@dataclass
class _References:
    retry_sources: frozenset[str]
    snapshot_text: str
    snapshot_digests: frozenset[str]


async def _read_references(sql: RetentionSql, backend: EvalRetentionBackend) -> _References:
    retry_sources = {
        row[0]
        for row in await sql.all(
            f"SELECT {backend.invocation_flag(sql, ('retry_of', 'run_id'))} FROM cayu_eval_runs"
        )
        if isinstance(row[0], str)
    }
    snapshot_text = ""
    if set(audit.SNAPSHOT_TABLES) <= await sql.existing_tables(audit.SNAPSHOT_TABLES):
        snapshot_text = await audit.snapshot_pin_text(sql)
    return _References(
        retry_sources=frozenset(retry_sources),
        snapshot_text=snapshot_text,
        snapshot_digests=frozenset(_DIGEST.findall(snapshot_text)),
    )


async def _protections(
    backend: EvalRetentionBackend,
    sql: RetentionSql,
    row: _RunRow,
    references: _References,
    caller_protected: ExternalProtections,
) -> tuple[RetentionProtection, ...]:
    reasons: set[RetentionProtection] = set(caller_protected.get(row.run_id, ()))
    if row.result_revision is not None:
        reasons.update(caller_protected.get(row.result_revision, ()))
    external_digests = {key for key in caller_protected if _DIGEST.fullmatch(key)}
    if external_digests:
        for digest in await _run_document_digests(backend, sql, row.run_id):
            reasons.update(caller_protected.get(digest, ()))
    if await sql.exists(
        "SELECT 1 FROM cayu_eval_result_records rr WHERE rr.fresh_run_id = ? AND ("
        "EXISTS (SELECT 1 FROM cayu_eval_baselines b WHERE b.result_revision = rr.revision) "
        "OR EXISTS (SELECT 1 FROM cayu_eval_baseline_mutations m "
        "WHERE m.selected_result_revision = rr.revision "
        "OR m.previous_result_revision = rr.revision))",
        (row.run_id,),
    ):
        reasons.add(RetentionProtection.BASELINE)
    retained = await sql.one(
        f"SELECT trial_checkpoint_count, "
        f"{backend.invocation_flag(sql, ('retain_trial_checkpoints',))} "
        "FROM cayu_eval_runs WHERE run_id = ?",
        (row.run_id,),
    )
    if retained is not None and (
        int(retained[0] or 0) > 0 or str(retained[1]).lower() in {"1", "true"}
    ):
        reasons.add(RetentionProtection.CAMPAIGN_EVIDENCE)
    if row.run_id in references.retry_sources:
        reasons.add(RetentionProtection.EVAL_REFERENCE)
    if references.snapshot_text and await _snapshot_references_run(backend, sql, row, references):
        reasons.add(RetentionProtection.SNAPSHOT_PIN)
    return tuple(sorted(reasons, key=lambda value: value.value))


async def _snapshot_references_run(
    backend: EvalRetentionBackend, sql: RetentionSql, row: _RunRow, references: _References
) -> bool:
    if row.run_id in references.snapshot_text:
        return True
    if row.result_revision is not None and row.result_revision in references.snapshot_text:
        return True
    if not references.snapshot_digests:
        return False
    # Snapshot result bindings name trial result digests, not run ids.
    return bool(references.snapshot_digests & await _run_document_digests(backend, sql, row.run_id))


async def _run_document_digests(
    backend: EvalRetentionBackend, sql: RetentionSql, run_id: str
) -> set[str]:
    digests: set[str] = set()
    for (document,) in await sql.all(
        f"SELECT {backend.result_document} FROM cayu_eval_results WHERE run_id = ? "
        "UNION ALL SELECT checkpoint_json FROM cayu_eval_run_trial_checkpoints WHERE run_id = ?",
        (run_id, run_id),
    ):
        if isinstance(document, str):
            digests.update(_DIGEST.findall(document))
    return digests


async def _run_size(
    backend: EvalRetentionBackend, sql: RetentionSql, run_id: str
) -> tuple[dict[str, int], int]:
    run = await sql.one(
        f"SELECT {sql.size('invocation_json')}, "
        f"COALESCE({sql.size('scenario_progress_json')}, 0), "
        f"COALESCE({sql.size('failure_diagnostic_json')}, 0), trial_checkpoint_count "
        "FROM cayu_eval_runs WHERE run_id = ?",
        (run_id,),
    )
    result = await sql.one(
        "SELECT COUNT(*), COALESCE(SUM(result_bytes), 0) FROM cayu_eval_results WHERE run_id = ?",
        (run_id,),
    )
    records = await sql.one(
        "SELECT COUNT(*), COALESCE(SUM(document_bytes), 0) FROM cayu_eval_result_records "
        "WHERE fresh_run_id = ?",
        (run_id,),
    )
    checkpoints = await sql.one(
        "SELECT COUNT(*), COALESCE(SUM(document_bytes), 0) "
        "FROM cayu_eval_run_trial_checkpoints WHERE run_id = ?",
        (run_id,),
    )
    assert run and result and records and checkpoints
    counts = {
        "runs_removed": 1,
        "results_removed": int(result[0]),
        "result_records_removed": int(records[0]),
        "trial_checkpoints_removed": int(checkpoints[0]),
    }
    size = (int(run[0]) + int(run[1]) + int(run[2]) + int(result[1]) + int(records[1])) + int(
        checkpoints[1]
    )
    return counts, size


async def apply_eval_retention_policy(
    backend: EvalRetentionBackend,
    policy: EvalRetentionPolicy,
    *,
    protected_ids: Collection[str] = (),
    references: Mapping[RetentionProtection, Collection[str]] | None = None,
    progress: RetentionProgressCallback | None = None,
) -> RetentionReport:
    if type(policy) is not EvalRetentionPolicy:
        raise TypeError("policy must be an EvalRetentionPolicy.")
    caller_protected = external_protections(protected_ids, references)
    if not policy.dry_run and backend.read_only:
        raise PermissionError("Retention apply requires a writable eval store.")
    async with backend.snapshot() as sql:
        if not policy.dry_run:
            await audit.require_audit_tables(sql)
        started_at = await backend.now(sql)
        cutoff = started_at - policy.older_than
        rows = sorted(
            (
                row
                for row in await _terminal_runs(sql, policy.statuses)
                if row.finished_at <= cutoff
            ),
            key=lambda row: (row.finished_at, row.run_id),
        )
        stored = await _read_references(sql, backend)
        plan = _Plan(cutoff=cutoff)
        selected_bytes = 0
        for index, row in enumerate(rows):
            if len(plan.selected) >= policy.max_items or (
                policy.max_bytes is not None and selected_bytes >= policy.max_bytes
            ):
                plan.deferred = len(rows) - index
                break
            reasons = await _protections(backend, sql, row, stored, caller_protected)
            if reasons:
                plan.protected.append(_protected_item(row, reasons))
                continue
            counts, size = await _run_size(backend, sql, row.run_id)
            plan.selected.append((row, counts, size))
            selected_bytes += size
    if progress is not None:
        await progress(
            RetentionProgress(
                store_kind=STORE_KIND,
                phase=RetentionPhase.PLANNED,
                dry_run=policy.dry_run,
                planned_items=len(plan.selected),
                protected=tuple(plan.protected[:MAX_RETENTION_ITEMS_PER_RUN]),
            )
        )
    if policy.dry_run:
        return _report(
            policy,
            plan,
            started_at,
            datetime.now(UTC),
            None,
            [
                RetentionItem(
                    item_id=row.run_id,
                    status=row.status,
                    last_updated_at=row.finished_at,
                    disposition=RetentionDisposition.SELECTED,
                    counts=counts,
                    bytes=size,
                )
                for row, counts, size in plan.selected
            ],
            plan.protected,
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
    for index, (row, _counts, _size) in enumerate(plan.selected):
        batch_applied: tuple[RetentionItem, ...] = ()
        batch_protected: tuple[RetentionItem, ...] = ()
        try:
            async with backend.write() as sql:
                item = await _delete_run(
                    backend, sql, policy, row, caller_protected, cutoff, audit_id
                )
            batch_applied = (item,)
        except _Blocked as blocked:
            batch_protected = (blocked.item,)
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
                    applied=batch_applied,
                    protected=batch_protected,
                )
            )
        await asyncio.sleep(0)
    async with backend.write() as sql:
        completed_at = await backend.now(sql)
        report = _report(policy, plan, started_at, completed_at, audit_id, applied, protected)
        await audit.complete_audit(
            sql,
            audit_id=audit_id,
            completed_at=completed_at,
            summary=retention_audit_summary(report),
        )
    return report


def _report(policy, plan, started_at, completed_at, audit_id, items, protected) -> RetentionReport:
    return RetentionReport(
        store_kind=STORE_KIND,
        mode=policy.mode,
        dry_run=policy.dry_run,
        policy=policy.policy_document(),
        started_at=started_at,
        completed_at=completed_at,
        cutoff=plan.cutoff,
        audit_id=audit_id,
        items=tuple(items),
        protected=tuple(protected[:MAX_RETENTION_ITEMS_PER_RUN]),
        protected_truncated=len(protected) > MAX_RETENTION_ITEMS_PER_RUN,
        deferred_count=plan.deferred,
    )


async def _delete_run(
    backend: EvalRetentionBackend,
    sql: RetentionSql,
    policy: EvalRetentionPolicy,
    planned: _RunRow,
    caller_protected: ExternalProtections,
    cutoff: datetime,
    audit_id: str,
) -> RetentionItem:
    await backend.lock_run(sql, planned.run_id)
    current = await _terminal_runs(sql, policy.statuses, planned.run_id)
    if not current or current[0] != planned or current[0].finished_at > cutoff:
        raise _Blocked(
            _protected_item(planned, (RetentionProtection.CHANGED,), "run changed after planning")
        )
    references = await _read_references(sql, backend)
    reasons = await _protections(backend, sql, planned, references, caller_protected)
    if reasons:
        raise _Blocked(_protected_item(planned, reasons))
    counts, size = await _run_size(backend, sql, planned.run_id)
    await sql.run("DELETE FROM cayu_eval_result_records WHERE fresh_run_id = ?", (planned.run_id,))
    await sql.run("DELETE FROM cayu_eval_results WHERE run_id = ?", (planned.run_id,))
    # Trial checkpoints cascade from the run row.
    deleted = await sql.run("DELETE FROM cayu_eval_runs WHERE run_id = ?", (planned.run_id,))
    if deleted != 1:
        raise _Blocked(
            _protected_item(planned, (RetentionProtection.CHANGED,), "run no longer exists")
        )
    recorded_at = await backend.now(sql)
    await audit.record_audit_entry(
        sql,
        audit_id=audit_id,
        item_id=planned.run_id,
        mode=policy.mode,
        counts=counts,
        size=size,
        recorded_at=recorded_at,
    )
    return RetentionItem(
        item_id=planned.run_id,
        status=planned.status,
        last_updated_at=planned.finished_at,
        disposition=RetentionDisposition.APPLIED,
        counts=counts,
        bytes=size,
    )


async def eval_session_references(backend: EvalRetentionBackend) -> frozenset[str]:
    """Session ids that stored eval runs, results and checkpoints name."""

    from cayu.storage._session_retention import (
        PostgresSessionRetentionBackend,
        SQLiteSessionRetentionBackend,
        _eval_session_atoms,
    )

    session_backend_type = (
        PostgresSessionRetentionBackend
        if isinstance(backend, PostgresEvalRetentionBackend)
        else SQLiteSessionRetentionBackend
    )
    found: set[str] = set()
    async with backend.snapshot() as sql:
        present = await sql.existing_tables(table for table, _ in session_backend_type.eval_sources)
        for table, column in session_backend_type.eval_sources:
            if table in present:
                found |= await _eval_session_atoms(sql, table, column)
    return frozenset(found)


async def eval_artifact_references(backend: EvalRetentionBackend) -> frozenset[str]:
    """Artifact ids that eval runs, results and trial checkpoints name."""

    sources = (
        ("cayu_eval_runs", "invocation_json"),
        ("cayu_eval_runs", "scenario_progress_json"),
        ("cayu_eval_results", backend.result_document),
        (
            "cayu_eval_result_records",
            "captured_result"
            if isinstance(backend, PostgresEvalRetentionBackend)
            else "captured_result_json",
        ),
        ("cayu_eval_run_trial_checkpoints", "checkpoint_json"),
    )
    found: set[str] = set()
    async with backend.snapshot() as sql:
        present = await sql.existing_tables(table for table, _ in sources)
        for table, column in sources:
            if table in present:
                found |= await audit.json_reference_atoms(
                    sql, table, column, key_suffix="artifact_id"
                )
    return frozenset(found)


__all__ = [
    "EvalRetentionBackend",
    "PostgresEvalRetentionBackend",
    "SQLiteEvalRetentionBackend",
    "apply_eval_retention_policy",
    "eval_artifact_references",
    "eval_session_references",
]
