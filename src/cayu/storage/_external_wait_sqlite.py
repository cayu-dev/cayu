"""SQLite external-wait transactions serialize across independent connections."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from datetime import datetime

from cayu.sessions._external_wait_records import (
    external_wait_session_identity,
    validate_external_wait_row,
)
from cayu.sessions._external_wait_transition import ExternalWaitMutation, transition
from cayu.sessions.external_waits import (
    ExternalWaitConflict,
    ExternalWaitPruneResult,
    ExternalWaitRecord,
    ExternalWaitRetirement,
    ExternalWaitRetirementRequest,
    ExternalWaitScope,
    encode_record,
)


class SQLiteExternalWaitMixin:
    _lock: asyncio.Lock
    _connection: sqlite3.Connection
    _ownership_clock: Callable[[], datetime]

    def _require_external_wait_admission_unlocked(self, session) -> None:
        from cayu.runtime._external_wait_admission import require_external_admission

        rows = self._connection.execute(
            "SELECT * FROM cayu_external_waits WHERE session_id=? AND session_instance_id=? "
            "AND pending_handoff=1 AND handoff='unbound' LIMIT 2",
            (session.id, session.instance_id),
        ).fetchall()
        require_external_admission((validate_external_wait_row(row) for row in rows), session)

    def _require_external_creation_unlocked(self, creation_request) -> None:
        from cayu.runtime._external_wait_creation import (
            current_external_creation,
            require_external_creation,
        )

        expected = current_external_creation()
        if expected is None:
            return
        registration, _ = expected
        request = registration.correlation.request
        row = self._connection.execute(
            "SELECT * FROM cayu_external_waits WHERE scope=? AND generation=? AND correlation_key=?",
            (request.scope.application_scope, request.scope.generation, request.correlation_key),
        ).fetchone()
        require_external_creation(
            None if row is None else validate_external_wait_row(row), creation_request
        )

    async def _mutate_external_wait(self, command: ExternalWaitMutation) -> ExternalWaitRecord:
        scope = command.request.scope
        namespace = (scope.application_scope, scope.generation)
        key = (*namespace, command.request.correlation_key)
        async with self._lock:
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                limits_json = command.limits.model_dump_json()
                self._connection.execute(
                    "INSERT OR IGNORE INTO cayu_external_wait_scopes(scope,generation,limits_json) VALUES (?,?,?)",
                    (*namespace, limits_json),
                )
                owner = self._connection.execute(
                    "SELECT limits_json,retired FROM cayu_external_wait_scopes WHERE scope=? AND generation=?",
                    namespace,
                ).fetchone()
                if owner[0] != limits_json or owner[1]:
                    raise ExternalWaitConflict(
                        "External wait scope is retired or its limits conflict."
                    )
                row = self._connection.execute(
                    "SELECT * FROM cayu_external_waits WHERE scope=? AND generation=? AND correlation_key=?",
                    key,
                ).fetchone()
                current = None if row is None else validate_external_wait_row(row)
                if (
                    command.kind == "prepare_execution"
                    and command.execution_intent is not None
                    and command.execution_intent.mode == "resume"
                ):
                    from cayu.runtime._external_wait_admission import require_resume_preparation
                    from cayu.storage import _sqlite_records as sqlite_records
                    from cayu.storage.sqlite import _load_checkpoint_state

                    session_id = command.execution_intent.session_id
                    if (
                        self._connection.execute(
                            "SELECT 1 FROM cayu_participant_session_bindings WHERE session_id=?",
                            (session_id,),
                        ).fetchone()
                        is not None
                    ):
                        raise PermissionError(
                            "External waits do not support participant-owned sessions."
                        )
                    session = sqlite_records.load_session(self._connection, session_id)
                    require_resume_preparation(
                        command,
                        current,
                        session,
                        _load_checkpoint_state(self._connection, session_id),
                    )
                    if current is None or current.execution is None:
                        self._require_external_wait_admission_unlocked(session)
                if command.kind == "exclude_execution":
                    from cayu.runtime._external_wait_creation import require_execution_exclusion
                    from cayu.storage import _sqlite_records as sqlite_records
                    from cayu.storage.sqlite import _load_checkpoint_state

                    session = (
                        None
                        if current is None or current.execution is None
                        else sqlite_records.load_session(
                            self._connection, current.execution.intent.session_id
                        )
                    )
                    require_execution_exclusion(
                        command,
                        current,
                        session,
                        None
                        if session is None
                        else _load_checkpoint_state(self._connection, session.id),
                    )
                if command.kind == "bind":
                    from cayu.runtime._external_wait_binding import (
                        require_binding_scope,
                        require_binding_writer,
                    )
                    from cayu.sessions._session_continuation import continuation_operation_key
                    from cayu.storage import _sqlite_records as sqlite_records
                    from cayu.storage.sqlite import _load_checkpoint_state

                    require_binding_scope(command)
                    if current is None or current.continuation is None:
                        assert command.continuation is not None
                        session_id = command.continuation.intent.session_id
                        if (
                            self._connection.execute(
                                "SELECT 1 FROM cayu_participant_session_bindings WHERE session_id=?",
                                (session_id,),
                            ).fetchone()
                            is not None
                        ):
                            raise PermissionError(
                                "External waits do not support participant-owned sessions."
                            )
                        native = self._connection.execute(
                            "SELECT record_json FROM cayu_session_operations WHERE session_id=? AND idempotency_key=?",
                            (session_id, continuation_operation_key(command.continuation.intent)),
                        ).fetchone()
                        require_binding_writer(
                            command,
                            sqlite_records.load_session(self._connection, session_id),
                            _load_checkpoint_state(self._connection, session_id),
                            None if native is None else native[0],
                            execution=None if current is None else current.execution,
                        )
                if command.kind in {
                    "settle",
                    "prepare_service",
                    "prepare_retirement",
                    "complete_retirement",
                    "reconcile_binding",
                }:
                    from cayu.runtime._external_wait_settlement import (
                        require_native_settlement,
                        require_settlement_scope,
                    )
                    from cayu.sessions._session_continuation import continuation_operation_key

                    require_settlement_scope(command)
                    assert command.continuation is not None
                    native = self._connection.execute(
                        "SELECT record_json FROM cayu_session_operations WHERE session_id=? AND idempotency_key=?",
                        (
                            command.continuation.intent.session_id,
                            continuation_operation_key(command.continuation.intent),
                        ),
                    ).fetchone()
                    require_native_settlement(
                        command, current, None if native is None else native[0]
                    )
                # Capacity is only consulted when admitting a new correlation.
                # Observation/settlement must not scan the scope under its lock.
                totals = (0, 0)
                if current is None and command.kind == "reserve":
                    totals = self._connection.execute(
                        "SELECT COUNT(*),COALESCE(SUM(reserved_bytes),0) FROM cayu_external_waits WHERE scope=? AND generation=?",
                        namespace,
                    ).fetchone()
                result = transition(
                    current,
                    command,
                    now_ms=int(self._ownership_clock().timestamp() * 1000),
                    count=totals[0],
                    reserved_bytes=totals[1],
                )
                if result != current:
                    self._connection.execute(
                        "INSERT INTO cayu_external_waits(scope,generation,correlation_key,source,incarnation,revision,reserved_bytes,handoff,pending_handoff,session_id,session_instance_id,record_json) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(scope,generation,correlation_key) DO UPDATE SET "
                        "revision=excluded.revision,reserved_bytes=excluded.reserved_bytes,handoff=excluded.handoff,pending_handoff=excluded.pending_handoff,session_id=excluded.session_id,session_instance_id=excluded.session_instance_id,record_json=excluded.record_json",
                        (
                            *key,
                            result.correlation.request.source,
                            result.correlation.incarnation,
                            result.revision,
                            result.reserved_bytes,
                            result.handoff,
                            int(result.pending_handoff),
                            *external_wait_session_identity(result),
                            encode_record(result),
                        ),
                    )
                return result

    async def _read_external_wait_retirement(
        self, scope: ExternalWaitScope
    ) -> ExternalWaitRetirement | None:
        from cayu.sessions._external_wait_retirement import read_scope_retirement

        async with self._lock:
            row = self._connection.execute(
                "SELECT retired,retirement_json,limits_json FROM cayu_external_wait_scopes "
                "WHERE scope=? AND generation=?",
                (scope.application_scope, scope.generation),
            ).fetchone()
            return read_scope_retirement(row, scope)

    async def _retire_external_wait_scope(
        self, request: ExternalWaitRetirementRequest
    ) -> ExternalWaitRetirement:
        from cayu.sessions._external_wait_retirement import RetirementAccumulator, read_retirement

        namespace = (request.scope.application_scope, request.scope.generation)
        async with self._lock:
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "INSERT OR IGNORE INTO cayu_external_wait_scopes(scope,generation,limits_json) VALUES (?,?,?)",
                    (*namespace, request.limits.model_dump_json()),
                )
                owner = self._connection.execute(
                    "SELECT limits_json,retired,retirement_json FROM cayu_external_wait_scopes WHERE scope=? AND generation=?",
                    namespace,
                ).fetchone()
                if owner[0] != request.limits.model_dump_json():
                    raise ExternalWaitConflict("External wait scope limits conflict.")
                if owner[1]:
                    return read_retirement(owner[2], request)
                if owner[2] is not None:
                    raise ExternalWaitConflict("External retirement state conflicts.")
                accumulator = RetirementAccumulator(request)
                cursor = self._connection.execute(
                    "SELECT * FROM cayu_external_waits WHERE scope=? AND generation=? ORDER BY correlation_key",
                    namespace,
                )
                for row in cursor:
                    accumulator.add(validate_external_wait_row(row))
                receipt = accumulator.finish(int(self._ownership_clock().timestamp() * 1000))
                self._connection.execute(
                    "UPDATE cayu_external_wait_scopes SET retired=1,retirement_json=? WHERE scope=? AND generation=?",
                    (receipt.model_dump_json(), *namespace),
                )
                return receipt

    async def _prune_external_wait_scope(
        self, retirement: ExternalWaitRetirement, *, limit: int
    ) -> ExternalWaitPruneResult:
        from cayu.sessions._external_wait_retirement import read_retirement, require_prune_limit

        require_prune_limit(limit)
        scope = retirement.request.scope
        namespace = (scope.application_scope, scope.generation)
        async with self._lock:
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                owner = self._connection.execute(
                    "SELECT retired,retirement_json,limits_json FROM cayu_external_wait_scopes WHERE scope=? AND generation=?",
                    namespace,
                ).fetchone()
                if (
                    owner is None
                    or owner[0] != 1
                    or owner[2] != retirement.request.limits.model_dump_json()
                    or read_retirement(owner[1], retirement.request) != retirement
                ):
                    raise ExternalWaitConflict("External retirement evidence conflicts.")
                keys = self._connection.execute(
                    "SELECT correlation_key FROM cayu_external_waits WHERE scope=? AND generation=? ORDER BY correlation_key LIMIT ?",
                    (*namespace, limit),
                ).fetchall()
                self._connection.executemany(
                    "DELETE FROM cayu_external_waits WHERE scope=? AND generation=? AND correlation_key=?",
                    ((*namespace, key[0]) for key in keys),
                )
                remaining = self._connection.execute(
                    "SELECT COUNT(*) FROM cayu_external_waits WHERE scope=? AND generation=?",
                    namespace,
                ).fetchone()[0]
                return ExternalWaitPruneResult(
                    retirement=retirement, removed=len(keys), remaining=remaining
                )

    async def _read_external_wait(
        self, scope: ExternalWaitScope, correlation_key: str
    ) -> ExternalWaitRecord | None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT * FROM cayu_external_waits WHERE scope=? AND generation=? AND correlation_key=?",
                (scope.application_scope, scope.generation, correlation_key),
            ).fetchone()
            return None if row is None else validate_external_wait_row(row)

    async def _list_external_waits(
        self, scope: ExternalWaitScope, *, source: str, after: str, limit: int
    ) -> tuple[ExternalWaitRecord, ...]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM cayu_external_waits WHERE scope=? AND generation=? AND source=? AND correlation_key>? ORDER BY correlation_key LIMIT ?",
                (scope.application_scope, scope.generation, source, after, limit),
            ).fetchall()
            return tuple(validate_external_wait_row(row) for row in rows)
