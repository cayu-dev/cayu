"""Memory transaction owner for external waits."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

from cayu.sessions._external_wait_records import (
    external_wait_session_identity,
)
from cayu.sessions._external_wait_transition import ExternalWaitMutation, transition
from cayu.sessions.external_waits import (
    ExternalWaitConflict,
    ExternalWaitLimits,
    ExternalWaitPruneResult,
    ExternalWaitRecord,
    ExternalWaitRetirement,
    ExternalWaitRetirementRequest,
    ExternalWaitScope,
)


class MemoryExternalWaitMixin:
    _lock: asyncio.Lock
    _ownership_clock: Callable[[], datetime]
    _external_wait_records: dict[tuple[str, int], dict[str, ExternalWaitRecord]]
    _external_wait_scopes: dict[tuple[str, int], ExternalWaitLimits]
    _external_wait_pending_sessions: dict[tuple[str, str], set[tuple[str, int, str]]]
    _external_wait_retirements: dict[tuple[str, int], ExternalWaitRetirement]
    _sessions: dict[str, Any]
    _checkpoints: dict[str, Any]
    _participant_session_bindings: dict[str, Any]
    _session_operation_records: dict[str, dict[str, Any]]

    def _require_external_creation_unlocked(self, creation_request) -> None:
        from cayu.runtime._external_wait_creation import (
            current_external_creation,
            require_external_creation,
        )

        expected = current_external_creation()
        if expected is None:
            return
        registration, _ = expected
        self._external_wait_initialize()
        request = registration.correlation.request
        record = self._external_wait_records.get(
            (request.scope.application_scope, request.scope.generation), {}
        ).get(request.correlation_key)
        require_external_creation(record, creation_request)

    def _external_wait_initialize(self) -> None:
        if not hasattr(self, "_external_wait_records"):
            self._external_wait_records = {}
            self._external_wait_scopes = {}
            self._external_wait_pending_sessions = {}
            self._external_wait_retirements = {}

    def _require_external_wait_quiescence_unlocked(self, session: Any) -> None:
        self._external_wait_initialize()
        if self._external_wait_pending_sessions.get((session.id, session.instance_id)):
            raise ValueError("Session has a pending external-wait handoff.")

    def _require_external_wait_admission_unlocked(self, session: Any) -> None:
        from cayu.runtime._external_wait_admission import require_external_admission

        self._external_wait_initialize()
        records = (
            self._external_wait_records[(scope, generation)][key]
            for scope, generation, key in self._external_wait_pending_sessions.get(
                (session.id, session.instance_id), ()
            )
        )
        require_external_admission(
            (record for record in records if record.handoff == "unbound"), session
        )

    async def _mutate_external_wait(self, command: ExternalWaitMutation) -> ExternalWaitRecord:
        async with self._lock:
            self._external_wait_initialize()
            scope = command.request.scope
            namespace = (scope.application_scope, scope.generation)
            key = command.request.correlation_key
            if namespace in self._external_wait_retirements:
                raise ExternalWaitConflict("External wait scope is retired.")
            if (
                namespace in self._external_wait_scopes
                and self._external_wait_scopes[namespace] != command.limits
            ):
                raise ExternalWaitConflict("External wait scope limits conflict.")
            records = self._external_wait_records.get(namespace, {})
            if (
                command.kind == "prepare_execution"
                and command.execution_intent is not None
                and command.execution_intent.mode == "resume"
            ):
                from cayu.runtime._external_wait_admission import require_resume_preparation

                session_id = command.execution_intent.session_id
                if session_id in self._participant_session_bindings:
                    raise PermissionError(
                        "External waits do not support participant-owned sessions."
                    )
                require_resume_preparation(
                    command,
                    records.get(key),
                    self._sessions.get(session_id),
                    self._checkpoints.get(session_id),
                )
                if records.get(key) is None or records[key].execution is None:
                    self._require_external_wait_admission_unlocked(self._sessions[session_id])
            if command.kind == "exclude_execution":
                from cayu.runtime._external_wait_creation import require_execution_exclusion

                current = records.get(key)
                session = (
                    None
                    if current is None or current.execution is None
                    else self._sessions.get(current.execution.intent.session_id)
                )
                require_execution_exclusion(
                    command,
                    current,
                    session,
                    None if session is None else self._checkpoints.get(session.id),
                )
            if command.kind == "bind":
                from cayu.runtime._external_wait_binding import (
                    require_binding_scope,
                    require_binding_writer,
                )
                from cayu.sessions._session_continuation import continuation_operation_key

                require_binding_scope(command)
                current = records.get(key)
                if current is None or current.continuation is None:
                    assert command.continuation is not None
                    session_id = command.continuation.intent.session_id
                    if session_id in self._participant_session_bindings:
                        raise PermissionError(
                            "External waits do not support participant-owned sessions."
                        )
                    require_binding_writer(
                        command,
                        self._sessions.get(session_id),
                        self._checkpoints.get(session_id),
                        self._session_operation_records.get(session_id, {}).get(
                            continuation_operation_key(command.continuation.intent)
                        ),
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
                require_native_settlement(
                    command,
                    records.get(key),
                    self._session_operation_records.get(
                        command.continuation.intent.session_id, {}
                    ).get(continuation_operation_key(command.continuation.intent)),
                )
            result = transition(
                records.get(key),
                command,
                now_ms=int(self._ownership_clock().timestamp() * 1000),
                count=len(records),
                reserved_bytes=(
                    sum(v.reserved_bytes for v in records.values())
                    if key not in records and command.kind == "reserve"
                    else 0
                ),
            )
            # Stage all validation before touching either dictionary.
            self._external_wait_scopes[namespace] = command.limits
            records[key] = result.model_copy(deep=True)
            self._external_wait_records[namespace] = records
            if result.continuation is not None or result.execution is not None:
                identity = external_wait_session_identity(result)
                pending = self._external_wait_pending_sessions.setdefault(identity, set())
                if result.pending_handoff:
                    pending.add((*namespace, key))
                else:
                    pending.discard((*namespace, key))
                if not pending:
                    self._external_wait_pending_sessions.pop(identity, None)
            return result.model_copy(deep=True)

    async def _read_external_wait_retirement(
        self, scope: ExternalWaitScope
    ) -> ExternalWaitRetirement | None:
        async with self._lock:
            self._external_wait_initialize()
            receipt = self._external_wait_retirements.get(
                (scope.application_scope, scope.generation)
            )
            return None if receipt is None else receipt.model_copy(deep=True)

    async def _retire_external_wait_scope(
        self, request: ExternalWaitRetirementRequest
    ) -> ExternalWaitRetirement:
        from cayu.sessions._external_wait_retirement import RetirementAccumulator

        async with self._lock:
            self._external_wait_initialize()
            namespace = (request.scope.application_scope, request.scope.generation)
            existing = self._external_wait_retirements.get(namespace)
            if existing is not None:
                if existing.request != request:
                    raise ExternalWaitConflict("External retirement operation conflicts.")
                return existing.model_copy(deep=True)
            limits = self._external_wait_scopes.get(namespace)
            if limits is not None and limits != request.limits:
                raise ExternalWaitConflict("External wait scope limits conflict.")
            records = self._external_wait_records.get(namespace, {})
            accumulator = RetirementAccumulator(request)
            for key in sorted(records):
                accumulator.add(records[key])
            receipt = accumulator.finish(int(self._ownership_clock().timestamp() * 1000))
            self._external_wait_scopes[namespace] = request.limits
            self._external_wait_retirements[namespace] = receipt
            return receipt.model_copy(deep=True)

    async def _prune_external_wait_scope(
        self, retirement: ExternalWaitRetirement, *, limit: int
    ) -> ExternalWaitPruneResult:
        from cayu.sessions._external_wait_retirement import require_prune_limit

        require_prune_limit(limit)
        async with self._lock:
            self._external_wait_initialize()
            scope = retirement.request.scope
            namespace = (scope.application_scope, scope.generation)
            if self._external_wait_retirements.get(namespace) != retirement:
                raise ExternalWaitConflict("External retirement evidence conflicts.")
            records = self._external_wait_records.get(namespace, {})
            keys = sorted(records)[:limit]
            for key in keys:
                del records[key]
            return ExternalWaitPruneResult(
                retirement=retirement, removed=len(keys), remaining=len(records)
            )

    async def _read_external_wait(
        self, scope: ExternalWaitScope, correlation_key: str
    ) -> ExternalWaitRecord | None:
        async with self._lock:
            self._external_wait_initialize()
            found = self._external_wait_records.get(
                (scope.application_scope, scope.generation), {}
            ).get(correlation_key)
            return None if found is None else found.model_copy(deep=True)

    async def _list_external_waits(
        self, scope: ExternalWaitScope, *, source: str, after: str, limit: int
    ) -> tuple[ExternalWaitRecord, ...]:
        async with self._lock:
            self._external_wait_initialize()
            records = self._external_wait_records.get(
                (scope.application_scope, scope.generation), {}
            )
            keys = sorted(
                key
                for key, record in records.items()
                if key > after and record.correlation.request.source == source
            )[:limit]
            return tuple(records[key].model_copy(deep=True) for key in keys)
