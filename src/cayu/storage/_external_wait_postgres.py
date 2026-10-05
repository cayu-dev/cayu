"""PostgreSQL external-wait owner: scope lock, then record, then store clock."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


class PostgresExternalWaitMixin:
    async def _require_external_wait_admission(self, cursor, session) -> None:
        from psycopg.rows import dict_row

        from cayu.runtime._external_wait_admission import (
            current_external_admission,
            require_external_admission,
        )

        async with cursor.connection.cursor(row_factory=dict_row) as reader:
            expected = current_external_admission()
            if expected is not None:
                scope = expected[0].correlation.request.scope
                await reader.execute(
                    "SELECT generation FROM cayu_external_wait_scopes WHERE scope=%s AND generation=%s FOR UPDATE",
                    (scope.application_scope, scope.generation),
                )
                if await reader.fetchone() is None:
                    raise ExternalWaitConflict("External admission scope is unavailable.")
            await reader.execute(
                "SELECT * FROM cayu_external_waits WHERE session_id=%s AND session_instance_id=%s "
                "AND pending_handoff=1 AND handoff='unbound' LIMIT 2",
                (session.id, session.instance_id),
            )
            rows = await reader.fetchall()
        require_external_admission((validate_external_wait_row(row) for row in rows), session)

    async def _require_external_creation(self, connection, creation_request) -> None:
        from psycopg.rows import dict_row

        from cayu.runtime._external_wait_creation import (
            current_external_creation,
            require_external_creation,
        )

        expected = current_external_creation()
        if expected is None:
            return
        registration, _ = expected
        request = registration.correlation.request
        namespace = (request.scope.application_scope, request.scope.generation)
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                "SELECT generation FROM cayu_external_wait_scopes WHERE scope=%s AND generation=%s FOR UPDATE",
                namespace,
            )
            if await cursor.fetchone() is None:
                raise ExternalWaitConflict("External creation scope is unavailable.")
            await cursor.execute(
                "SELECT * FROM cayu_external_waits WHERE scope=%s AND generation=%s AND correlation_key=%s",
                (*namespace, request.correlation_key),
            )
            row = await cursor.fetchone()
            require_external_creation(
                None if row is None else validate_external_wait_row(row), creation_request
            )

    if TYPE_CHECKING:

        async def _ensure_ready(self) -> None: ...

        def _connection(self) -> Any: ...
        async def _load_for_update(self, cur: Any, session_id: str) -> Any: ...
        async def _load_checkpoint(self, cur: Any, session_id: str) -> Any: ...

    async def _mutate_external_wait(self, command: ExternalWaitMutation) -> ExternalWaitRecord:
        from psycopg.rows import dict_row

        await self._ensure_ready()
        scope = command.request.scope
        namespace = (scope.application_scope, scope.generation)
        key = (*namespace, command.request.correlation_key)
        async with (
            self._connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            session = checkpoint = None
            if command.kind == "exclude_execution" or (
                command.kind == "prepare_execution"
                and command.execution_intent is not None
                and command.execution_intent.mode == "resume"
            ):
                from cayu.runtime._external_wait_execution_scope import (
                    require_execution_preparation,
                )

                require_execution_preparation(command)
                assert command.execution_intent is not None
                # Existing writers release under the session lock. Take it
                # before the wait-scope lock, matching native binding/settlement.
                async with connection.cursor() as session_cursor:
                    session = await self._load_for_update(
                        session_cursor, command.execution_intent.session_id
                    )
                    checkpoint = await self._load_checkpoint(
                        session_cursor, command.execution_intent.session_id
                    )
            if command.kind in {
                "bind",
                "settle",
                "prepare_service",
                "prepare_retirement",
                "complete_retirement",
                "reconcile_binding",
            }:
                from cayu.runtime._external_wait_binding import require_binding_scope
                from cayu.runtime._external_wait_settlement import require_settlement_scope

                if command.kind == "bind":
                    require_binding_scope(command)
                else:
                    require_settlement_scope(command)
                assert command.continuation is not None
                # Match the session owner's lock order: session before wait scope.
                async with connection.cursor() as session_cursor:
                    session = await self._load_for_update(
                        session_cursor, command.continuation.intent.session_id
                    )
                    checkpoint = await self._load_checkpoint(
                        session_cursor, command.continuation.intent.session_id
                    )
            limits_json = command.limits.model_dump_json()
            await cursor.execute(
                "INSERT INTO cayu_external_wait_scopes(scope,generation,limits_json) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                (*namespace, limits_json),
            )
            await cursor.execute(
                "SELECT limits_json,retired FROM cayu_external_wait_scopes WHERE scope=%s AND generation=%s FOR UPDATE",
                namespace,
            )
            owner = await cursor.fetchone()
            if owner["limits_json"] != limits_json or owner["retired"]:
                raise ExternalWaitConflict("External wait scope is retired or its limits conflict.")
            await cursor.execute(
                "SELECT * FROM cayu_external_waits WHERE scope=%s AND generation=%s AND correlation_key=%s",
                key,
            )
            row = await cursor.fetchone()
            current = None if row is None else validate_external_wait_row(row)
            if (
                command.kind == "prepare_execution"
                and command.execution_intent is not None
                and command.execution_intent.mode == "resume"
            ):
                from cayu.runtime._external_wait_admission import require_resume_preparation

                await cursor.execute(
                    "SELECT 1 FROM cayu_participant_session_bindings WHERE session_id=%s",
                    (command.execution_intent.session_id,),
                )
                if await cursor.fetchone() is not None:
                    raise PermissionError(
                        "External waits do not support participant-owned sessions."
                    )
                require_resume_preparation(command, current, session, checkpoint)
                if current is None or current.execution is None:
                    await self._require_external_wait_admission(cursor, session)
            if command.kind == "exclude_execution":
                from cayu.runtime._external_wait_creation import require_execution_exclusion

                # If the first read found no row, CREATE might have committed
                # before we obtained its wait-scope lock. Re-read while holding
                # that lock; do not acquire a session lock in the reverse order.
                if session is None and current is not None and current.execution is not None:
                    await cursor.execute(
                        "SELECT instance_id FROM cayu_sessions WHERE id=%s",
                        (current.execution.intent.session_id,),
                    )
                    existing_row = await cursor.fetchone()
                    if (
                        existing_row is not None
                        and existing_row["instance_id"] == current.execution.session_instance_id
                    ):
                        raise ExternalWaitConflict(
                            "External execution was created; reconcile its native writer."
                        )
                require_execution_exclusion(command, current, session, checkpoint)
            if command.kind == "bind" and (current is None or current.continuation is None):
                from cayu.runtime._external_wait_binding import require_binding_writer

                assert command.continuation is not None
                await cursor.execute(
                    "SELECT 1 FROM cayu_participant_session_bindings WHERE session_id=%s",
                    (command.continuation.intent.session_id,),
                )
                if await cursor.fetchone() is not None:
                    raise PermissionError(
                        "External waits do not support participant-owned sessions."
                    )

                from cayu.sessions._session_continuation import continuation_operation_key

                await cursor.execute(
                    "SELECT record FROM cayu_session_operations WHERE session_id=%s AND idempotency_key=%s",
                    (
                        command.continuation.intent.session_id,
                        continuation_operation_key(command.continuation.intent),
                    ),
                )
                native = await cursor.fetchone()
                require_binding_writer(
                    command,
                    session,
                    checkpoint,
                    None if native is None else native["record"],
                    execution=None if current is None else current.execution,
                )
            if command.kind in {
                "settle",
                "prepare_service",
                "prepare_retirement",
                "complete_retirement",
                "reconcile_binding",
            }:
                from cayu.runtime._external_wait_settlement import require_native_settlement
                from cayu.sessions._session_continuation import continuation_operation_key

                assert command.continuation is not None
                await cursor.execute(
                    "SELECT record FROM cayu_session_operations WHERE session_id=%s AND idempotency_key=%s",
                    (
                        command.continuation.intent.session_id,
                        continuation_operation_key(command.continuation.intent),
                    ),
                )
                native = await cursor.fetchone()
                require_native_settlement(
                    command, current, None if native is None else native["record"]
                )
            totals = {"count": 0, "bytes": 0}
            if current is None and command.kind == "reserve":
                await cursor.execute(
                    "SELECT COUNT(*) AS count,COALESCE(SUM(reserved_bytes),0) AS bytes FROM cayu_external_waits WHERE scope=%s AND generation=%s",
                    namespace,
                )
                totals = await cursor.fetchone()
            # transaction_timestamp() can predate waiting for a competing
            # writer. Sample clock_timestamp only AFTER acquiring ownership.
            await cursor.execute(
                "SELECT floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint AS now"
            )
            now = (await cursor.fetchone())["now"]
            result = transition(
                current,
                command,
                now_ms=now,
                count=totals["count"],
                reserved_bytes=totals["bytes"],
            )
            if result != current:
                await cursor.execute(
                    "INSERT INTO cayu_external_waits(scope,generation,correlation_key,source,incarnation,revision,reserved_bytes,handoff,pending_handoff,session_id,session_instance_id,record_json) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(scope,generation,correlation_key) DO UPDATE SET "
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

        await self._ensure_ready()
        async with self._connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                "SELECT retired,retirement_json,limits_json FROM cayu_external_wait_scopes "
                "WHERE scope=%s AND generation=%s",
                (scope.application_scope, scope.generation),
            )
            return read_scope_retirement(await cursor.fetchone(), scope)

    async def _retire_external_wait_scope(
        self, request: ExternalWaitRetirementRequest
    ) -> ExternalWaitRetirement:
        from psycopg.rows import dict_row

        from cayu.sessions._external_wait_retirement import RetirementAccumulator, read_retirement

        await self._ensure_ready()
        namespace = (request.scope.application_scope, request.scope.generation)
        async with (
            self._connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                "INSERT INTO cayu_external_wait_scopes(scope,generation,limits_json) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                (*namespace, request.limits.model_dump_json()),
            )
            await cursor.execute(
                "SELECT limits_json,retired,retirement_json FROM cayu_external_wait_scopes WHERE scope=%s AND generation=%s FOR UPDATE",
                namespace,
            )
            owner = await cursor.fetchone()
            if owner["limits_json"] != request.limits.model_dump_json():
                raise ExternalWaitConflict("External wait scope limits conflict.")
            if owner["retired"]:
                return read_retirement(owner["retirement_json"], request)
            if owner["retirement_json"] is not None:
                raise ExternalWaitConflict("External retirement state conflicts.")
            accumulator = RetirementAccumulator(request)
            after = ""
            while True:
                await cursor.execute(
                    'SELECT * FROM cayu_external_waits WHERE scope=%s AND generation=%s AND correlation_key COLLATE "C">%s ORDER BY correlation_key COLLATE "C" LIMIT 32',
                    (*namespace, after),
                )
                rows = await cursor.fetchall()
                if not rows:
                    break
                for row in rows:
                    accumulator.add(validate_external_wait_row(row))
                after = accumulator.last_key
            await cursor.execute(
                "SELECT floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint AS now"
            )
            receipt = accumulator.finish((await cursor.fetchone())["now"])
            await cursor.execute(
                "UPDATE cayu_external_wait_scopes SET retired=1,retirement_json=%s WHERE scope=%s AND generation=%s",
                (receipt.model_dump_json(), *namespace),
            )
            return receipt

    async def _prune_external_wait_scope(
        self, retirement: ExternalWaitRetirement, *, limit: int
    ) -> ExternalWaitPruneResult:
        from psycopg.rows import dict_row

        from cayu.sessions._external_wait_retirement import read_retirement, require_prune_limit

        require_prune_limit(limit)
        await self._ensure_ready()
        scope = retirement.request.scope
        namespace = (scope.application_scope, scope.generation)
        async with (
            self._connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                "SELECT retired,retirement_json,limits_json FROM cayu_external_wait_scopes WHERE scope=%s AND generation=%s FOR UPDATE",
                namespace,
            )
            owner = await cursor.fetchone()
            if (
                owner is None
                or owner["retired"] != 1
                or owner["limits_json"] != retirement.request.limits.model_dump_json()
                or read_retirement(owner["retirement_json"], retirement.request) != retirement
            ):
                raise ExternalWaitConflict("External retirement evidence conflicts.")
            await cursor.execute(
                "DELETE FROM cayu_external_waits WHERE scope=%s AND generation=%s AND correlation_key IN "
                "(SELECT correlation_key FROM cayu_external_waits WHERE scope=%s AND generation=%s ORDER BY correlation_key LIMIT %s) RETURNING correlation_key",
                (*namespace, *namespace, limit),
            )
            removed = len(await cursor.fetchall())
            await cursor.execute(
                "SELECT COUNT(*) AS count FROM cayu_external_waits WHERE scope=%s AND generation=%s",
                namespace,
            )
            return ExternalWaitPruneResult(
                retirement=retirement, removed=removed, remaining=(await cursor.fetchone())["count"]
            )

    async def _read_external_wait(
        self, scope: ExternalWaitScope, correlation_key: str
    ) -> ExternalWaitRecord | None:
        from psycopg.rows import dict_row

        await self._ensure_ready()
        async with (
            self._connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                "SELECT * FROM cayu_external_waits WHERE scope=%s AND generation=%s AND correlation_key=%s",
                (scope.application_scope, scope.generation, correlation_key),
            )
            row = await cursor.fetchone()
            return None if row is None else validate_external_wait_row(row)

    async def _list_external_waits(
        self, scope: ExternalWaitScope, *, source: str, after: str, limit: int
    ) -> tuple[ExternalWaitRecord, ...]:
        from psycopg.rows import dict_row

        await self._ensure_ready()
        async with (
            self._connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                "SELECT * FROM cayu_external_waits WHERE scope=%s AND generation=%s AND source=%s AND correlation_key>%s ORDER BY correlation_key LIMIT %s",
                (scope.application_scope, scope.generation, source, after, limit),
            )
            return tuple(validate_external_wait_row(row) for row in await cursor.fetchall())
