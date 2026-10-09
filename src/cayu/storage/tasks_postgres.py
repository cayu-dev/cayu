"""PostgreSQL task persistence, admission notifications and terminal receipts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Literal, LiteralString, cast
from uuid import uuid4
from weakref import ReferenceType, ref

from cayu._resource_store_surface import model_store_surface
from cayu.storage import _postgres_base as postgres_base
from cayu.storage import _postgres_support as pg_support
from cayu.storage._phase_timing import PostgresTimingScope

if TYPE_CHECKING:
    from cayu.tasks.groups import (
        TaskGroupCreate,
        TaskGroupCreationReceipt,
        TaskGroupEvent,
        TaskGroupInvocationObligation,
        TaskGroupQuiescenceResolution,
        TaskGroupSnapshot,
    )
try:
    from psycopg import AsyncConnection, sql
    from psycopg.errors import UniqueViolation
    from psycopg.types.json import Jsonb
    from psycopg_pool import AsyncConnectionPool  # noqa: TC002 — keep runtime annotation resolution
except ModuleNotFoundError as exc:
    raise RuntimeError(
        'Cayu\'s Postgres stores require the optional psycopg packages. Install them with `pip install "cayu[postgres]"`.'
    ) from exc
from cayu._clock import normalize_utc_datetime, utc_clock
from cayu._validation import copy_durable_json_object, require_nonblank, revalidate_model_input
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.aggregates import EXACT_AGGREGATE
from cayu.runtime._task_admission_wakeup import TaskAdmissionWakeup
from cayu.runtime._task_lease_authority import managed_task_lease_mutation
from cayu.runtime.local_execution_attempts import (
    LocalExecutionAttemptAuthority,
    LocalExecutionAttemptConflict,
    LocalExecutionAttemptListCursor,
    LocalExecutionAttemptRecord,
    LocalExecutionAttemptRecoveryClaim,
    LocalExecutionAttemptSettlement,
    LocalExecutionAttemptStart,
    _copy_authenticated_local_execution_attempt_settlement,
    _copy_local_execution_attempt_authority,
    _copy_local_execution_attempt_list_cursor,
    _copy_local_execution_attempt_recovery_claim,
    _copy_local_execution_attempt_start,
    advance_local_execution_attempt_start,
    claim_local_execution_attempt_recovery_record,
    prepare_local_execution_attempt_record,
    require_local_execution_recovery_eligible,
    require_local_execution_task_authority,
    settle_local_execution_attempt_record,
)
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.sessions.invocation import SessionInvocationBinding, TaskInvocation
from cayu.storage import migrations as schema
from cayu.storage._postgres_verified_work import (
    PostgresVerifiedWorkMixin,
    _PostgresMutationConnectionOwner,
    _require_quiescent_postgres_mutation_pool,
)
from cayu.tasks import _verified_work_policy as verified_work_support
from cayu.tasks._cancellation import (
    _expired_dispatched_task_cancellation,
    _reconciled_task_cancellation,
    _reconciled_task_retry_cancellation,
    _task_cancellation_requested_task,
    _task_retry_cancellation_requested_task,
    _validate_ordinary_task_terminalization_against_cancellation,
    _validated_task_cancellation,
    _validated_task_retry_cancellation,
)
from cayu.tasks._lifecycle import (
    _can_attach_claimed_task_state,
    _copy_optional_status_payload,
    _copy_optional_status_reason,
    _ensure_can_hold_task,
    _ensure_can_resume_task,
    _ensure_can_transition,
    _ensure_exact_owned_active_task_lease,
    _ensure_owned_active_task_lease,
    _ensure_recovered_attached_task_failure_authority,
    _ensure_recovered_attached_task_session,
    _ensure_retry_series_queue_attempt,
    _ensure_task_handoff_authority,
    _ensure_task_terminalization_lease_authority,
    _raise_task_claim_attach_error,
    _require_active_attached_task_worker,
    _require_direct_attached_task_resume,
)
from cayu.tasks._scheduling import (
    admitted_schedule,
    require_schedule_mutation,
    rescheduled_task,
    schedule_creation_digest,
    schedule_mutation_digest,
    schedule_receipt,
    schedule_revision_after,
    schedule_transition_events,
)
from cayu.tasks.access import runtime_collection_read, runtime_task_creation, runtime_task_mutation
from cayu.tasks.admission import WorkAttemptExecutionClaimLost
from cayu.tasks.cancellation import (
    _TASK_CANCELLATION_REQUESTED_REASON,
    _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
    TaskCancellationReconciliationRequest,
    TaskCancellationReconciliationResult,
    TaskRetryCancellationReconciliationRequest,
    _copy_task_cancellation_reconciliation_result,
    _rejected_task_cancellation_reconciliation,
    _rejected_task_retry_cancellation_reconciliation,
    _replay_task_cancellation_reconciliation,
    _replay_task_cancellation_reconciliation_rejection,
    _replay_task_retry_cancellation_reconciliation_rejection,
    _task_cancellation_reconciliation_conflict,
    _task_cancellation_reconciliation_rejection_record,
    _task_cancellation_requested,
    _task_retry_cancellation_reconciliation_conflict,
    _task_retry_cancellation_reconciliation_rejection_record,
    _task_retry_reconciliation_identity_is_bounded,
    _TaskCancellationReconciliationRejectionRecord,
    _TaskRetryCancellationReconciliationRejectionRecord,
    prepare_task_cancellation_reconciliation,
    prepare_task_retry_cancellation_reconciliation,
)
from cayu.tasks.contracts import WorkCompletionConflict
from cayu.tasks.creation import (
    TaskCreate,
    TaskInvocationSnapshot,
    _copy_optional_session_binding,
    _copy_required_session_binding,
    _running_task_from_create,
    _task_from_create,
    _task_invocation_for_attachment,
    _task_session_id_for_start,
    _task_session_instance_for_attachment,
    copy_task_create,
)
from cayu.tasks.graphs import (
    TaskGraphCreate,
    TaskGraphCreationReceipt,
    TaskGraphEvent,
    TaskGraphSnapshot,
)
from cayu.tasks.handoff import (
    _TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    InterruptedTaskContinuationClaimPage,
    TaskInterruptedHandoffConflict,
    TaskInterruptedHandoffReceipt,
    TaskInterruptedHandoffRequest,
    _interrupted_task_continuation_handoff_id_sha256,
    _replay_interrupted_task_handoff_receipt,
    _require_interrupted_task_handoff_authority,
    prepare_interrupted_task_continuation_claim_page,
    prepare_interrupted_task_handoff,
    prepare_interrupted_task_handoff_candidate_page,
    prepare_interrupted_task_handoff_receipt_lookup,
)
from cayu.tasks.queries import (
    TaskAggregateFilter,
    TaskOperationalSnapshot,
    TaskOrder,
    TaskQuery,
    TaskStatusCounts,
    _ensure_claim_query_supported,
    _task_matches_claim_filter,
    copy_task_aggregate_filter,
    copy_task_query,
    task_query_from_aggregate_filter,
)
from cayu.tasks.records import (
    Task,
    TaskClaimLost,
    TaskRetrySeriesDisposition,
    TaskSessionClosureClaim,
    TaskStatus,
    copy_task_session_closure_claim,
)
from cayu.tasks.retry import (
    TaskRetrySettlementRequest,
    TaskRetrySettlementResult,
    _cancelled_task_retry_settlement,
    _claimed_task_retry_attempt_elapsed,
    _elapsed_claimed_task_retry_settlement,
    _expired_task_retry_settlement,
    _replay_task_retry_cancellation_reconciliation,
    _replay_task_retry_settlement,
    _scheduled_task_nonexecution,
    _settled_task_retry_attempt,
    _task_retry_events,
    _validated_task_retry_terminal_accounting,
    prepare_task_retry_settlement,
)
from cayu.tasks.scheduling import (
    TaskRescheduleRequest,
    TaskScheduleCancelRequest,
    TaskScheduleConflict,
    TaskScheduleEligibility,
    TaskScheduleEvent,
    TaskScheduleEventType,
    TaskScheduleReceipt,
    TaskScheduleWakeup,
    task_schedule_eligibility,
)
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    _replay_task_terminalization_receipt,
    prepare_task_terminalization,
    prepare_task_terminalization_receipt_lookup,
)
from cayu.tasks.topology import (
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    TaskTopologyInconsistent,
    TaskTopologyNode,
    TaskTopologyQuery,
    TaskTopologyStoreResult,
    _allocate_task_topology_branch_limits,
    _bounded_optional_task_topology_parent_id,
    _validate_task_topology_ancestry,
    build_task_topology_result,
    decode_task_topology_cursor,
)

_POSTGRES_TASK_MIN_REQUIRED_REVISION = 117
_TASK_ADMISSION_NOTIFY_CHANNEL = "cayu_task_admission_v1"
_TASK_ADMISSION_LISTENER_INITIAL_RECONNECT_DELAY_S = 0.1
_TASK_ADMISSION_LISTENER_MAX_RECONNECT_DELAY_S = 5.0
_TASK_RETURNING_COLUMNS = (
    "task.id, task.type, task.title, task.description, task.status, task.session_id, "
    "task.session_instance_id, task.parent_task_id, task.assigned_agent_name, "
    "task.available_at, task.worker_id, task.lease_expires_at, task.interrupted_handoff_id, "
    "task.status_reason, task.status_payload, task.input, task.result, task.error, task.metadata, "
    "task.created_at, task.updated_at, task.started_at, task.completed_at, task.invocation, "
    "task.retry_series, task.work_contract, task.schedule, task.graph_id, task.prerequisite_task_ids"
)


def _ilike_contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


@model_store_surface("tasks")
class PostgresTaskStore(PostgresVerifiedWorkMixin, postgres_base._PostgresStoreBase, TaskStore):
    """Postgres-backed task store for durable multi-tenant work items.

    Task-admission wakeups use one dedicated ``LISTEN`` connection outside the
    connection pool. A store that owns its DSN listens on that DSN, or on
    ``task_admission_listener_conninfo`` when given (for example a direct server
    address behind a transaction-pooling proxy). A store built on a caller-owned
    ``pool`` listens only when ``task_admission_listener_conninfo`` is supplied;
    otherwise workers fall back to bounded claim polling.
    """

    task_access_version: ClassVar[int | None] = 1

    supports_delayed_availability: ClassVar[bool] = True
    supports_task_graphs: ClassVar[bool] = True
    supports_task_groups: ClassVar[bool] = True
    supports_task_group_quiescence: ClassVar[bool] = True

    async def list_task_group_reconciliation_candidates(
        self,
        *,
        after_group_id: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        from cayu.storage._postgres_task_groups import reconciliation_candidates

        return await reconciliation_candidates(self, after_group_id=after_group_id, limit=limit)

    async def _settle_task_group_execution(self, task: Task) -> None:
        from cayu.storage._postgres_task_groups import settle_execution

        await settle_execution(self, task)

    async def _task_group_cancellation_requested(self, task_id: str) -> bool:
        from cayu.storage._postgres_task_groups import cancellation_requested

        return await cancellation_requested(self, task_id)

    async def _task_group_retains_execution(self, task_id: str) -> bool:
        from cayu.storage._postgres_task_groups import retains_execution

        return await retains_execution(self, task_id)

    async def _observe_task_group_invocation(
        self, invocation: TaskGroupInvocationObligation
    ) -> None:
        from cayu.storage._postgres_task_groups import observe_invocation

        await observe_invocation(self, invocation)

    async def _observe_task_group_result_resolution(
        self, task_id: str, decision_id: str, owner_id: str, *, settled: bool
    ) -> None:
        from cayu.storage._postgres_task_groups import observe_result_resolution

        await observe_result_resolution(self, task_id, decision_id, owner_id, settled=settled)

    async def reconcile_task_group(self, group_id: str) -> TaskGroupSnapshot:
        from cayu.storage._postgres_task_groups import reconcile

        return await reconcile(self, group_id)

    async def resolve_task_group_quiescence(
        self,
        request: TaskGroupQuiescenceResolution,
    ) -> TaskGroupSnapshot:
        from cayu.storage._postgres_task_groups import reconcile

        return await reconcile(self, request.group_id, resolution=request)

    @runtime_task_creation
    async def create_task_group(self, request: TaskGroupCreate) -> TaskGroupCreationReceipt:
        from cayu.storage._postgres_task_groups import create_group

        return await create_group(self, request)

    @runtime_collection_read
    async def load_task_group(self, group_id: str) -> TaskGroupSnapshot | None:
        from cayu.storage._postgres_task_groups import load_group

        return await load_group(self, group_id)

    @runtime_collection_read
    async def list_task_group_events(
        self, group_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskGroupEvent]:
        from cayu.storage._postgres_task_groups import list_events

        return await list_events(self, group_id, after_sequence=after_sequence, limit=limit)

    supports_task_scheduling: ClassVar[bool] = True
    supports_task_topology: ClassVar[bool] = True
    supports_idempotent_terminalization: ClassVar[bool] = True
    supports_attached_task_recovery_terminalization: ClassVar[bool] = True
    supports_interrupted_task_handoffs: ClassVar[bool] = True
    supports_exact_interrupted_task_handoffs: ClassVar[bool] = True
    supports_task_cancellation_reconciliation: ClassVar[bool] = True
    supports_task_retry_series: ClassVar[bool] = True
    supports_verified_work_contracts: ClassVar[bool] = True
    supports_completion_verifier_dispatches: ClassVar[bool] = True
    supports_completion_evaluations: ClassVar[bool] = True
    supports_work_attempt_admission: ClassVar[bool] = True
    supports_verified_task_worker: ClassVar[bool] = True
    supports_local_execution_attempts: ClassVar[bool] = True
    supports_session_closure_deletion: ClassVar[bool] = True
    supports_session_closure_claims: ClassVar[bool] = True
    verified_work_mutations_are_cancellation_quiescent: ClassVar[bool] = True
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.DURABLE
    _min_required_revision = _POSTGRES_TASK_MIN_REQUIRED_REVISION

    @runtime_task_creation
    async def create_task_graph(self, request: TaskGraphCreate) -> TaskGraphCreationReceipt:
        from cayu.storage._postgres_task_graphs import create_graph

        return await create_graph(self, request)

    @runtime_collection_read
    async def load_task_graph(self, graph_id: str) -> TaskGraphSnapshot | None:
        from cayu.storage._postgres_task_graphs import load_graph

        return await load_graph(self, graph_id)

    @runtime_collection_read
    async def list_task_graph_events(
        self, graph_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskGraphEvent]:
        from cayu.storage._postgres_task_graphs import list_events

        return await list_events(self, graph_id, after_sequence=after_sequence, limit=limit)

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        clock: Callable[[], datetime] | None = None,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        read_only: bool = False,
        task_admission_listener_conninfo: str | None = None,
    ) -> None:
        if task_admission_listener_conninfo is not None:
            if type(task_admission_listener_conninfo) is not str:
                raise TypeError("task_admission_listener_conninfo must be a string.")
            task_admission_listener_conninfo = require_nonblank(
                task_admission_listener_conninfo,
                "task_admission_listener_conninfo",
            )
        if pool is not None:
            _require_quiescent_postgres_mutation_pool(pool)
        super().__init__(
            conninfo,
            pool=pool,
            min_size=min_size,
            max_size=max_size,
            schema_mode=schema_mode,
            read_only=read_only,
        )
        self.service_durability = (
            RuntimeStoreDurability.READ_ONLY if read_only else RuntimeStoreDurability.DURABLE
        )
        self._postgres_mutation_allowed_configure = (
            None if pool is not None else postgres_base._configure_store_connection
        )
        _require_quiescent_postgres_mutation_pool(
            self._pool,
            allowed_configure=self._postgres_mutation_allowed_configure,
        )
        self._clock = utc_clock(clock)
        self._clock_is_injected = clock is not None
        self._enable_task_admission_wakeups()
        # A caller-owned pool has no DSN of its own, so shared-pool callers pass
        # the address for one dedicated LISTEN connection outside that pool.
        self._task_admission_listener_conninfo = (
            task_admission_listener_conninfo
            if task_admission_listener_conninfo is not None
            else self._conninfo
        )
        self._task_admission_listener_task: asyncio.Task[None] | None = None
        self._task_admission_listener_first_attempt: asyncio.Event | None = None
        self._task_admission_listener_connection: Any | None = None
        self._task_admission_listener_closing = False
        self._task_admission_notification_senders: dict[int, tuple[ReferenceType[Any], int]] = {}

    async def _task_admission_wakeup(
        self,
        queries: Iterable[TaskQuery | None],
    ) -> TaskAdmissionWakeup | None:
        self._ensure_task_admission_listener()
        return await TaskStore._task_admission_wakeup(self, queries)

    def _ensure_task_admission_listener(self) -> None:
        """Start one empty-payload LISTEN connection when a listener DSN is known."""

        if self._task_admission_listener_conninfo is None or self._task_admission_listener_closing:
            return
        listener = self._task_admission_listener_task
        if listener is None or listener.done():
            first_attempt = asyncio.Event()
            self._task_admission_listener_first_attempt = first_attempt
            listener = asyncio.create_task(
                self._run_task_admission_listener(first_attempt),
                name="cayu-postgres-task-admission-listener",
            )
            self._task_admission_listener_task = listener

    async def _run_task_admission_listener(self, first_attempt: asyncio.Event) -> None:
        conninfo = self._task_admission_listener_conninfo
        if conninfo is None:
            first_attempt.set()
            return
        reconnect_delay = _TASK_ADMISSION_LISTENER_INITIAL_RECONNECT_DELAY_S
        while not self._task_admission_listener_closing:
            connection: Any | None = None
            try:
                connection = await AsyncConnection.connect(
                    conninfo,
                    autocommit=True,
                )
                connection.prepare_threshold = None
                await connection.execute(
                    sql.SQL("LISTEN {}").format(sql.Identifier(_TASK_ADMISSION_NOTIFY_CHANNEL))
                )
                self._task_admission_listener_connection = connection
                first_attempt.set()
                reconnect_delay = _TASK_ADMISSION_LISTENER_INITIAL_RECONNECT_DELAY_S
                async for notification in connection.notifies():
                    if (
                        notification.channel == _TASK_ADMISSION_NOTIFY_CHANNEL
                        and notification.payload == ""
                        and not self._task_admission_notification_is_self(notification.pid)
                    ):
                        self._publish_task_admission_broadcast()
                    if self._task_admission_listener_closing:
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                # Notifications are an optimization. Do not surface listener
                # failures into worker authority; bounded claim polling remains.
                first_attempt.set()
            finally:
                if self._task_admission_listener_connection is connection:
                    self._task_admission_listener_connection = None
                if connection is not None:
                    with suppress(Exception):
                        await connection.close()
            if not self._task_admission_listener_closing:
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 2,
                    _TASK_ADMISSION_LISTENER_MAX_RECONNECT_DELAY_S,
                )

    async def close(self) -> None:
        self._task_admission_listener_closing = True
        listener = self._task_admission_listener_task
        if listener is not None and not listener.done():
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener
        self._task_admission_listener_task = None
        self._task_admission_listener_connection = None
        self._task_admission_notification_senders.clear()
        await super().close()

    def _register_task_admission_notification_sender(self, connection: Any) -> int | None:
        """Count one content-free NOTIFY expected from this store's connection."""

        sender_pid = connection.info.backend_pid
        if type(sender_pid) is not int or sender_pid <= 0:
            return None
        existing = self._task_admission_notification_senders.get(sender_pid)
        count = 0
        if existing is not None and existing[0]() is connection:
            count = existing[1]
        self._task_admission_notification_senders[sender_pid] = (ref(connection), count + 1)
        return sender_pid

    def _discard_task_admission_notification_sender(
        self,
        sender_pid: int,
        connection: Any,
    ) -> None:
        existing = self._task_admission_notification_senders.get(sender_pid)
        if existing is None or existing[0]() is not connection:
            return
        if existing[1] > 1:
            self._task_admission_notification_senders[sender_pid] = (
                existing[0],
                existing[1] - 1,
            )
        else:
            self._task_admission_notification_senders.pop(sender_pid, None)

    def _task_admission_notification_is_self(self, sender_pid: int) -> bool:
        existing = self._task_admission_notification_senders.get(sender_pid)
        if existing is None:
            return False
        connection_ref, count = existing
        connection = connection_ref()
        if connection is None or connection.closed:
            self._task_admission_notification_senders.pop(sender_pid, None)
            return False
        try:
            is_self = connection.info.backend_pid == sender_pid
        except Exception:
            is_self = False
        if not is_self:
            self._task_admission_notification_senders.pop(sender_pid, None)
            return False
        if count > 1:
            self._task_admission_notification_senders[sender_pid] = (
                connection_ref,
                count - 1,
            )
        else:
            self._task_admission_notification_senders.pop(sender_pid, None)
        return True

    async def _ensure_ready(self) -> None:
        # Re-authenticate before every store entrance so private configuration
        # drift cannot run an extension callback during lazy readiness.
        _require_quiescent_postgres_mutation_pool(
            self._pool,
            allowed_configure=self._postgres_mutation_allowed_configure,
        )
        await super()._ensure_ready()

    async def _load_local_execution_attempt_row(
        self,
        cur: Any,
        attempt_id: str,
        *,
        for_update: bool = False,
    ) -> LocalExecutionAttemptRecord | None:
        await cur.execute(
            "SELECT attempt_id, task_id, retry_series_id, effect_lineage_id, "
            "request_sha256, phase, quiescence, retry_admissible, "
            "recovery_generation, recovery_owner_id, recovery_owner_expires_at, "
            "record_json, created_at, updated_at "
            "FROM cayu_local_execution_attempts WHERE attempt_id = %s"
            + (" FOR UPDATE" if for_update else ""),
            (attempt_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            record = LocalExecutionAttemptRecord.model_validate(pg_support._json_obj(row[11]))
        except (TypeError, ValueError):
            raise LocalExecutionAttemptConflict(
                "Stored local execution attempt content is malformed."
            ) from None
        if (
            record.authority.attempt_id != row[0]
            or record.authority.task_id != row[1]
            or record.authority.retry_series_id != row[2]
            or record.authority.effect_lineage_id != row[3]
            or record.authority.request_sha256 != row[4]
            or record.phase.value != row[5]
            or record.quiescence.value != row[6]
            or record.retry_admissible is not row[7]
            or record.recovery_generation != row[8]
            or record.recovery_owner_id != row[9]
            or record.recovery_owner_expires_at != pg_support.to_utc_optional(row[10])
            or record.created_at != pg_support.to_utc(row[12])
            or record.updated_at != pg_support.to_utc(row[13])
        ):
            raise LocalExecutionAttemptConflict(
                "Stored local execution attempt indexes conflict with canonical content."
            )
        return record

    async def _latest_local_execution_attempt_row(
        self,
        cur: Any,
        authority: LocalExecutionAttemptAuthority,
    ) -> LocalExecutionAttemptRecord | None:
        if authority.retry_series_id is None:
            await cur.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts "
                "WHERE retry_series_id IS NULL AND task_id = %s "
                "AND effect_lineage_id = %s "
                "ORDER BY retry_admissible ASC, created_at DESC, attempt_id DESC LIMIT 1",
                (authority.task_id, authority.effect_lineage_id),
            )
        else:
            await cur.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts "
                "WHERE retry_series_id = %s AND effect_lineage_id = %s "
                "ORDER BY retry_admissible ASC, created_at DESC, attempt_id DESC LIMIT 1",
                (authority.retry_series_id, authority.effect_lineage_id),
            )
        row = await cur.fetchone()
        return None if row is None else await self._load_local_execution_attempt_row(cur, row[0])

    async def _store_local_execution_attempt_row(
        self,
        cur: Any,
        record: LocalExecutionAttemptRecord,
        *,
        insert: bool,
    ) -> None:
        values = (
            record.authority.attempt_id,
            record.authority.task_id,
            record.authority.retry_series_id,
            record.authority.effect_lineage_id,
            record.authority.request_sha256,
            record.phase.value,
            record.quiescence.value,
            record.retry_admissible,
            record.recovery_generation,
            record.recovery_owner_id,
            record.recovery_owner_expires_at,
            pg_support._dumps(record.model_dump(mode="json", warnings=False)),
            record.created_at,
            record.updated_at,
        )
        if insert:
            await cur.execute(
                "INSERT INTO cayu_local_execution_attempts ("
                "attempt_id, task_id, retry_series_id, effect_lineage_id, request_sha256, "
                "phase, quiescence, retry_admissible, recovery_generation, "
                "recovery_owner_id, recovery_owner_expires_at, record_json, created_at, "
                "updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s)",
                values,
            )
            return
        await cur.execute(
            "UPDATE cayu_local_execution_attempts SET task_id = %s, "
            "retry_series_id = %s, effect_lineage_id = %s, request_sha256 = %s, "
            "phase = %s, quiescence = %s, retry_admissible = %s, "
            "recovery_generation = %s, recovery_owner_id = %s, "
            "recovery_owner_expires_at = %s, record_json = %s, created_at = %s, "
            "updated_at = %s WHERE attempt_id = %s AND request_sha256 = %s",
            (*values[1:], values[0], values[4]),
        )
        if cur.rowcount != 1:
            raise LocalExecutionAttemptConflict(
                "Local execution attempt changed during durable publication."
            )

    @staticmethod
    def _local_execution_retry_fence_scope(task: Task) -> str:
        retry_series = task.retry_series
        return f"retry:{retry_series.series_id}" if retry_series is not None else f"task:{task.id}"

    async def _lock_local_execution_retry_fence(self, cur: Any, task: Task) -> None:
        await self._lock_verified_work_identity(
            cur,
            "local-execution-retry-admission",
            self._local_execution_retry_fence_scope(task),
        )

    async def _local_execution_attempt_fences_task(
        self,
        cur: Any,
        task: Task,
    ) -> bool:
        retry_series = task.retry_series
        if retry_series is None:
            await cur.execute(
                "SELECT 1 FROM cayu_local_execution_attempts "
                "WHERE task_id = %s AND NOT retry_admissible LIMIT 1",
                (task.id,),
            )
        else:
            await cur.execute(
                "SELECT 1 FROM cayu_local_execution_attempts "
                "WHERE NOT retry_admissible AND (task_id = %s OR retry_series_id = %s) "
                "LIMIT 1",
                (task.id, retry_series.series_id),
            )
        return await cur.fetchone() is not None

    async def _require_no_competing_local_execution_task_owner(
        self,
        cur: Any,
        task: Task,
    ) -> None:
        retry_series = task.retry_series
        if retry_series is None:
            return
        await cur.execute(
            "SELECT 1 FROM cayu_tasks WHERE id <> %s AND retry_series IS NOT NULL "
            "AND retry_series->>'series_id' = %s AND status IN (%s, %s) LIMIT 1",
            (
                task.id,
                retry_series.series_id,
                str(TaskStatus.CLAIMED),
                str(TaskStatus.RUNNING),
            ),
        )
        if await cur.fetchone() is not None:
            raise LocalExecutionAttemptConflict(
                "Local execution attempt conflicts with another active retry-series task."
            )

    async def prepare_local_execution_attempt(
        self,
        authority: LocalExecutionAttemptAuthority,
    ) -> LocalExecutionAttemptRecord:
        authority = _copy_local_execution_attempt_authority(authority)
        await self._ensure_ready()

        async def mutation(_conn: Any, cur: Any) -> LocalExecutionAttemptRecord:
            await self._lock_verified_work_identity(
                cur,
                "local-execution-attempt",
                authority.attempt_id,
            )
            scope = (
                f"retry:{authority.retry_series_id}"
                if authority.retry_series_id is not None
                else f"task:{authority.task_id}"
            )
            await self._lock_verified_work_identity(
                cur,
                "local-execution-lineage",
                f"{scope}:{authority.effect_lineage_id}",
            )
            existing = await self._load_local_execution_attempt_row(
                cur,
                authority.attempt_id,
                for_update=True,
            )
            task = (
                None
                if existing is not None
                else await self._load_task_locked(cur, authority.task_id)
            )
            if task is not None:
                # PostgreSQL statements retain their original READ COMMITTED
                # snapshot after a row-lock wait.  Share one transaction-scoped
                # fence with lease reclamation and task claiming, then inspect
                # retry-series ownership again before publishing the attempt.
                await self._lock_local_execution_retry_fence(cur, task)
                await self._require_no_competing_local_execution_task_owner(cur, task)
            prior = await self._latest_local_execution_attempt_row(cur, authority)
            now = await self._database_now(cur)
            record = prepare_local_execution_attempt_record(
                authority=authority,
                task=task,
                existing=existing,
                prior=prior,
                evidence_now=now,
                lease_now=now,
            )
            if existing is None:
                await self._store_local_execution_attempt_row(cur, record, insert=True)
            return record.model_copy(deep=True)

        return await self._run_verified_work_mutation(mutation)

    async def start_local_execution_attempt(
        self,
        start: LocalExecutionAttemptStart,
    ) -> LocalExecutionAttemptRecord:
        start = _copy_local_execution_attempt_start(start)
        await self._ensure_ready()

        async def mutation(_conn: Any, cur: Any) -> LocalExecutionAttemptRecord:
            await self._lock_verified_work_identity(
                cur,
                "local-execution-attempt",
                start.attempt_id,
            )
            record = await self._load_local_execution_attempt_row(
                cur,
                start.attempt_id,
                for_update=True,
            )
            if record is None:
                raise LocalExecutionAttemptConflict(
                    "Local execution start has no prepared attempt."
                )
            authority_task = None
            if record.start is None:
                authority_task = await self._load_task_locked(
                    cur,
                    record.authority.task_id,
                )
            now = await self._database_now(cur)
            if authority_task is not None:
                require_local_execution_task_authority(
                    authority_task,
                    record.authority,
                    now=now,
                )
            updated = advance_local_execution_attempt_start(
                record,
                start,
                evidence_now=now,
                lease_now=now,
            )
            if updated != record:
                await self._store_local_execution_attempt_row(cur, updated, insert=False)
            return updated.model_copy(deep=True)

        return await self._run_verified_work_mutation(mutation)

    async def settle_local_execution_attempt(
        self,
        settlement: LocalExecutionAttemptSettlement,
    ) -> LocalExecutionAttemptRecord:
        settlement = _copy_authenticated_local_execution_attempt_settlement(settlement)
        await self._ensure_ready()

        async def mutation(_conn: Any, cur: Any) -> LocalExecutionAttemptRecord:
            await self._lock_verified_work_identity(
                cur,
                "local-execution-attempt",
                settlement.attempt_id,
            )
            record = await self._load_local_execution_attempt_row(
                cur,
                settlement.attempt_id,
                for_update=True,
            )
            if record is None:
                raise LocalExecutionAttemptConflict(
                    "Local execution settlement has no prepared attempt."
                )
            now = await self._database_now(cur)
            updated = settle_local_execution_attempt_record(
                record,
                settlement,
                evidence_now=now,
                lease_now=now,
            )
            if updated != record:
                await self._store_local_execution_attempt_row(cur, updated, insert=False)
            return updated.model_copy(deep=True)

        return await self._run_verified_work_mutation(mutation)

    async def load_local_execution_attempt(
        self,
        attempt_id: str,
    ) -> LocalExecutionAttemptRecord | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            record = await self._load_local_execution_attempt_row(cur, attempt_id)
        return None if record is None else record.model_copy(deep=True)

    async def list_unsettled_local_execution_attempts(
        self,
        *,
        limit: int = 100,
        after: LocalExecutionAttemptListCursor | None = None,
    ) -> tuple[LocalExecutionAttemptRecord, ...]:
        limit = _validate_task_positive_int(limit, "limit")
        after = _copy_local_execution_attempt_list_cursor(after)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            predicate = "(phase <> %s OR quiescence IN (%s, %s))"
            parameters: list[Any] = [
                "terminal",
                "terminal_not_quiescent",
                "unavailable",
            ]
            if after is not None:
                predicate += " AND (created_at > %s OR (created_at = %s AND attempt_id > %s))"
                parameters.extend(
                    (
                        after.created_at,
                        after.created_at,
                        after.attempt_id,
                    )
                )
            parameters.append(limit)
            await cur.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts WHERE "
                f"{predicate} "
                "ORDER BY created_at ASC, attempt_id ASC LIMIT %s",
                parameters,
            )
            rows = await cur.fetchall()
            records = [await self._load_local_execution_attempt_row(cur, row[0]) for row in rows]
        return tuple(record.model_copy(deep=True) for record in records if record is not None)

    async def claim_local_execution_attempt_recovery(
        self,
        claim: LocalExecutionAttemptRecoveryClaim,
    ) -> LocalExecutionAttemptRecord:
        claim = _copy_local_execution_attempt_recovery_claim(claim)
        await self._ensure_ready()

        async def mutation(_conn: Any, cur: Any) -> LocalExecutionAttemptRecord:
            await self._lock_verified_work_identity(
                cur,
                "local-execution-attempt",
                claim.attempt_id,
            )
            record = await self._load_local_execution_attempt_row(
                cur,
                claim.attempt_id,
                for_update=True,
            )
            if record is None:
                raise LocalExecutionAttemptConflict(
                    "Local execution recovery attempt was not found."
                )
            task = await self._load_task_locked(cur, record.authority.task_id)
            now = await self._database_now(cur)
            require_local_execution_recovery_eligible(
                task,
                record,
                now=now,
            )
            updated = claim_local_execution_attempt_recovery_record(
                record,
                claim,
                evidence_now=now,
                lease_now=now,
            )
            if updated != record:
                await self._store_local_execution_attempt_row(cur, updated, insert=False)
            return updated.model_copy(deep=True)

        return await self._run_verified_work_mutation(mutation)

    @runtime_task_creation
    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        if request.schedule_policy is not None and not self.supports_task_scheduling:
            raise NotImplementedError("This store does not support managed task scheduling.")
        await self._ensure_ready()
        task = await self._insert_task(request, running=False)
        return task.model_copy(deep=True)

    @runtime_task_creation
    async def create_running_task(
        self,
        request: TaskCreate,
        *,
        session_invocation: SessionInvocationBinding,
    ) -> Task:
        request = copy_task_create(request)
        session_binding = _copy_required_session_binding(session_invocation)
        await self._ensure_ready()
        task = await self._insert_task(
            request,
            running=True,
            session_invocation=session_binding,
        )
        return task.model_copy(deep=True)

    async def _insert_task(
        self,
        request: TaskCreate,
        *,
        running: bool,
        session_invocation: SessionInvocationBinding | None = None,
    ) -> Task:
        task_id = request.task_id or str(uuid4())
        notification_sender_pid: int | None = None
        notification_sender_connection: Any | None = None

        async def operation(conn: Any, cur: Any) -> tuple[Task, bool]:
            nonlocal notification_sender_connection, notification_sender_pid
            await self._lock_verified_work_task(cur, task_id)
            from cayu.storage._postgres_task_graphs import require_unreserved_identity

            await require_unreserved_identity(cur, task_id)
            if request.schedule_policy is not None:
                existing = await self._load_task(cur, task_id)
                if existing is not None:
                    if (
                        existing.schedule is None
                        or existing.schedule.creation_sha256 != schedule_creation_digest(request)
                    ):
                        raise TaskScheduleConflict("Task creation identity has different content.")
                    return existing, False
            retry_started_at = await self._verified_evidence_now(cur)
            parent: TaskInvocationSnapshot | None = None
            if request.parent_task_id is not None:
                if request.parent_task_id == task_id:
                    raise ValueError("Task cannot be its own parent.")
                await cur.execute(
                    """
                    SELECT id, session_id, session_instance_id, invocation
                    FROM cayu_tasks
                    WHERE id = %s
                    FOR KEY SHARE
                    """,
                    (request.parent_task_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise ValueError(f"Parent task not found: {request.parent_task_id}")
                invocation_value = row[3]
                if isinstance(invocation_value, str):
                    invocation_value = json.loads(invocation_value)
                parent = TaskInvocationSnapshot(
                    id=row[0],
                    session_id=row[1],
                    session_instance_id=row[2],
                    invocation=TaskInvocation.model_validate(invocation_value),
                )
            if request.work_contract is not None:
                contract = await self._load_work_contract_row(
                    cur,
                    request.work_contract,
                    for_update=True,
                )
                verified_work_support.require_contract_reference(
                    contract,
                    request.work_contract,
                )
                if request.session_id is not None:
                    await self._ensure_session_authority(
                        cur,
                        request.session_id,
                        "contracted",
                    )
            if running:
                if session_invocation is None:
                    raise AssertionError(
                        "Running task insertion requires session invocation provenance."
                    )
                task = _running_task_from_create(
                    request,
                    task_id=task_id,
                    parent_task=parent,
                    session_invocation=session_invocation,
                    retry_started_at=retry_started_at,
                    supports_verified_work_contracts=True,
                )
            else:
                task = _task_from_create(
                    request,
                    task_id=task_id,
                    parent_task=parent,
                    retry_started_at=retry_started_at,
                    supports_verified_work_contracts=True,
                )
            await cur.execute(
                f"""
                INSERT INTO cayu_tasks ({pg_support.TASK_COLUMNS})
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                """,
                pg_support.task_insert_values(task),
            )
            await self._record_task_transition(cur, None, task)
            publish_admission_wakeup = not running and (
                task.available_at is None or task.available_at <= retry_started_at
            )
            if publish_admission_wakeup:
                notification_sender_connection = conn
                notification_sender_pid = self._register_task_admission_notification_sender(conn)
                await cur.execute(
                    "SELECT pg_notify(%s, %s)",
                    (_TASK_ADMISSION_NOTIFY_CHANNEL, ""),
                )
            return task, publish_admission_wakeup

        try:
            task, publish_admission_wakeup = await self._run_verified_work_mutation(operation)
        except BaseException as exc:
            if notification_sender_pid is not None:
                self._discard_task_admission_notification_sender(
                    notification_sender_pid,
                    notification_sender_connection,
                )
            if isinstance(exc, UniqueViolation):
                raise ValueError(f"Task already exists: {task_id}") from exc
            raise
        if publish_admission_wakeup:
            self._publish_task_admission_wakeup(
                task,
                now=task.available_at or task.created_at,
            )
        return task

    async def _record_task_transition(
        self,
        cur: Any,
        prior: Task | None,
        current: Task,
        *,
        operation_id: str | None = None,
        settled_execution: tuple[str, str, datetime] | None = None,
    ) -> None:
        from cayu.storage._postgres_task_graphs import record_transition

        await record_transition(self, cur, prior, current, settled_execution=settled_execution)
        await self._record_schedule_transition(cur, prior, current, operation_id=operation_id)

    async def _record_schedule_transition(
        self, cur: Any, prior: Task | None, current: Task, *, operation_id: str | None = None
    ) -> None:
        """Append evidence while the caller owns the task's native transaction lock."""
        if current.schedule is None:
            return
        await cur.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM cayu_task_schedule_events WHERE task_id = %s",
            (current.id,),
        )
        row = await cur.fetchone()
        events = schedule_transition_events(
            prior, current, first_sequence=row[0] + 1, operation_id=operation_id
        )
        for event in events:
            await cur.execute(
                "INSERT INTO cayu_task_schedule_events (task_id, sequence, event_json) "
                "VALUES (%s, %s, %s)",
                (event.task_id, event.sequence, json.dumps(event.model_dump(mode="json"))),
            )

    async def reschedule_task(self, request: TaskRescheduleRequest) -> TaskScheduleReceipt:
        if type(request) is not TaskRescheduleRequest:
            raise TypeError("A typed task reschedule request is required.")
        request = revalidate_model_input(request, TaskRescheduleRequest)
        digest = schedule_mutation_digest(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> TaskScheduleReceipt:
            del conn
            await self._lock_verified_work_task(cur, request.task_id)
            await cur.execute(
                "SELECT receipt_json FROM cayu_task_schedule_receipts "
                "WHERE task_id = %s AND operation_id = %s",
                (request.task_id, request.operation_id),
            )
            row = await cur.fetchone()
            if row is not None:
                retained = TaskScheduleReceipt.model_validate(
                    json.loads(row[0]) if isinstance(row[0], str) else row[0]
                )
                if retained.request_sha256 != digest:
                    raise TaskScheduleConflict("Schedule operation identity has different content.")
                return retained
            current = await self._load_task_locked(cur, request.task_id)
            await cur.execute(
                "SELECT 1 FROM cayu_local_execution_attempts "
                "WHERE task_id = %s AND retry_admissible = FALSE LIMIT 1",
                (current.id,),
            )
            if await cur.fetchone() is not None:
                raise TaskScheduleConflict("Task has unsettled execution authority.")
            now = await self._database_now(cur)
            updated = rescheduled_task(current, request, now=now)
            receipt = schedule_receipt(
                updated, request, now=now, kind=TaskScheduleEventType.RESCHEDULED
            )
            await self._update_task_snapshot(cur, updated)
            await self._record_task_transition(
                cur, current, updated, operation_id=request.operation_id
            )
            await cur.execute(
                "INSERT INTO cayu_task_schedule_receipts (task_id, operation_id, receipt_json) "
                "VALUES (%s, %s, %s)",
                (
                    request.task_id,
                    request.operation_id,
                    json.dumps(receipt.model_dump(mode="json")),
                ),
            )
            await cur.execute("SELECT pg_notify(%s, %s)", (_TASK_ADMISSION_NOTIFY_CHANNEL, ""))
            return receipt

        receipt = await self._run_verified_work_mutation(operation)
        self._publish_task_admission_broadcast()
        return receipt

    async def cancel_scheduled_task(
        self, request: TaskScheduleCancelRequest
    ) -> TaskScheduleReceipt:
        if type(request) is not TaskScheduleCancelRequest:
            raise TypeError("A typed task schedule cancellation is required.")
        request = revalidate_model_input(request, TaskScheduleCancelRequest)
        digest = schedule_mutation_digest(request)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> TaskScheduleReceipt:
            del conn
            await self._lock_verified_work_task(cur, request.task_id)
            await cur.execute(
                "SELECT receipt_json FROM cayu_task_schedule_receipts "
                "WHERE task_id = %s AND operation_id = %s",
                (request.task_id, request.operation_id),
            )
            row = await cur.fetchone()
            if row is not None:
                retained = TaskScheduleReceipt.model_validate(
                    json.loads(row[0]) if isinstance(row[0], str) else row[0]
                )
                if retained.request_sha256 != digest:
                    raise TaskScheduleConflict("Schedule operation identity has different content.")
                return retained
            prior = await self._load_task_locked(cur, request.task_id)
            state = require_schedule_mutation(prior, request.expected_revision)
            updated = await self._finish_task_in_transaction(
                cur, prior.id, TaskStatus.CANCELLED, result=None, error=None
            )
            updated = updated.model_copy(
                update={
                    "schedule": state.model_copy(
                        update={"revision": schedule_revision_after(state)}
                    )
                }
            )
            await self._update_task_snapshot(cur, updated)
            if updated.retry_series is not None and updated.status is TaskStatus.CANCELLED:
                assert updated.status_payload is not None
                settlement_key = updated.status_payload["settlement_idempotency_key"]
                await cur.execute(
                    "SELECT receipt_json FROM cayu_task_retry_settlements "
                    "WHERE task_id = %s AND idempotency_key = %s",
                    (updated.id, settlement_key),
                )
                row = await cur.fetchone()
                if row is None:
                    raise TaskScheduleConflict("Retry cancellation has no settlement evidence.")
                settled = TaskRetrySettlementResult.model_validate(
                    json.loads(row[0]) if isinstance(row[0], str) else row[0]
                )
                if settled.task.model_copy(update={"schedule": updated.schedule}) != updated:
                    raise TaskScheduleConflict("Retry settlement has contradictory authority.")
                settled = settled.model_copy(update={"task": updated})
                await cur.execute(
                    "UPDATE cayu_task_retry_settlements SET receipt_json = %s "
                    "WHERE task_id = %s AND idempotency_key = %s",
                    (json.dumps(settled.model_dump(mode="json")), updated.id, settlement_key),
                )
            receipt = schedule_receipt(
                updated,
                request,
                now=updated.updated_at,
                kind=TaskScheduleEventType.CANCELLED
                if updated.status is TaskStatus.CANCELLED
                else TaskScheduleEventType.CANCELLATION_REQUESTED,
            )
            await self._record_task_transition(
                cur, prior, updated, operation_id=request.operation_id
            )
            await cur.execute(
                "INSERT INTO cayu_task_schedule_receipts (task_id, operation_id, receipt_json) "
                "VALUES (%s, %s, %s)",
                (
                    request.task_id,
                    request.operation_id,
                    json.dumps(receipt.model_dump(mode="json")),
                ),
            )
            return receipt

        return await self._run_verified_work_mutation(operation)

    async def list_task_schedule_events(
        self, task_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskScheduleEvent]:
        task_id = require_clean_nonblank(task_id, "task_id")
        if type(after_sequence) is not int or not 0 <= after_sequence <= 9007199254740991:
            raise ValueError("after_sequence must be a bounded nonnegative integer.")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Schedule event limit must be between 1 and 1000.")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT event_json FROM cayu_task_schedule_events "
                "WHERE task_id = %s AND sequence > %s ORDER BY sequence LIMIT %s",
                (task_id, after_sequence, limit),
            )
            return [
                TaskScheduleEvent.model_validate(
                    json.loads(row[0]) if isinstance(row[0], str) else row[0]
                )
                for row in await cur.fetchall()
            ]

    async def next_task_schedule_wakeup(self, query: TaskQuery | None = None) -> TaskScheduleWakeup:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            now = self._clock() if self._clock_is_injected else await self._database_now(cur)
            if query.status is not None and query.status is not TaskStatus.PENDING:
                return TaskScheduleWakeup(as_of=now)
            clauses, params = self._task_filter_clauses(query)
            scope = " AND ".join(
                [
                    "session_id IS NULL",
                    "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                    "WHERE NOT attempt.retry_admissible AND (attempt.task_id = cayu_tasks.id OR "
                    "(cayu_tasks.retry_series IS NOT NULL AND attempt.retry_series_id = "
                    "cayu_tasks.retry_series->>'series_id')))",
                    *clauses,
                ]
            )
            await cur.execute(
                cast(
                    "LiteralString",
                    f"SELECT MIN(available_at) FROM cayu_tasks WHERE {scope} "
                    "AND status = 'pending' AND available_at > %s "
                    "AND (schedule IS NULL OR schedule->>'admitted_at' IS NULL) "
                    "AND (retry_series IS NULL OR retry_series->>'elapsed_deadline' IS NULL "
                    "OR (retry_series->>'elapsed_deadline')::timestamptz > %s)",
                ),
                [*params, now, now],
            )
            due = (await cur.fetchone())[0]
            await cur.execute(
                cast(
                    "LiteralString",
                    "SELECT MIN(CASE WHEN (schedule->'policy'->>'expires_at')::timestamptz > %s "
                    "THEN (schedule->'policy'->>'expires_at')::timestamptz END), "
                    "COALESCE(BOOL_OR((schedule->'policy'->>'expires_at')::timestamptz <= %s "
                    "OR (schedule->'policy'->>'misfire_policy' = 'skip' AND "
                    "EXTRACT(EPOCH FROM (%s::timestamptz - available_at)) > "
                    "(schedule->'policy'->>'misfire_grace_seconds')::bigint)), FALSE) "
                    f"FROM cayu_tasks WHERE {scope} "
                    "AND status IN ('pending', 'paused', 'blocked', 'needs_attention') "
                    "AND schedule IS NOT NULL AND schedule->>'admitted_at' IS NULL",
                ),
                [now, now, now, *params],
            )
            expiry, maintenance = await cur.fetchone()
            return TaskScheduleWakeup(
                as_of=now,
                next_available_at=due,
                next_expiry_at=expiry,
                maintenance_required=maintenance,
            )

    async def load_task(self, task_id: str, *, _access_bounds=None) -> Task | None:
        if _access_bounds is None:
            from cayu.resource_access import current_data_bounds

            _access_bounds = await current_data_bounds("tasks")
        task_id = require_clean_nonblank(task_id, "task_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            task = await self._load_task(cur, task_id)
            if _access_bounds is not None:
                from cayu.tasks.access import require_read

                require_read(task, _access_bounds)
            return task

    async def load_active_attached_task_worker(
        self,
        task_id: str,
        worker_id: str,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        session_instance_id = require_clean_nonblank(
            session_instance_id,
            "session_instance_id",
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            task = await self._load_task(cur, task_id)
            now = await self._database_now(cur)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return _require_active_attached_task_worker(
            task,
            worker_id=worker_id,
            session_id=session_id,
            session_instance_id=session_instance_id,
            now=now,
        )

    async def load_direct_attached_task_resume(
        self,
        task_id: str,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        session_instance_id = require_clean_nonblank(
            session_instance_id,
            "session_instance_id",
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            task = await self._load_task(cur, task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return _require_direct_attached_task_resume(
            task,
            session_id=session_id,
            session_instance_id=session_instance_id,
        )

    async def load_invocation_snapshot(
        self,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT id, session_id, session_instance_id, invocation "
                "FROM cayu_tasks WHERE id = %s",
                (task_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            invocation_value = row[3]
            if isinstance(invocation_value, str):
                invocation_value = json.loads(invocation_value)
            return TaskInvocationSnapshot(
                id=row[0],
                session_id=row[1],
                session_instance_id=row[2],
                invocation=TaskInvocation.model_validate(invocation_value),
            )

    async def list_tasks(
        self, query: TaskQuery | None = None, *, _access_bounds=None
    ) -> list[Task]:
        if _access_bounds is None:
            from cayu.resource_access import current_data_bounds

            _access_bounds = await current_data_bounds("tasks")
        query = copy_task_query(query)
        clauses: list[str] = []
        params: list[object] = []
        if _access_bounds is not None:
            from cayu.tasks.access import sql_predicate

            access_sql, access_params = sql_predicate(_access_bounds, postgres=True)
            clauses.append(access_sql)
            params.extend(access_params)

        if query.q is not None:
            like = _ilike_contains_pattern(query.q)
            clauses.append(
                """
                (
                    id ILIKE %s ESCAPE '\\'
                    OR type ILIKE %s ESCAPE '\\'
                    OR title ILIKE %s ESCAPE '\\'
                    OR description ILIKE %s ESCAPE '\\'
                    OR status ILIKE %s ESCAPE '\\'
                    OR session_id ILIKE %s ESCAPE '\\'
                    OR parent_task_id ILIKE %s ESCAPE '\\'
                    OR assigned_agent_name ILIKE %s ESCAPE '\\'
                    OR worker_id ILIKE %s ESCAPE '\\'
                    OR status_reason ILIKE %s ESCAPE '\\'
                )
                """
            )
            params.extend([like] * 10)
        if query.status is not None:
            clauses.append("status = %s")
            params.append(str(query.status))
        if query.type is not None:
            clauses.append("type = %s")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = %s")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = %s")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = %s")
            params.append(query.assigned_agent_name)

        if query.has_work_contract is not None:
            clauses.append(
                "work_contract IS NOT NULL" if query.has_work_contract else "work_contract IS NULL"
            )
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = pg_support.task_order_sql(query.order_by)
        params.extend([query.limit, query.offset])

        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            # Interpolations are trusted: TASK_COLUMNS is a constant, order_sql is an
            # enum-derived literal, where_sql is hard-coded clauses; values bind via %s.
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT {pg_support.TASK_COLUMNS}
                    FROM cayu_tasks
                    {where_sql}
                    ORDER BY {order_sql}, id ASC
                    LIMIT %s OFFSET %s
                    """,
                ),
                params,
            )
            rows = await cur.fetchall()
            return [pg_support.task_from_row(row) for row in rows]

    async def load_session_closure_claim(self, session_id: str) -> TaskSessionClosureClaim | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT plan_id, claim_json FROM cayu_task_session_closure_claims "
                "WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            claim = TaskSessionClosureClaim.model_validate(row[1])
            if claim.session_id != session_id or claim.plan_id != row[0]:
                raise ValueError("Task closure claim conflicts with its retained authority.")
            return claim

    async def claim_session_closure(
        self, claim: TaskSessionClosureClaim
    ) -> TaskSessionClosureClaim:
        claim = copy_task_session_closure_claim(claim)
        await self._ensure_ready()
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                from cayu.storage._postgres_task_graphs import (
                    lock_task_graphs,
                    require_deletion_ready,
                )

                await lock_task_graphs(cur, claim.task_ids)
                await require_deletion_ready(cur, claim.task_ids)
                # Writers acquire a shared lock in their INSERT/UPDATE trigger;
                # closure exclusively excludes all of them before inspecting.
                # Do not take task row locks here: a writer can already own one
                # while waiting in that trigger. The claim excludes its commit.
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    ("cayu-task-session-closure:" + claim.session_id,),
                )
                await cur.execute(
                    "SELECT plan_id, claim_json FROM cayu_task_session_closure_claims "
                    "WHERE session_id = %s",
                    (claim.session_id,),
                )
                row = await cur.fetchone()
                if row is not None:
                    existing = TaskSessionClosureClaim.model_validate(row[1])
                    if existing != claim or row[0] != claim.plan_id:
                        raise ValueError(
                            "Task closure claim conflicts with its retained authority."
                        )
                    return existing
                await cur.execute(
                    "SELECT id, status, worker_id, lease_expires_at FROM cayu_tasks "
                    "WHERE session_id = %s LIMIT %s",
                    (claim.session_id, len(claim.task_ids) + 1),
                )
                rows = await cur.fetchall()
                if {row[0] for row in rows} != set(claim.task_ids):
                    raise ValueError("Task closure set changed before admission.")
                if any(
                    row[1] not in {"completed", "failed", "cancelled", "dependency_skipped"}
                    or row[2] is not None
                    or row[3] is not None
                    for row in rows
                ):
                    raise ValueError("Task closure requires quiescent terminal tasks.")
                await cur.execute(
                    "INSERT INTO cayu_task_session_closure_claims "
                    "(session_id, plan_id, claim_json) VALUES (%s, %s, %s)",
                    (claim.session_id, claim.plan_id, Jsonb(claim.model_dump(mode="json"))),
                )
            await conn.commit()
        return claim

    async def delete_session_tasks(
        self,
        session_id: str,
        *,
        task_ids: tuple[str, ...],
        policy: Any,
    ) -> None:
        """Delete a bounded, quiescent session task graph transactionally."""
        from cayu.storage._postgres_task_graphs import lock_task_graphs, require_deletion_ready
        from cayu.storage._session_closure_sql import (
            TASK_CLOSURE_DEPENDENCIES,
            task_closure_deletion_steps,
        )

        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await lock_task_graphs(cur, task_ids)
                await require_deletion_ready(cur, task_ids)
                await cur.execute(
                    "SELECT plan_id, claim_json FROM cayu_task_session_closure_claims "
                    "WHERE session_id = %s",
                    (session_id,),
                )
                claim_row = await cur.fetchone()
                claim = None
                if claim_row is not None:
                    claim = TaskSessionClosureClaim.model_validate(claim_row[1])
                    if (
                        claim.session_id != session_id
                        or claim.plan_id != claim_row[0]
                        or set(task_ids) != set(claim.task_ids)
                    ):
                        raise ValueError("Task deletion conflicts with the retained closure set.")
                if not task_ids:
                    return
                await cur.execute(
                    "SELECT id, status, worker_id, lease_expires_at, session_id "
                    "FROM cayu_tasks WHERE id = ANY(%s) "
                    "FOR UPDATE",
                    (list(task_ids),),
                )
                rows = await cur.fetchall()
                if (claim is None and {row[0] for row in rows} != set(task_ids)) or any(
                    row[4] != session_id for row in rows
                ):
                    raise ValueError("Task closure authority changed during deletion.")
                if any(
                    row[1]
                    not in {
                        TaskStatus.COMPLETED.value,
                        TaskStatus.FAILED.value,
                        TaskStatus.CANCELLED.value,
                        TaskStatus.DEPENDENCY_SKIPPED.value,
                    }
                    or row[2] is not None
                    or row[3] is not None
                    for row in rows
                ):
                    raise ValueError("Task closure requires quiescent terminal tasks.")
                await cur.execute(
                    """
                    SELECT DISTINCT kcu.table_name, kcu.column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON tc.constraint_name = ccu.constraint_name
                     AND tc.constraint_schema = ccu.constraint_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = current_schema()
                      AND ccu.table_name = 'cayu_tasks'
                      AND ccu.column_name = 'id'
                    """
                )
                discovered = {
                    (table, column)
                    for table, column in await cur.fetchall()
                    if (table, column) in TASK_CLOSURE_DEPENDENCIES
                }
                for table, column, parent in task_closure_deletion_steps(discovered):
                    if parent is None:
                        statement = sql.SQL("DELETE FROM {} WHERE {} = ANY(%s)").format(
                            sql.Identifier(table), sql.Identifier(column)
                        )
                    else:
                        statement = sql.SQL(
                            "DELETE FROM {} WHERE {} IN (SELECT {} FROM {} WHERE task_id = ANY(%s))"
                        ).format(
                            sql.Identifier(table),
                            sql.Identifier(column),
                            sql.Identifier(column),
                            sql.Identifier(parent),
                        )
                    await cur.execute(statement, (list(task_ids),))
                await cur.execute(
                    "DELETE FROM cayu_tasks WHERE session_id = %s AND id = ANY(%s)",
                    (session_id, list(task_ids)),
                )
            await conn.commit()

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology queries must be TaskTopologyQuery instances.")
        query = TaskTopologyQuery.model_validate(query.model_dump(mode="python"))
        session_branch_limits, child_branch_limits = _allocate_task_topology_branch_limits(query)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    await cur.execute("SELECT transaction_timestamp()")
                    observed_row = await cur.fetchone()
                    if observed_row is None:
                        raise RuntimeError("Postgres did not return a topology snapshot timestamp.")

                    expanded_parents: list[TaskTopologyNode] = []
                    if query.expanded_parent_ids:
                        await cur.execute(
                            cast(
                                "LiteralString",
                                f"""
                                SELECT {pg_support.TASK_TOPOLOGY_COLUMNS}
                                FROM cayu_tasks
                                WHERE id = ANY(%s)
                                """,
                            ),
                            (list(query.expanded_parent_ids),),
                        )
                        parents_by_id = {
                            row[0]: pg_support.task_topology_node_from_row(row)
                            for row in await cur.fetchall()
                        }
                        for parent_id in query.expanded_parent_ids:
                            parent = parents_by_id.get(parent_id)
                            if parent is None:
                                raise KeyError(f"Task not found: {parent_id}")
                            expanded_parents.append(parent)

                    async def read_branch_candidates(
                        *,
                        branch_ids: tuple[str, ...],
                        cursors: dict[str, str],
                        scope_kind: Literal["session", "parent_task"],
                        scope_column: Literal["session_id", "parent_task_id"],
                        branch_limits: tuple[int, ...],
                    ) -> list[list[TaskTopologyNode]]:
                        candidates: list[list[TaskTopologyNode]] = [[] for _ in branch_ids]
                        if not branch_ids:
                            return candidates
                        cursor_created_ats: list[datetime | None] = []
                        cursor_ids: list[str | None] = []
                        for branch_id in branch_ids:
                            cursor = cursors.get(branch_id)
                            if cursor is None:
                                cursor_created_ats.append(None)
                                cursor_ids.append(None)
                                continue
                            cursor_created_at, cursor_id = decode_task_topology_cursor(
                                cursor,
                                scope_kind=scope_kind,
                                scope_id=branch_id,
                            )
                            cursor_created_ats.append(cursor_created_at)
                            cursor_ids.append(cursor_id)
                        branch_sql: LiteralString = f"""
                                WITH requested_branches AS (
                                    SELECT branch_id, cursor_created_at, cursor_id,
                                           candidate_limit, branch_order
                                    FROM unnest(
                                        %s::text[],
                                        %s::timestamptz[],
                                        %s::text[],
                                        %s::integer[]
                                    ) WITH ORDINALITY AS requested(
                                        branch_id,
                                        cursor_created_at,
                                        cursor_id,
                                        candidate_limit,
                                        branch_order
                                    )
                                )
                                SELECT requested.branch_order, candidate.*
                                FROM requested_branches AS requested
                                CROSS JOIN LATERAL (
                                    SELECT {pg_support.TASK_TOPOLOGY_COLUMNS}
                                    FROM cayu_tasks
                                    WHERE cayu_tasks.{scope_column} = requested.branch_id
                                      AND (
                                          requested.cursor_created_at IS NULL
                                          OR cayu_tasks.created_at >
                                             requested.cursor_created_at
                                          OR (
                                              cayu_tasks.created_at =
                                                  requested.cursor_created_at
                                              AND cayu_tasks.id COLLATE "C" >
                                                  requested.cursor_id COLLATE "C"
                                          )
                                      )
                                    ORDER BY cayu_tasks.created_at ASC,
                                             cayu_tasks.id COLLATE "C" ASC
                                    LIMIT requested.candidate_limit
                                ) AS candidate
                                ORDER BY requested.branch_order ASC,
                                         candidate.topology_created_at ASC,
                                         candidate.topology_id COLLATE "C" ASC
                                """
                        await cur.execute(
                            branch_sql,
                            (
                                list(branch_ids),
                                cursor_created_ats,
                                cursor_ids,
                                [limit + 1 for limit in branch_limits],
                            ),
                        )
                        for row in await cur.fetchall():
                            branch_index = int(row[0]) - 1
                            candidates[branch_index].append(
                                pg_support.task_topology_node_from_row(row[1:])
                            )
                        return candidates

                    session_candidates = await read_branch_candidates(
                        branch_ids=query.linked_session_ids,
                        cursors=query.session_cursors,
                        scope_kind="session",
                        scope_column="session_id",
                        branch_limits=session_branch_limits,
                    )
                    child_candidates = await read_branch_candidates(
                        branch_ids=query.expanded_parent_ids,
                        cursors=query.child_cursors,
                        scope_kind="parent_task",
                        scope_column="parent_task_id",
                        branch_limits=child_branch_limits,
                    )

                    async def load_parent_links(
                        task_ids: tuple[str, ...],
                    ) -> dict[str, str | None]:
                        await cur.execute(
                            cast(
                                "LiteralString",
                                f"""
                                SELECT
                                    id,
                                    CASE
                                        WHEN parent_task_id IS NULL
                                          OR octet_length(parent_task_id)
                                             <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                        THEN parent_task_id
                                    END AS topology_parent_task_id,
                                    parent_task_id IS NOT NULL
                                      AND octet_length(parent_task_id)
                                          > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                        AS topology_parent_task_id_oversized
                                FROM cayu_tasks
                                WHERE id = ANY(%s)
                                """,
                            ),
                            (list(task_ids),),
                        )
                        links: dict[str, str | None] = {}
                        for task_id, parent_task_id, parent_id_oversized in await cur.fetchall():
                            if parent_id_oversized:
                                raise TaskTopologyInconsistent(
                                    "A task topology ancestor contains an oversized "
                                    "parent identifier."
                                )
                            links[task_id] = _bounded_optional_task_topology_parent_id(
                                parent_task_id
                            )
                        return links

                    await _validate_task_topology_ancestry(
                        (
                            *expanded_parents,
                            *(task for branch in session_candidates for task in branch),
                            *(task for branch in child_candidates for task in branch),
                        ),
                        load_parent_links,
                    )
                    result = build_task_topology_result(
                        observed_at=observed_row[0],
                        linked_session_ids=query.linked_session_ids,
                        session_branch_candidates=session_candidates,
                        session_branch_limits=session_branch_limits,
                        expanded_parents=expanded_parents,
                        child_branch_candidates=child_candidates,
                        child_branch_limits=child_branch_limits,
                        session_task_limit=query.session_task_limit,
                        child_limit=query.child_limit,
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return result

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        filters = copy_task_aggregate_filter(filters)
        query = task_query_from_aggregate_filter(filters)
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (
            ("type", query.type),
            ("session_id", query.session_id),
            ("parent_task_id", query.parent_task_id),
            ("assigned_agent_name", query.assigned_agent_name),
        ):
            if value is not None:
                clauses.append(f"{column} = %s")
                params.append(value)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    await cur.execute("SELECT transaction_timestamp()")
                    as_of_row = await cur.fetchone()
                    if as_of_row is None:
                        raise RuntimeError("Postgres did not return a snapshot timestamp.")
                    as_of = self._clock() if self._clock_is_injected else as_of_row[0]
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            WITH
                            matching_tasks AS (
                                SELECT id, status, session_id, available_at, retry_series
                                FROM cayu_tasks
                                {where_sql}
                            ),
                            status_counts AS (
                                SELECT status, COUNT(*) AS status_count
                                FROM matching_tasks
                                GROUP BY status
                            ),
                            pending_counts AS (
                                SELECT
                                    COUNT(*) FILTER (
                                        WHERE status = 'pending'
                                          AND session_id IS NULL
                                          AND (available_at IS NULL OR available_at <= %s)
                                          AND NOT EXISTS (
                                              SELECT 1
                                              FROM cayu_local_execution_attempts AS attempt
                                              WHERE NOT attempt.retry_admissible
                                                AND (
                                                    attempt.task_id = matching_tasks.id
                                                    OR (
                                                        matching_tasks.retry_series IS NOT NULL
                                                        AND attempt.retry_series_id =
                                                            matching_tasks.retry_series->>'series_id'
                                                    )
                                                )
                                          )
                                    ) AS claimable_pending_count,
                                    COUNT(*) FILTER (
                                        WHERE status = 'pending'
                                          AND available_at > %s
                                    ) AS scheduled_pending_count
                                FROM matching_tasks
                            )
                            SELECT
                                status_counts.status,
                                status_counts.status_count,
                                pending_counts.claimable_pending_count,
                                pending_counts.scheduled_pending_count
                            FROM pending_counts
                            LEFT JOIN status_counts ON TRUE
                            """,
                        ),
                        [*params, as_of, as_of],
                    )
                    rows = await cur.fetchall()
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        counts = {status: 0 for status in TaskStatus}
        for row in rows:
            if row[0] is not None:
                counts[TaskStatus(row[0])] = row[1]
        return TaskOperationalSnapshot(
            as_of=as_of,
            total_count=sum(counts.values()),
            counts_by_status=TaskStatusCounts.model_validate(counts),
            claimable_pending_count=rows[0][2],
            scheduled_pending_count=rows[0][3],
            accuracy=EXACT_AGGREGATE.model_copy(),
        )

    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
        session_invocation: SessionInvocationBinding | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        session_binding = _copy_optional_session_binding(session_invocation)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> Task:
            del conn
            await self._lock_verified_work_task(cur, task_id)
            task = await self._load_task_locked(cur, task_id)
            _ensure_retry_series_queue_attempt(task.retry_series)
            _ensure_can_transition(task, TaskStatus.RUNNING)
            effective_session_id = _task_session_id_for_start(
                task_id=task_id,
                stored_session_id=task.session_id,
                requested_session_id=session_id,
            )
            if task.work_contract is not None:
                await self._require_task_contract(cur, task, task.work_contract)
                if effective_session_id is None:
                    raise WorkCompletionConflict(
                        "Contracted tasks require a session binding before starting."
                    )
                await self._ensure_session_authority(
                    cur,
                    effective_session_id,
                    "contracted",
                )
            _task_invocation_for_attachment(
                task.invocation,
                session_id=effective_session_id,
                session_binding=session_binding,
            )
            session_instance_id = _task_session_instance_for_attachment(
                stored_session_instance_id=task.session_instance_id,
                session_id=effective_session_id,
                session_binding=session_binding,
            )
            now = await self._database_now(cur)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "session_id": effective_session_id,
                    "session_instance_id": session_instance_id,
                    "started_at": task.started_at or now,
                    "updated_at": now,
                }
            )
            await self._update_task_snapshot(cur, updated)
            await self._record_task_transition(cur, task, updated)
            return updated.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        session_invocation: SessionInvocationBinding,
        worker_id: str,
        lease_expires_at: datetime | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = (
            None
            if lease_expires_at is None
            else normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        )
        session_binding = _copy_required_session_binding(session_invocation)
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> Task:
            del conn
            await self._lock_verified_work_task(cur, task_id)
            task = await self._load_task_locked(cur, task_id)
            _ensure_retry_series_queue_attempt(task.retry_series)
            if task.work_contract is not None:
                await self._require_task_contract(cur, task, task.work_contract)
                await self._ensure_session_authority(cur, session_id, "contracted")
            _task_invocation_for_attachment(
                task.invocation,
                session_id=session_id,
                session_binding=session_binding,
            )
            session_instance_id = _task_session_instance_for_attachment(
                stored_session_instance_id=task.session_instance_id,
                session_id=session_id,
                session_binding=session_binding,
            )
            # Contract and session-authority lookups may wait behind another
            # transaction.  Sample ownership time only after those locks so an
            # expired queue worker cannot attach using a stale pre-wait value.
            now = await self._database_now(cur)
            if not _can_attach_claimed_task_state(
                status=task.status,
                session_id=task.session_id,
                worker_id=task.worker_id,
                lease_expires_at=task.lease_expires_at,
                expected_worker_id=worker_id,
                now=now,
            ):
                await self._raise_task_claim_attach_error(
                    cur,
                    task_id,
                    worker_id,
                    now=now,
                )
            if expected_lease is None:
                raise TaskClaimLost("Task attachment requires its exact worker lease.")
            _ensure_exact_owned_active_task_lease(
                task,
                worker_id,
                expected_lease,
                now=now,
            )
            updated = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "session_id": session_id,
                    "session_instance_id": session_instance_id,
                    "started_at": task.started_at or now,
                    "updated_at": now,
                }
            )
            await self._update_task_snapshot(cur, updated)
            await self._record_task_transition(cur, task, updated)
            return updated.model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any],
        *,
        worker_id: str | None = None,
        lease_expires_at: datetime | None = None,
        handoff_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_durable_json_object(result, "result")
        if worker_id is not None and lease_expires_at is None:
            raise TaskClaimLost(
                "Worker-owned task terminalization requires its exact lease generation."
            )
        async with managed_task_lease_mutation(
            task_id=task_id,
            worker_id=worker_id,
            handoff_id=handoff_id,
            presented_lease_expires_at=lease_expires_at,
        ) as effective_lease:
            return await self._finish_task(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
                worker_id=worker_id,
                expected_lease_expires_at=effective_lease,
                handoff_id=handoff_id,
            )

    async def fail_task(
        self,
        task_id: str,
        error: dict[str, Any],
        *,
        worker_id: str | None = None,
        lease_expires_at: datetime | None = None,
        handoff_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_durable_json_object(error, "error")
        if worker_id is not None and lease_expires_at is None:
            raise TaskClaimLost(
                "Worker-owned task terminalization requires its exact lease generation."
            )
        async with managed_task_lease_mutation(
            task_id=task_id,
            worker_id=worker_id,
            handoff_id=handoff_id,
            presented_lease_expires_at=lease_expires_at,
        ) as effective_lease:
            return await self._finish_task(
                task_id,
                TaskStatus.FAILED,
                result=None,
                error=error,
                worker_id=worker_id,
                expected_lease_expires_at=effective_lease,
                handoff_id=handoff_id,
            )

    async def terminalize_task(self, request: TaskTerminalizationRequest) -> Task:
        request, request_sha256 = prepare_task_terminalization(request)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    task = await self._load_task_locked(cur, request.task_id)

                    await cur.execute(
                        "SELECT request_sha256, worker_id, terminal_kind, "
                        "task_json, committed_at "
                        "FROM cayu_task_terminalization_receipts "
                        "WHERE task_id = %s AND idempotency_key = %s",
                        (request.task_id, request.idempotency_key),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is not None:
                        receipt = _postgres_task_terminalization_receipt(
                            task_id=request.task_id,
                            idempotency_key=request.idempotency_key,
                            row=receipt_row,
                        )
                        replayed = _replay_task_terminalization_receipt(
                            request=request,
                            request_sha256=request_sha256,
                            receipt=receipt,
                            current_task=task,
                        )
                        await conn.commit()
                        return replayed

                    await cur.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                        (request.task_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise WorkAttemptExecutionClaimLost(
                            "Admitted work attempts cannot use ordinary terminalization."
                        )
                    if task.retry_series is not None:
                        raise ValueError(
                            "Retry-series tasks require settle_task_retry_attempt for "
                            "completion or failure."
                        )
                    if task.status in {
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                    }:
                        raise TaskTerminalizationConflict(
                            "Task is terminal without the matching terminalization receipt."
                        )
                    now = await self._database_now(cur)
                    _ensure_task_terminalization_lease_authority(task, request, now=now)
                    _ensure_task_handoff_authority(task, request.handoff_id)
                    _validate_ordinary_task_terminalization_against_cancellation(task, request)
                    status = TaskStatus(request.kind.value)
                    verified_work_support.require_contracted_completion_authority(
                        task,
                        status,
                    )
                    await cur.execute(
                        f"""
                        UPDATE cayu_tasks
                        SET status = %s,
                            status_reason = NULL,
                            status_payload = NULL,
                            result = %s,
                            error = %s,
                            worker_id = NULL,
                            lease_expires_at = NULL,
                            interrupted_handoff_id = NULL,
                            started_at = COALESCE(started_at, %s),
                            completed_at = %s,
                            updated_at = %s
                        WHERE id = %s
                          AND status IN (%s, %s)
                          AND worker_id = %s
                          AND interrupted_handoff_id IS NOT DISTINCT FROM %s
                          AND lease_expires_at IS NOT NULL
                          AND lease_expires_at > %s
                        RETURNING {pg_support.TASK_COLUMNS}
                        """,
                        (
                            str(status),
                            None if request.result is None else pg_support._dumps(request.result),
                            None if request.error is None else pg_support._dumps(request.error),
                            now,
                            now,
                            now,
                            request.task_id,
                            str(TaskStatus.CLAIMED),
                            str(TaskStatus.RUNNING),
                            request.worker_id,
                            request.handoff_id,
                            now,
                        ),
                    )
                    terminal_row = await cur.fetchone()
                    if terminal_row is None:
                        await self._raise_task_active_lease_error(
                            cur,
                            request.task_id,
                            request.worker_id,
                            now=now,
                        )
                    assert terminal_row is not None
                    terminal_task = pg_support.task_from_row(terminal_row)
                    await self._record_task_transition(cur, task, terminal_task)
                    await cur.execute(
                        "INSERT INTO cayu_task_terminalization_receipts "
                        "(task_id, idempotency_key, request_sha256, worker_id, "
                        "terminal_kind, task_json, committed_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            request.task_id,
                            request.idempotency_key,
                            request_sha256,
                            request.worker_id,
                            request.kind.value,
                            pg_support._dumps(terminal_task.model_dump(mode="json")),
                            now,
                        ),
                    )
                await conn.commit()
                return terminal_task.model_copy(deep=True)
            except BaseException:
                await conn.rollback()
                raise

    async def recover_attached_task_failure(
        self,
        request: TaskTerminalizationRequest,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        request, request_sha256 = prepare_task_terminalization(request)
        session_id = require_clean_nonblank(session_id, "session_id")
        session_instance_id = require_clean_nonblank(
            session_instance_id,
            "session_instance_id",
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    task = await self._load_task_locked(cur, request.task_id)
                    await cur.execute(
                        "SELECT request_sha256, worker_id, terminal_kind, "
                        "task_json, committed_at "
                        "FROM cayu_task_terminalization_receipts "
                        "WHERE task_id = %s AND idempotency_key = %s",
                        (request.task_id, request.idempotency_key),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is not None:
                        receipt = _postgres_task_terminalization_receipt(
                            task_id=request.task_id,
                            idempotency_key=request.idempotency_key,
                            row=receipt_row,
                        )
                        replayed = _replay_task_terminalization_receipt(
                            request=request,
                            request_sha256=request_sha256,
                            receipt=receipt,
                            current_task=task,
                        )
                        _ensure_recovered_attached_task_session(
                            replayed,
                            session_id=session_id,
                            session_instance_id=session_instance_id,
                        )
                        await conn.commit()
                        return replayed

                    await cur.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                        (request.task_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise WorkAttemptExecutionClaimLost(
                            "Admitted work attempts cannot use attached-task recovery "
                            "terminalization."
                        )
                    now = await self._database_now(cur)
                    _ensure_recovered_attached_task_failure_authority(
                        task,
                        request,
                        session_id=session_id,
                        session_instance_id=session_instance_id,
                        now=now,
                    )
                    verified_work_support.require_contracted_completion_authority(
                        task,
                        TaskStatus.FAILED,
                    )
                    await cur.execute(
                        f"""
                        UPDATE cayu_tasks
                        SET status = %s,
                            status_reason = NULL,
                            status_payload = NULL,
                            result = NULL,
                            error = %s,
                            worker_id = NULL,
                            lease_expires_at = NULL,
                            interrupted_handoff_id = NULL,
                            started_at = COALESCE(started_at, %s),
                            completed_at = %s,
                            updated_at = %s
                        WHERE id = %s
                          AND status = %s
                          AND session_id = %s
                          AND session_instance_id = %s
                          AND worker_id = %s
                          AND lease_expires_at = %s
                          AND lease_expires_at <= %s
                          AND interrupted_handoff_id IS NOT DISTINCT FROM %s
                        RETURNING {pg_support.TASK_COLUMNS}
                        """,
                        (
                            str(TaskStatus.FAILED),
                            pg_support._dumps(request.error),
                            now,
                            now,
                            now,
                            request.task_id,
                            str(TaskStatus.RUNNING),
                            session_id,
                            session_instance_id,
                            request.worker_id,
                            request.lease_expires_at,
                            now,
                            request.handoff_id,
                        ),
                    )
                    terminal_row = await cur.fetchone()
                    if terminal_row is None:
                        raise TaskClaimLost(
                            "Attached-task recovery lost its exact durable authority."
                        )
                    terminal_task = pg_support.task_from_row(terminal_row)
                    await self._record_task_transition(cur, task, terminal_task)
                    await cur.execute(
                        "INSERT INTO cayu_task_terminalization_receipts "
                        "(task_id, idempotency_key, request_sha256, worker_id, "
                        "terminal_kind, task_json, committed_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            request.task_id,
                            request.idempotency_key,
                            request_sha256,
                            request.worker_id,
                            request.kind.value,
                            pg_support._dumps(terminal_task.model_dump(mode="json")),
                            now,
                        ),
                    )
                await conn.commit()
                return terminal_task.model_copy(deep=True)
            except BaseException:
                await conn.rollback()
                raise

    async def load_task_terminalization_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskTerminalizationReceipt | None:
        task_id, idempotency_key = prepare_task_terminalization_receipt_lookup(
            task_id,
            idempotency_key,
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                "FROM cayu_task_terminalization_receipts "
                "WHERE task_id = %s AND idempotency_key = %s",
                (task_id, idempotency_key),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return _postgres_task_terminalization_receipt(
            task_id=task_id,
            idempotency_key=idempotency_key,
            row=row,
        )

    async def release_interrupted_task_worker(
        self,
        request: TaskInterruptedHandoffRequest,
    ) -> TaskInterruptedHandoffReceipt:
        return await self._settle_interrupted_task_handoff(
            request,
            recover_expired=False,
        )

    async def recover_interrupted_task_worker(
        self,
        request: TaskInterruptedHandoffRequest,
    ) -> TaskInterruptedHandoffReceipt:
        return await self._settle_interrupted_task_handoff(
            request,
            recover_expired=True,
        )

    async def _settle_interrupted_task_handoff(
        self,
        request: TaskInterruptedHandoffRequest,
        *,
        recover_expired: bool,
    ) -> TaskInterruptedHandoffReceipt:
        request, request_sha256 = prepare_interrupted_task_handoff(request)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT request_sha256, request_json, task_json, committed_at "
                        "FROM cayu_task_interrupted_handoff_receipts "
                        "WHERE task_id = %s AND handoff_id = %s",
                        (request.task_id, request.handoff_id),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is not None:
                        receipt = _postgres_interrupted_task_handoff_receipt(
                            task_id=request.task_id,
                            handoff_id=request.handoff_id,
                            row=receipt_row,
                        )
                        replayed = _replay_interrupted_task_handoff_receipt(
                            request=request,
                            request_sha256=request_sha256,
                            receipt=receipt,
                        )
                        await conn.commit()
                        return replayed

                    task = await self._load_task_locked(cur, request.task_id)
                    # A concurrent exact publisher can commit while this caller
                    # waits for the task row. Re-read after acquiring that lock
                    # so acknowledgement-loss replay converges instead of
                    # misclassifying the released task as conflicting authority.
                    await cur.execute(
                        "SELECT request_sha256, request_json, task_json, committed_at "
                        "FROM cayu_task_interrupted_handoff_receipts "
                        "WHERE task_id = %s AND handoff_id = %s",
                        (request.task_id, request.handoff_id),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is not None:
                        receipt = _postgres_interrupted_task_handoff_receipt(
                            task_id=request.task_id,
                            handoff_id=request.handoff_id,
                            row=receipt_row,
                        )
                        replayed = _replay_interrupted_task_handoff_receipt(
                            request=request,
                            request_sha256=request_sha256,
                            receipt=receipt,
                        )
                        await conn.commit()
                        return replayed
                    await cur.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                        (request.task_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise TaskInterruptedHandoffConflict(
                            "Admitted work attempts do not use interrupted-task handoff release."
                        )
                    now = await self._database_now(cur)
                    _require_interrupted_task_handoff_authority(
                        task,
                        request,
                        now=now,
                        recover_expired=recover_expired,
                    )
                    lease_comparison = "<=" if recover_expired else ">"
                    await cur.execute(
                        f"""
                        UPDATE cayu_tasks
                        SET worker_id = NULL,
                            lease_expires_at = NULL,
                            interrupted_handoff_id = %s,
                            updated_at = %s
                        WHERE id = %s
                          AND status = %s
                          AND session_id = %s
                          AND session_instance_id = %s
                          AND worker_id = %s
                          AND lease_expires_at = %s
                          AND lease_expires_at {lease_comparison} %s
                        RETURNING {pg_support.TASK_COLUMNS}
                        """,
                        (
                            request.handoff_id,
                            now,
                            request.task_id,
                            str(TaskStatus.RUNNING),
                            request.session_id,
                            request.session_instance_id,
                            request.worker_id,
                            request.lease_expires_at,
                            now,
                        ),
                    )
                    released_row = await cur.fetchone()
                    if released_row is None:
                        raise TaskInterruptedHandoffConflict(
                            "Interrupted-task handoff lost its exact durable authority."
                        )
                    released = pg_support.task_from_row(released_row)
                    receipt = TaskInterruptedHandoffReceipt(
                        request=request,
                        request_sha256=request_sha256,
                        task=released,
                        committed_at=now,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_task_interrupted_handoff_receipts "
                        "(task_id, handoff_id, request_sha256, request_json, "
                        "task_json, committed_at) VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            request.task_id,
                            request.handoff_id,
                            request_sha256,
                            pg_support._dumps(request.model_dump(mode="json")),
                            pg_support._dumps(released.model_dump(mode="json")),
                            now,
                        ),
                    )
                await conn.commit()
                return receipt
            except BaseException:
                await conn.rollback()
                raise

    async def load_interrupted_task_handoff_receipt(
        self,
        task_id: str,
        handoff_id: str,
    ) -> TaskInterruptedHandoffReceipt | None:
        task_id, handoff_id = prepare_interrupted_task_handoff_receipt_lookup(
            task_id,
            handoff_id,
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT request_sha256, request_json, task_json, committed_at "
                "FROM cayu_task_interrupted_handoff_receipts "
                "WHERE task_id = %s AND handoff_id = %s",
                (task_id, handoff_id),
            )
            row = await cur.fetchone()
        return (
            None
            if row is None
            else _postgres_interrupted_task_handoff_receipt(
                task_id=task_id,
                handoff_id=handoff_id,
                row=row,
            )
        )

    async def list_expired_interrupted_task_handoff_candidates(
        self,
        *,
        after: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Task]:
        after, limit = prepare_interrupted_task_handoff_candidate_page(
            after=after,
            limit=limit,
        )
        after_clause = ""
        after_params: tuple[datetime, str] | tuple[()] = ()
        if after is not None:
            after_clause = "AND (lease_expires_at, id) > (%s, %s)"
            after_params = after
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            now = await self._database_now(cur)
            await cur.execute(
                f"""
                SELECT {pg_support.TASK_COLUMNS}
                FROM cayu_tasks
                WHERE status = %s
                  AND session_id IS NOT NULL
                  AND session_instance_id IS NOT NULL
                  AND worker_id IS NOT NULL
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= %s
                  AND status_reason IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM cayu_work_attempt_admissions
                      WHERE cayu_work_attempt_admissions.task_id = cayu_tasks.id
                  )
                  {after_clause}
                ORDER BY lease_expires_at ASC, id ASC
                LIMIT %s
                """,
                (str(TaskStatus.RUNNING), now, *after_params, limit),
            )
            rows = await cur.fetchall()
        return [pg_support.task_from_row(row) for row in rows]

    async def load_expired_interrupted_task_handoff_candidate(
        self,
        task_id: str,
    ) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                        SELECT {_TASK_RETURNING_COLUMNS}
                        FROM cayu_tasks AS task
                        WHERE task.id = %s
                          AND task.status = %s
                          AND task.session_id IS NOT NULL
                          AND task.session_instance_id IS NOT NULL
                          AND task.worker_id IS NOT NULL
                          AND task.lease_expires_at IS NOT NULL
                          AND task.lease_expires_at <= transaction_timestamp()
                          AND task.status_reason IS NULL
                          AND NOT EXISTS (
                              SELECT 1 FROM cayu_work_attempt_admissions
                              WHERE cayu_work_attempt_admissions.task_id = task.id
                          )
                    """,
                ),
                (task_id, str(TaskStatus.RUNNING)),
            )
            row = await cur.fetchone()
        return None if row is None else pg_support.task_from_row(row)

    async def claim_interrupted_task_continuation(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        handoff_id: str,
        task_id: str | None = None,
        lease_seconds: int = 300,
        after: tuple[datetime, str] | None = None,
        scan_limit: int = _TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    ) -> InterruptedTaskContinuationClaimPage:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        handoff_id = require_clean_nonblank(handoff_id, "handoff_id")
        if task_id is not None:
            task_id = require_clean_nonblank(task_id, "task_id")
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        after, scan_limit = prepare_interrupted_task_continuation_claim_page(
            after=after,
            limit=scan_limit,
        )
        await self._ensure_ready()
        connection_owner = _PostgresMutationConnectionOwner(
            self._pool,
            allowed_configure=self._postgres_mutation_allowed_configure,
        )
        return await self._await_owned_store_mutation(
            self._claim_interrupted_task_continuation_unowned(
                worker_id,
                query=query,
                handoff_id=handoff_id,
                task_id=task_id,
                lease_seconds=lease_seconds,
                after=after,
                scan_limit=scan_limit,
                connection_owner=connection_owner,
            ),
            connection_owner=connection_owner,
        )

    async def _claim_interrupted_task_continuation_unowned(
        self,
        worker_id: str,
        *,
        query: TaskQuery,
        handoff_id: str,
        task_id: str | None,
        lease_seconds: int,
        after: tuple[datetime, str] | None,
        scan_limit: int,
        connection_owner: _PostgresMutationConnectionOwner,
    ) -> InterruptedTaskContinuationClaimPage:
        handoff_id_sha256 = _interrupted_task_continuation_handoff_id_sha256(handoff_id)
        async with self._owned_store_connection(connection_owner) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (handoff_id_sha256,),
                )
                await cur.execute(
                    "SELECT task_id, worker_id "
                    "FROM cayu_task_interrupted_continuation_claims "
                    "WHERE handoff_id_sha256 = %s",
                    (handoff_id_sha256,),
                )
                prior_claim_row = await cur.fetchone()
                if prior_claim_row is not None:
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"SELECT {_TASK_RETURNING_COLUMNS} FROM cayu_tasks AS task "
                            "WHERE task.id = %s FOR UPDATE OF task",
                        ),
                        (prior_claim_row[0],),
                    )
                    existing_row = await cur.fetchone()
                    existing = (
                        None if existing_row is None else pg_support.task_from_row(existing_row)
                    )
                    now = await self._database_now(cur)
                    if (
                        prior_claim_row[1] != worker_id
                        or existing is None
                        or existing.interrupted_handoff_id != handoff_id
                        or existing.worker_id != worker_id
                        or existing.status is not TaskStatus.RUNNING
                        or existing.session_id is None
                        or existing.session_instance_id is None
                        or existing.lease_expires_at is None
                        or existing.lease_expires_at <= now
                        or (task_id is not None and existing.id != task_id)
                        or not _task_matches_claim_filter(existing, query)
                    ):
                        raise TaskClaimLost(
                            "Interrupted-task continuation claim generation is no longer live."
                        )
                    await conn.commit()
                    return InterruptedTaskContinuationClaimPage(
                        task=existing,
                        next_after=(existing.created_at, existing.id),
                        scanned_candidates=0,
                        rejected_candidates=0,
                        replayed=True,
                        exhausted=False,
                    )
                await cur.execute(
                    cast(
                        "LiteralString",
                        "SELECT 1 FROM cayu_tasks AS task "
                        "WHERE task.interrupted_handoff_id = %s LIMIT 1 FOR UPDATE OF task",
                    ),
                    (handoff_id,),
                )
                existing_row = await cur.fetchone()
                if existing_row is not None:
                    raise TaskClaimLost(
                        "Interrupted-task continuation claim generation is already in use."
                    )
                if query.status is not None and query.status is not TaskStatus.RUNNING:
                    await conn.commit()
                    return InterruptedTaskContinuationClaimPage(
                        scanned_candidates=0,
                        rejected_candidates=0,
                        exhausted=True,
                    )
                cursor = after
                after_sql = ""
                after_params: list[object] = []
                if cursor is not None:
                    after_sql = "AND (created_at, id) > (%s, %s)"
                    after_params = [cursor[0], cursor[1]]
                task_id_sql = "" if task_id is None else "AND task.id = %s"
                task_id_params: list[object] = [] if task_id is None else [task_id]
                await cur.execute(
                    f"""
                        SELECT {_TASK_RETURNING_COLUMNS}
                        FROM cayu_tasks AS task
                        WHERE status = %s
                          AND session_id IS NOT NULL
                          AND session_instance_id IS NOT NULL
                          AND status_reason IS NULL
                          AND worker_id IS NULL
                          AND lease_expires_at IS NULL
                          AND interrupted_handoff_id IS NOT NULL
                          {task_id_sql}
                          {after_sql}
                        ORDER BY created_at ASC, id ASC
                        FOR UPDATE OF task
                        LIMIT %s
                        """,
                    [
                        str(TaskStatus.RUNNING),
                        *task_id_params,
                        *after_params,
                        scan_limit,
                    ],
                )
                rows = await cur.fetchall()
                rejected = 0
                filtered = 0
                result: InterruptedTaskContinuationClaimPage | None = None
                last_observed: Task | None = None
                for index, row in enumerate(rows):
                    observed = pg_support.task_from_row(row)
                    last_observed = observed
                    if not _task_matches_claim_filter(observed, query):
                        filtered += 1
                        continue
                    candidate_handoff_id = observed.interrupted_handoff_id
                    if candidate_handoff_id is None:
                        raise AssertionError("Continuation candidate lost its handoff generation.")
                    await cur.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                        (observed.id,),
                    )
                    if await cur.fetchone() is not None:
                        rejected += 1
                        continue
                    await cur.execute(
                        "SELECT request_sha256, request_json, task_json, committed_at "
                        "FROM cayu_task_interrupted_handoff_receipts "
                        "WHERE task_id = %s AND handoff_id = %s",
                        (observed.id, candidate_handoff_id),
                    )
                    receipt_row = await cur.fetchone()
                    try:
                        receipt = (
                            None
                            if receipt_row is None
                            else _postgres_interrupted_task_handoff_receipt(
                                task_id=observed.id,
                                handoff_id=candidate_handoff_id,
                                row=receipt_row,
                            )
                        )
                    except TaskInterruptedHandoffConflict:
                        receipt = None
                    if receipt is None or receipt.task != observed:
                        rejected += 1
                        continue
                    await cur.execute(
                        "INSERT INTO cayu_task_interrupted_continuation_claims ("
                        "handoff_id_sha256, task_id, worker_id, claimed_at"
                        ") VALUES (%s, %s, %s, transaction_timestamp())",
                        (handoff_id_sha256, observed.id, worker_id),
                    )
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            UPDATE cayu_tasks AS task
                            SET worker_id = %s,
                                lease_expires_at = transaction_timestamp()
                                    + (%s * INTERVAL '1 second'),
                                interrupted_handoff_id = %s,
                                updated_at = transaction_timestamp()
                            WHERE task.id = %s
                              AND task.status = %s
                              AND task.worker_id IS NULL
                              AND task.lease_expires_at IS NULL
                              AND task.interrupted_handoff_id = %s
                            RETURNING {_TASK_RETURNING_COLUMNS}
                            """,
                        ),
                        (
                            worker_id,
                            lease_seconds,
                            handoff_id,
                            observed.id,
                            str(TaskStatus.RUNNING),
                            observed.interrupted_handoff_id,
                        ),
                    )
                    claimed_row = await cur.fetchone()
                    if claimed_row is None:
                        raise RuntimeError(
                            "PostgreSQL continuation claim lost its locked candidate."
                        )
                    claimed = pg_support.task_from_row(claimed_row)
                    result = InterruptedTaskContinuationClaimPage(
                        task=claimed,
                        next_after=(observed.created_at, observed.id),
                        scanned_candidates=index + 1,
                        rejected_candidates=rejected,
                        filtered_candidates=filtered,
                        exhausted=index == len(rows) - 1 and len(rows) < scan_limit,
                    )
                    break
                if result is None:
                    result = InterruptedTaskContinuationClaimPage(
                        next_after=(
                            (last_observed.created_at, last_observed.id)
                            if last_observed is not None
                            else None
                        ),
                        scanned_candidates=len(rows),
                        rejected_candidates=rejected,
                        filtered_candidates=filtered,
                        exhausted=len(rows) < scan_limit,
                    )
            await conn.commit()
        return result

    async def reconcile_task_cancellation(
        self,
        request: TaskCancellationReconciliationRequest,
    ) -> TaskCancellationReconciliationResult:
        request, request_sha256 = prepare_task_cancellation_reconciliation(request)
        from cayu.storage._postgres_task_graphs import lock_task_graphs

        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await lock_task_graphs(cur, (request.task_id,))
                    await cur.execute(
                        f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks "
                        "WHERE id = %s FOR UPDATE",
                        (request.task_id,),
                    )
                    task_row = await cur.fetchone()
                    task = None if task_row is None else pg_support.task_from_row(task_row)
                    await cur.execute(
                        "SELECT request_sha256, record_json "
                        "FROM cayu_task_retry_reconciliation_rejections "
                        "WHERE task_id = %s AND reconciliation_idempotency_key = %s",
                        (request.task_id, request.reconciliation_idempotency_key),
                    )
                    rejection_row = await cur.fetchone()
                    await cur.execute(
                        "SELECT request_sha256, worker_id, terminal_kind, "
                        "task_json, committed_at "
                        "FROM cayu_task_terminalization_receipts "
                        "WHERE task_id = %s AND idempotency_key = %s",
                        (request.task_id, request.cancellation_idempotency_key),
                    )
                    receipt_row = await cur.fetchone()
                    now = await self._database_now(cur)
                    if rejection_row is not None:
                        rejection = _TaskCancellationReconciliationRejectionRecord.model_validate(
                            pg_support._json_obj(rejection_row[1])
                        )
                        if rejection.request_sha256 != rejection_row[0]:
                            raise RuntimeError(
                                "Postgres cancellation reconciliation rejection contains "
                                "invalid durable material."
                            )
                        raise _replay_task_cancellation_reconciliation_rejection(
                            request,
                            request_sha256=request_sha256,
                            record=rejection,
                        )
                    if receipt_row is not None:
                        receipt = _postgres_task_terminalization_receipt(
                            task_id=request.task_id,
                            idempotency_key=request.cancellation_idempotency_key,
                            row=receipt_row,
                        )
                        replayed = _replay_task_cancellation_reconciliation(
                            request=request,
                            request_sha256=request_sha256,
                            receipt=receipt,
                            current_task=task,
                        )
                        await conn.commit()
                        return replayed
                    if task is None:
                        raise _task_cancellation_reconciliation_conflict(
                            request,
                            "Task cancellation reconciliation task was not found.",
                        )
                    await cur.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                        (request.task_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise WorkAttemptExecutionClaimLost(
                            "Admitted work attempts cannot use ordinary cancellation "
                            "reconciliation."
                        )
                    rejection = _task_cancellation_reconciliation_rejection_record(
                        request,
                        request_sha256=request_sha256,
                        recorded_at=now,
                    )
                    if rejection is not None:
                        _validated_task_cancellation(
                            task,
                            request,
                            now=now,
                            require_owner_lost=False,
                        )
                        await cur.execute(
                            "INSERT INTO cayu_task_retry_reconciliation_rejections "
                            "(task_id, reconciliation_idempotency_key, request_sha256, "
                            "record_json, recorded_at) VALUES (%s, %s, %s, %s, %s)",
                            (
                                rejection.task_id,
                                rejection.reconciliation_idempotency_key,
                                rejection.request_sha256,
                                pg_support._dumps(rejection.model_dump(mode="json")),
                                rejection.recorded_at,
                            ),
                        )
                        await conn.commit()
                        raise _rejected_task_cancellation_reconciliation(rejection)

                    result = _reconciled_task_cancellation(
                        task,
                        request,
                        request_sha256=request_sha256,
                        committed_at=now,
                    )
                    settled = result.task
                    await cur.execute(
                        f"""
                        UPDATE cayu_tasks
                        SET status = %s, status_reason = NULL, status_payload = %s,
                            result = NULL, error = %s, worker_id = NULL,
                            lease_expires_at = NULL, interrupted_handoff_id = NULL,
                            started_at = %s, completed_at = %s, updated_at = %s
                        WHERE id = %s AND status IN (%s, %s) AND status_reason = %s
                          AND worker_id = %s AND lease_expires_at = %s
                          AND lease_expires_at <= %s AND retry_series IS NULL
                        RETURNING {pg_support.TASK_COLUMNS}
                        """,
                        (
                            str(settled.status),
                            pg_support._dumps(settled.status_payload),
                            pg_support._dumps(settled.error),
                            settled.started_at,
                            settled.completed_at,
                            settled.updated_at,
                            request.task_id,
                            str(TaskStatus.CLAIMED),
                            str(TaskStatus.RUNNING),
                            request.expected_status_reason,
                            request.original_worker_id,
                            request.original_lease_expires_at,
                            now,
                        ),
                    )
                    durable_row = await cur.fetchone()
                    if durable_row is None:
                        raise _task_cancellation_reconciliation_conflict(
                            request,
                            "Task cancellation reconciliation lost its fenced transition.",
                        )
                    durable_task = pg_support.task_from_row(durable_row)
                    receipt = result.terminalization_receipt.model_copy(
                        # The corresponding journal entry is committed below
                        # with this exact returned task snapshot.
                        update={"task": durable_task},
                        deep=True,
                    )
                    durable_result = TaskCancellationReconciliationResult(
                        request_sha256=request_sha256,
                        task=durable_task,
                        terminalization_receipt=receipt,
                        reconciliation=result.reconciliation,
                        committed_at=now,
                    )
                    await self._record_task_transition(
                        cur,
                        task,
                        durable_task,
                        settled_execution=(task.id, task.worker_id, task.started_at)
                        if task.worker_id is not None and task.started_at is not None
                        else None,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_task_terminalization_receipts "
                        "(task_id, idempotency_key, request_sha256, worker_id, "
                        "terminal_kind, task_json, committed_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            receipt.task_id,
                            receipt.idempotency_key,
                            receipt.request_sha256,
                            receipt.worker_id,
                            receipt.kind.value,
                            pg_support._dumps(receipt.task.model_dump(mode="json")),
                            receipt.committed_at,
                        ),
                    )
                await conn.commit()
                return _copy_task_cancellation_reconciliation_result(durable_result)
            except BaseException:
                await conn.rollback()
                raise

    async def settle_task_retry_attempt(
        self,
        request: TaskRetrySettlementRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_settlement(request)
        await self._ensure_ready()
        notification_sender_pid: int | None = None
        notification_sender_connection: Any | None = None
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    task = await self._load_task_locked(cur, request.task_id)
                    await cur.execute(
                        "SELECT request_sha256, receipt_json "
                        "FROM cayu_task_retry_settlements "
                        "WHERE task_id = %s AND idempotency_key = %s",
                        (request.task_id, request.idempotency_key),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is not None:
                        receipt = TaskRetrySettlementResult.model_validate(
                            pg_support._json_obj(receipt_row[1])
                        )
                        replayed = _replay_task_retry_settlement(
                            request=request,
                            request_sha256=request_sha256,
                            receipt=receipt,
                            current_task=task,
                        )
                        await conn.commit()
                        return replayed

                    now = await self._database_now(cur)
                    series_now = self._clock() if self._clock_is_injected else now
                    if request.lease_expires_at is None:
                        raise TaskClaimLost(
                            "Task retry settlement requires its exact worker lease."
                        )
                    _ensure_exact_owned_active_task_lease(
                        task,
                        request.worker_id,
                        request.lease_expires_at,
                        now=now,
                    )
                    settled, successor = _settled_task_retry_attempt(
                        task,
                        request,
                        now=now,
                        series_now=series_now,
                    )
                    assert settled.retry_series is not None
                    await cur.execute(
                        f"""
                        UPDATE cayu_tasks
                        SET status = %s, status_reason = %s, status_payload = %s,
                            result = %s, error = %s, worker_id = NULL,
                            lease_expires_at = NULL, started_at = %s,
                            completed_at = %s, updated_at = %s, retry_series = %s
                        WHERE id = %s AND status IN (%s, %s) AND worker_id = %s
                          AND lease_expires_at IS NOT NULL AND lease_expires_at > %s
                        RETURNING {pg_support.TASK_COLUMNS}
                        """,
                        (
                            str(settled.status),
                            settled.status_reason,
                            pg_support._dumps(settled.status_payload),
                            None if settled.result is None else pg_support._dumps(settled.result),
                            None if settled.error is None else pg_support._dumps(settled.error),
                            settled.started_at,
                            settled.completed_at,
                            settled.updated_at,
                            pg_support._dumps(settled.retry_series.model_dump(mode="json")),
                            request.task_id,
                            str(TaskStatus.CLAIMED),
                            str(TaskStatus.RUNNING),
                            request.worker_id,
                            now,
                        ),
                    )
                    settled_row = await cur.fetchone()
                    if settled_row is None:
                        await self._raise_task_active_lease_error(
                            cur,
                            request.task_id,
                            request.worker_id,
                            now=now,
                        )
                    assert settled_row is not None
                    durable_task = pg_support.task_from_row(settled_row)
                    if successor is not None:
                        await cur.execute(
                            f"""
                            INSERT INTO cayu_tasks ({pg_support.TASK_COLUMNS})
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s
                            )
                            """,
                            pg_support.task_insert_values(successor),
                        )
                        from cayu.storage._postgres_task_graphs import register_retry_successor

                        await register_retry_successor(cur, durable_task, successor)
                        if successor.available_at is None or successor.available_at <= series_now:
                            notification_sender_connection = conn
                            notification_sender_pid = (
                                self._register_task_admission_notification_sender(conn)
                            )
                            await cur.execute(
                                "SELECT pg_notify(%s, %s)",
                                (_TASK_ADMISSION_NOTIFY_CHANNEL, ""),
                            )
                    await self._record_task_transition(cur, task, durable_task)
                    if successor is not None:
                        current_successor = await self._load_task(cur, successor.id)
                        assert current_successor is not None
                        await self._record_task_transition(cur, successor, current_successor)
                    receipt = TaskRetrySettlementResult(
                        task_id=request.task_id,
                        idempotency_key=request.idempotency_key,
                        request_sha256=request_sha256,
                        task=durable_task,
                        successor=successor,
                        events=_task_retry_events(durable_task, occurred_at=now),
                        committed_at=now,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_task_retry_settlements "
                        "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            request.task_id,
                            request.idempotency_key,
                            request_sha256,
                            pg_support._dumps(receipt.model_dump(mode="json")),
                            receipt.committed_at,
                        ),
                    )
                await conn.commit()
                committed = receipt.model_copy(deep=True)
            except BaseException:
                if notification_sender_pid is not None:
                    self._discard_task_admission_notification_sender(
                        notification_sender_pid,
                        notification_sender_connection,
                    )
                await conn.rollback()
                raise
        if successor is not None:
            self._publish_task_admission_wakeup(successor, now=series_now)
        return committed

    async def load_task_retry_settlement(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskRetrySettlementResult | None:
        task_id, idempotency_key = prepare_task_terminalization_receipt_lookup(
            task_id,
            idempotency_key,
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT receipt_json FROM cayu_task_retry_settlements "
                "WHERE task_id = %s AND idempotency_key = %s",
                (task_id, idempotency_key),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return TaskRetrySettlementResult.model_validate(pg_support._json_obj(row[0]))

    async def reconcile_task_retry_cancellation(
        self,
        request: TaskRetryCancellationReconciliationRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_cancellation_reconciliation(request)
        from cayu.storage._postgres_task_graphs import lock_task_graphs

        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await lock_task_graphs(cur, (request.task_id,))
                    await cur.execute(
                        f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks "
                        "WHERE id = %s FOR UPDATE",
                        (request.task_id,),
                    )
                    task_row = await cur.fetchone()
                    task = None if task_row is None else pg_support.task_from_row(task_row)
                    await cur.execute(
                        "SELECT request_sha256, record_json "
                        "FROM cayu_task_retry_reconciliation_rejections "
                        "WHERE task_id = %s AND reconciliation_idempotency_key = %s",
                        (request.task_id, request.reconciliation_idempotency_key),
                    )
                    rejection_row = await cur.fetchone()
                    await cur.execute(
                        "SELECT request_sha256, receipt_json "
                        "FROM cayu_task_retry_settlements "
                        "WHERE task_id = %s AND idempotency_key = %s",
                        (request.task_id, request.cancellation_idempotency_key),
                    )
                    receipt_row = await cur.fetchone()
                    now = await self._database_now(cur)
                    if rejection_row is not None:
                        rejection = (
                            _TaskRetryCancellationReconciliationRejectionRecord.model_validate(
                                pg_support._json_obj(rejection_row[1])
                            )
                        )
                        if rejection.request_sha256 != rejection_row[0]:
                            raise RuntimeError(
                                "Postgres retry reconciliation rejection contains invalid "
                                "durable material."
                            )
                        raise _replay_task_retry_cancellation_reconciliation_rejection(
                            request,
                            request_sha256=request_sha256,
                            record=rejection,
                        )
                    if receipt_row is not None:
                        receipt = TaskRetrySettlementResult.model_validate(
                            pg_support._json_obj(receipt_row[1])
                        )
                        replayed = _replay_task_retry_cancellation_reconciliation(
                            request=request,
                            request_sha256=request_sha256,
                            receipt=receipt,
                            current_task=task,
                        )
                        await conn.commit()
                        return replayed
                    if task is None:
                        raise _task_retry_cancellation_reconciliation_conflict(
                            request,
                            "Task retry cancellation reconciliation task was not found.",
                        )
                    rejection = _task_retry_cancellation_reconciliation_rejection_record(
                        request,
                        request_sha256=request_sha256,
                        recorded_at=now,
                    )
                    if rejection is not None:
                        _validated_task_retry_cancellation(
                            task,
                            request,
                            now=now,
                            require_owner_lost=False,
                        )
                        await cur.execute(
                            "INSERT INTO cayu_task_retry_reconciliation_rejections "
                            "(task_id, reconciliation_idempotency_key, request_sha256, "
                            "record_json, recorded_at) VALUES (%s, %s, %s, %s, %s)",
                            (
                                rejection.task_id,
                                rejection.reconciliation_idempotency_key,
                                rejection.request_sha256,
                                pg_support._dumps(rejection.model_dump(mode="json")),
                                rejection.recorded_at,
                            ),
                        )
                        await conn.commit()
                        raise _rejected_task_retry_cancellation_reconciliation(rejection)

                    receipt = _reconciled_task_retry_cancellation(
                        task,
                        request,
                        request_sha256=request_sha256,
                        committed_at=now,
                    )
                    settled = receipt.task
                    assert settled.retry_series is not None
                    await cur.execute(
                        f"""
                        UPDATE cayu_tasks
                        SET status = %s, status_reason = %s, status_payload = %s,
                            result = NULL, error = %s, worker_id = NULL,
                            lease_expires_at = NULL, started_at = %s,
                            completed_at = %s, updated_at = %s, retry_series = %s
                        WHERE id = %s AND status IN (%s, %s) AND status_reason = %s
                          AND worker_id = %s AND lease_expires_at = %s
                          AND lease_expires_at <= %s
                        RETURNING {pg_support.TASK_COLUMNS}
                        """,
                        (
                            str(settled.status),
                            settled.status_reason,
                            pg_support._dumps(settled.status_payload),
                            pg_support._dumps(settled.error),
                            settled.started_at,
                            settled.completed_at,
                            settled.updated_at,
                            pg_support._dumps(settled.retry_series.model_dump(mode="json")),
                            request.task_id,
                            str(TaskStatus.CLAIMED),
                            str(TaskStatus.RUNNING),
                            request.expected_status_reason,
                            request.original_worker_id,
                            request.original_lease_expires_at,
                            now,
                        ),
                    )
                    durable_row = await cur.fetchone()
                    if durable_row is None:
                        raise _task_retry_cancellation_reconciliation_conflict(
                            request,
                            "Task retry cancellation reconciliation lost its fenced transition.",
                        )
                    durable_task = pg_support.task_from_row(durable_row)
                    receipt = TaskRetrySettlementResult(
                        task_id=request.task_id,
                        idempotency_key=request.cancellation_idempotency_key,
                        request_sha256=request_sha256,
                        task=durable_task,
                        successor=None,
                        reconciliation=receipt.reconciliation,
                        events=_task_retry_events(durable_task, occurred_at=now),
                        committed_at=now,
                    )
                    await self._record_task_transition(
                        cur,
                        task,
                        durable_task,
                        settled_execution=(task.id, task.worker_id, task.started_at)
                        if task.worker_id is not None and task.started_at is not None
                        else None,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_task_retry_settlements "
                        "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            request.task_id,
                            request.cancellation_idempotency_key,
                            request_sha256,
                            pg_support._dumps(receipt.model_dump(mode="json")),
                            receipt.committed_at,
                        ),
                    )
                await conn.commit()
                return receipt.model_copy(deep=True)
            except BaseException:
                await conn.rollback()
                raise

    async def enforce_task_retry_deadline(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
        token_count: int = 0,
        estimated_cost: Decimal = Decimal(0),
    ) -> TaskRetrySettlementResult | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        token_count, estimated_cost = _validated_task_retry_terminal_accounting(
            token_count=token_count,
            estimated_cost=estimated_cost,
        )
        from cayu.storage._postgres_task_graphs import lock_task_graphs

        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await lock_task_graphs(cur, (task_id,))
                    await cur.execute(
                        f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks "
                        "WHERE id = %s FOR UPDATE",
                        (task_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise KeyError(f"Task not found: {task_id}")
                    await cur.execute("SELECT clock_timestamp()")
                    timestamp_row = await cur.fetchone()
                    if timestamp_row is None:
                        raise RuntimeError("Postgres did not return a current timestamp.")
                    lease_now = timestamp_row[0]
                    series_now = self._clock() if self._clock_is_injected else lease_now
                    task = pg_support.task_from_row(row)
                    _ensure_exact_owned_active_task_lease(
                        task,
                        worker_id,
                        expected_lease,
                        now=lease_now,
                    )
                    if not _claimed_task_retry_attempt_elapsed(task, series_now=series_now):
                        await conn.commit()
                        return None
                    receipt = _elapsed_claimed_task_retry_settlement(
                        task,
                        committed_at=lease_now,
                        token_count=token_count,
                        estimated_cost=estimated_cost,
                    )
                    settled = receipt.task
                    assert settled.retry_series is not None
                    await cur.execute(
                        f"""
                        UPDATE cayu_tasks
                        SET status = %s, status_reason = %s, status_payload = %s,
                            result = NULL, error = %s, worker_id = NULL,
                            lease_expires_at = NULL, started_at = %s,
                            completed_at = %s, updated_at = %s, retry_series = %s
                        WHERE id = %s AND status IN (%s, %s) AND worker_id = %s
                          AND lease_expires_at = %s AND lease_expires_at > %s
                        RETURNING {pg_support.TASK_COLUMNS}
                        """,
                        (
                            str(settled.status),
                            settled.status_reason,
                            pg_support._dumps(settled.status_payload),
                            pg_support._dumps(settled.error),
                            settled.started_at,
                            settled.completed_at,
                            settled.updated_at,
                            pg_support._dumps(settled.retry_series.model_dump(mode="json")),
                            task_id,
                            str(TaskStatus.CLAIMED),
                            str(TaskStatus.RUNNING),
                            worker_id,
                            expected_lease,
                            lease_now,
                        ),
                    )
                    if await cur.fetchone() is None:
                        await self._raise_task_active_lease_error(
                            cur,
                            task_id,
                            worker_id,
                            now=lease_now,
                        )
                    await self._record_task_transition(cur, task, settled)
                    await cur.execute(
                        "INSERT INTO cayu_task_retry_settlements "
                        "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            receipt.task_id,
                            receipt.idempotency_key,
                            receipt.request_sha256,
                            pg_support._dumps(receipt.model_dump(mode="json")),
                            receipt.committed_at,
                        ),
                    )
                await conn.commit()
                return receipt.model_copy(deep=True)
            except BaseException:
                await conn.rollback()
                raise

    async def task_retry_deadline_elapsed(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> bool:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks "
                        "WHERE id = %s FOR UPDATE",
                        (task_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise KeyError(f"Task not found: {task_id}")
                    await cur.execute("SELECT clock_timestamp()")
                    timestamp_row = await cur.fetchone()
                    if timestamp_row is None:
                        raise RuntimeError("Postgres did not return a current timestamp.")
                    lease_now = timestamp_row[0]
                    series_now = self._clock() if self._clock_is_injected else lease_now
                    task = pg_support.task_from_row(row)
                    _ensure_exact_owned_active_task_lease(
                        task,
                        worker_id,
                        expected_lease,
                        now=lease_now,
                    )
                    elapsed = _claimed_task_retry_attempt_elapsed(
                        task,
                        series_now=series_now,
                    )
                await conn.commit()
                return elapsed
            except BaseException:
                await conn.rollback()
                raise

    @runtime_task_mutation
    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        return await self._finish_task(
            task_id, TaskStatus.CANCELLED, result=None, error=copied_error
        )

    async def request_claimed_task_cancellation(
        self,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        return await self._finish_task(
            task_id,
            TaskStatus.CANCELLED,
            result=None,
            error=copied_error,
            worker_id=worker_id,
            expected_lease_expires_at=expected_lease,
            request_claimed_cancellation=True,
        )

    async def mark_claimed_task_execution_started(
        self,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> Task:
            del conn
            await self._lock_verified_work_task(cur, task_id)
            current = await self._load_task_locked(cur, task_id)
            now = await self._database_now(cur)
            if current.worker_id != worker_id or current.lease_expires_at != expected_lease:
                raise TaskClaimLost(
                    "Claimed-task execution no longer owns the expected worker lease."
                )
            _ensure_owned_active_task_lease(current, worker_id, now=now)
            if (
                current.status is not TaskStatus.CLAIMED
                or current.session_id is not None
                or _task_cancellation_requested(current)
                or current.status_reason == _TASK_RETRY_CANCELLATION_REQUESTED_REASON
            ):
                raise TaskTerminalizationConflict(
                    "Claimed task cannot begin ordinary worker execution."
                )
            if current.started_at is not None:
                return current.model_copy(deep=True)
            started = current.model_copy(update={"started_at": now, "updated_at": now})
            await self._update_task_snapshot(cur, started)
            await self._record_task_transition(cur, current, started)
            return (await self._require_task(cur, task_id)).model_copy(deep=True)

        return await self._run_verified_work_mutation(operation)

    @runtime_task_mutation
    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.PAUSED,
            reason=reason,
            payload=payload,
        )

    @runtime_task_mutation
    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.BLOCKED,
            reason=reason,
            payload=payload,
        )

    @runtime_task_mutation
    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.NEEDS_ATTENTION,
            reason=reason,
            payload=payload,
        )

    @runtime_task_mutation
    async def resume_task(self, task_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            async with conn.cursor() as cur:
                prior = await self._load_task_locked(cur, task_id)
                await cur.execute(
                    "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                    (task_id,),
                )
                if await cur.fetchone() is not None:
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts cannot use ordinary task resumption."
                    )
                now = await self._database_now(cur)
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status = %s,
                        status_reason = NULL,
                        status_payload = NULL,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = %s
                    WHERE id = %s
                      AND status IN (%s, %s, %s)
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        str(TaskStatus.PENDING),
                        now,
                        task_id,
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    task = await self._require_task(cur, task_id)
                    _ensure_can_resume_task(task)
                    raise ValueError(f"Task {task.id} cannot resume from {task.status}")
                updated = pg_support.task_from_row(row)
                await self._record_task_transition(cur, prior, updated)
                updated = await self._require_task(cur, task_id)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        # Finish lazy readiness before creating the bounded mutation owner so
        # a first-use schema wait cannot be mistaken for an already-dispatched
        # task claim or escape as a detached mutation owner.
        await self._ensure_ready()
        connection_owner = _PostgresMutationConnectionOwner(
            self._pool,
            allowed_configure=self._postgres_mutation_allowed_configure,
        )
        return await self._await_owned_store_mutation(
            self._claim_task_unowned(
                worker_id,
                query,
                lease_seconds=lease_seconds,
                connection_owner=connection_owner,
            ),
            connection_owner=connection_owner,
        )

    async def _settle_schedule_nonexecution(
        self, cur: Any, task: Task, *, eligibility: TaskScheduleEligibility, now: datetime
    ) -> None:
        settled, cancellation = _scheduled_task_nonexecution(task, eligibility=eligibility, now=now)
        await self._update_task_snapshot(cur, settled)
        await self._record_task_transition(cur, task, settled)
        if cancellation is not None:
            await cur.execute(
                "INSERT INTO cayu_task_retry_settlements "
                "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    cancellation.task_id,
                    cancellation.idempotency_key,
                    cancellation.request_sha256,
                    pg_support._dumps(cancellation.model_dump(mode="json")),
                    cancellation.committed_at,
                ),
            )

    def _held_schedule_scope(self, query: TaskQuery, *, now: datetime) -> tuple[str, list[Any]]:
        clauses, params = self._task_filter_clauses(query.model_copy(update={"status": None}))
        scope = " AND ".join(
            [
                "status IN ('paused', 'blocked', 'needs_attention', 'waiting_dependencies', 'waiting_group')",
                "session_id IS NULL",
                "worker_id IS NULL",
                "schedule IS NOT NULL AND schedule->>'admitted_at' IS NULL",
                "((schedule->'policy'->>'expires_at')::timestamptz <= %s OR "
                "(schedule->'policy'->>'misfire_policy' = 'skip' AND "
                "EXTRACT(EPOCH FROM (%s::timestamptz - available_at)) > "
                "(schedule->'policy'->>'misfire_grace_seconds')::bigint))",
                "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                "WHERE NOT attempt.retry_admissible AND (attempt.task_id = cayu_tasks.id OR "
                "(cayu_tasks.retry_series IS NOT NULL AND attempt.retry_series_id = "
                "cayu_tasks.retry_series->>'series_id')))",
                *clauses,
            ]
        )
        return scope, [now, now, *params]

    async def _expire_held_schedules(
        self, cur: Any, query: TaskQuery, *, now: datetime, candidate_ids: tuple[str, ...]
    ) -> None:
        scope, params = self._held_schedule_scope(query, now=now)
        await cur.execute(
            cast(
                "LiteralString",
                f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks WHERE {scope} AND id = ANY(%s) "
                "ORDER BY created_at ASC, id ASC FOR UPDATE SKIP LOCKED LIMIT 100",
            ),
            [*params, list(candidate_ids)],
        )
        tasks = [pg_support.task_from_row(row) for row in await cur.fetchall()]
        by_scope = {self._local_execution_retry_fence_scope(task): task for task in tasks}
        for scope_id in sorted(by_scope):
            await self._lock_local_execution_retry_fence(cur, by_scope[scope_id])
        for task in tasks:
            # Earlier source outcomes may already have terminalized descendants.
            task = await self._require_task(cur, task.id)
            if task.status not in {
                TaskStatus.PAUSED,
                TaskStatus.BLOCKED,
                TaskStatus.NEEDS_ATTENTION,
                TaskStatus.WAITING_DEPENDENCIES,
                TaskStatus.WAITING_GROUP,
            }:
                continue
            if await self._local_execution_attempt_fences_task(cur, task):
                continue
            assert task.schedule is not None and task.available_at is not None
            eligibility = task_schedule_eligibility(
                available_at=task.available_at, policy=task.schedule.policy, as_of=now
            )
            if eligibility in {TaskScheduleEligibility.EXPIRED, TaskScheduleEligibility.SKIPPED}:
                await self._settle_schedule_nonexecution(
                    cur, task, eligibility=eligibility, now=now
                )

    async def _claim_task_unowned(
        self,
        worker_id: str,
        query: TaskQuery,
        *,
        lease_seconds: int = 300,
        connection_owner: _PostgresMutationConnectionOwner,
    ) -> Task | None:
        clauses, params = self._task_filter_clauses(query)
        retry_worker_id_is_bounded = _task_retry_reconciliation_identity_is_bounded(worker_id)
        if self._clock_is_injected:
            series_now = self._clock()
            availability_clause = "(available_at IS NULL OR available_at <= %s)"
            availability_params: list[Any] = [series_now]
            retry_deadline_clause = (
                "(retry_series IS NULL "
                "OR retry_series->>'elapsed_deadline' IS NULL "
                "OR (retry_series->>'elapsed_deadline')::timestamptz > %s)"
            )
            retry_deadline_params: list[Any] = [series_now]
            expiration_deadline_clause = "(retry_series->>'elapsed_deadline')::timestamptz <= %s"
            expiration_deadline_params: list[Any] = [series_now]
        else:
            # Production eligibility and lease timestamps share PostgreSQL's
            # transaction clock. A skewed worker clock must never make a
            # future task claimable before the authoritative store says so.
            availability_clause = (
                "(available_at IS NULL OR available_at <= transaction_timestamp())"
            )
            availability_params = []
            retry_deadline_clause = (
                "(retry_series IS NULL "
                "OR retry_series->>'elapsed_deadline' IS NULL "
                "OR (retry_series->>'elapsed_deadline')::timestamptz "
                "> transaction_timestamp())"
            )
            retry_deadline_params = []
            expiration_deadline_clause = (
                "(retry_series->>'elapsed_deadline')::timestamptz <= transaction_timestamp()"
            )
            expiration_deadline_params = []
        if not retry_worker_id_is_bounded:
            retry_deadline_clause = "retry_series IS NULL"
            retry_deadline_params = []
        lease_expires_sql = "transaction_timestamp() + (%s * INTERVAL '1 second')"
        updated_at_sql = "transaction_timestamp()"
        mutation_params = [lease_seconds]
        where_sql = " AND ".join(
            [
                "status = %s",
                "session_id IS NULL",
                availability_clause,
                retry_deadline_clause,
                "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                "WHERE NOT attempt.retry_admissible AND ("
                "attempt.task_id = cayu_tasks.id OR (cayu_tasks.retry_series IS NOT NULL "
                "AND attempt.retry_series_id = cayu_tasks.retry_series->>'series_id')))",
                *clauses,
            ]
        )
        # Claiming is always FIFO by creation time, independent of the query's
        # display ordering, so the oldest pending task is dispatched first.
        order_sql = pg_support.task_order_sql(TaskOrder.CREATED_AT_ASC)
        expiration_scope = (
            "status = %s AND session_id IS NULL AND retry_series IS NOT NULL "
            "AND retry_series->>'disposition' = %s "
            "AND retry_series->>'elapsed_deadline' IS NOT NULL "
            f"AND {expiration_deadline_clause} AND NOT EXISTS ("
            "SELECT 1 FROM cayu_local_execution_attempts AS attempt "
            "WHERE NOT attempt.retry_admissible AND (attempt.task_id = cayu_tasks.id "
            "OR attempt.retry_series_id = retry_series->>'series_id'))"
        )
        async with self._owned_store_connection(connection_owner) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT transaction_timestamp()")
                transaction_timestamp_row = await cur.fetchone()
                if transaction_timestamp_row is None:
                    raise RuntimeError("Postgres did not return a transaction timestamp.")
                now = transaction_timestamp_row[0]
                authoritative_series_now = series_now if self._clock_is_injected else now
                from cayu.storage._postgres_task_graphs import (
                    TaskCandidateQuery,
                    select_graph_scopes,
                )

                held_scope, held_params = self._held_schedule_scope(
                    query, now=authoritative_series_now
                )
                held_ids, expired_ids, pending_ids = await select_graph_scopes(
                    cur,
                    (
                        TaskCandidateQuery(
                            cast(
                                "LiteralString",
                                f"SELECT id FROM cayu_tasks WHERE {held_scope} ORDER BY created_at, id LIMIT 100",
                            ),
                            tuple(held_params),
                        ),
                        TaskCandidateQuery(
                            f"SELECT id FROM cayu_tasks WHERE {expiration_scope} ORDER BY created_at, id LIMIT 100",
                            (
                                str(TaskStatus.PENDING),
                                str(TaskRetrySeriesDisposition.ACTIVE),
                                *expiration_deadline_params,
                            ),
                        ),
                        TaskCandidateQuery(
                            cast(
                                "LiteralString",
                                f"SELECT id FROM cayu_tasks WHERE {where_sql} ORDER BY {order_sql}, id ASC LIMIT 1",
                            ),
                            (
                                str(TaskStatus.PENDING),
                                *availability_params,
                                *retry_deadline_params,
                                *params,
                            ),
                        ),
                    ),
                )
                await self._expire_held_schedules(
                    cur, query, now=authoritative_series_now, candidate_ids=held_ids
                )
                await cur.execute(
                    f"""
                        SELECT {pg_support.TASK_COLUMNS}
                        FROM cayu_tasks
                        WHERE {expiration_scope} AND id = ANY(%s)
                        ORDER BY created_at ASC, id ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 100
                        """,
                    [
                        str(TaskStatus.PENDING),
                        str(TaskRetrySeriesDisposition.ACTIVE),
                        *expiration_deadline_params,
                        list(expired_ids),
                    ],
                )
                expired_rows = await cur.fetchall()
                for expired_row in expired_rows:
                    expired_task = pg_support.task_from_row(expired_row)
                    expiration = _expired_task_retry_settlement(
                        expired_task,
                        committed_at=now,
                        series_now=authoritative_series_now,
                    )
                    assert expiration.task.retry_series is not None
                    await cur.execute(
                        """
                        UPDATE cayu_tasks
                        SET status = %s, status_reason = %s, status_payload = %s,
                            result = NULL, error = %s, worker_id = NULL,
                            lease_expires_at = NULL, started_at = %s, completed_at = %s,
                            updated_at = %s, retry_series = %s
                        WHERE id = %s AND status = %s AND session_id IS NULL
                        """,
                        (
                            str(expiration.task.status),
                            expiration.task.status_reason,
                            pg_support._dumps(expiration.task.status_payload),
                            pg_support._dumps(expiration.task.error),
                            expiration.task.started_at,
                            expiration.task.completed_at,
                            expiration.task.updated_at,
                            pg_support._dumps(expiration.task.retry_series.model_dump(mode="json")),
                            expiration.task_id,
                            str(TaskStatus.PENDING),
                        ),
                    )
                    if cur.rowcount != 1:
                        raise TaskTerminalizationConflict(
                            "Elapsed task retry attempt changed during claim admission."
                        )
                    await self._record_task_transition(cur, expired_task, expiration.task)
                    await cur.execute(
                        "INSERT INTO cayu_task_retry_settlements "
                        "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            expiration.task_id,
                            expiration.idempotency_key,
                            expiration.request_sha256,
                            pg_support._dumps(expiration.model_dump(mode="json")),
                            expiration.committed_at,
                        ),
                    )
                await cur.execute(
                    cast(
                        "LiteralString",
                        f"""
                        WITH candidate AS (
                            SELECT id, status AS prior_status, worker_id AS prior_worker_id,
                                   lease_expires_at AS prior_lease, updated_at AS prior_updated_at
                            FROM cayu_tasks
                            WHERE {where_sql} AND (
                                (graph_id IS NULL AND NOT EXISTS (
                                    SELECT 1 FROM cayu_task_group_retry_lineage lineage
                                    WHERE lineage.task_id = cayu_tasks.id
                                )) OR COALESCE(graph_id, (
                                    SELECT lineage.graph_id FROM cayu_task_group_retry_lineage lineage
                                    WHERE lineage.task_id = cayu_tasks.id
                                )) IN (
                                    SELECT member.graph_id FROM cayu_task_graph_members AS member
                                    WHERE member.task_id = ANY(%s)
                                    UNION SELECT lineage.graph_id FROM cayu_task_group_retry_lineage lineage
                                    WHERE lineage.task_id = ANY(%s)
                                )
                            )
                            ORDER BY {order_sql}, id ASC
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                        )
                        UPDATE cayu_tasks AS task
                        SET status = %s,
                            worker_id = %s,
                            lease_expires_at = {lease_expires_sql},
                            updated_at = {updated_at_sql}
                        FROM candidate
                        WHERE task.id = candidate.id
                        RETURNING {_TASK_RETURNING_COLUMNS}, candidate.prior_status,
                                  candidate.prior_worker_id, candidate.prior_lease,
                                  candidate.prior_updated_at
                        """,
                    ),
                    [
                        str(TaskStatus.PENDING),
                        *availability_params,
                        *retry_deadline_params,
                        *params,
                        list(pending_ids),
                        list(pending_ids),
                        str(TaskStatus.CLAIMED),
                        worker_id,
                        *mutation_params,
                    ],
                )
                row = await cur.fetchone()
                claimed = None if row is None else pg_support.task_from_row(row)
                prior = (
                    None
                    if claimed is None
                    else claimed.model_copy(
                        update={
                            "status": TaskStatus(row[-4]),
                            "worker_id": row[-3],
                            "lease_expires_at": row[-2],
                            "updated_at": row[-1],
                        }
                    )
                )
                fenced = False
                if claimed is not None:
                    await self._lock_local_execution_retry_fence(cur, claimed)
                    # Recheck in a fresh statement after the advisory-lock wait.
                    # The candidate CTE's snapshot may predate an attempt that
                    # committed while its task row was locked by preparation.
                    fenced = await self._local_execution_attempt_fences_task(
                        cur,
                        claimed,
                    )
                    if not fenced:
                        post_fence_now = await self._database_now(cur)
                        post_fence_series_now = (
                            self._clock() if self._clock_is_injected else post_fence_now
                        )
                        retry_elapsed = _claimed_task_retry_attempt_elapsed(
                            claimed,
                            series_now=post_fence_series_now,
                        )
                    else:
                        retry_elapsed = False
                    schedule_eligibility = None
                    if (
                        not fenced
                        and claimed.schedule is not None
                        and claimed.schedule.admitted_at is None
                    ):
                        assert claimed.available_at is not None
                        schedule_eligibility = task_schedule_eligibility(
                            available_at=claimed.available_at,
                            policy=claimed.schedule.policy,
                            as_of=post_fence_series_now,
                        )
                    if not retry_elapsed and schedule_eligibility in {
                        TaskScheduleEligibility.EXPIRED,
                        TaskScheduleEligibility.SKIPPED,
                    }:
                        assert prior is not None
                        await self._settle_schedule_nonexecution(
                            cur, prior, eligibility=schedule_eligibility, now=post_fence_series_now
                        )
                        claimed = None
                    elif retry_elapsed:
                        expiration = _elapsed_claimed_task_retry_settlement(
                            claimed,
                            committed_at=post_fence_now,
                        )
                        settled = expiration.task
                        assert settled.retry_series is not None
                        await cur.execute(
                            """
                            UPDATE cayu_tasks
                            SET status = %s, status_reason = %s, status_payload = %s,
                                result = NULL, error = %s, worker_id = NULL,
                                lease_expires_at = NULL, started_at = %s,
                                completed_at = %s, updated_at = %s, retry_series = %s
                            WHERE id = %s AND status = %s AND worker_id = %s
                            """,
                            (
                                str(settled.status),
                                settled.status_reason,
                                pg_support._dumps(settled.status_payload),
                                pg_support._dumps(settled.error),
                                settled.started_at,
                                settled.completed_at,
                                settled.updated_at,
                                pg_support._dumps(settled.retry_series.model_dump(mode="json")),
                                claimed.id,
                                str(TaskStatus.CLAIMED),
                                worker_id,
                            ),
                        )
                        if cur.rowcount != 1:
                            raise TaskTerminalizationConflict(
                                "Elapsed task retry attempt changed during claim admission."
                            )
                        await self._record_task_transition(cur, prior, settled)
                        await cur.execute(
                            "INSERT INTO cayu_task_retry_settlements "
                            "(task_id, idempotency_key, request_sha256, receipt_json, "
                            "committed_at) VALUES (%s, %s, %s, %s, %s)",
                            (
                                expiration.task_id,
                                expiration.idempotency_key,
                                expiration.request_sha256,
                                pg_support._dumps(expiration.model_dump(mode="json")),
                                expiration.committed_at,
                            ),
                        )
                        claimed = None
                    elif not fenced:
                        # ``transaction_timestamp()`` used by the claim statement is
                        # frozen before the advisory-lock wait.  Restamp the newly
                        # acquired lease only after the retry fence is known clear so
                        # the returned ownership cannot already be expired at commit.
                        await cur.execute(
                            f"""
                            WITH authority AS (
                                SELECT %s::timestamptz AS now
                            )
                            UPDATE cayu_tasks AS task
                            SET lease_expires_at = authority.now
                                    + (%s * INTERVAL '1 second'),
                                updated_at = authority.now
                            FROM authority
                            WHERE task.id = %s
                              AND task.status = %s
                              AND task.worker_id = %s
                            RETURNING {_TASK_RETURNING_COLUMNS}
                            """,
                            (
                                post_fence_now,
                                lease_seconds,
                                claimed.id,
                                str(TaskStatus.CLAIMED),
                                worker_id,
                            ),
                        )
                        refreshed_row = await cur.fetchone()
                        if refreshed_row is None:
                            raise RuntimeError(
                                "Task claim changed while its retry fence was acquired."
                            )
                        claimed = pg_support.task_from_row(refreshed_row)
                        if claimed.schedule is not None:
                            claimed = claimed.model_copy(
                                update={
                                    "schedule": admitted_schedule(
                                        claimed, now=post_fence_series_now
                                    )
                                }
                            )
                            await self._update_task_snapshot(cur, claimed)
                        await self._record_task_transition(cur, prior, claimed)
            if fenced:
                await conn.rollback()
                return None
            await conn.commit()
        return None if claimed is None else claimed.model_copy(deep=True)

    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
        handoff_id: str | None = None,
        extend_seconds: int = 300,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        extend_seconds = _validate_task_positive_int(extend_seconds, "extend_seconds")

        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            async with conn.cursor() as cur:
                task = await self._load_task_locked(cur, task_id)
                now = await self._database_now(cur)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost("Task heartbeat no longer owns the expected worker lease.")
                _ensure_task_handoff_authority(task, handoff_id)
                await cur.execute(
                    "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                    (task_id,),
                )
                if await cur.fetchone() is not None:
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts require claim-fenced lease renewal."
                    )
                lease_expires_at = now + timedelta(seconds=extend_seconds)
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET lease_expires_at = %s,
                        updated_at = %s
                    WHERE id = %s AND worker_id = %s
                      AND interrupted_handoff_id IS NOT DISTINCT FROM %s
                      AND status IN (%s, %s)
                      AND lease_expires_at = %s AND lease_expires_at > %s
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        lease_expires_at,
                        now,
                        task_id,
                        worker_id,
                        handoff_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        expected_lease,
                        now,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    await self._raise_task_active_lease_error(
                        cur,
                        task_id,
                        worker_id,
                        now=now,
                    )
                assert row is not None
                updated = pg_support.task_from_row(row)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def release_task(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")

        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            async with conn.cursor() as cur:
                task = await self._load_task_locked(cur, task_id)
                now = await self._database_now(cur)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost("Task release no longer owns the expected worker lease.")
                await cur.execute(
                    "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                    (task_id,),
                )
                if await cur.fetchone() is not None:
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts release ownership through proposal publication."
                    )
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status = %s,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        started_at = NULL,
                        updated_at = %s
                    WHERE id = %s AND worker_id = %s AND status = %s
                      AND session_id IS NULL
                      AND status_reason IS DISTINCT FROM %s
                      AND status_reason IS DISTINCT FROM %s
                      AND lease_expires_at = %s AND lease_expires_at > %s
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        str(TaskStatus.PENDING),
                        now,
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        expected_lease,
                        now,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    await self._raise_task_release_error(
                        cur,
                        task_id,
                        worker_id,
                        now=now,
                    )
                assert row is not None
                updated = pg_support.task_from_row(row)
                await self._record_task_transition(cur, task, updated)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def release_attached_task_worker(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")

        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            async with conn.cursor() as cur:
                task = await self._load_task_locked(cur, task_id)
                now = await self._database_now(cur)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost(
                        "Attached-task release no longer owns the expected worker lease."
                    )
                if task.interrupted_handoff_id is not None:
                    raise TaskInterruptedHandoffConflict(
                        "Recovery-owned attached tasks must publish an interrupted handoff."
                    )
                await cur.execute(
                    "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                    (task_id,),
                )
                if await cur.fetchone() is not None:
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts release ownership through proposal publication."
                    )
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = %s
                    WHERE id = %s AND worker_id = %s AND status = %s
                      AND session_id IS NOT NULL
                      AND status_reason IS DISTINCT FROM %s
                      AND lease_expires_at = %s AND lease_expires_at > %s
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        now,
                        task_id,
                        worker_id,
                        str(TaskStatus.RUNNING),
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        expected_lease,
                        now,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    await self._raise_attached_task_worker_release_error(
                        cur,
                        task_id,
                        worker_id,
                        now=now,
                    )
                assert row is not None
                updated = pg_support.task_from_row(row)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        max_reclaims = _validate_task_positive_int(max_reclaims, "max_reclaims")
        if query.status is not None and query.status is not TaskStatus.CLAIMED:
            return []
        # Keep lazy schema reconciliation outside the mutation owner for the
        # same first-use cancellation contract as claim_task().
        await self._ensure_ready()
        connection_owner = _PostgresMutationConnectionOwner(
            self._pool,
            allowed_configure=self._postgres_mutation_allowed_configure,
        )
        return await self._await_owned_store_mutation(
            self._reclaim_expired_unowned(
                query=query,
                max_reclaims=max_reclaims,
                connection_owner=connection_owner,
            ),
            connection_owner=connection_owner,
        )

    async def _reclaim_expired_unowned(
        self,
        *,
        query: TaskQuery,
        max_reclaims: int = 100,
        connection_owner: _PostgresMutationConnectionOwner,
    ) -> list[Task]:
        clauses, params = self._task_filter_clauses(query)
        where_sql = " AND ".join(
            [
                "status = %s",
                "session_id IS NULL",
                "lease_expires_at IS NOT NULL",
                "lease_expires_at <= timing.now",
                "status_reason IS DISTINCT FROM %s",
                "status_reason IS DISTINCT FROM %s",
                "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                "WHERE NOT attempt.retry_admissible AND ("
                "attempt.task_id = task.id OR (task.retry_series IS NOT NULL "
                "AND attempt.retry_series_id = task.retry_series->>'series_id')))",
                *clauses,
            ]
        )

        async with self._owned_store_connection(connection_owner) as conn:
            async with conn.cursor() as cur:
                from cayu.storage._postgres_task_graphs import (
                    TaskCandidateQuery,
                    select_graph_scopes,
                )

                (candidate_ids,) = await select_graph_scopes(
                    cur,
                    (
                        TaskCandidateQuery(
                            cast(
                                "LiteralString",
                                f"WITH timing AS MATERIALIZED (SELECT clock_timestamp() AS now) "
                                f"SELECT task.id FROM cayu_tasks AS task CROSS JOIN timing WHERE {where_sql} "
                                "ORDER BY task.lease_expires_at ASC, task.id ASC LIMIT %s",
                            ),
                            (
                                str(TaskStatus.CLAIMED),
                                _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                                _TASK_CANCELLATION_REQUESTED_REASON,
                                *params,
                                max_reclaims,
                            ),
                        ),
                    ),
                )
                await cur.execute(
                    cast(
                        "LiteralString",
                        f"""
                        WITH timing AS MATERIALIZED (
                            SELECT clock_timestamp() AS now
                        )
                        SELECT {_TASK_RETURNING_COLUMNS}, timing.now
                        FROM cayu_tasks AS task
                        CROSS JOIN timing
                        WHERE {where_sql} AND task.id = ANY(%s)
                        ORDER BY task.lease_expires_at ASC, task.id ASC
                        FOR UPDATE OF task SKIP LOCKED
                        LIMIT %s
                        """,
                    ),
                    [
                        str(TaskStatus.CLAIMED),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        *params,
                        list(candidate_ids),
                        max_reclaims,
                    ],
                )
                rows = await cur.fetchall()
                expired = [pg_support.task_from_row(row) for row in rows]
                # Acquire scope locks in canonical order so a multi-row reclaim
                # cannot deadlock another reclaimer.  Every eligibility check is
                # then repeated in a new statement after any lock wait.
                tasks_by_scope = {
                    self._local_execution_retry_fence_scope(task): task for task in expired
                }
                for scope in sorted(tasks_by_scope):
                    await self._lock_local_execution_retry_fence(
                        cur,
                        tasks_by_scope[scope],
                    )
                fenced = any(
                    [await self._local_execution_attempt_fences_task(cur, task) for task in expired]
                )
                reclaimed: list[Task] = []
                if not fenced:
                    now = rows[0][-1] if rows else None
                    for task in expired:
                        assert now is not None
                        if task.started_at is not None:
                            updated = _expired_dispatched_task_cancellation(
                                task,
                                updated_at=now,
                            )
                        else:
                            updated = task.model_copy(
                                update={
                                    "status": TaskStatus.PENDING,
                                    "worker_id": None,
                                    "lease_expires_at": None,
                                    "updated_at": now,
                                }
                            )
                            reclaimed.append(updated)
                        await self._update_task_snapshot(cur, updated)
                        await self._record_task_transition(cur, task, updated)
            if fenced:
                # Roll back every row selected by this batch.  A later reclaim
                # starts from a fresh snapshot and can still settle unrelated
                # expired tasks without ever publishing a stale retry window.
                await conn.rollback()
                return []
            await conn.commit()
        return [task.model_copy(deep=True) for task in reclaimed]

    # -- internal helpers -------------------------------------------------

    async def _hold_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        reason: str | None,
        payload: dict[str, Any] | None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        reason = _copy_optional_status_reason(reason)
        payload = _copy_optional_status_payload(payload)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            async with conn.cursor() as cur:
                prior = await self._load_task_locked(cur, task_id)
                await cur.execute(
                    "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
                    (task_id,),
                )
                if await cur.fetchone() is not None:
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts cannot use ordinary task holds."
                    )
                now = await self._database_now(cur)
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status = %s,
                        status_reason = %s,
                        status_payload = %s,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = %s
                    WHERE id = %s
                      AND (
                        status = %s
                        OR status = %s
                        OR status = %s
                        OR status = %s
                        OR status = %s
                        OR (status = %s AND session_id IS NULL)
                        OR status IN ('waiting_dependencies', 'waiting_group')
                      )
                      AND status_reason IS DISTINCT FROM %s
                      AND status_reason IS DISTINCT FROM %s
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        str(status),
                        reason,
                        None if payload is None else pg_support._dumps(payload),
                        now,
                        task_id,
                        str(TaskStatus.PENDING),
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                        str(TaskStatus.RUNNING),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    task = await self._require_task(cur, task_id)
                    _ensure_can_hold_task(task, status)
                    raise ValueError(f"Task {task.id} cannot transition to {status}")
                updated = pg_support.task_from_row(row)
                await self._record_task_transition(cur, prior, updated)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def _finish_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
        handoff_id: str | None = None,
        expected_lease_expires_at: datetime | None = None,
        request_claimed_cancellation: bool = False,
    ) -> Task:
        await self._ensure_ready()

        async def operation(conn: Any, cur: Any) -> Task:
            del conn
            await self._lock_verified_work_task(cur, task_id)
            prior = await self._load_task_locked(cur, task_id)
            if prior.schedule is not None and status is TaskStatus.CANCELLED and worker_id is None:
                raise TaskScheduleConflict("Managed cancellation requires its schedule revision.")
            updated = await self._finish_task_in_transaction(
                cur,
                task_id,
                status,
                result=result,
                error=error,
                worker_id=worker_id,
                handoff_id=handoff_id,
                expected_lease_expires_at=expected_lease_expires_at,
                request_claimed_cancellation=request_claimed_cancellation,
            )
            await self._record_task_transition(cur, prior, updated)
            return updated

        return await self._run_verified_work_mutation(operation)

    async def _finish_task_in_transaction(
        self,
        cur: Any,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
        handoff_id: str | None = None,
        expected_lease_expires_at: datetime | None = None,
        request_claimed_cancellation: bool = False,
    ) -> Task:
        if expected_lease_expires_at is not None:
            expected_lease_expires_at = normalize_utc_datetime(
                expected_lease_expires_at,
                "lease_expires_at",
            )
        # When a worker_id is given, only terminalize if that worker still owns an active
        # lease — a worker that lost its lease must not clobber a task another has reclaimed.
        owner_clause = ""
        owner_params: list[Any] = []
        effective_handoff_id = handoff_id
        if worker_id is not None:
            owner_clause = (
                "\n                      AND worker_id = %s"
                "\n                      AND lease_expires_at IS NOT NULL AND lease_expires_at > %s"
                "\n                      AND interrupted_handoff_id IS NOT DISTINCT FROM %s"
            )
        task = await self._load_task_locked(cur, task_id)
        now = await self._database_now(cur)
        if (
            worker_id is not None
            and expected_lease_expires_at is not None
            and (task.worker_id != worker_id or task.lease_expires_at != expected_lease_expires_at)
        ):
            raise TaskClaimLost("Task terminalization no longer owns the expected worker lease.")
        await cur.execute(
            "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = %s LIMIT 1",
            (task_id,),
        )
        if await cur.fetchone() is not None:
            raise WorkAttemptExecutionClaimLost(
                "Admitted work attempts cannot use ordinary terminalization."
            )
        verified_work_support.require_contracted_completion_authority(task, status)
        if request_claimed_cancellation and (
            _task_cancellation_requested(task)
            or task.status_reason == _TASK_RETRY_CANCELLATION_REQUESTED_REASON
        ):
            return task.model_copy(deep=True)
        if worker_id is not None:
            if not (request_claimed_cancellation and task.started_at is not None):
                _ensure_owned_active_task_lease(task, worker_id, now=now)
            effective_handoff_id = (
                task.interrupted_handoff_id if request_claimed_cancellation else handoff_id
            )
            _ensure_task_handoff_authority(task, effective_handoff_id)
            owner_params = [worker_id, now, effective_handoff_id]
        cancellation = None
        if task.retry_series is not None:
            if status is not TaskStatus.CANCELLED:
                raise ValueError(
                    "Retry-series tasks require settle_task_retry_attempt for "
                    "completion or failure."
                )
            if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                cancellation_requested = _task_retry_cancellation_requested_task(
                    task,
                    error=error,
                    updated_at=now,
                )
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status_reason = %s, status_payload = %s, updated_at = %s
                    WHERE id = %s AND status IN (%s, %s)
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        cancellation_requested.status_reason,
                        pg_support._dumps(cancellation_requested.status_payload),
                        cancellation_requested.updated_at,
                        task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    raise TaskTerminalizationConflict(
                        "Task retry cancellation lost active ownership."
                    )
                updated = pg_support.task_from_row(row)
                return updated.model_copy(deep=True)
            cancellation = _cancelled_task_retry_settlement(
                task,
                error=error,
                committed_at=now,
            )
            terminal_task = cancellation.task
        elif (
            task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
            and status is TaskStatus.CANCELLED
            and task.worker_id is not None
            and task.lease_expires_at is not None
            and not _task_cancellation_requested(task)
        ):
            cancellation_requested = _task_cancellation_requested_task(
                task,
                error=error,
                updated_at=now,
            )
            await cur.execute(
                f"""
                UPDATE cayu_tasks
                SET status_reason = %s, status_payload = %s, updated_at = %s
                WHERE id = %s AND status IN (%s, %s)
                RETURNING {pg_support.TASK_COLUMNS}
                """,
                (
                    cancellation_requested.status_reason,
                    pg_support._dumps(cancellation_requested.status_payload),
                    cancellation_requested.updated_at,
                    task_id,
                    str(TaskStatus.CLAIMED),
                    str(TaskStatus.RUNNING),
                ),
            )
            row = await cur.fetchone()
            if row is None:
                raise TaskTerminalizationConflict("Task cancellation lost active ownership.")
            updated = pg_support.task_from_row(row)
            return updated.model_copy(deep=True)
        else:
            if _task_cancellation_requested(task):
                raise TaskTerminalizationConflict(
                    "Task cancellation is still draining under its current owner."
                )
            terminal_task = task.model_copy(
                update={
                    "status": status,
                    "status_reason": None,
                    "status_payload": None,
                    "result": result,
                    "error": error,
                    "started_at": task.started_at or now,
                    "completed_at": now,
                    "updated_at": now,
                    "interrupted_handoff_id": None,
                }
            )
        await cur.execute(
            f"""
            UPDATE cayu_tasks
            SET status = %s,
                status_reason = %s,
                status_payload = %s,
                result = %s,
                error = %s,
                worker_id = NULL,
                lease_expires_at = NULL,
                interrupted_handoff_id = NULL,
                started_at = COALESCE(started_at, %s),
                completed_at = %s,
                updated_at = %s,
                retry_series = %s
            WHERE id = %s
              AND status NOT IN (%s, %s, %s){owner_clause}
            """,
            (
                str(status),
                terminal_task.status_reason,
                (
                    None
                    if terminal_task.status_payload is None
                    else pg_support._dumps(terminal_task.status_payload)
                ),
                None if terminal_task.result is None else pg_support._dumps(terminal_task.result),
                None if terminal_task.error is None else pg_support._dumps(terminal_task.error),
                terminal_task.started_at,
                terminal_task.completed_at,
                terminal_task.updated_at,
                (
                    None
                    if terminal_task.retry_series is None
                    else pg_support._dumps(terminal_task.retry_series.model_dump(mode="json"))
                ),
                task_id,
                str(TaskStatus.COMPLETED),
                str(TaskStatus.FAILED),
                str(TaskStatus.CANCELLED),
                *owner_params,
            ),
        )
        if cur.rowcount != 1:
            if worker_id is not None:
                await self._raise_task_active_lease_error(
                    cur,
                    task_id,
                    worker_id,
                    now=now,
                )
                current = await self._require_task(cur, task_id)
                _ensure_task_handoff_authority(current, effective_handoff_id)
            task = await self._require_task(cur, task_id)
            _ensure_can_transition(task, status)
            raise ValueError(f"Task {task.id} cannot transition from {task.status}")
        if cancellation is not None:
            await cur.execute(
                "INSERT INTO cayu_task_retry_settlements "
                "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    cancellation.task_id,
                    cancellation.idempotency_key,
                    cancellation.request_sha256,
                    pg_support._dumps(cancellation.model_dump(mode="json")),
                    cancellation.committed_at,
                ),
            )
        updated = await self._require_task(cur, task_id)
        return updated.model_copy(deep=True)

    async def _load_task(self, cur: Any, task_id: str) -> Task | None:
        await cur.execute(
            f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks WHERE id = %s",
            (task_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return pg_support.task_from_row(row)

    async def _require_task(self, cur: Any, task_id: str) -> Task:
        task = await self._load_task(cur, task_id)
        from cayu.tasks.access import require_mutation

        require_mutation(task)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _task_filter_clauses(self, query: TaskQuery) -> tuple[list[str], list[object]]:
        clauses: list[str] = []
        params: list[object] = []
        if query.has_work_contract is not None:
            clauses.append(
                "work_contract IS NOT NULL" if query.has_work_contract else "work_contract IS NULL"
            )
        if query.type is not None:
            clauses.append("type = %s")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = %s")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = %s")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = %s")
            params.append(query.assigned_agent_name)
        return clauses, params

    async def _raise_task_active_lease_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = await self._require_task(cur, task_id)
        _ensure_owned_active_task_lease(task, worker_id, now=now)
        raise RuntimeError(f"Task {task.id} active-lease mutation did not update a row.")

    async def _raise_task_release_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = await self._require_task(cur, task_id)
        _ensure_owned_active_task_lease(task, worker_id, now=now)
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.status is not TaskStatus.CLAIMED:
            raise ValueError(f"Task {task.id} is not claimed.")
        if task.status_reason in {
            _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
            _TASK_CANCELLATION_REQUESTED_REASON,
        }:
            raise TaskTerminalizationConflict(
                "Task cancellation is still draining under its current owner."
            )
        raise RuntimeError(f"Task {task.id} active claim could not be released.")

    async def _raise_attached_task_worker_release_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = await self._require_task(cur, task_id)
        _ensure_owned_active_task_lease(task, worker_id, now=now)
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} is not running.")
        if task.session_id is None:
            raise ValueError(f"Task {task.id} is not attached to a session.")
        if _task_cancellation_requested(task):
            raise TaskTerminalizationConflict(
                "Task cancellation is still draining under its current owner."
            )
        raise RuntimeError(f"Task {task.id} active attached claim could not be released.")

    async def _raise_task_claim_attach_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = await self._require_task(cur, task_id)
        _raise_task_claim_attach_error(task, worker_id, now=now)


def _postgres_task_terminalization_receipt(
    *,
    task_id: str,
    idempotency_key: str,
    row: Any,
) -> TaskTerminalizationReceipt:
    try:
        return TaskTerminalizationReceipt(
            task_id=task_id,
            idempotency_key=idempotency_key,
            request_sha256=row[0],
            worker_id=row[1],
            kind=row[2],
            task=Task.model_validate(pg_support._json_obj(row[3])),
            committed_at=pg_support.to_utc(row[4]),
        )
    except Exception as exc:
        raise TaskTerminalizationConflict("Task terminalization receipt is malformed.") from exc


def _postgres_interrupted_task_handoff_receipt(
    *,
    task_id: str,
    handoff_id: str,
    row: Any,
) -> TaskInterruptedHandoffReceipt:
    try:
        receipt = TaskInterruptedHandoffReceipt(
            request=TaskInterruptedHandoffRequest.model_validate(pg_support._json_obj(row[1])),
            request_sha256=row[0],
            task=Task.model_validate(pg_support._json_obj(row[2])),
            committed_at=pg_support.to_utc(row[3]),
        )
        if receipt.request.task_id != task_id or receipt.request.handoff_id != handoff_id:
            raise ValueError("Interrupted-task handoff receipt conflicts with its storage key.")
        return receipt
    except Exception as exc:
        raise TaskInterruptedHandoffConflict(
            "Interrupted-task handoff receipt is malformed."
        ) from exc


def _validate_task_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value
