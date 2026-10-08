"""SQLite task persistence, separate from the session adapter."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar
from uuid import uuid4

from cayu._resource_store_surface import model_store_surface

if TYPE_CHECKING:
    from cayu.tasks.groups import (
        TaskGroupCreate,
        TaskGroupCreationReceipt,
        TaskGroupEvent,
        TaskGroupInvocationObligation,
        TaskGroupQuiescenceResolution,
        TaskGroupSnapshot,
    )

from pydantic import ValidationError

from cayu._clock import normalize_utc_datetime, utc_clock
from cayu._validation import (
    copy_durable_json_object,
    require_nonblank,
    revalidate_model_input,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.budgets.aggregates import EXACT_AGGREGATE
from cayu.runtime._task_lease_authority import managed_task_lease_mutation
from cayu.runtime._work_attempt_lifecycle_policy import (
    plan_work_attempt_execution_entry,
    plan_work_attempt_execution_stop,
    plan_work_attempt_lifecycle_settlement,
    plan_work_attempt_preparation_hold,
)
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
from cayu.runtime.work_attempt_lifecycle import (
    WorkAttemptLifecycleSettlement,
    WorkAttemptPreparationHold,
    copy_work_attempt_lifecycle_settlement,
    copy_work_attempt_preparation_hold,
    work_attempt_lifecycle_settlement_sha256,
    work_attempt_preparation_hold_sha256,
)
from cayu.sessions.invocation import SessionInvocationBinding, TaskInvocation
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage._phase_timing import TimedStoreLock
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.storage.targets import require_sqlite_store_allowed
from cayu.tasks import _verified_work_policy as verified_work_support
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
from cayu.tasks.admission import (
    WORK_ATTEMPT_RENEWABLE_STATES,
    AdmittedCompletionProposalRequest,
    WorkAttemptAdmission,
    WorkAttemptAdmissionActivate,
    WorkAttemptAdmissionConflict,
    WorkAttemptAdmissionPrepare,
    WorkAttemptAdmissionState,
    WorkAttemptContinuationContext,
    WorkAttemptExecutionClaim,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionClaimRequest,
    WorkAttemptExecutionEntryDisposition,
    WorkAttemptExecutionEntryRequest,
    WorkAttemptExecutionEntryResult,
    WorkAttemptExecutionStopRequest,
    WorkAttemptRecoveryActivate,
    copy_admitted_completion_proposal_request,
    copy_work_attempt_admission_activate,
    copy_work_attempt_admission_prepare,
    copy_work_attempt_execution_claim_request,
    copy_work_attempt_execution_entry_request,
    copy_work_attempt_execution_stop_request,
    copy_work_attempt_recovery_activate,
    renewed_work_attempt_execution_claim,
    work_attempt_admission_prepare_matches_sha256,
    work_attempt_admission_prepare_sha256,
    work_attempt_execution_claim_request_sha256,
)
from cayu.tasks.base import (
    _TASK_CANCELLATION_REQUESTED_REASON,
    _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
    CompletionDecisionApplicationReceipt,
    TaskAggregateFilter,
    TaskCancellationReconciliationRequest,
    TaskCancellationReconciliationResult,
    TaskCreate,
    TaskInvocationSnapshot,
    TaskOperationalSnapshot,
    TaskOrder,
    TaskQuery,
    TaskRetryCancellationReconciliationRequest,
    TaskRetrySettlementRequest,
    TaskRetrySettlementResult,
    TaskSessionClosureClaim,
    TaskStatusCounts,
    TaskStore,
    WorkAttemptLifecycleReceipt,
    WorkAttemptPreparationHoldReceipt,
    _can_attach_claimed_task_state,
    _cancelled_task_retry_settlement,
    _claimed_task_retry_attempt_elapsed,
    _copy_optional_session_binding,
    _copy_optional_status_payload,
    _copy_optional_status_reason,
    _copy_required_session_binding,
    _copy_task_cancellation_reconciliation_result,
    _elapsed_claimed_task_retry_settlement,
    _ensure_can_hold_task,
    _ensure_can_resume_task,
    _ensure_can_transition,
    _ensure_claim_query_supported,
    _ensure_exact_owned_active_task_lease,
    _ensure_owned_active_task_lease,
    _ensure_recovered_attached_task_failure_authority,
    _ensure_recovered_attached_task_session,
    _ensure_retry_series_queue_attempt,
    _ensure_task_handoff_authority,
    _ensure_task_terminalization_lease_authority,
    _expired_dispatched_task_cancellation,
    _expired_task_retry_settlement,
    _raise_task_claim_attach_error,
    _reconciled_task_cancellation,
    _reconciled_task_retry_cancellation,
    _rejected_task_cancellation_reconciliation,
    _rejected_task_retry_cancellation_reconciliation,
    _replay_task_cancellation_reconciliation,
    _replay_task_cancellation_reconciliation_rejection,
    _replay_task_retry_cancellation_reconciliation,
    _replay_task_retry_cancellation_reconciliation_rejection,
    _replay_task_retry_settlement,
    _require_active_attached_task_worker,
    _require_direct_attached_task_resume,
    _require_interrupted_task_handoff_authority,
    _running_task_from_create,
    _scheduled_task_nonexecution,
    _settled_task_retry_attempt,
    _task_cancellation_reconciliation_conflict,
    _task_cancellation_reconciliation_rejection_record,
    _task_cancellation_requested,
    _task_cancellation_requested_task,
    _task_from_create,
    _task_invocation_for_attachment,
    _task_matches_claim_filter,
    _task_retry_cancellation_reconciliation_conflict,
    _task_retry_cancellation_reconciliation_rejection_record,
    _task_retry_cancellation_requested_task,
    _task_retry_events,
    _task_retry_reconciliation_identity_is_bounded,
    _task_session_id_for_start,
    _task_session_instance_for_attachment,
    _TaskCancellationReconciliationRejectionRecord,
    _TaskRetryCancellationReconciliationRejectionRecord,
    _validate_ordinary_task_terminalization_against_cancellation,
    _validated_task_cancellation,
    _validated_task_retry_cancellation,
    _validated_task_retry_terminal_accounting,
    _work_attempt_discovery_query,
    copy_task_aggregate_filter,
    copy_task_create,
    copy_task_query,
    copy_task_session_closure_claim,
    prepare_task_cancellation_reconciliation,
    prepare_task_retry_cancellation_reconciliation,
    prepare_task_retry_settlement,
    task_query_from_aggregate_filter,
)
from cayu.tasks.completion_evaluations import (
    CompletionEvaluationRun,
    CompletionEvaluationRunRequest,
    CompletionEvaluationSettlementRequest,
    completion_evaluation_run_document,
    completion_evaluation_run_from_document,
    completion_evaluation_run_from_request,
    copy_completion_evaluation_run,
    copy_completion_evaluation_run_request,
    copy_completion_evaluation_settlement_request,
    replay_completion_evaluation_run,
    require_completion_evaluation_admission,
    settle_completion_evaluation_run_record,
)
from cayu.tasks.completion_verifier_dispatches import (
    CompletionVerifierDispatch,
    CompletionVerifierDispatchRequest,
    CompletionVerifierDispatchSettlementRequest,
    completion_verifier_dispatch_document,
    completion_verifier_dispatch_from_document,
    completion_verifier_dispatch_from_request,
    copy_completion_verifier_dispatch,
    copy_completion_verifier_dispatch_request,
    copy_completion_verifier_dispatch_settlement_request,
    replay_completion_verifier_dispatch,
    require_completion_verifier_dispatch_admission,
    settle_completion_verifier_dispatch_record,
)
from cayu.tasks.completion_verifier_profiles import (
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierProfileRecord,
    completion_verifier_profile_preparation_request_sha256,
    completion_verifier_profile_record_from_document,
    completion_verifier_profile_record_from_preparation,
    copy_completion_verifier_profile_preparation_request,
    copy_completion_verifier_profile_record,
    require_completion_verifier_profile_transition,
)
from cayu.tasks.contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionVerdict,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    TaskCompletionDecisionRequired,
    WorkAttempt,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    WorkContractRef,
    _GroupVerificationAdmissionRefused,
    completion_decision_application_request_sha256,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
    completion_proposal_request_sha256,
    completion_verification_claim_authority_sha256,
    completion_verification_claim_request_sha256,
    copy_completion_decision_application_request,
    copy_completion_decision_create,
    copy_completion_proposal_create,
    copy_completion_verification_claim_request,
    copy_work_attempt_create,
    copy_work_contract,
    copy_work_contract_ref,
    validate_completion_decision_contract,
    validate_work_completion_idempotency_key,
    work_attempt_request_sha256,
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
    prepare_interrupted_task_continuation_claim_page,
    prepare_interrupted_task_handoff,
    prepare_interrupted_task_handoff_candidate_page,
    prepare_interrupted_task_handoff_receipt_lookup,
)
from cayu.tasks.records import (
    Task,
    TaskClaimLost,
    TaskRetrySeriesDisposition,
    TaskStatus,
    copy_task,
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

_T = TypeVar("_T")


_SQLITE_TASK_MIN_REQUIRED_REVISION = 117


@model_store_surface("tasks")
class SQLiteTaskStore(TaskStore):
    """SQLite-backed task store for durable local work items."""

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
        from cayu.storage._sqlite_task_groups import reconciliation_candidates

        return await reconciliation_candidates(self, after_group_id=after_group_id, limit=limit)

    async def _settle_task_group_execution(self, task: Task) -> None:
        from cayu.storage._sqlite_task_groups import settle_execution

        await settle_execution(self, task)

    async def _task_group_cancellation_requested(self, task_id: str) -> bool:
        from cayu.storage._sqlite_task_groups import cancellation_requested

        return await cancellation_requested(self, task_id)

    async def _task_group_retains_execution(self, task_id: str) -> bool:
        from cayu.storage._sqlite_task_groups import retains_execution

        return await retains_execution(self, task_id)

    async def _observe_task_group_invocation(
        self, invocation: TaskGroupInvocationObligation
    ) -> None:
        from cayu.storage._sqlite_task_groups import observe_invocation

        await observe_invocation(self, invocation)

    async def _observe_task_group_result_resolution(
        self, task_id: str, decision_id: str, owner_id: str, *, settled: bool
    ) -> None:
        from cayu.storage._sqlite_task_groups import observe_result_resolution

        await observe_result_resolution(self, task_id, decision_id, owner_id, settled=settled)

    async def reconcile_task_group(self, group_id: str) -> TaskGroupSnapshot:
        from cayu.storage._sqlite_task_groups import reconcile

        return await reconcile(self, group_id)

    async def resolve_task_group_quiescence(
        self,
        request: TaskGroupQuiescenceResolution,
    ) -> TaskGroupSnapshot:
        from cayu.storage._sqlite_task_groups import reconcile

        return await reconcile(self, request.group_id, resolution=request)

    @runtime_task_creation
    async def create_task_group(self, request: TaskGroupCreate) -> TaskGroupCreationReceipt:
        from cayu.storage._sqlite_task_groups import create_group

        return await create_group(self, request)

    @runtime_collection_read
    async def load_task_group(self, group_id: str) -> TaskGroupSnapshot | None:
        from cayu.storage._sqlite_task_groups import load_group

        return await load_group(self, group_id)

    @runtime_collection_read
    async def list_task_group_events(
        self, group_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskGroupEvent]:
        from cayu.storage._sqlite_task_groups import list_events

        return await list_events(self, group_id, after_sequence=after_sequence, limit=limit)

    @runtime_task_creation
    async def create_task_graph(self, request: TaskGraphCreate) -> TaskGraphCreationReceipt:
        from cayu.storage._sqlite_task_graphs import create_graph

        return await create_graph(self, request)

    @runtime_collection_read
    async def load_task_graph(self, graph_id: str) -> TaskGraphSnapshot | None:
        from cayu.storage._sqlite_task_graphs import load_graph

        return await load_graph(self, graph_id)

    @runtime_collection_read
    async def list_task_graph_events(
        self, graph_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskGraphEvent]:
        from cayu.storage._sqlite_task_graphs import list_events

        return await list_events(self, graph_id, after_sequence=after_sequence, limit=limit)

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

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        ownership_clock: Callable[[], datetime] | None = None,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        require_sqlite_store_allowed("SQLiteTaskStore")
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteTaskStore path must be a string or Path.")
        self.service_durability = (
            RuntimeStoreDurability.DEVELOPMENT
            if str(db_path) == ":memory:"
            else RuntimeStoreDurability.DURABLE
        )
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")

        self.path = db_path
        diagnostic_source_missing = sqlite_connection.diagnostic_sqlite_source_missing(db_path)
        self._diagnostic_source_missing = diagnostic_source_missing
        self._schema_mode = schema.SchemaMode.CREATE if diagnostic_source_missing else schema_mode
        self._clock = utc_clock(clock)
        self._enable_task_admission_wakeups()
        self._ownership_clock = utc_clock(ownership_clock)
        self._lock = TimedStoreLock()
        effective_db_path = Path(":memory:") if diagnostic_source_missing else db_path
        self._connection = self._connect(effective_db_path)
        try:
            self._initialize_schema()
            if diagnostic_source_missing:
                self._connection.execute("PRAGMA query_only = ON")
        except BaseException:
            self._connection.close()
            raise

    def _verified_transaction_unlocked(self):
        return sqlite_connection._transaction(self._connection)

    def _load_local_execution_attempt_unlocked(
        self,
        attempt_id: str,
    ) -> LocalExecutionAttemptRecord | None:
        row = self._connection.execute(
            "SELECT attempt_id, task_id, retry_series_id, effect_lineage_id, "
            "request_sha256, phase, quiescence, retry_admissible, "
            "recovery_generation, recovery_owner_id, recovery_owner_expires_at, "
            "record_json, created_at, updated_at "
            "FROM cayu_local_execution_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            record = LocalExecutionAttemptRecord.model_validate(json.loads(row["record_json"]))
        except (json.JSONDecodeError, ValidationError):
            raise LocalExecutionAttemptConflict(
                "Stored local execution attempt content is malformed."
            ) from None
        if (
            record.authority.attempt_id != row["attempt_id"]
            or record.authority.task_id != row["task_id"]
            or record.authority.retry_series_id != row["retry_series_id"]
            or record.authority.effect_lineage_id != row["effect_lineage_id"]
            or record.authority.request_sha256 != row["request_sha256"]
            or record.phase.value != row["phase"]
            or record.quiescence.value != row["quiescence"]
            or int(record.retry_admissible) != row["retry_admissible"]
            or record.recovery_generation != row["recovery_generation"]
            or record.recovery_owner_id != row["recovery_owner_id"]
            or record.recovery_owner_expires_at
            != sqlite_records.parse_optional_datetime(row["recovery_owner_expires_at"])
            or record.created_at != sqlite_records.parse_datetime(row["created_at"])
            or record.updated_at != sqlite_records.parse_datetime(row["updated_at"])
        ):
            raise LocalExecutionAttemptConflict(
                "Stored local execution attempt indexes conflict with canonical content."
            )
        return record

    def _latest_local_execution_attempt_unlocked(
        self,
        authority: LocalExecutionAttemptAuthority,
    ) -> LocalExecutionAttemptRecord | None:
        if authority.retry_series_id is None:
            row = self._connection.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts "
                "WHERE retry_series_id IS NULL AND task_id = ? AND effect_lineage_id = ? "
                "ORDER BY retry_admissible ASC, created_at DESC, attempt_id DESC LIMIT 1",
                (authority.task_id, authority.effect_lineage_id),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts "
                "WHERE retry_series_id = ? AND effect_lineage_id = ? "
                "ORDER BY retry_admissible ASC, created_at DESC, attempt_id DESC LIMIT 1",
                (authority.retry_series_id, authority.effect_lineage_id),
            ).fetchone()
        return (
            None if row is None else self._load_local_execution_attempt_unlocked(row["attempt_id"])
        )

    def _store_local_execution_attempt_unlocked(
        self,
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
            int(record.retry_admissible),
            record.recovery_generation,
            record.recovery_owner_id,
            sqlite_records.format_optional_datetime(record.recovery_owner_expires_at),
            sqlite_records.json_dumps(record.model_dump(mode="json", warnings=False)),
            sqlite_records.format_datetime(record.created_at),
            sqlite_records.format_datetime(record.updated_at),
        )
        if insert:
            self._connection.execute(
                "INSERT INTO cayu_local_execution_attempts ("
                "attempt_id, task_id, retry_series_id, effect_lineage_id, request_sha256, "
                "phase, quiescence, retry_admissible, recovery_generation, "
                "recovery_owner_id, recovery_owner_expires_at, record_json, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            return
        cursor = self._connection.execute(
            "UPDATE cayu_local_execution_attempts SET task_id = ?, retry_series_id = ?, "
            "effect_lineage_id = ?, request_sha256 = ?, phase = ?, quiescence = ?, "
            "retry_admissible = ?, recovery_generation = ?, recovery_owner_id = ?, "
            "recovery_owner_expires_at = ?, record_json = ?, created_at = ?, updated_at = ? "
            "WHERE attempt_id = ? AND request_sha256 = ?",
            (*values[1:], values[0], values[4]),
        )
        if cursor.rowcount != 1:
            raise LocalExecutionAttemptConflict(
                "Local execution attempt changed during durable publication."
            )

    async def prepare_local_execution_attempt(
        self,
        authority: LocalExecutionAttemptAuthority,
    ) -> LocalExecutionAttemptRecord:
        authority = _copy_local_execution_attempt_authority(authority)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._load_local_execution_attempt_unlocked(authority.attempt_id)
                task = (
                    None if existing is not None else self._require_task_unlocked(authority.task_id)
                )
                prior = self._latest_local_execution_attempt_unlocked(authority)
                evidence_now = self._clock()
                lease_now = self._ownership_clock()
                record = prepare_local_execution_attempt_record(
                    authority=authority,
                    task=task,
                    existing=existing,
                    prior=prior,
                    evidence_now=evidence_now,
                    lease_now=lease_now,
                )
                if existing is None:
                    self._store_local_execution_attempt_unlocked(record, insert=True)
                self._connection.commit()
                return record.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def start_local_execution_attempt(
        self,
        start: LocalExecutionAttemptStart,
    ) -> LocalExecutionAttemptRecord:
        start = _copy_local_execution_attempt_start(start)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._load_local_execution_attempt_unlocked(start.attempt_id)
                if record is None:
                    raise LocalExecutionAttemptConflict(
                        "Local execution start has no prepared attempt."
                    )
                evidence_now = self._clock()
                lease_now = self._ownership_clock()
                if record.start is None:
                    require_local_execution_task_authority(
                        self._require_task_unlocked(record.authority.task_id),
                        record.authority,
                        now=lease_now,
                    )
                updated = advance_local_execution_attempt_start(
                    record,
                    start,
                    evidence_now=evidence_now,
                    lease_now=lease_now,
                )
                if updated != record:
                    self._store_local_execution_attempt_unlocked(updated, insert=False)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def settle_local_execution_attempt(
        self,
        settlement: LocalExecutionAttemptSettlement,
    ) -> LocalExecutionAttemptRecord:
        settlement = _copy_authenticated_local_execution_attempt_settlement(settlement)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._load_local_execution_attempt_unlocked(settlement.attempt_id)
                if record is None:
                    raise LocalExecutionAttemptConflict(
                        "Local execution settlement has no prepared attempt."
                    )
                updated = settle_local_execution_attempt_record(
                    record,
                    settlement,
                    evidence_now=self._clock(),
                    lease_now=self._ownership_clock(),
                )
                if updated != record:
                    self._store_local_execution_attempt_unlocked(updated, insert=False)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def load_local_execution_attempt(
        self,
        attempt_id: str,
    ) -> LocalExecutionAttemptRecord | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            record = self._load_local_execution_attempt_unlocked(attempt_id)
            return None if record is None else record.model_copy(deep=True)

    async def list_unsettled_local_execution_attempts(
        self,
        *,
        limit: int = 100,
        after: LocalExecutionAttemptListCursor | None = None,
    ) -> tuple[LocalExecutionAttemptRecord, ...]:
        limit = _validate_task_positive_int(limit, "limit")
        after = _copy_local_execution_attempt_list_cursor(after)
        async with self._lock:
            predicate = "(phase <> ? OR quiescence IN (?, ?))"
            parameters: list[Any] = [
                "terminal",
                "terminal_not_quiescent",
                "unavailable",
            ]
            if after is not None:
                predicate += " AND (created_at > ? OR (created_at = ? AND attempt_id > ?))"
                created_at = sqlite_records.format_datetime(after.created_at)
                parameters.extend(
                    (
                        created_at,
                        created_at,
                        after.attempt_id,
                    )
                )
            parameters.append(limit)
            rows = self._connection.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts WHERE "
                f"{predicate} "
                "ORDER BY created_at ASC, attempt_id ASC LIMIT ?",
                parameters,
            ).fetchall()
            records = [
                self._load_local_execution_attempt_unlocked(row["attempt_id"]) for row in rows
            ]
            return tuple(record.model_copy(deep=True) for record in records if record is not None)

    async def claim_local_execution_attempt_recovery(
        self,
        claim: LocalExecutionAttemptRecoveryClaim,
    ) -> LocalExecutionAttemptRecord:
        claim = _copy_local_execution_attempt_recovery_claim(claim)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._load_local_execution_attempt_unlocked(claim.attempt_id)
                if record is None:
                    raise LocalExecutionAttemptConflict(
                        "Local execution recovery attempt was not found."
                    )
                evidence_now = self._clock()
                lease_now = self._ownership_clock()
                task = self._require_task_unlocked(record.authority.task_id)
                require_local_execution_recovery_eligible(
                    task,
                    record,
                    now=lease_now,
                )
                updated = claim_local_execution_attempt_recovery_record(
                    record,
                    claim,
                    evidence_now=evidence_now,
                    lease_now=lease_now,
                )
                if updated != record:
                    self._store_local_execution_attempt_unlocked(updated, insert=False)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    def _load_work_contract_unlocked(
        self,
        reference: WorkContractRef,
    ) -> WorkContract | None:
        row = self._connection.execute(
            "SELECT contract_id, version, fingerprint, contract_json "
            "FROM cayu_work_contracts WHERE contract_id = ? AND version = ?",
            (reference.contract_id, reference.version),
        ).fetchone()
        if row is None:
            return None
        contract = WorkContract.model_validate(json.loads(row["contract_json"]))
        if (
            contract.contract_id != row["contract_id"]
            or contract.version != row["version"]
            or contract.fingerprint != row["fingerprint"]
        ):
            raise WorkContractConflict(
                "Stored work-contract indexes conflict with canonical content."
            )
        return verified_work_support.require_contract_reference(contract, reference)

    def _load_work_attempt_unlocked(self, attempt_id: str) -> WorkAttempt | None:
        row = self._connection.execute(
            "SELECT attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json "
            "FROM cayu_work_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        attempt = WorkAttempt.model_validate(json.loads(row["attempt_json"]))
        if (
            attempt.attempt_id != row["attempt_id"]
            or attempt.task_id != row["task_id"]
            or attempt.ordinal != row["ordinal"]
            or attempt.request_sha256 != row["request_sha256"]
            or attempt.started_at != sqlite_records.parse_datetime(row["started_at"])
        ):
            raise WorkCompletionConflict(
                "Stored work-attempt indexes conflict with canonical content."
            )
        return attempt

    def _load_work_attempt_admission_unlocked(
        self,
        admission_id: str,
    ) -> WorkAttemptAdmission | None:
        row = self._connection.execute(
            "SELECT admission.admission_id, admission.attempt_id, admission.task_id, "
            "admission.session_id, admission.interaction_id, admission.state, "
            "admission.prepare_request_sha256, admission.current_claim_id, "
            "admission.current_generation, admission.lease_expires_at, "
            "admission.admission_json, claim.claim_id AS durable_claim_id, "
            "claim.admission_id AS durable_claim_admission_id, "
            "claim.generation AS durable_claim_generation, "
            "claim.request_sha256 AS durable_claim_request_sha256, "
            "claim.lease_expires_at AS durable_claim_lease_expires_at, "
            "claim.is_current AS durable_claim_is_current, "
            "claim.claim_json AS durable_claim_json "
            "FROM cayu_work_attempt_admissions AS admission "
            "LEFT JOIN cayu_work_attempt_execution_claims AS claim "
            "ON claim.admission_id = admission.admission_id AND claim.is_current = 1 "
            "WHERE admission.admission_id = ?",
            (admission_id,),
        ).fetchone()
        if row is None:
            return None
        admission = WorkAttemptAdmission.model_validate(json.loads(row["admission_json"]))
        if (
            admission.admission_id != row["admission_id"]
            or admission.attempt_id != row["attempt_id"]
            or admission.task_id != row["task_id"]
            or admission.session_id != row["session_id"]
            or admission.interaction_id != row["interaction_id"]
            or admission.state.value != row["state"]
            or admission.prepare_request_sha256 != row["prepare_request_sha256"]
            or admission.claim.claim_id != row["current_claim_id"]
            or admission.claim.generation != row["current_generation"]
            or admission.claim.lease_expires_at
            != sqlite_records.parse_datetime(row["lease_expires_at"])
        ):
            raise WorkAttemptAdmissionConflict(
                "Stored work-attempt admission indexes conflict with canonical content."
            )
        if row["durable_claim_json"] is None:
            raise WorkAttemptAdmissionConflict(
                "Stored admission has no durable current execution claim."
            )
        durable_claim = WorkAttemptExecutionClaim.model_validate(
            json.loads(row["durable_claim_json"])
        )
        if (
            durable_claim != admission.claim
            or durable_claim.claim_id != row["durable_claim_id"]
            or durable_claim.admission_id != row["durable_claim_admission_id"]
            or durable_claim.generation != row["durable_claim_generation"]
            or durable_claim.request_sha256 != row["durable_claim_request_sha256"]
            or durable_claim.lease_expires_at
            != sqlite_records.parse_datetime(row["durable_claim_lease_expires_at"])
            or row["durable_claim_is_current"] != 1
        ):
            raise WorkAttemptAdmissionConflict(
                "Stored execution-claim authority conflicts with its admission."
            )
        return admission

    def _load_work_attempt_admission_for_attempt_unlocked(
        self,
        attempt_id: str,
    ) -> WorkAttemptAdmission | None:
        row = self._connection.execute(
            "SELECT admission_id FROM cayu_work_attempt_admissions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        return (
            None if row is None else self._load_work_attempt_admission_unlocked(row["admission_id"])
        )

    def _load_work_attempt_execution_claim_unlocked(
        self,
        claim_id: str,
    ) -> WorkAttemptExecutionClaim | None:
        row = self._connection.execute(
            "SELECT claim_id, admission_id, generation, request_sha256, "
            "lease_expires_at, claim_json FROM cayu_work_attempt_execution_claims "
            "WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if row is None:
            return None
        claim = WorkAttemptExecutionClaim.model_validate(json.loads(row["claim_json"]))
        if (
            claim.claim_id != row["claim_id"]
            or claim.admission_id != row["admission_id"]
            or claim.generation != row["generation"]
            or claim.request_sha256 != row["request_sha256"]
            or claim.lease_expires_at != sqlite_records.parse_datetime(row["lease_expires_at"])
        ):
            raise WorkAttemptAdmissionConflict(
                "Stored execution-claim indexes conflict with canonical content."
            )
        return claim

    def _insert_work_attempt_execution_claim_unlocked(
        self,
        claim: WorkAttemptExecutionClaim,
    ) -> None:
        self._connection.execute(
            "INSERT INTO cayu_work_attempt_execution_claims "
            "(claim_id, admission_id, generation, request_sha256, lease_expires_at, "
            "is_current, claim_json) VALUES (?, ?, ?, ?, ?, 1, ?)",
            (
                claim.claim_id,
                claim.admission_id,
                claim.generation,
                claim.request_sha256,
                sqlite_records.format_datetime(claim.lease_expires_at),
                sqlite_records.json_dumps(claim.model_dump(mode="json", warnings=False)),
            ),
        )

    def _update_work_attempt_admission_unlocked(
        self,
        admission: WorkAttemptAdmission,
    ) -> None:
        cursor = self._connection.execute(
            "UPDATE cayu_work_attempt_admissions SET state = ?, current_claim_id = ?, "
            "current_generation = ?, lease_expires_at = ?, admission_json = ? "
            "WHERE admission_id = ?",
            (
                admission.state.value,
                admission.claim.claim_id,
                admission.claim.generation,
                sqlite_records.format_datetime(admission.claim.lease_expires_at),
                sqlite_records.json_dumps(admission.model_dump(mode="json", warnings=False)),
                admission.admission_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Work-attempt admission not found: {admission.admission_id}")

    @staticmethod
    def _ensure_live_work_attempt_admission_claim(
        admission: WorkAttemptAdmission,
        *,
        now: datetime,
    ) -> None:
        if admission.claim.lease_expires_at <= now:
            raise WorkAttemptExecutionClaimLost("Work-attempt execution claim has expired.")

    def _latest_work_attempt_id_unlocked(self, task_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT attempt_id FROM cayu_work_attempts "
            "WHERE task_id = ? ORDER BY ordinal DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return None if row is None else row["attempt_id"]

    def _load_completion_proposal_unlocked(
        self,
        proposal_id: str,
    ) -> CompletionProposal | None:
        row = self._connection.execute(
            "SELECT proposal_id, attempt_id, task_id, request_sha256, proposed_at, "
            "proposal_json FROM cayu_completion_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        proposal = CompletionProposal.model_validate(json.loads(row["proposal_json"]))
        if (
            proposal.proposal_id != row["proposal_id"]
            or proposal.attempt_id != row["attempt_id"]
            or proposal.task_id != row["task_id"]
            or proposal.request_sha256 != row["request_sha256"]
            or proposal.proposed_at != sqlite_records.parse_datetime(row["proposed_at"])
        ):
            raise WorkCompletionConflict(
                "Stored completion-proposal indexes conflict with canonical content."
            )
        return proposal

    def _load_completion_claim_unlocked(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        row = self._connection.execute(
            "SELECT claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, request_sha256, "
            "lease_expires_at, claim_json FROM cayu_completion_verification_claims "
            "WHERE proposal_id = ? AND is_current = 1",
            (proposal_id,),
        ).fetchone()
        return None if row is None else self._completion_claim_from_row(row)

    def _load_completion_verifier_profile_unlocked(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        row = self._connection.execute(
            "SELECT proposal_id, task_id, attempt_id, profile_fingerprint, "
            "request_sha256, prepared_at, profile_json "
            "FROM cayu_completion_verifier_profiles WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        profile = completion_verifier_profile_record_from_document(json.loads(row["profile_json"]))
        if (
            profile.proposal_id != row["proposal_id"]
            or profile.task_id != row["task_id"]
            or profile.attempt_id != row["attempt_id"]
            or profile.profile.fingerprint != row["profile_fingerprint"]
            or profile.request_sha256 != row["request_sha256"]
            or profile.prepared_at != sqlite_records.parse_datetime(row["prepared_at"])
        ):
            raise WorkCompletionConflict(
                "Stored completion-verifier profile indexes conflict with canonical content."
            )
        return profile

    def _load_completion_verifier_adoption_unlocked(
        self,
        *,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionVerifierProfileRecord | None:
        rows = self._connection.execute(
            "SELECT proposal_id FROM cayu_completion_verifier_profiles "
            "WHERE task_id = ? "
            "AND json_extract(profile_json, '$.adoption.idempotency_key') = ? "
            "ORDER BY proposal_id LIMIT 2",
            (task_id, idempotency_key),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise WorkCompletionConflict(
                "Stored completion-verifier adoption idempotency authority is ambiguous."
            )
        profile = self._load_completion_verifier_profile_unlocked(rows[0]["proposal_id"])
        if (
            profile is None
            or profile.task_id != task_id
            or profile.adoption is None
            or profile.adoption.idempotency_key != idempotency_key
        ):
            raise WorkCompletionConflict(
                "Stored completion-verifier adoption idempotency authority is invalid."
            )
        return profile

    def _load_completion_claim_by_id_unlocked(
        self,
        claim_id: str,
    ) -> CompletionVerificationClaim | None:
        row = self._connection.execute(
            "SELECT claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, request_sha256, "
            "lease_expires_at, claim_json FROM cayu_completion_verification_claims "
            "WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        return None if row is None else self._completion_claim_from_row(row)

    @staticmethod
    def _completion_claim_from_row(row: sqlite3.Row) -> CompletionVerificationClaim:
        claim = CompletionVerificationClaim.model_validate(json.loads(row["claim_json"]))
        if (
            claim.claim_id != row["claim_id"]
            or claim.proposal_id != row["proposal_id"]
            or claim.attempt_number != row["attempt_number"]
            or claim.verifier_profile_fingerprint != row["verifier_profile_fingerprint"]
            or claim.request_sha256 != row["request_sha256"]
            or claim.lease_expires_at != sqlite_records.parse_datetime(row["lease_expires_at"])
        ):
            raise WorkCompletionConflict(
                "Stored verification-claim indexes conflict with canonical content."
            )
        return claim

    def _load_completion_decision_unlocked(
        self,
        decision_id: str,
    ) -> CompletionDecision | None:
        row = self._connection.execute(
            "SELECT decision_id, proposal_id, task_id, attempt_id, claim_id, verifier_profile_fingerprint, verdict, "
            "gap_fingerprint, request_sha256, decided_at, decision_json "
            "FROM cayu_completion_decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        return None if row is None else self._completion_decision_from_row(row)

    def _load_completion_decision_for_proposal_unlocked(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        row = self._connection.execute(
            "SELECT decision_id, proposal_id, task_id, attempt_id, claim_id, verifier_profile_fingerprint, verdict, "
            "gap_fingerprint, request_sha256, decided_at, decision_json "
            "FROM cayu_completion_decisions WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        return None if row is None else self._completion_decision_from_row(row)

    @staticmethod
    def _completion_decision_from_row(row: sqlite3.Row) -> CompletionDecision:
        decision = CompletionDecision.model_validate(json.loads(row["decision_json"]))
        if (
            decision.decision_id != row["decision_id"]
            or decision.proposal_id != row["proposal_id"]
            or decision.task_id != row["task_id"]
            or decision.attempt_id != row["attempt_id"]
            or decision.claim_id != row["claim_id"]
            or decision.verifier_profile_fingerprint != row["verifier_profile_fingerprint"]
            or decision.verdict.value != row["verdict"]
            or decision.gap_fingerprint != row["gap_fingerprint"]
            or decision.request_sha256 != row["request_sha256"]
            or decision.decided_at != sqlite_records.parse_datetime(row["decided_at"])
        ):
            raise WorkCompletionConflict(
                "Stored completion-decision indexes conflict with canonical content."
            )
        return decision

    def _load_decision_application_receipt_unlocked(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        row = self._connection.execute(
            "SELECT task_id, idempotency_key, decision_id, request_sha256, applied_at, "
            "receipt_json FROM cayu_completion_decision_application_receipts "
            "WHERE task_id = ? AND idempotency_key = ?",
            (task_id, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        receipt = CompletionDecisionApplicationReceipt.model_validate(
            json.loads(row["receipt_json"])
        )
        if (
            receipt.task_id != row["task_id"]
            or receipt.idempotency_key != row["idempotency_key"]
            or receipt.decision_id != row["decision_id"]
            or receipt.request_sha256 != row["request_sha256"]
            or receipt.applied_at != sqlite_records.parse_datetime(row["applied_at"])
        ):
            raise WorkCompletionConflict(
                "Stored decision-application receipt indexes conflict with canonical content."
            )
        return receipt

    def _ensure_session_execution_authority_unlocked(
        self,
        session_id: str,
        authority_kind: Literal["ordinary", "contracted"],
    ) -> None:
        now = self._ownership_clock()
        self._connection.execute(
            "INSERT OR IGNORE INTO cayu_task_session_execution_authority "
            "(session_id, authority_kind, committed_at) VALUES (?, ?, ?)",
            (session_id, authority_kind, sqlite_records.format_datetime(now)),
        )
        row = self._connection.execute(
            "SELECT authority_kind FROM cayu_task_session_execution_authority WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise TaskTopologyInconsistent("Session execution authority was not persisted.")
        if row["authority_kind"] != authority_kind:
            if authority_kind == "ordinary":
                raise TaskCompletionDecisionRequired(
                    "Contracted tasks require the verifier-aware execution entrance."
                )
            raise WorkCompletionConflict(
                "Work-contract attachment conflicts with prior ordinary session execution."
            )

    def _require_task_contract_unlocked(
        self,
        task: Task,
        reference: WorkContractRef,
    ) -> WorkContract:
        return verified_work_support.require_task_contract(
            task,
            reference,
            self._load_work_contract_unlocked(reference),
        )

    def _update_task_snapshot_unlocked(self, task: Task) -> None:
        if task.work_contract is not None:
            task = copy_task(task)
        cursor = self._connection.execute(
            """
            UPDATE cayu_tasks
            SET status = ?, session_id = ?, session_instance_id = ?,
                worker_id = ?, lease_expires_at = ?,
                status_reason = ?, status_payload_json = ?, result_json = ?, error_json = ?,
                updated_at = ?, started_at = ?, completed_at = ?, retry_series_json = ?,
                work_contract_json = ?, available_at = ?, schedule_json = ?
            WHERE id = ?
            """,
            (
                str(task.status),
                task.session_id,
                task.session_instance_id,
                task.worker_id,
                sqlite_records.format_optional_datetime(task.lease_expires_at),
                task.status_reason,
                None
                if task.status_payload is None
                else sqlite_records.json_dumps(task.status_payload),
                None if task.result is None else sqlite_records.json_dumps(task.result),
                None if task.error is None else sqlite_records.json_dumps(task.error),
                sqlite_records.format_datetime(task.updated_at),
                sqlite_records.format_optional_datetime(task.started_at),
                sqlite_records.format_optional_datetime(task.completed_at),
                None
                if task.retry_series is None
                else sqlite_records.json_dumps(task.retry_series.model_dump(mode="json")),
                None
                if task.work_contract is None
                else sqlite_records.json_dumps(
                    task.work_contract.model_dump(mode="json", warnings=False)
                ),
                sqlite_records.format_optional_datetime(task.available_at),
                None
                if task.schedule is None
                else sqlite_records.json_dumps(task.schedule.model_dump(mode="json")),
                task.id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Task not found: {task.id}")

    async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
        contract = copy_work_contract(contract)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_work_contract_unlocked(contract.reference())
                if existing is not None:
                    if existing != contract:
                        raise WorkContractConflict(
                            "Work-contract identity is already bound to different content."
                        )
                    return copy_work_contract(existing)
                if contract.supersedes is not None:
                    predecessor = self._load_work_contract_unlocked(contract.supersedes)
                    verified_work_support.require_contract_reference(
                        predecessor,
                        contract.supersedes,
                    )
                self._connection.execute(
                    "INSERT INTO cayu_work_contracts "
                    "(contract_id, version, fingerprint, contract_json) VALUES (?, ?, ?, ?)",
                    (
                        contract.contract_id,
                        contract.version,
                        contract.fingerprint,
                        sqlite_records.json_dumps(contract.model_dump(mode="json", warnings=False)),
                    ),
                )
                return copy_work_contract(contract)

    async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
        copied = copy_work_contract_ref(reference)
        if copied is None:
            raise TypeError("reference must be a WorkContractRef.")
        async with self._lock:
            contract = self._load_work_contract_unlocked(copied)
            return None if contract is None else copy_work_contract(contract)

    async def load_active_work_contract_task_for_session(
        self,
        session_id: str,
    ) -> Task | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            authority = self._connection.execute(
                "SELECT authority_kind FROM cayu_task_session_execution_authority "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if authority is None or authority["authority_kind"] == "ordinary":
                return None
            row = self._connection.execute(
                "SELECT * FROM cayu_tasks WHERE session_id = ? "
                "AND work_contract_json IS NOT NULL AND NOT EXISTS ("
                "SELECT 1 FROM cayu_work_attempt_lifecycle_receipts AS receipt "
                "WHERE receipt.task_id = cayu_tasks.id AND receipt.retired_contract_binding = 1) "
                "ORDER BY created_at, id LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                raise TaskTopologyInconsistent(
                    "Contracted session authority has no matching durable task."
                )
            return sqlite_records.task_from_row(row)

    async def admit_ordinary_session_execution(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            with self._verified_transaction_unlocked():
                self._ensure_session_execution_authority_unlocked(session_id, "ordinary")

    async def hold_claimed_work_contract_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        lease_expires_at: datetime | None = None,
        contract: WorkContractRef,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = (
            None
            if lease_expires_at is None
            else normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        )
        copied_contract = copy_work_contract_ref(contract)
        if copied_contract is None:
            raise TypeError("contract must be a WorkContractRef.")
        async with self._lock:
            with self._verified_transaction_unlocked():
                task = self._require_task_unlocked(task_id)
                now = self._ownership_clock()
                if expected_lease is None:
                    raise TaskClaimLost("Contracted task parking requires its exact worker lease.")
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease,
                    now=now,
                )
                if task.status is not TaskStatus.CLAIMED or task.session_id is not None:
                    raise TaskClaimLost(
                        "Only the current worker may park its unattached claimed task."
                    )
                self._require_task_contract_unlocked(task, copied_contract)
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.NEEDS_ATTENTION,
                        "status_reason": "verified_work_contract_runner_required",
                        "status_payload": {
                            "contract_id": copied_contract.contract_id,
                            "contract_version": copied_contract.version,
                        },
                        "worker_id": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    }
                )
                self._update_task_snapshot_unlocked(updated)
                self._record_task_transition_unlocked(task, updated)
                return updated.model_copy(deep=True)

    def _work_attempt_continuation_context_unlocked(
        self,
        task: Task,
        contract: WorkContract,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptContinuationContext | None:
        prior_attempt_id = self._latest_work_attempt_id_unlocked(task.id)
        if prior_attempt_id is None:
            return None
        prior_admission = self._load_work_attempt_admission_for_attempt_unlocked(prior_attempt_id)
        if (
            prior_admission is None
            or prior_admission.state is not WorkAttemptAdmissionState.RELEASED
            or prior_admission.task_id != task.id
            or prior_admission.session_id != task.session_id
        ):
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no exact released admission authority."
            )
        if prior_admission.run_semantics != request.run_semantics:
            raise WorkAttemptAdmissionConflict(
                "Continuation admission cannot change the source run settings."
            )
        row = self._connection.execute(
            "SELECT proposal.proposal_id, decision.decision_id, "
            "receipt.idempotency_key FROM cayu_completion_proposals AS proposal "
            "LEFT JOIN cayu_completion_decisions AS decision "
            "ON decision.proposal_id = proposal.proposal_id "
            "LEFT JOIN cayu_completion_decision_application_receipts AS receipt "
            "ON receipt.decision_id = decision.decision_id "
            "WHERE proposal.attempt_id = ?",
            (prior_attempt_id,),
        ).fetchone()
        if row is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no durable completion proposal."
            )
        if row["decision_id"] is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no durable completion decision."
            )
        if row["idempotency_key"] is None:
            raise WorkAttemptAdmissionConflict(
                "The latest completion decision has not been applied durably."
            )
        decision = self._load_completion_decision_unlocked(row["decision_id"])
        if decision is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work-attempt decision index is incomplete."
            )
        receipt = self._load_decision_application_receipt_unlocked(
            task.id,
            row["idempotency_key"],
        )
        if receipt is None or receipt.decision_id != decision.decision_id or receipt.task != task:
            raise WorkAttemptAdmissionConflict(
                "The latest decision application conflicts with continuation authority."
            )
        if (
            decision.verdict is not CompletionVerdict.REJECTED
            or contract.continuation_policy.rejection_action
            is not CompletionRejectionAction.CONTINUE
        ):
            raise WorkAttemptAdmissionConflict(
                "The latest completion decision does not authorize continuation."
            )
        return WorkAttemptContinuationContext(
            prior_admission_id=prior_admission.admission_id,
            prior_attempt_id=prior_attempt_id,
            proposal_id=row["proposal_id"],
            decision=decision,
            application_idempotency_key=row["idempotency_key"],
            gap_fingerprint=decision.gap_fingerprint,
        )

    async def prepare_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptAdmission:
        from cayu.storage._sqlite_task_groups import cancellation_requested_unlocked

        request = copy_work_attempt_admission_prepare(request)
        if request.generation != 1:
            raise WorkAttemptAdmissionConflict(
                "A new work-attempt admission must start at execution generation 1."
            )
        request_sha256 = work_attempt_admission_prepare_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_work_attempt_admission_unlocked(request.admission_id)
                if existing is not None:
                    if not work_attempt_admission_prepare_matches_sha256(
                        request,
                        existing.prepare_request_sha256,
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "Work-attempt admission identity is bound to another request."
                        )
                    return existing.model_copy(deep=True)
                if (
                    self._load_work_attempt_admission_for_attempt_unlocked(request.attempt_id)
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Work-attempt identity is already bound to another admission."
                    )
                if self._load_work_attempt_unlocked(request.attempt_id) is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Work-attempt identity already exists without this admission."
                    )
                occupied_claim = self._connection.execute(
                    "SELECT admission_id FROM cayu_work_attempt_execution_claims "
                    "WHERE claim_id = ?",
                    (request.claim_id,),
                ).fetchone()
                if occupied_claim is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Execution-claim identity is bound to another admission."
                    )
                occupied_interaction = self._connection.execute(
                    "SELECT admission_id FROM cayu_work_attempt_admissions "
                    "WHERE session_id = ? AND interaction_id = ?",
                    (request.session_id, request.interaction_id),
                ).fetchone()
                if occupied_interaction is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Session interaction is already bound to another admission."
                    )
                occupied_session = self._connection.execute(
                    "SELECT admission_id FROM cayu_work_attempt_admissions "
                    "WHERE session_id = ? AND state != 'released' LIMIT 1",
                    (request.session_id,),
                ).fetchone()
                if occupied_session is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Session already has an unreleased work-attempt admission."
                    )

                task = self._require_task_unlocked(request.task_id)
                unreleased_admission_row = self._connection.execute(
                    "SELECT admission_id FROM cayu_work_attempt_admissions "
                    "WHERE task_id = ? AND state != 'released' LIMIT 1",
                    (request.task_id,),
                ).fetchone()
                if unreleased_admission_row is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Task already has an unreleased work-attempt admission."
                    )
                contract = self._require_task_contract_unlocked(task, request.contract)
                lease_now = self._ownership_clock()
                availability_now = self._clock()
                if cancellation_requested_unlocked(self, task.id):
                    raise WorkAttemptAdmissionConflict(
                        "A decided group loser cannot admit another execution."
                    )
                continuation = self._work_attempt_continuation_context_unlocked(
                    task,
                    contract,
                    request,
                )
                if continuation is None:
                    if request.kind != "initial":
                        raise WorkAttemptAdmissionConflict(
                            "Continuation admission requires a rejected/continue decision."
                        )
                    if task.status not in {TaskStatus.PENDING, TaskStatus.CLAIMED}:
                        raise WorkAttemptAdmissionConflict(
                            "Initial admission requires a pending or claimed contracted task."
                        )
                    if task.available_at is not None and task.available_at > availability_now:
                        raise WorkAttemptAdmissionConflict(
                            "Contracted task is not yet available for admission."
                        )
                    if task.status is TaskStatus.CLAIMED:
                        if request.task_lease_expires_at is None:
                            raise TaskClaimLost(
                                "Work-attempt admission requires the claimed task's exact lease."
                            )
                        _ensure_exact_owned_active_task_lease(
                            task,
                            request.worker_id,
                            request.task_lease_expires_at,
                            now=lease_now,
                        )
                    elif task.worker_id is not None or task.lease_expires_at is not None:
                        raise WorkAttemptAdmissionConflict(
                            "Pending contracted task has conflicting worker ownership."
                        )
                    elif request.task_lease_expires_at is not None:
                        raise WorkAttemptAdmissionConflict(
                            "Pending work-attempt admission cannot consume worker lease authority."
                        )
                    if task.session_id not in {None, request.session_id}:
                        raise WorkAttemptAdmissionConflict(
                            "Initial admission conflicts with the task's session."
                        )
                else:
                    if request.task_lease_expires_at is not None:
                        raise WorkAttemptAdmissionConflict(
                            "Continuation admission cannot consume prior worker lease authority."
                        )
                    if request.kind != "continuation":
                        raise WorkAttemptAdmissionConflict(
                            "Initial admission cannot consume continuation authority."
                        )
                    if continuation.prior_admission_id != request.predecessor_admission_id:
                        raise WorkAttemptAdmissionConflict(
                            "Continuation admission selected another predecessor admission."
                        )
                    if (
                        task.status is not TaskStatus.RUNNING
                        or task.session_id != request.session_id
                        or task.worker_id is not None
                        or task.lease_expires_at is not None
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "Continuation admission requires an unowned running task on its exact session."
                        )
                    if (
                        contract.continuation_policy.rejection_action
                        is not CompletionRejectionAction.CONTINUE
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "The frozen contract does not authorize continuation."
                        )

                self._ensure_session_execution_authority_unlocked(
                    request.session_id,
                    "contracted",
                )
                _task_invocation_for_attachment(
                    task.invocation,
                    session_id=request.session_id,
                    session_binding=request.session_invocation,
                )
                session_instance_id = _task_session_instance_for_attachment(
                    stored_session_instance_id=task.session_instance_id,
                    session_id=request.session_id,
                    session_binding=request.session_invocation,
                )
                claim_request = WorkAttemptExecutionClaimRequest(
                    admission_id=request.admission_id,
                    claim_id=request.claim_id,
                    worker_id=request.worker_id,
                    execution_owner_id=request.execution_owner_id,
                    generation=request.generation,
                    lease_seconds=request.lease_seconds,
                )
                claim = WorkAttemptExecutionClaim(
                    admission_id=request.admission_id,
                    claim_id=request.claim_id,
                    worker_id=request.worker_id,
                    execution_owner_id=request.execution_owner_id,
                    generation=request.generation,
                    request_sha256=work_attempt_execution_claim_request_sha256(claim_request),
                    claimed_at=lease_now,
                    lease_expires_at=lease_now + timedelta(seconds=request.lease_seconds),
                )
                admission = WorkAttemptAdmission(
                    admission_id=request.admission_id,
                    prepare_request_sha256=request_sha256,
                    state=WorkAttemptAdmissionState.PREPARING,
                    attempt_id=request.attempt_id,
                    task_id=request.task_id,
                    session_id=request.session_id,
                    interaction_id=request.interaction_id,
                    kind=request.kind,
                    source_request_sha256=request.source_request_sha256,
                    contract=request.contract,
                    session_invocation=request.session_invocation,
                    source_execution_profile_fingerprint=(
                        request.source_execution_profile_fingerprint
                    ),
                    run_semantics=request.run_semantics,
                    source_request=request.source_request,
                    claim=claim,
                    continuation=continuation,
                    prepared_at=lease_now,
                )
                updated_task = task.model_copy(
                    update={
                        "status": TaskStatus.RUNNING,
                        "session_id": request.session_id,
                        "session_instance_id": session_instance_id,
                        "worker_id": request.worker_id,
                        "lease_expires_at": claim.lease_expires_at,
                        "started_at": task.started_at or lease_now,
                        "updated_at": lease_now,
                    }
                )
                self._update_task_snapshot_unlocked(updated_task)
                self._record_task_transition_unlocked(task, updated_task)
                self._connection.execute(
                    "INSERT INTO cayu_work_attempt_admissions "
                    "(admission_id, attempt_id, task_id, session_id, interaction_id, state, "
                    "prepare_request_sha256, current_claim_id, current_generation, "
                    "lease_expires_at, admission_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        admission.admission_id,
                        admission.attempt_id,
                        admission.task_id,
                        admission.session_id,
                        admission.interaction_id,
                        admission.state.value,
                        admission.prepare_request_sha256,
                        admission.claim.claim_id,
                        admission.claim.generation,
                        sqlite_records.format_datetime(admission.claim.lease_expires_at),
                        sqlite_records.json_dumps(
                            admission.model_dump(mode="json", warnings=False)
                        ),
                    ),
                )
                self._insert_work_attempt_execution_claim_unlocked(claim)
                return admission.model_copy(deep=True)

    async def activate_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionActivate,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_admission_activate(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                if admission.prepare_request_sha256 != request.prepare_request_sha256:
                    raise WorkAttemptAdmissionConflict(
                        "Admission activation conflicts with its prepared request."
                    )
                if admission.claim.claim_id != request.claim_id:
                    raise WorkAttemptExecutionClaimLost(
                        "Admission activation no longer owns the prepared execution claim."
                    )
                task = self._require_task_unlocked(admission.task_id)
                if (
                    task.session_id != admission.session_id
                    or task.session_instance_id != admission.session_invocation.session_instance_id
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Prepared admission conflicts with exact task-session authority."
                    )
                if admission.state is WorkAttemptAdmissionState.ACTIVE:
                    if admission.session_evidence_sha256 != request.session_evidence_sha256:
                        raise WorkAttemptAdmissionConflict(
                            "Admission activation conflicts with durable session evidence."
                        )
                    return admission.model_copy(deep=True)
                if admission.state is not WorkAttemptAdmissionState.PREPARING:
                    raise WorkAttemptAdmissionConflict(
                        "Only a prepared admission can publish its work attempt."
                    )
                lease_now = self._ownership_clock()
                self._ensure_live_work_attempt_admission_claim(admission, now=lease_now)
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.worker_id != admission.claim.worker_id
                    or task.lease_expires_at != admission.claim.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Prepared admission conflicts with current task ownership."
                    )
                contract = self._require_task_contract_unlocked(task, admission.contract)
                prior_id = self._latest_work_attempt_id_unlocked(task.id)
                prior = None if prior_id is None else self._load_work_attempt_unlocked(prior_id)
                ordinal = 1 if prior is None else prior.ordinal + 1
                if ordinal > contract.continuation_policy.max_attempts:
                    raise WorkAttemptAdmissionConflict(
                        "Work-contract attempt limit forbids activation."
                    )
                attempt_request = WorkAttemptCreate(
                    attempt_id=admission.attempt_id,
                    task_id=admission.task_id,
                    session_id=admission.session_id,
                    contract=admission.contract,
                    execution_profile_fingerprint=(admission.source_execution_profile_fingerprint),
                    worker_id=admission.claim.worker_id,
                )
                attempt = WorkAttempt(
                    **attempt_request.model_dump(mode="python"),
                    ordinal=ordinal,
                    request_sha256=work_attempt_request_sha256(attempt_request),
                    started_at=(evidence_now := self._clock()),
                )
                activated = WorkAttemptAdmission.model_validate(
                    admission.model_copy(
                        update={
                            "state": WorkAttemptAdmissionState.ACTIVE,
                            "attempt": attempt,
                            "session_evidence_sha256": request.session_evidence_sha256,
                            "activated_at": evidence_now,
                        }
                    ).model_dump(mode="python", warnings=False)
                )
                self._connection.execute(
                    "INSERT INTO cayu_work_attempts "
                    "(attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        attempt.attempt_id,
                        attempt.task_id,
                        attempt.ordinal,
                        attempt.request_sha256,
                        sqlite_records.format_datetime(attempt.started_at),
                        sqlite_records.json_dumps(attempt.model_dump(mode="json", warnings=False)),
                    ),
                )
                self._update_work_attempt_admission_unlocked(activated)
                return activated.model_copy(deep=True)

    async def load_work_attempt_admission(
        self,
        admission_id: str,
    ) -> WorkAttemptAdmission | None:
        admission_id = require_clean_nonblank(admission_id, "admission_id")
        async with self._lock:
            admission = self._load_work_attempt_admission_unlocked(admission_id)
            return None if admission is None else admission.model_copy(deep=True)

    async def load_work_attempt_execution_claim(
        self,
        claim_id: str,
    ) -> WorkAttemptExecutionClaim | None:
        claim_id = require_clean_nonblank(claim_id, "claim_id")
        async with self._lock:
            claim = self._load_work_attempt_execution_claim_unlocked(claim_id)
            return None if claim is None else claim.model_copy(deep=True)

    async def load_latest_work_attempt_admission(
        self,
        task_id: str,
    ) -> WorkAttemptAdmission | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            return self._load_latest_work_attempt_admission_unlocked(task_id)

    def _load_latest_work_attempt_admission_unlocked(
        self, task_id: str
    ) -> WorkAttemptAdmission | None:
        rows = self._connection.execute(
            "SELECT current.admission_id FROM cayu_work_attempt_admissions AS current "
            "WHERE current.task_id = ? AND NOT EXISTS ("
            "SELECT 1 FROM cayu_work_attempt_admissions AS successor "
            "WHERE successor.task_id = current.task_id AND "
            "json_extract(successor.admission_json, '$.continuation.prior_admission_id') "
            "= current.admission_id) LIMIT 2",
            (task_id,),
        ).fetchall()
        if len(rows) > 1:
            raise WorkAttemptAdmissionConflict("Task admission history has no unique successor.")
        if not rows:
            return None
        admission = self._load_work_attempt_admission_unlocked(rows[0]["admission_id"])
        if admission is None or admission.task_id != task_id:
            raise WorkAttemptAdmissionConflict("Latest admission conflicts with its task.")
        return admission.model_copy(deep=True)

    def _load_work_attempt_lifecycle_receipt_unlocked(
        self, admission_id: str
    ) -> WorkAttemptLifecycleReceipt | None:
        row = self._connection.execute(
            "SELECT * FROM cayu_work_attempt_lifecycle_receipts WHERE admission_id = ?",
            (admission_id,),
        ).fetchone()
        if row is None:
            return None
        receipt = WorkAttemptLifecycleReceipt.model_validate_json(row["receipt_json"])
        if (
            receipt.request.admission_id != admission_id
            or receipt.request.settlement_id != row["settlement_id"]
            or receipt.task.id != row["task_id"]
            or receipt.request_sha256 != row["request_sha256"]
            or int(receipt.retired_contract_binding) != row["retired_contract_binding"]
            or receipt.settled_at != sqlite_records.parse_datetime(row["settled_at"])
        ):
            raise WorkAttemptAdmissionConflict(
                "Lifecycle receipt indexes conflict with canonical content."
            )
        return receipt

    async def load_work_attempt_lifecycle_receipt(
        self, admission_id: str
    ) -> WorkAttemptLifecycleReceipt | None:
        admission_id = require_clean_nonblank(admission_id, "admission_id")
        async with self._lock:
            return self._load_work_attempt_lifecycle_receipt_unlocked(admission_id)

    async def enter_work_attempt_execution(
        self, request: WorkAttemptExecutionEntryRequest
    ) -> WorkAttemptExecutionEntryResult:
        from cayu.storage._sqlite_task_groups import cancellation_requested_unlocked

        request = copy_work_attempt_execution_entry_request(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise WorkAttemptAdmissionConflict("Execution entry has no admission.")
                result = plan_work_attempt_execution_entry(
                    request,
                    admission=admission,
                    task=self._require_task_unlocked(admission.task_id),
                    now=self._ownership_clock(),
                    group_cancelled=cancellation_requested_unlocked(self, admission.task_id),
                )
                if result.disposition is WorkAttemptExecutionEntryDisposition.ENTERED:
                    self._update_work_attempt_admission_unlocked(result.admission)
                return result

    async def record_work_attempt_execution_stop(
        self, request: WorkAttemptExecutionStopRequest
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_execution_stop_request(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise WorkAttemptAdmissionConflict("Execution stop has no admission.")
                result = plan_work_attempt_execution_stop(
                    request,
                    admission=admission,
                    task=self._require_task_unlocked(admission.task_id),
                    now=self._ownership_clock(),
                )
                if admission.execution_stop is None:
                    self._update_work_attempt_admission_unlocked(result)
                return result

    def _load_work_attempt_preparation_hold_unlocked(
        self, hold_id: str
    ) -> WorkAttemptPreparationHoldReceipt | None:
        row = self._connection.execute(
            "SELECT task_id, request_sha256, receipt_json "
            "FROM cayu_work_attempt_preparation_holds WHERE hold_id = ?",
            (hold_id,),
        ).fetchone()
        if row is None:
            return None
        receipt = WorkAttemptPreparationHoldReceipt.model_validate_json(row["receipt_json"])
        if (
            receipt.request.hold_id != hold_id
            or receipt.task.id != row["task_id"]
            or receipt.request_sha256 != row["request_sha256"]
        ):
            raise WorkAttemptAdmissionConflict(
                "Preparation hold indexes conflict with canonical content."
            )
        return receipt

    async def load_work_attempt_preparation_hold_receipt(
        self, hold_id: str
    ) -> WorkAttemptPreparationHoldReceipt | None:
        hold_id = validate_work_completion_idempotency_key(hold_id)
        async with self._lock:
            return self._load_work_attempt_preparation_hold_unlocked(hold_id)

    async def hold_work_attempt_preparation(
        self, request: WorkAttemptPreparationHold
    ) -> WorkAttemptPreparationHoldReceipt:
        from cayu.storage._sqlite_task_groups import cancellation_requested_unlocked

        request = copy_work_attempt_preparation_hold(request)
        digest = work_attempt_preparation_hold_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_work_attempt_preparation_hold_unlocked(request.hold_id)
                if existing is not None:
                    if existing.request_sha256 != digest:
                        raise WorkAttemptAdmissionConflict(
                            "Preparation hold conflicts with its receipt."
                        )
                    return existing
                task = self._require_task_unlocked(request.task_id)
                self._require_task_contract_unlocked(task, request.contract)
                updated, receipt = plan_work_attempt_preparation_hold(
                    request,
                    task=task,
                    has_attempt=self._latest_work_attempt_id_unlocked(task.id) is not None,
                    now=self._ownership_clock(),
                    group_cancelled=cancellation_requested_unlocked(self, task.id),
                )
                encoded = receipt.model_dump_json(warnings=False)
                self._update_task_snapshot_unlocked(updated)
                self._record_task_transition_unlocked(
                    task,
                    updated,
                    settled_execution=(task.id, request.worker_id, task.started_at)
                    if task.started_at is not None
                    else None,
                )
                self._connection.execute(
                    "INSERT INTO cayu_work_attempt_preparation_holds "
                    "(hold_id, task_id, request_sha256, receipt_json) VALUES (?, ?, ?, ?)",
                    (request.hold_id, task.id, digest, encoded),
                )
                return receipt

    async def list_unsettled_work_attempt_admissions(
        self,
        *,
        task_filter: TaskAggregateFilter | None = None,
        limit: int = 100,
        after: str | None = None,
    ) -> list[WorkAttemptAdmission]:
        query, after = _work_attempt_discovery_query(task_filter, limit=limit, after=after)
        clauses, params = self._task_filter_clauses(query)
        cursor_clause = "" if after is None else "AND admission.admission_id COLLATE BINARY > ? "
        if after is not None:
            params.append(after)
        params.append(limit)

        def read(connection: sqlite3.Connection) -> list[WorkAttemptAdmission]:
            rows = connection.execute(
                "SELECT admission.admission_id FROM cayu_work_attempt_admissions AS admission "
                "JOIN (SELECT id FROM cayu_tasks WHERE "
                + " AND ".join(clauses)
                + ") AS task ON task.id = admission.task_id "
                "WHERE NOT EXISTS (SELECT 1 FROM cayu_work_attempt_lifecycle_receipts AS receipt "
                "WHERE receipt.admission_id = admission.admission_id) "
                "AND NOT EXISTS (SELECT 1 FROM cayu_work_attempt_admissions AS successor "
                "WHERE successor.task_id = admission.task_id AND "
                "json_extract(successor.admission_json, '$.continuation.prior_admission_id') = admission.admission_id) "
                + cursor_clause
                + "ORDER BY admission.admission_id COLLATE BINARY LIMIT ?",
                params,
            ).fetchall()
            admissions = []
            for row in rows:
                admission = self._load_work_attempt_admission_unlocked(row["admission_id"])
                if admission is None:
                    raise WorkAttemptAdmissionConflict(
                        "Discovered admission lost its durable record."
                    )
                admissions.append(admission)
            return admissions

        return await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, read, interrupt_on_cancellation=True
        )

    async def settle_work_attempt_lifecycle(
        self, request: WorkAttemptLifecycleSettlement
    ) -> WorkAttemptLifecycleReceipt:
        request = copy_work_attempt_lifecycle_settlement(request)
        request_sha256 = work_attempt_lifecycle_settlement_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_work_attempt_lifecycle_receipt_unlocked(request.admission_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkAttemptAdmissionConflict(
                            "Lifecycle settlement conflicts with its receipt."
                        )
                    return existing
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_lifecycle_receipts WHERE settlement_id = ?",
                        (request.settlement_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Lifecycle settlement identity is already bound."
                    )
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise WorkAttemptAdmissionConflict("Lifecycle settlement has no admission.")
                task = self._require_task_unlocked(request.task_id)
                latest = self._load_latest_work_attempt_admission_unlocked(task.id)
                row = self._connection.execute(
                    "SELECT proposal.proposal_id, decision.decision_id "
                    "FROM cayu_completion_proposals AS proposal "
                    "LEFT JOIN cayu_completion_decisions AS decision ON decision.proposal_id = proposal.proposal_id "
                    "WHERE proposal.attempt_id = ?",
                    (admission.attempt_id,),
                ).fetchone()
                proposal = (
                    None
                    if row is None
                    else self._load_completion_proposal_unlocked(row["proposal_id"])
                )
                decision = (
                    None
                    if row is None or row["decision_id"] is None
                    else self._load_completion_decision_unlocked(row["decision_id"])
                )
                application = self._load_decision_application_receipt_unlocked(
                    task.id, request.application_idempotency_key or ""
                )
                from cayu.storage._sqlite_task_groups import cancellation_requested_unlocked

                updated, settled_admission, receipt = plan_work_attempt_lifecycle_settlement(
                    request,
                    task=task,
                    admission=admission,
                    latest_admission_id="" if latest is None else latest.admission_id,
                    proposal=proposal,
                    decision=decision,
                    application=application,
                    now=self._ownership_clock(),
                    group_cancelled=cancellation_requested_unlocked(self, task.id),
                    verification_claim=None
                    if proposal is None
                    else self._load_completion_claim_unlocked(proposal.proposal_id),
                )
                encoded = receipt.model_dump_json(warnings=False)
                self._update_task_snapshot_unlocked(updated)
                self._record_task_transition_unlocked(
                    task,
                    updated,
                    settled_execution=(task.id, admission.claim.worker_id, task.started_at)
                    if task.started_at is not None
                    else None,
                )
                self._update_work_attempt_admission_unlocked(settled_admission)
                self._connection.execute(
                    "INSERT INTO cayu_work_attempt_lifecycle_receipts "
                    "(admission_id, settlement_id, task_id, request_sha256, retired_contract_binding, settled_at, receipt_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        request.admission_id,
                        request.settlement_id,
                        task.id,
                        request_sha256,
                        int(receipt.retired_contract_binding),
                        sqlite_records.format_datetime(receipt.settled_at),
                        encoded,
                    ),
                )
                if receipt.retired_contract_binding:
                    self._connection.execute(
                        "DELETE FROM cayu_task_session_execution_authority WHERE session_id = ? "
                        "AND authority_kind = 'contracted' AND NOT EXISTS ("
                        "SELECT 1 FROM cayu_tasks AS task WHERE task.session_id = ? AND task.work_contract_json IS NOT NULL "
                        "AND NOT EXISTS (SELECT 1 FROM cayu_work_attempt_lifecycle_receipts AS receipt "
                        "WHERE receipt.task_id = task.id AND receipt.retired_contract_binding = 1))",
                        (admission.session_id, admission.session_id),
                    )
                return receipt

    async def renew_work_attempt_execution_claim(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_execution_claim_request(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                claim = admission.claim
                if (
                    admission.state not in WORK_ATTEMPT_RENEWABLE_STATES
                    or claim.claim_id != request.claim_id
                    or claim.worker_id != request.worker_id
                    or claim.execution_owner_id != request.execution_owner_id
                    or claim.generation != request.generation
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Execution-claim renewal conflicts with current authority."
                    )
                now = self._ownership_clock()
                self._ensure_live_work_attempt_admission_claim(admission, now=now)
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_completion_proposals WHERE attempt_id = ?",
                        (admission.attempt_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Completion proposal has already closed execution authority."
                    )
                task = self._require_task_unlocked(admission.task_id)
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.session_id != admission.session_id
                    or task.session_instance_id != admission.session_invocation.session_instance_id
                    or task.worker_id != claim.worker_id
                    or task.lease_expires_at != claim.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Execution-claim renewal lost exact task ownership."
                    )
                renewed_claim = renewed_work_attempt_execution_claim(claim, request, now=now)
                renewed = WorkAttemptAdmission.model_validate(
                    admission.model_copy(update={"claim": renewed_claim}).model_dump(
                        mode="python", warnings=False
                    )
                )
                self._connection.execute(
                    "UPDATE cayu_work_attempt_execution_claims "
                    "SET lease_expires_at = ?, claim_json = ? "
                    "WHERE claim_id = ? AND is_current = 1",
                    (
                        sqlite_records.format_datetime(renewed_claim.lease_expires_at),
                        sqlite_records.json_dumps(
                            renewed_claim.model_dump(mode="json", warnings=False)
                        ),
                        renewed_claim.claim_id,
                    ),
                )
                self._update_work_attempt_admission_unlocked(renewed)
                self._update_task_snapshot_unlocked(
                    task.model_copy(
                        update={
                            "lease_expires_at": renewed_claim.lease_expires_at,
                            "updated_at": now,
                        }
                    )
                )
                return renewed.model_copy(deep=True)

    async def claim_work_attempt_recovery(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_execution_claim_request(request)
        request_sha256 = work_attempt_execution_claim_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                preparing = admission.state is WorkAttemptAdmissionState.PREPARING
                if not preparing and (
                    admission.attempt is None
                    or admission.state
                    not in {
                        WorkAttemptAdmissionState.ACTIVE,
                        WorkAttemptAdmissionState.RECOVERING,
                    }
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Only a prepared or published admission can enter recovery."
                    )
                current = admission.claim
                now = self._ownership_clock()
                exact_current_request = (
                    current.claim_id == request.claim_id
                    and current.worker_id == request.worker_id
                    and current.execution_owner_id == request.execution_owner_id
                    and current.generation == request.generation
                    and current.request_sha256 == request_sha256
                )
                task = self._require_task_unlocked(admission.task_id)
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.session_id != admission.session_id
                    or task.session_instance_id != admission.session_invocation.session_instance_id
                    or task.worker_id != current.worker_id
                    or task.lease_expires_at != current.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Recovery conflicts with current task ownership."
                    )
                if preparing and exact_current_request:
                    if current.lease_expires_at <= now:
                        raise WorkAttemptExecutionClaimLost(
                            "The prepared execution claim expired and must be replaced."
                        )
                    return admission.model_copy(deep=True)
                if admission.state is WorkAttemptAdmissionState.ACTIVE and exact_current_request:
                    if current.lease_expires_at <= now:
                        raise WorkAttemptExecutionClaimLost(
                            "The active execution claim expired and must be replaced."
                        )
                    return admission.model_copy(deep=True)
                if admission.state is WorkAttemptAdmissionState.RECOVERING:
                    if exact_current_request:
                        if current.lease_expires_at <= now:
                            raise WorkAttemptExecutionClaimLost(
                                "The recovery claim expired and must be replaced."
                            )
                        return admission.model_copy(deep=True)
                    if current.lease_expires_at > now:
                        raise WorkAttemptExecutionClaimLost(
                            "Another execution generation already owns live recovery."
                        )
                if current.lease_expires_at > now:
                    raise WorkAttemptExecutionClaimLost(
                        "The prior execution generation still owns a live lease."
                    )
                if request.generation != current.generation + 1:
                    raise WorkAttemptAdmissionConflict(
                        "Recovery must advance the execution generation exactly once."
                    )
                from cayu.storage._sqlite_task_groups import cancellation_requested_unlocked

                # Exact live replay above does not acquire new authority. A losing
                # execution must retain its original owner until positive settlement.
                if cancellation_requested_unlocked(self, admission.task_id):
                    raise WorkAttemptAdmissionConflict(
                        "Task-group cancellation forbids replacement execution authority."
                    )
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_completion_proposals WHERE attempt_id = ?",
                        (admission.attempt_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "A proposed attempt cannot acquire replacement execution authority."
                    )
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_execution_claims WHERE claim_id = ?",
                        (request.claim_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict("Execution-claim identity is already bound.")
                replacement = WorkAttemptExecutionClaim(
                    admission_id=request.admission_id,
                    claim_id=request.claim_id,
                    worker_id=request.worker_id,
                    execution_owner_id=request.execution_owner_id,
                    generation=request.generation,
                    request_sha256=request_sha256,
                    claimed_at=now,
                    lease_expires_at=now + timedelta(seconds=request.lease_seconds),
                )
                recovering = WorkAttemptAdmission.model_validate(
                    admission.model_copy(
                        update={
                            "state": (
                                WorkAttemptAdmissionState.PREPARING
                                if preparing
                                else WorkAttemptAdmissionState.RECOVERING
                            ),
                            "claim": replacement,
                            "recovery_evidence_sha256": None,
                        }
                    ).model_dump(mode="python", warnings=False)
                )
                retired = self._connection.execute(
                    "UPDATE cayu_work_attempt_execution_claims SET is_current = 0 "
                    "WHERE admission_id = ? AND is_current = 1",
                    (admission.admission_id,),
                )
                if retired.rowcount != 1:
                    raise WorkAttemptAdmissionConflict(
                        "Recovery could not retire the prior execution claim."
                    )
                self._insert_work_attempt_execution_claim_unlocked(replacement)
                self._update_work_attempt_admission_unlocked(recovering)
                self._update_task_snapshot_unlocked(
                    task.model_copy(
                        update={
                            "worker_id": request.worker_id,
                            "lease_expires_at": replacement.lease_expires_at,
                            "updated_at": now,
                        }
                    )
                )
                return recovering.model_copy(deep=True)

    async def activate_work_attempt_recovery(
        self,
        request: WorkAttemptRecoveryActivate,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_recovery_activate(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                if admission.state is WorkAttemptAdmissionState.ACTIVE:
                    if not (
                        admission.claim.claim_id == request.claim_id
                        and admission.claim.generation == request.generation
                        and admission.recovery_evidence_sha256 == request.recovery_evidence_sha256
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "Recovery activation conflicts with current active authority."
                        )
                elif (
                    admission.state is not WorkAttemptAdmissionState.RECOVERING
                    or admission.claim.claim_id != request.claim_id
                    or admission.claim.generation != request.generation
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Recovery activation no longer owns the replacement claim."
                    )
                task = self._require_task_unlocked(admission.task_id)
                if (
                    task.session_id != admission.session_id
                    or task.session_instance_id != admission.session_invocation.session_instance_id
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Recovery activation conflicts with exact task-session authority."
                    )
                if admission.state is WorkAttemptAdmissionState.ACTIVE:
                    return admission.model_copy(deep=True)
                now = self._ownership_clock()
                self._ensure_live_work_attempt_admission_claim(admission, now=now)
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.worker_id != admission.claim.worker_id
                    or task.lease_expires_at != admission.claim.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Recovery activation conflicts with current task ownership."
                    )
                if admission.attempt is None:
                    raise WorkAttemptAdmissionConflict(
                        "Recovery activation requires a published work attempt."
                    )
                contract = self._load_work_contract_unlocked(admission.contract)
                verified_work_support.require_attempt_state_current(
                    task,
                    admission.attempt,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                active = WorkAttemptAdmission.model_validate(
                    admission.model_copy(
                        update={
                            "state": WorkAttemptAdmissionState.ACTIVE,
                            "recovery_evidence_sha256": request.recovery_evidence_sha256,
                        }
                    ).model_dump(mode="python", warnings=False)
                )
                self._update_work_attempt_admission_unlocked(active)
                return active.model_copy(deep=True)

    async def begin_work_attempt(self, request: WorkAttemptCreate) -> WorkAttempt:
        request = copy_work_attempt_create(request)
        request_sha256 = work_attempt_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                if (
                    self._load_work_attempt_admission_for_attempt_unlocked(request.attempt_id)
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Admitted work attempts are published only by admission activation."
                    )
                existing = self._load_work_attempt_unlocked(request.attempt_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Work-attempt identity is already bound to another request."
                        )
                    return existing.model_copy(deep=True)
                task = self._require_task_unlocked(request.task_id)
                governed_admission = self._connection.execute(
                    "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                    (task.id,),
                ).fetchone()
                if governed_admission is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Task is permanently governed by runtime-owned work-attempt admission."
                    )
                contract = self._require_task_contract_unlocked(task, request.contract)
                if task.status is not TaskStatus.RUNNING:
                    raise ValueError("Work attempts require a running contracted task.")
                if task.session_id != request.session_id:
                    raise WorkCompletionConflict(
                        "Work attempt is bound to a different task session."
                    )
                verified_work_support.require_attempt_worker(
                    task,
                    request.worker_id,
                    now=self._ownership_clock(),
                )
                prior_id = self._latest_work_attempt_id_unlocked(task.id)
                prior = None if prior_id is None else self._load_work_attempt_unlocked(prior_id)
                ordinal = 1 if prior is None else prior.ordinal + 1
                if ordinal > contract.continuation_policy.max_attempts:
                    raise WorkCompletionConflict(
                        "Work-contract attempt limit forbids another work attempt."
                    )
                if prior is not None:
                    row = self._connection.execute(
                        "SELECT decision.decision_id, receipt.decision_id AS applied_decision_id "
                        "FROM cayu_completion_proposals AS proposal "
                        "LEFT JOIN cayu_completion_decisions AS decision "
                        "ON decision.proposal_id = proposal.proposal_id "
                        "LEFT JOIN cayu_completion_decision_application_receipts AS receipt "
                        "ON receipt.decision_id = decision.decision_id "
                        "WHERE proposal.attempt_id = ?",
                        (prior.attempt_id,),
                    ).fetchone()
                    if row is None or row["decision_id"] is None:
                        raise WorkCompletionConflict(
                            "A prior work attempt has not reached a durable decision."
                        )
                    if row["applied_decision_id"] is None:
                        raise WorkCompletionConflict(
                            "A prior verifier decision has not reached durable task application."
                        )
                attempt = WorkAttempt(
                    attempt_id=request.attempt_id,
                    task_id=request.task_id,
                    session_id=request.session_id,
                    contract=request.contract,
                    execution_profile_fingerprint=request.execution_profile_fingerprint,
                    worker_id=request.worker_id,
                    ordinal=ordinal,
                    request_sha256=request_sha256,
                    started_at=self._clock(),
                )
                self._connection.execute(
                    "INSERT INTO cayu_work_attempts "
                    "(attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        attempt.attempt_id,
                        attempt.task_id,
                        attempt.ordinal,
                        attempt.request_sha256,
                        sqlite_records.format_datetime(attempt.started_at),
                        sqlite_records.json_dumps(attempt.model_dump(mode="json", warnings=False)),
                    ),
                )
                return attempt.model_copy(deep=True)

    async def load_work_attempt(self, attempt_id: str) -> WorkAttempt | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            attempt = self._load_work_attempt_unlocked(attempt_id)
            return None if attempt is None else attempt.model_copy(deep=True)

    async def submit_completion_proposal(
        self,
        request: CompletionProposalCreate,
    ) -> CompletionProposal:
        request = copy_completion_proposal_create(request)
        request_sha256 = completion_proposal_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                if (
                    self._load_work_attempt_admission_for_attempt_unlocked(request.attempt_id)
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Admitted work attempts require claim-fenced proposal publication."
                    )
                existing = self._load_completion_proposal_unlocked(request.proposal_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Completion-proposal identity is already bound to another request."
                        )
                    return existing.model_copy(deep=True)
                occupied = self._connection.execute(
                    "SELECT proposal_id FROM cayu_completion_proposals WHERE attempt_id = ?",
                    (request.attempt_id,),
                ).fetchone()
                if occupied is not None:
                    raise WorkCompletionConflict(
                        "Work attempt already has a different completion proposal."
                    )
                attempt = self._load_work_attempt_unlocked(request.attempt_id)
                if attempt is None:
                    raise KeyError(f"Work attempt not found: {request.attempt_id}")
                task = self._require_task_unlocked(attempt.task_id)
                contract = self._load_work_contract_unlocked(attempt.contract)
                verified_work_support.require_attempt_current(
                    task,
                    attempt,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                    now=self._ownership_clock(),
                )
                proposal = CompletionProposal(
                    proposal_id=request.proposal_id,
                    attempt_id=request.attempt_id,
                    result=request.result,
                    evidence_references=request.evidence_references,
                    task_id=attempt.task_id,
                    contract=attempt.contract,
                    request_sha256=request_sha256,
                    proposed_at=self._clock(),
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_proposals "
                    "(proposal_id, attempt_id, task_id, request_sha256, proposed_at, "
                    "proposal_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        proposal.proposal_id,
                        proposal.attempt_id,
                        proposal.task_id,
                        proposal.request_sha256,
                        sqlite_records.format_datetime(proposal.proposed_at),
                        sqlite_records.json_dumps(proposal.model_dump(mode="json", warnings=False)),
                    ),
                )
                return proposal.model_copy(deep=True)

    async def submit_admitted_completion_proposal(
        self,
        request: AdmittedCompletionProposalRequest,
    ) -> CompletionProposal:
        from cayu.storage._sqlite_task_groups import cancellation_requested_unlocked

        request = copy_admitted_completion_proposal_request(request)
        proposal_request = request.proposal
        proposal_sha256 = completion_proposal_request_sha256(proposal_request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                exact_claim_authority = (
                    admission.attempt is not None
                    and admission.attempt_id == proposal_request.attempt_id
                    and admission.claim.claim_id == request.claim_id
                    and admission.claim.generation == request.generation
                )
                if not exact_claim_authority:
                    raise WorkAttemptExecutionClaimLost(
                        "Completion proposal no longer owns the exact active admission."
                    )
                existing = self._load_completion_proposal_unlocked(proposal_request.proposal_id)
                if admission.state is WorkAttemptAdmissionState.RELEASED:
                    prior = self._connection.execute(
                        "SELECT proposal_id FROM cayu_completion_proposals WHERE attempt_id = ?",
                        (admission.attempt_id,),
                    ).fetchone()
                    if (
                        existing is None
                        or existing.request_sha256 != proposal_sha256
                        or prior is None
                        or prior["proposal_id"] != proposal_request.proposal_id
                    ):
                        raise WorkCompletionConflict(
                            "Released admission conflicts with the requested proposal replay."
                        )
                    return existing.model_copy(deep=True)
                if (
                    admission.state is not WorkAttemptAdmissionState.ACTIVE
                    or admission.claim.execution_owner_id != request.execution_owner_id
                    or admission.execution_stop is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Completion proposal no longer owns the exact active admission."
                    )
                lease_now = self._ownership_clock()
                self._ensure_live_work_attempt_admission_claim(admission, now=lease_now)
                if existing is not None:
                    if existing.request_sha256 != proposal_sha256:
                        raise WorkCompletionConflict(
                            "Completion-proposal identity is bound to another request."
                        )
                    return existing.model_copy(deep=True)
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_completion_proposals WHERE attempt_id = ?",
                        (admission.attempt_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkCompletionConflict(
                        "Work attempt already has a different completion proposal."
                    )
                task = self._require_task_unlocked(admission.task_id)
                contract = self._load_work_contract_unlocked(admission.contract)
                if cancellation_requested_unlocked(self, task.id):
                    raise WorkAttemptAdmissionConflict(
                        "A decided group loser cannot submit a new proposal."
                    )
                verified_work_support.require_attempt_state_current(
                    task,
                    admission.attempt,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                if (
                    task.session_instance_id != admission.session_invocation.session_instance_id
                    or task.worker_id != admission.claim.worker_id
                    or task.lease_expires_at != admission.claim.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Completion proposal lost exact task-worker lease ownership."
                    )
                proposal = CompletionProposal(
                    proposal_id=proposal_request.proposal_id,
                    attempt_id=proposal_request.attempt_id,
                    result=proposal_request.result,
                    evidence_references=proposal_request.evidence_references,
                    task_id=admission.task_id,
                    contract=admission.contract,
                    request_sha256=proposal_sha256,
                    proposed_at=self._clock(),
                )
                released = WorkAttemptAdmission.model_validate(
                    admission.model_copy(
                        update={"state": WorkAttemptAdmissionState.RELEASED}
                    ).model_dump(mode="python", warnings=False)
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_proposals "
                    "(proposal_id, attempt_id, task_id, request_sha256, proposed_at, "
                    "proposal_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        proposal.proposal_id,
                        proposal.attempt_id,
                        proposal.task_id,
                        proposal.request_sha256,
                        sqlite_records.format_datetime(proposal.proposed_at),
                        sqlite_records.json_dumps(proposal.model_dump(mode="json", warnings=False)),
                    ),
                )
                self._update_work_attempt_admission_unlocked(released)
                self._update_task_snapshot_unlocked(
                    task.model_copy(
                        update={
                            "worker_id": None,
                            "lease_expires_at": None,
                            "updated_at": lease_now,
                        }
                    )
                )
                return proposal.model_copy(deep=True)

    async def load_completion_proposal(self, proposal_id: str) -> CompletionProposal | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            proposal = self._load_completion_proposal_unlocked(proposal_id)
            return None if proposal is None else proposal.model_copy(deep=True)

    async def load_completion_proposal_for_attempt(
        self, attempt_id: str
    ) -> CompletionProposal | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            row = self._connection.execute(
                "SELECT proposal_id FROM cayu_completion_proposals WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                return None
            proposal = self._load_completion_proposal_unlocked(row["proposal_id"])
            if proposal is None or proposal.attempt_id != attempt_id:
                raise WorkCompletionConflict("Proposal index conflicts with its attempt.")
            return proposal.model_copy(deep=True)

    def _load_prior_completion_verifier_profile_unlocked(
        self,
        proposal: CompletionProposal,
    ) -> CompletionVerifierProfileRecord | None:
        row = self._connection.execute(
            "SELECT prior_proposal.proposal_id "
            "FROM cayu_work_attempts AS current_attempt "
            "JOIN cayu_work_attempts AS prior_attempt "
            "ON prior_attempt.task_id = current_attempt.task_id "
            "AND prior_attempt.ordinal = current_attempt.ordinal - 1 "
            "LEFT JOIN cayu_completion_proposals AS prior_proposal "
            "ON prior_proposal.attempt_id = prior_attempt.attempt_id "
            "WHERE current_attempt.attempt_id = ?",
            (proposal.attempt_id,),
        ).fetchone()
        if row is None:
            return None
        if row["proposal_id"] is None:
            raise WorkCompletionConflict("Prior work attempt has no completion proposal authority.")
        profile = self._load_completion_verifier_profile_unlocked(row["proposal_id"])
        if profile is None:
            raise WorkCompletionConflict("Prior work attempt has no verifier-profile authority.")
        return profile

    async def prepare_completion_verifier_profile(
        self,
        request: CompletionVerifierProfilePreparationRequest,
    ) -> CompletionVerifierProfileRecord:
        request = copy_completion_verifier_profile_preparation_request(request)
        request_sha256 = completion_verifier_profile_preparation_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_completion_verifier_profile_unlocked(request.proposal_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Completion-verifier profile is already bound to another request."
                        )
                    return copy_completion_verifier_profile_record(existing)
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                contract = verified_work_support.require_contract_reference(
                    self._load_work_contract_unlocked(proposal.contract),
                    proposal.contract,
                )
                if (
                    request.task_id != proposal.task_id
                    or request.attempt_id != attempt.attempt_id
                    or request.attempt_request_sha256 != attempt.request_sha256
                    or request.source_execution_profile_fingerprint
                    != attempt.execution_profile_fingerprint
                    or request.proposal_request_sha256 != proposal.request_sha256
                    or request.contract != contract.reference()
                    or request.profile.verifier != contract.verifier
                ):
                    raise WorkCompletionConflict(
                        "Completion-verifier profile conflicts with its durable proposal authority."
                    )
                prior = self._load_prior_completion_verifier_profile_unlocked(proposal)
                require_completion_verifier_profile_transition(request, prior)
                adoption = request.adoption
                if (
                    adoption is not None
                    and self._load_completion_verifier_adoption_unlocked(
                        task_id=request.task_id,
                        idempotency_key=adoption.idempotency_key,
                    )
                    is not None
                ):
                    raise WorkCompletionConflict(
                        "Completion-verifier profile adoption idempotency key is already "
                        "bound to another proposal."
                    )
                record = completion_verifier_profile_record_from_preparation(
                    request,
                    request_sha256=request_sha256,
                    prepared_at=self._clock(),
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_verifier_profiles "
                    "(proposal_id, task_id, attempt_id, profile_fingerprint, "
                    "request_sha256, prepared_at, profile_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.proposal_id,
                        record.task_id,
                        record.attempt_id,
                        record.profile.fingerprint,
                        record.request_sha256,
                        sqlite_records.format_datetime(record.prepared_at),
                        sqlite_records.json_dumps(record.model_dump(mode="json", warnings=False)),
                    ),
                )
                return copy_completion_verifier_profile_record(record)

    async def load_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            profile = self._load_completion_verifier_profile_unlocked(proposal_id)
            return None if profile is None else copy_completion_verifier_profile_record(profile)

    async def load_prior_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            proposal = self._load_completion_proposal_unlocked(proposal_id)
            if proposal is None:
                raise KeyError(f"Completion proposal not found: {proposal_id}")
            profile = self._load_prior_completion_verifier_profile_unlocked(proposal)
            return None if profile is None else copy_completion_verifier_profile_record(profile)

    async def claim_completion_verification(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        from cayu.storage._sqlite_task_groups import cancellation_requested_unlocked

        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                claim_by_id = self._load_completion_claim_by_id_unlocked(request.claim_id)
                if claim_by_id is not None and (
                    claim_by_id.proposal_id != request.proposal_id
                    or claim_by_id.request_sha256 != request_sha256
                ):
                    raise WorkCompletionConflict(
                        "Verification-claim identity is already bound to another request."
                    )
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                contract = self._load_work_contract_unlocked(proposal.contract)
                contract = verified_work_support.require_contract_reference(
                    contract,
                    proposal.contract,
                )
                if request.verifier != contract.verifier:
                    raise WorkCompletionConflict(
                        "Verification claim uses a verifier other than the frozen contract verifier."
                    )
                profile = self._load_completion_verifier_profile_unlocked(request.proposal_id)
                if (
                    profile is None
                    or profile.profile.fingerprint != request.verifier_profile_fingerprint
                ):
                    raise WorkCompletionConflict(
                        "Verification claim requires the exact prepared verifier profile."
                    )
                now = self._ownership_clock()
                current = self._load_completion_claim_unlocked(request.proposal_id)
                decision = self._load_completion_decision_for_proposal_unlocked(request.proposal_id)
                if decision is None and cancellation_requested_unlocked(self, proposal.task_id):
                    raise _GroupVerificationAdmissionRefused(
                        "A decided group loser cannot dispatch verification."
                    )
                if (
                    current is not None
                    and current.claim_id == request.claim_id
                    and current.request_sha256 == request_sha256
                ):
                    if current.lease_expires_at > now or decision is not None:
                        return current.model_copy(deep=True)
                    raise CompletionVerificationClaimLost(
                        "Verification claim expired and cannot regain authority by replay."
                    )
                if decision is not None:
                    raise WorkCompletionConflict(
                        "Completion proposal already has a durable decision."
                    )
                if current is not None and current.lease_expires_at > now:
                    raise CompletionVerificationClaimLost(
                        "Completion proposal is owned by another live verifier claim."
                    )
                if claim_by_id is not None:
                    raise CompletionVerificationClaimLost(
                        "Verification claim expired and cannot regain authority by replay."
                    )
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                task = self._require_task_unlocked(proposal.task_id)
                verified_work_support.require_proposal_chain(
                    proposal,
                    attempt,
                    task,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                attempt_number = 1 if current is None else current.attempt_number + 1
                claim = CompletionVerificationClaim(
                    claim_id=request.claim_id,
                    lease_seconds=request.lease_seconds,
                    proposal_id=request.proposal_id,
                    worker_id=request.worker_id,
                    execution_owner_id=request.execution_owner_id,
                    execution_timeout_seconds=request.execution_timeout_seconds,
                    verifier=request.verifier,
                    verifier_profile_fingerprint=request.verifier_profile_fingerprint,
                    attempt_number=attempt_number,
                    request_sha256=request_sha256,
                    claimed_at=now,
                    lease_expires_at=now + timedelta(seconds=request.lease_seconds),
                )
                self._connection.execute(
                    "UPDATE cayu_completion_verification_claims SET is_current = 0 "
                    "WHERE proposal_id = ? AND is_current = 1",
                    (request.proposal_id,),
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_verification_claims "
                    "(claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, "
                    "request_sha256, lease_expires_at, is_current, claim_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    (
                        claim.claim_id,
                        claim.proposal_id,
                        claim.attempt_number,
                        claim.verifier_profile_fingerprint,
                        claim.request_sha256,
                        sqlite_records.format_datetime(claim.lease_expires_at),
                        sqlite_records.json_dumps(claim.model_dump(mode="json", warnings=False)),
                    ),
                )
                return claim.model_copy(deep=True)

    async def load_completion_verification_claim(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            claim = self._load_completion_claim_unlocked(proposal_id)
            return None if claim is None else claim.model_copy(deep=True)

    async def renew_completion_verification_claim(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                current = self._load_completion_claim_unlocked(request.proposal_id)
                now = self._ownership_clock()
                if (
                    current is None
                    or current.claim_id != request.claim_id
                    or current.worker_id != request.worker_id
                    or current.execution_owner_id != request.execution_owner_id
                    or current.execution_timeout_seconds != request.execution_timeout_seconds
                    or current.verifier != request.verifier
                    or current.verifier_profile_fingerprint != request.verifier_profile_fingerprint
                    or current.request_sha256 != request_sha256
                    or current.lease_expires_at <= now
                    or self._load_completion_decision_for_proposal_unlocked(proposal.proposal_id)
                    is not None
                ):
                    raise CompletionVerificationClaimLost(
                        "Verification claim cannot be renewed without exact current live authority."
                    )
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                task = self._require_task_unlocked(proposal.task_id)
                contract = self._load_work_contract_unlocked(proposal.contract)
                verified_work_support.require_proposal_chain(
                    proposal,
                    attempt,
                    task,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                renewed = current.model_copy(
                    update={
                        "lease_expires_at": max(
                            current.lease_expires_at,
                            now + timedelta(seconds=request.lease_seconds),
                        )
                    }
                )
                self._connection.execute(
                    "UPDATE cayu_completion_verification_claims "
                    "SET lease_expires_at = ?, claim_json = ? "
                    "WHERE claim_id = ? AND is_current = 1",
                    (
                        sqlite_records.format_datetime(renewed.lease_expires_at),
                        sqlite_records.json_dumps(renewed.model_dump(mode="json", warnings=False)),
                        renewed.claim_id,
                    ),
                )
                return renewed.model_copy(deep=True)

    async def record_completion_decision(
        self,
        request: CompletionDecisionCreate,
    ) -> CompletionDecision:
        request = copy_completion_decision_create(request)
        request_sha256 = completion_decision_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_completion_decision_unlocked(request.decision_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Completion-decision identity is already bound to another request."
                        )
                    return existing.model_copy(deep=True)
                prior = self._load_completion_decision_for_proposal_unlocked(request.proposal_id)
                if prior is not None:
                    raise WorkCompletionConflict(
                        "Completion proposal already has a different durable decision."
                    )
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                claim = self._load_completion_claim_unlocked(proposal.proposal_id)
                profile = self._load_completion_verifier_profile_unlocked(proposal.proposal_id)
                lease_now = self._ownership_clock()
                evidence_now = self._clock()
                if (
                    claim is None
                    or claim.claim_id != request.claim_id
                    or claim.worker_id != request.worker_id
                    or claim.verifier != request.verifier
                    or claim.verifier_profile_fingerprint != request.verifier_profile_fingerprint
                    or profile is None
                    or profile.profile.fingerprint != request.verifier_profile_fingerprint
                    or claim.lease_expires_at <= lease_now
                ):
                    raise CompletionVerificationClaimLost(
                        "Completion decision requires the current live verifier claim."
                    )
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                task = self._require_task_unlocked(proposal.task_id)
                contract = self._load_work_contract_unlocked(proposal.contract)
                contract = verified_work_support.require_proposal_chain(
                    proposal,
                    attempt,
                    task,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                validate_completion_decision_contract(contract, request)
                decision = CompletionDecision(
                    decision_id=request.decision_id,
                    proposal_id=request.proposal_id,
                    claim_id=request.claim_id,
                    worker_id=request.worker_id,
                    verifier=request.verifier,
                    verifier_profile_fingerprint=request.verifier_profile_fingerprint,
                    decision_version=request.decision_version,
                    verdict=request.verdict,
                    criterion_outcomes=request.criterion_outcomes,
                    constraint_outcomes=request.constraint_outcomes,
                    gaps=request.gaps,
                    evidence_references=request.evidence_references,
                    task_id=proposal.task_id,
                    attempt_id=proposal.attempt_id,
                    contract=proposal.contract,
                    claim_authority_sha256=completion_verification_claim_authority_sha256(claim),
                    request_sha256=request_sha256,
                    gap_fingerprint=completion_gap_fingerprint(request),
                    decided_at=evidence_now,
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_decisions "
                    "(decision_id, proposal_id, task_id, attempt_id, claim_id, "
                    "verifier_profile_fingerprint, verdict, "
                    "gap_fingerprint, request_sha256, decided_at, decision_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        decision.decision_id,
                        decision.proposal_id,
                        decision.task_id,
                        decision.attempt_id,
                        decision.claim_id,
                        decision.verifier_profile_fingerprint,
                        decision.verdict.value,
                        decision.gap_fingerprint,
                        decision.request_sha256,
                        sqlite_records.format_datetime(decision.decided_at),
                        sqlite_records.json_dumps(decision.model_dump(mode="json", warnings=False)),
                    ),
                )
                return decision.model_copy(deep=True)

    async def load_completion_decision(
        self,
        decision_id: str,
    ) -> CompletionDecision | None:
        decision_id = require_clean_nonblank(decision_id, "decision_id")
        async with self._lock:
            decision = self._load_completion_decision_unlocked(decision_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def load_completion_decision_for_proposal(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            decision = self._load_completion_decision_for_proposal_unlocked(proposal_id)
            return None if decision is None else decision.model_copy(deep=True)

    @staticmethod
    def _completion_verifier_dispatch_from_row(row: sqlite3.Row) -> CompletionVerifierDispatch:
        dispatch = completion_verifier_dispatch_from_document(json.loads(row["record_json"]))
        if (
            dispatch.dispatch_id != row["dispatch_id"]
            or dispatch.proposal_id != row["proposal_id"]
            or dispatch.task_id != row["task_id"]
            or dispatch.ordinal != row["ordinal"]
            or dispatch.request_sha256 != row["request_sha256"]
            or (dispatch.settlement is not None) != bool(row["settled"])
        ):
            raise WorkCompletionConflict(
                "Stored completion-verifier dispatch indexes conflict with canonical content."
            )
        return dispatch

    def _load_completion_verifier_dispatch_unlocked(
        self,
        dispatch_id: str,
    ) -> CompletionVerifierDispatch | None:
        row = self._connection.execute(
            "SELECT dispatch_id, proposal_id, task_id, ordinal, request_sha256, settled, "
            "record_json FROM cayu_completion_verifier_dispatches WHERE dispatch_id = ?",
            (dispatch_id,),
        ).fetchone()
        return None if row is None else self._completion_verifier_dispatch_from_row(row)

    def _list_completion_verifier_dispatches_unlocked(
        self,
        proposal_id: str,
    ) -> tuple[CompletionVerifierDispatch, ...]:
        rows = self._connection.execute(
            "SELECT dispatch_id, proposal_id, task_id, ordinal, request_sha256, settled, "
            "record_json FROM cayu_completion_verifier_dispatches WHERE proposal_id = ? "
            "ORDER BY ordinal",
            (proposal_id,),
        ).fetchall()
        dispatches = tuple(self._completion_verifier_dispatch_from_row(row) for row in rows)
        if tuple(item.ordinal for item in dispatches) != tuple(range(1, len(dispatches) + 1)):
            raise WorkCompletionConflict("Stored completion-verifier dispatch order has gaps.")
        return dispatches

    async def record_completion_verifier_dispatch(
        self,
        request: CompletionVerifierDispatchRequest,
    ) -> CompletionVerifierDispatch:
        request = copy_completion_verifier_dispatch_request(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_completion_verifier_dispatch_unlocked(request.dispatch_id)
                if existing is not None:
                    return replay_completion_verifier_dispatch(existing, request)
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                task = self._require_task_unlocked(proposal.task_id)
                contract = verified_work_support.require_proposal_chain(
                    proposal,
                    attempt,
                    task,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=self._load_work_contract_unlocked(proposal.contract),
                )
                profile = self._load_completion_verifier_profile_unlocked(proposal.proposal_id)
                prior = self._list_completion_verifier_dispatches_unlocked(proposal.proposal_id)
                require_completion_verifier_dispatch_admission(
                    request,
                    proposal=proposal,
                    contract_verifier=contract.verifier,
                    claim=self._load_completion_claim_unlocked(proposal.proposal_id),
                    profile_fingerprint=None if profile is None else profile.profile.fingerprint,
                    decided=self._load_completion_decision_for_proposal_unlocked(
                        proposal.proposal_id
                    )
                    is not None,
                    existing=prior,
                    lease_now=self._ownership_clock(),
                )
                record = completion_verifier_dispatch_from_request(
                    request,
                    proposal=proposal,
                    ordinal=len(prior) + 1,
                    dispatched_at=self._clock(),
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_verifier_dispatches "
                    "(dispatch_id, proposal_id, task_id, ordinal, request_sha256, settled, "
                    "record_json) VALUES (?, ?, ?, ?, ?, 0, ?)",
                    (
                        record.dispatch_id,
                        record.proposal_id,
                        record.task_id,
                        record.ordinal,
                        record.request_sha256,
                        sqlite_records.json_dumps(completion_verifier_dispatch_document(record)),
                    ),
                )
                return copy_completion_verifier_dispatch(record)

    async def settle_completion_verifier_dispatch(
        self,
        request: CompletionVerifierDispatchSettlementRequest,
    ) -> CompletionVerifierDispatch:
        request = copy_completion_verifier_dispatch_settlement_request(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_completion_verifier_dispatch_unlocked(request.dispatch_id)
                if existing is None:
                    raise KeyError(f"Completion verifier dispatch not found: {request.dispatch_id}")
                updated, changed = settle_completion_verifier_dispatch_record(
                    existing,
                    request,
                    settled_at=self._clock(),
                )
                if changed:
                    cursor = self._connection.execute(
                        "UPDATE cayu_completion_verifier_dispatches "
                        "SET settled = 1, record_json = ? "
                        "WHERE dispatch_id = ? AND settled = 0",
                        (
                            sqlite_records.json_dumps(
                                completion_verifier_dispatch_document(updated)
                            ),
                            updated.dispatch_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise WorkCompletionConflict(
                            "Completion verifier dispatch settlement lost its record."
                        )
                return copy_completion_verifier_dispatch(updated)

    async def list_completion_verifier_dispatches(
        self,
        proposal_id: str,
    ) -> tuple[CompletionVerifierDispatch, ...]:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            return self._list_completion_verifier_dispatches_unlocked(proposal_id)

    @staticmethod
    def _completion_evaluation_run_from_row(row: sqlite3.Row) -> CompletionEvaluationRun:
        run = completion_evaluation_run_from_document(json.loads(row["record_json"]))
        if (
            run.effect_id != row["effect_id"]
            or run.proposal_id != row["proposal_id"]
            or run.task_id != row["task_id"]
            or run.run_ordinal != row["run_ordinal"]
            or run.request_sha256 != row["request_sha256"]
            or (run.settlement is not None) != bool(row["settled"])
        ):
            raise WorkCompletionConflict(
                "Stored completion-evaluation indexes conflict with canonical content."
            )
        return run

    def _load_completion_evaluation_run_unlocked(
        self,
        effect_id: str,
    ) -> CompletionEvaluationRun | None:
        row = self._connection.execute(
            "SELECT effect_id, proposal_id, task_id, run_ordinal, request_sha256, settled, "
            "record_json FROM cayu_completion_evaluation_runs WHERE effect_id = ?",
            (effect_id,),
        ).fetchone()
        return None if row is None else self._completion_evaluation_run_from_row(row)

    def _list_completion_evaluation_runs_unlocked(
        self,
        proposal_id: str,
    ) -> tuple[CompletionEvaluationRun, ...]:
        rows = self._connection.execute(
            "SELECT effect_id, proposal_id, task_id, run_ordinal, request_sha256, settled, "
            "record_json FROM cayu_completion_evaluation_runs WHERE proposal_id = ? "
            "ORDER BY run_ordinal",
            (proposal_id,),
        ).fetchall()
        runs = tuple(self._completion_evaluation_run_from_row(row) for row in rows)
        if tuple(item.run_ordinal for item in runs) != tuple(range(1, len(runs) + 1)):
            raise WorkCompletionConflict("Stored completion-evaluation run order has gaps.")
        return runs

    async def record_completion_evaluation_run(
        self,
        request: CompletionEvaluationRunRequest,
    ) -> CompletionEvaluationRun:
        request = copy_completion_evaluation_run_request(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_completion_evaluation_run_unlocked(request.effect_id)
                if existing is not None:
                    return replay_completion_evaluation_run(existing, request)
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                task = self._require_task_unlocked(proposal.task_id)
                contract = verified_work_support.require_proposal_chain(
                    proposal,
                    attempt,
                    task,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=self._load_work_contract_unlocked(proposal.contract),
                )
                prior = self._list_completion_evaluation_runs_unlocked(proposal.proposal_id)
                require_completion_evaluation_admission(
                    request,
                    proposal=proposal,
                    contract_evaluation=contract.evaluation,
                    claim=self._load_completion_claim_unlocked(proposal.proposal_id),
                    decided=self._load_completion_decision_for_proposal_unlocked(
                        proposal.proposal_id
                    )
                    is not None,
                    existing=prior,
                    lease_now=self._ownership_clock(),
                )
                record = completion_evaluation_run_from_request(
                    request, proposal=proposal, started_at=self._clock()
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_evaluation_runs "
                    "(effect_id, proposal_id, task_id, run_ordinal, request_sha256, settled, "
                    "record_json) VALUES (?, ?, ?, ?, ?, 0, ?)",
                    (
                        record.effect_id,
                        record.proposal_id,
                        record.task_id,
                        record.run_ordinal,
                        record.request_sha256,
                        sqlite_records.json_dumps(completion_evaluation_run_document(record)),
                    ),
                )
                return copy_completion_evaluation_run(record)

    async def settle_completion_evaluation_run(
        self,
        request: CompletionEvaluationSettlementRequest,
    ) -> CompletionEvaluationRun:
        request = copy_completion_evaluation_settlement_request(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_completion_evaluation_run_unlocked(request.effect_id)
                if existing is None:
                    raise KeyError(f"Completion evaluation run not found: {request.effect_id}")
                updated, changed = settle_completion_evaluation_run_record(
                    existing, request, settled_at=self._clock()
                )
                if changed:
                    cursor = self._connection.execute(
                        "UPDATE cayu_completion_evaluation_runs "
                        "SET settled = 1, record_json = ? "
                        "WHERE effect_id = ? AND settled = 0",
                        (
                            sqlite_records.json_dumps(completion_evaluation_run_document(updated)),
                            updated.effect_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise WorkCompletionConflict(
                            "Completion evaluation settlement lost its record."
                        )
                return copy_completion_evaluation_run(updated)

    async def list_completion_evaluation_runs(
        self,
        proposal_id: str,
    ) -> tuple[CompletionEvaluationRun, ...]:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            return self._list_completion_evaluation_runs_unlocked(proposal_id)

    async def apply_completion_decision(
        self,
        request: CompletionDecisionApplicationRequest,
    ) -> Task:
        try:
            copied_request = copy_completion_decision_application_request(request)
        except BaseException:
            del request
            raise
        request = copied_request
        del copied_request
        request_sha256 = completion_decision_application_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                receipt = self._load_decision_application_receipt_unlocked(
                    request.task_id,
                    request.idempotency_key,
                )
                if receipt is not None:
                    if receipt.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Decision-application identity is already bound to another request."
                        )
                    return receipt.task.model_copy(deep=True)
                prior = self._connection.execute(
                    "SELECT task_id, idempotency_key FROM "
                    "cayu_completion_decision_application_receipts WHERE decision_id = ?",
                    (request.decision_id,),
                ).fetchone()
                if prior is not None:
                    raise WorkCompletionConflict(
                        "Completion decision was already applied under another identity."
                    )
                task = self._require_task_unlocked(request.task_id)
                decision = self._load_completion_decision_unlocked(request.decision_id)
                if decision is None:
                    raise KeyError(f"Completion decision not found: {request.decision_id}")
                if decision.task_id != task.id:
                    raise WorkCompletionConflict("Completion decision belongs to another task.")
                contract = self._require_task_contract_unlocked(task, decision.contract)
                attempt = self._load_work_attempt_unlocked(decision.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion decision has no work attempt.")
                verified_work_support.require_decision_attempt_current(
                    task,
                    attempt,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                proposal = self._load_completion_proposal_unlocked(decision.proposal_id)
                if proposal is None:
                    raise WorkCompletionConflict("Completion decision has no completion proposal.")
                profile = self._load_completion_verifier_profile_unlocked(proposal.proposal_id)
                if (
                    profile is None
                    or profile.profile.fingerprint != decision.verifier_profile_fingerprint
                ):
                    raise WorkCompletionConflict(
                        "Completion decision has no exact verifier-profile authority."
                    )
                row = self._connection.execute(
                    "SELECT COUNT(*) AS matching FROM cayu_completion_decisions "
                    "WHERE task_id = ? AND verdict = ? AND gap_fingerprint = ?",
                    (task.id, CompletionVerdict.REJECTED.value, decision.gap_fingerprint),
                ).fetchone()
                matching_gap_count = 0 if row is None else int(row["matching"])
                updated, receipt = verified_work_support.plan_decision_application(
                    request,
                    request_sha256=request_sha256,
                    task=task,
                    decision=decision,
                    proposal=proposal,
                    attempt=attempt,
                    contract=contract,
                    matching_gap_count=matching_gap_count,
                    now=self._ownership_clock(),
                )
                if updated != task:
                    self._update_task_snapshot_unlocked(updated)
                    self._record_task_transition_unlocked(task, updated)
                self._connection.execute(
                    "INSERT INTO cayu_completion_decision_application_receipts "
                    "(task_id, idempotency_key, decision_id, request_sha256, applied_at, "
                    "receipt_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        receipt.task_id,
                        receipt.idempotency_key,
                        receipt.decision_id,
                        receipt.request_sha256,
                        sqlite_records.format_datetime(receipt.applied_at),
                        sqlite_records.json_dumps(receipt.model_dump(mode="json", warnings=False)),
                    ),
                )
                return updated.model_copy(deep=True)

    async def load_completion_decision_application_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        idempotency_key = validate_work_completion_idempotency_key(idempotency_key)
        async with self._lock:
            receipt = self._load_decision_application_receipt_unlocked(
                task_id,
                idempotency_key,
            )
            return None if receipt is None else receipt.model_copy(deep=True)

    @runtime_task_creation
    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        if request.schedule_policy is not None and not self.supports_task_scheduling:
            raise NotImplementedError("This store does not support managed task scheduling.")
        async with self._lock:
            with self._verified_transaction_unlocked():
                task_id = request.task_id or str(uuid4())
                if request.schedule_policy is not None:
                    existing = self._load_task_unlocked(task_id)
                    if existing is not None:
                        if (
                            existing.schedule is None
                            or existing.schedule.creation_sha256
                            != schedule_creation_digest(request)
                        ):
                            raise TaskScheduleConflict(
                                "Task schedule creation conflicts with retained intent."
                            )
                        return existing.model_copy(deep=True)
                parent = self._task_parent_for_create_unlocked(request, task_id=task_id)
                if request.work_contract is not None:
                    contract = self._load_work_contract_unlocked(request.work_contract)
                    verified_work_support.require_contract_reference(
                        contract,
                        request.work_contract,
                    )
                    if request.session_id is not None:
                        self._ensure_session_execution_authority_unlocked(
                            request.session_id,
                            "contracted",
                        )
                admission_now = self._clock()
                task = _task_from_create(
                    request,
                    task_id=task_id,
                    parent_task=parent,
                    retry_started_at=admission_now,
                    supports_verified_work_contracts=True,
                )
                self._insert_task_unlocked(task)
                self._record_task_transition_unlocked(None, task)
                created = task.model_copy(deep=True)
        self._publish_task_admission_wakeup(task, now=admission_now)
        return created

    def _record_task_transition_unlocked(
        self,
        prior: Task | None,
        current: Task,
        *,
        operation_id: str | None = None,
        settled_execution: tuple[str, str, datetime] | None = None,
    ) -> None:
        from cayu.storage._sqlite_task_graphs import record_transition

        record_transition(self, prior, current, settled_execution=settled_execution)
        self._record_schedule_transition_unlocked(prior, current, operation_id=operation_id)

    def _record_schedule_transition_unlocked(
        self, prior: Task | None, current: Task, *, operation_id: str | None = None
    ) -> None:
        if current.schedule is None:
            return
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM cayu_task_schedule_events WHERE task_id = ?",
            (current.id,),
        ).fetchone()
        events = schedule_transition_events(
            prior, current, first_sequence=row[0] + 1, operation_id=operation_id
        )
        self._connection.executemany(
            "INSERT INTO cayu_task_schedule_events (task_id, sequence, event_json) VALUES (?, ?, ?)",
            [
                (
                    event.task_id,
                    event.sequence,
                    sqlite_records.json_dumps(event.model_dump(mode="json")),
                )
                for event in events
            ],
        )

    async def reschedule_task(self, request: TaskRescheduleRequest) -> TaskScheduleReceipt:
        if type(request) is not TaskRescheduleRequest:
            raise TypeError("A typed task reschedule request is required.")
        request = revalidate_model_input(request, TaskRescheduleRequest)
        digest = schedule_mutation_digest(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                row = self._connection.execute(
                    "SELECT receipt_json FROM cayu_task_schedule_receipts "
                    "WHERE task_id = ? AND operation_id = ?",
                    (request.task_id, request.operation_id),
                ).fetchone()
                if row is not None:
                    retained = TaskScheduleReceipt.model_validate_json(row[0])
                    if retained.request_sha256 != digest:
                        raise TaskScheduleConflict(
                            "Schedule operation identity has different content."
                        )
                    return retained
                current = self._require_task_unlocked(request.task_id)
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_local_execution_attempts "
                        "WHERE task_id = ? AND retry_admissible = 0 LIMIT 1",
                        (current.id,),
                    ).fetchone()
                    is not None
                ):
                    raise TaskScheduleConflict("Task has unsettled execution authority.")
                now = self._clock()
                updated = rescheduled_task(current, request, now=now)
                receipt = schedule_receipt(
                    updated, request, now=now, kind=TaskScheduleEventType.RESCHEDULED
                )
                self._update_task_snapshot_unlocked(updated)
                self._record_task_transition_unlocked(
                    current, updated, operation_id=request.operation_id
                )
                self._connection.execute(
                    "INSERT INTO cayu_task_schedule_receipts "
                    "(task_id, operation_id, receipt_json) VALUES (?, ?, ?)",
                    (
                        request.task_id,
                        request.operation_id,
                        sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                    ),
                )
        self._publish_task_admission_broadcast()
        return receipt

    async def list_task_schedule_events(
        self, task_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskScheduleEvent]:
        task_id = require_clean_nonblank(task_id, "task_id")
        if type(after_sequence) is not int or not 0 <= after_sequence <= 9007199254740991:
            raise ValueError("after_sequence must be a bounded nonnegative integer.")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Schedule event limit must be between 1 and 1000.")
        async with self._lock:
            rows = self._connection.execute(
                "SELECT event_json FROM cayu_task_schedule_events "
                "WHERE task_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (task_id, after_sequence, limit),
            ).fetchall()
            return [TaskScheduleEvent.model_validate_json(row[0]) for row in rows]

    async def cancel_scheduled_task(
        self, request: TaskScheduleCancelRequest
    ) -> TaskScheduleReceipt:
        if type(request) is not TaskScheduleCancelRequest:
            raise TypeError("A typed task schedule cancellation is required.")
        request = revalidate_model_input(request, TaskScheduleCancelRequest)
        digest = schedule_mutation_digest(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                row = self._connection.execute(
                    "SELECT receipt_json FROM cayu_task_schedule_receipts "
                    "WHERE task_id = ? AND operation_id = ?",
                    (request.task_id, request.operation_id),
                ).fetchone()
                if row is not None:
                    retained = TaskScheduleReceipt.model_validate_json(row[0])
                    if retained.request_sha256 != digest:
                        raise TaskScheduleConflict(
                            "Schedule operation identity has different content."
                        )
                    return retained
                prior = self._require_task_unlocked(request.task_id)
                state = require_schedule_mutation(prior, request.expected_revision)
                updated = self._finish_task_in_transaction_unlocked(
                    prior.id, TaskStatus.CANCELLED, result=None, error=None
                )
                updated = updated.model_copy(
                    update={
                        "schedule": state.model_copy(
                            update={"revision": schedule_revision_after(state)}
                        )
                    }
                )
                self._update_task_snapshot_unlocked(updated)
                # Native retry cancellation may have persisted a receipt in this
                # same transaction. Its task must name the accepted schedule revision.
                if updated.retry_series is not None and updated.status is TaskStatus.CANCELLED:
                    assert updated.status_payload is not None
                    settlement_key = updated.status_payload["settlement_idempotency_key"]
                    row = self._connection.execute(
                        "SELECT receipt_json FROM cayu_task_retry_settlements "
                        "WHERE task_id = ? AND idempotency_key = ?",
                        (updated.id, settlement_key),
                    ).fetchone()
                    if row is None:
                        raise TaskScheduleConflict("Retry cancellation has no settlement evidence.")
                    settled = TaskRetrySettlementResult.model_validate_json(row[0])
                    if settled.task.model_copy(update={"schedule": updated.schedule}) != updated:
                        raise TaskScheduleConflict("Retry settlement has contradictory authority.")
                    settled = settled.model_copy(update={"task": updated})
                    self._connection.execute(
                        "UPDATE cayu_task_retry_settlements SET receipt_json = ? "
                        "WHERE task_id = ? AND idempotency_key = ?",
                        (
                            sqlite_records.json_dumps(settled.model_dump(mode="json")),
                            updated.id,
                            settlement_key,
                        ),
                    )
                receipt = schedule_receipt(
                    updated,
                    request,
                    now=updated.updated_at,
                    kind=TaskScheduleEventType.CANCELLED
                    if updated.status is TaskStatus.CANCELLED
                    else TaskScheduleEventType.CANCELLATION_REQUESTED,
                )
                self._record_task_transition_unlocked(
                    prior, updated, operation_id=request.operation_id
                )
                self._connection.execute(
                    "INSERT INTO cayu_task_schedule_receipts (task_id, operation_id, receipt_json) "
                    "VALUES (?, ?, ?)",
                    (
                        updated.id,
                        request.operation_id,
                        sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                    ),
                )
                return receipt

    async def next_task_schedule_wakeup(self, query: TaskQuery | None = None) -> TaskScheduleWakeup:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._clock()
                if query.status is not None and query.status is not TaskStatus.PENDING:
                    return TaskScheduleWakeup(as_of=now)
                clauses, params = self._task_filter_clauses(
                    query.model_copy(update={"status": None})
                )
                scope = " AND ".join(
                    [
                        "session_id IS NULL",
                        "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                        "WHERE attempt.retry_admissible = 0 AND (attempt.task_id = cayu_tasks.id OR "
                        "(cayu_tasks.retry_series_json IS NOT NULL AND attempt.retry_series_id = "
                        "json_extract(cayu_tasks.retry_series_json, '$.series_id'))))",
                        *clauses,
                    ]
                )
                stamp = sqlite_records.format_datetime(now)
                due = self._connection.execute(
                    f"SELECT MIN(available_at) FROM cayu_tasks WHERE {scope} "
                    "AND status = 'pending' AND available_at > ? "
                    "AND (schedule_json IS NULL OR json_extract(schedule_json, '$.admitted_at') IS NULL) "
                    "AND (retry_series_json IS NULL OR json_extract(retry_series_json, '$.elapsed_deadline') IS NULL "
                    "OR julianday(json_extract(retry_series_json, '$.elapsed_deadline')) > julianday(?))",
                    [*params, stamp, stamp],
                ).fetchone()[0]
                # JSON uses Z whereas column timestamps use +00:00. Normalize
                # the UTC suffix, including before MIN, without rounding away
                # microseconds through SQLite's date/time functions.
                expiry, maintenance = self._connection.execute(
                    "SELECT MIN(CASE WHEN replace(json_extract(schedule_json, '$.policy.expires_at'), 'Z', '+00:00') > ? "
                    "THEN replace(json_extract(schedule_json, '$.policy.expires_at'), 'Z', '+00:00') END), "
                    "COALESCE(MAX(CASE WHEN replace(json_extract(schedule_json, '$.policy.expires_at'), 'Z', '+00:00') <= ? "
                    "OR (json_extract(schedule_json, '$.policy.misfire_policy') = 'skip' AND "
                    "(julianday(?) - julianday(available_at)) * 86400 > "
                    "json_extract(schedule_json, '$.policy.misfire_grace_seconds')) THEN 1 ELSE 0 END), 0) "
                    f"FROM cayu_tasks WHERE {scope} "
                    "AND status IN ('pending', 'paused', 'blocked', 'needs_attention') "
                    "AND schedule_json IS NOT NULL AND json_extract(schedule_json, '$.admitted_at') IS NULL",
                    [stamp, stamp, stamp, *params],
                ).fetchone()
                return TaskScheduleWakeup(
                    as_of=now,
                    next_available_at=sqlite_records.parse_optional_datetime(due),
                    next_expiry_at=sqlite_records.parse_optional_datetime(expiry),
                    maintenance_required=bool(maintenance),
                )

    @runtime_task_creation
    async def create_running_task(
        self,
        request: TaskCreate,
        *,
        session_invocation: SessionInvocationBinding,
    ) -> Task:
        request = copy_task_create(request)
        session_binding = _copy_required_session_binding(session_invocation)
        async with self._lock:
            with self._verified_transaction_unlocked():
                task_id = request.task_id or str(uuid4())
                parent = self._task_parent_for_create_unlocked(request, task_id=task_id)
                if request.work_contract is not None:
                    contract = self._load_work_contract_unlocked(request.work_contract)
                    verified_work_support.require_contract_reference(
                        contract,
                        request.work_contract,
                    )
                    if request.session_id is not None:
                        self._ensure_session_execution_authority_unlocked(
                            request.session_id,
                            "contracted",
                        )
                task = _running_task_from_create(
                    request,
                    task_id=task_id,
                    parent_task=parent,
                    session_invocation=session_binding,
                    retry_started_at=self._clock(),
                    supports_verified_work_contracts=True,
                )
                self._insert_task_unlocked(task)
                return task.model_copy(deep=True)

    def _insert_task_unlocked(self, task: Task) -> None:
        from cayu.storage._sqlite_task_graphs import require_unreserved_identity

        require_unreserved_identity(self, task.id)
        try:
            self._connection.execute(
                """
                INSERT INTO cayu_tasks (
                    id,
                    type,
                    title,
                    description,
                    status,
                    session_id,
                    session_instance_id,
                    parent_task_id,
                    assigned_agent_name,
                    available_at,
                    worker_id,
                    lease_expires_at,
                    interrupted_handoff_id,
                    status_reason,
                    status_payload_json,
                    input_json,
                    result_json,
                    error_json,
                    metadata_json,
                    created_at,
                    updated_at,
                    started_at,
                    completed_at,
                    invocation_json,
                    retry_series_json,
                    work_contract_json,
                    schedule_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                sqlite_records.task_to_row_values(task),
            )
        except sqlite3.IntegrityError as exc:
            if self._task_exists_unlocked(task.id):
                raise ValueError(f"Task already exists: {task.id}") from exc
            raise

    async def load_task(self, task_id: str, *, _access_bounds=None) -> Task | None:
        if _access_bounds is None:
            from cayu.resource_access import current_data_bounds

            _access_bounds = await current_data_bounds("tasks")
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._load_task_unlocked(task_id)
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
        async with self._lock:
            return _require_active_attached_task_worker(
                self._require_task_unlocked(task_id),
                worker_id=worker_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
                now=self._ownership_clock(),
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
        async with self._lock:
            return _require_direct_attached_task_resume(
                self._require_task_unlocked(task_id),
                session_id=session_id,
                session_instance_id=session_instance_id,
            )

    async def load_invocation_snapshot(
        self,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            row = self._connection.execute(
                "SELECT id, session_id, session_instance_id, invocation_json "
                "FROM cayu_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            return TaskInvocationSnapshot(
                id=row["id"],
                session_id=row["session_id"],
                session_instance_id=row["session_instance_id"],
                invocation=TaskInvocation.model_validate(json.loads(row["invocation_json"])),
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

            access_sql, access_params = sql_predicate(_access_bounds, postgres=False)
            clauses.append(access_sql)
            params.extend(access_params)

        if query.q is not None:
            like = _like_contains_pattern(query.q)
            clauses.append(
                """
                (
                    id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR type COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR title COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR description COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR status COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR session_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR parent_task_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR assigned_agent_name COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR worker_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR status_reason COLLATE NOCASE LIKE ? ESCAPE '\\'
                )
                """
            )
            params.extend([like] * 10)
        if query.status is not None:
            clauses.append("status = ?")
            params.append(str(query.status))
        if query.type is not None:
            clauses.append("type = ?")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = ?")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = ?")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = ?")
            params.append(query.assigned_agent_name)

        if query.has_work_contract is not None:
            clauses.append(
                "work_contract_json IS NOT NULL"
                if query.has_work_contract
                else "work_contract_json IS NULL"
            )
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = sqlite_records.task_order_sql(query.order_by)
        params.extend([query.limit, query.offset])

        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM cayu_tasks
                {where_sql}
                ORDER BY {order_sql}, id ASC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            return [sqlite_records.task_from_row(row) for row in rows]

    async def load_session_closure_claim(self, session_id: str) -> TaskSessionClosureClaim | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            row = self._connection.execute(
                "SELECT plan_id, claim_json FROM cayu_task_session_closure_claims "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            claim = TaskSessionClosureClaim.model_validate_json(row["claim_json"])
            if claim.session_id != session_id or claim.plan_id != row["plan_id"]:
                raise ValueError("Task closure claim conflicts with its retained authority.")
            return claim

    async def claim_session_closure(
        self, claim: TaskSessionClosureClaim
    ) -> TaskSessionClosureClaim:
        claim = copy_task_session_closure_claim(claim)
        async with self._lock:
            with self._verified_transaction_unlocked():
                from cayu.storage._sqlite_task_graphs import require_deletion_ready

                require_deletion_ready(self, claim.task_ids)
                row = self._connection.execute(
                    "SELECT plan_id, claim_json FROM cayu_task_session_closure_claims "
                    "WHERE session_id = ?",
                    (claim.session_id,),
                ).fetchone()
                if row is not None:
                    existing = TaskSessionClosureClaim.model_validate_json(row["claim_json"])
                    if existing != claim or row["plan_id"] != claim.plan_id:
                        raise ValueError(
                            "Task closure claim conflicts with its retained authority."
                        )
                    return existing
                rows = self._connection.execute(
                    "SELECT id, status, worker_id, lease_expires_at FROM cayu_tasks "
                    "WHERE session_id = ? LIMIT ?",
                    (claim.session_id, len(claim.task_ids) + 1),
                ).fetchall()
                if {row["id"] for row in rows} != set(claim.task_ids):
                    raise ValueError("Task closure set changed before admission.")
                if any(
                    row["status"] not in {"completed", "failed", "cancelled", "dependency_skipped"}
                    or row["worker_id"] is not None
                    or row["lease_expires_at"] is not None
                    for row in rows
                ):
                    raise ValueError("Task closure requires quiescent terminal tasks.")
                self._connection.execute(
                    "INSERT INTO cayu_task_session_closure_claims "
                    "(session_id, plan_id, claim_json) VALUES (?, ?, ?)",
                    (claim.session_id, claim.plan_id, claim.model_dump_json()),
                )
                return claim

    async def delete_session_tasks(
        self,
        session_id: str,
        *,
        task_ids: tuple[str, ...],
        policy: Any,
    ) -> None:
        """Delete one session's terminal task graph in the store transaction.

        The closure coordinator supplies a bounded, revalidated identity set;
        this method owns the SQL dependency cleanup so receipts and attempt
        records cannot outlive their task rows.
        """
        from cayu.storage._session_closure_sql import (
            TASK_CLOSURE_DEPENDENCIES,
            task_closure_deletion_steps,
        )
        from cayu.storage._sqlite_task_graphs import require_deletion_ready

        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            with self._verified_transaction_unlocked():
                claim_row = self._connection.execute(
                    "SELECT plan_id, claim_json FROM cayu_task_session_closure_claims "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                claim = None
                if claim_row is not None:
                    claim = TaskSessionClosureClaim.model_validate_json(claim_row["claim_json"])
                    if (
                        claim.session_id != session_id
                        or claim.plan_id != claim_row["plan_id"]
                        or set(task_ids) != set(claim.task_ids)
                    ):
                        raise ValueError("Task deletion conflicts with the retained closure set.")
                if not task_ids:
                    return
                require_deletion_ready(self, task_ids)
                placeholders = ", ".join("?" for _ in task_ids)
                rows = self._connection.execute(
                    f"SELECT id, session_id, status, worker_id, lease_expires_at "
                    f"FROM cayu_tasks WHERE id IN ({placeholders})",
                    task_ids,
                ).fetchall()
                if (claim is None and {row["id"] for row in rows} != set(task_ids)) or any(
                    row["session_id"] != session_id for row in rows
                ):
                    raise ValueError("Task closure authority changed during deletion.")
                if any(
                    row["status"] not in {"completed", "failed", "cancelled", "dependency_skipped"}
                    or row["worker_id"] is not None
                    or row["lease_expires_at"] is not None
                    for row in rows
                ):
                    raise ValueError("Task closure requires quiescent terminal tasks.")
                if rows:
                    table_rows = self._connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                    discovered: set[tuple[str, str]] = set()
                    for table_row in table_rows:
                        table = table_row["name"]
                        if table == "cayu_tasks":
                            continue
                        foreign_keys = self._connection.execute(
                            f"PRAGMA foreign_key_list('{table.replace(chr(39), chr(39) * 2)}')"
                        ).fetchall()
                        for foreign_key in foreign_keys:
                            if foreign_key["table"] != "cayu_tasks":
                                continue
                            column = foreign_key["from"]
                            if (table, column) in TASK_CLOSURE_DEPENDENCIES:
                                discovered.add((table, column))
                    for table, column, parent in task_closure_deletion_steps(discovered):
                        quoted_table = table.replace(chr(34), chr(34) * 2)
                        quoted_column = column.replace(chr(34), chr(34) * 2)
                        if parent is None:
                            owned = f"({placeholders})"
                        else:
                            quoted_parent = parent.replace(chr(34), chr(34) * 2)
                            owned = (
                                f'(SELECT "{quoted_column}" FROM "{quoted_parent}" '
                                f"WHERE task_id IN ({placeholders}))"
                            )
                        self._connection.execute(
                            f'DELETE FROM "{quoted_table}" WHERE "{quoted_column}" IN {owned}',
                            task_ids,
                        )
                    self._connection.execute(
                        f"DELETE FROM cayu_tasks WHERE session_id = ? AND id IN ({placeholders})",
                        (session_id, *task_ids),
                    )

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology queries must be TaskTopologyQuery instances.")
        query = TaskTopologyQuery.model_validate(query.model_dump(mode="python"))
        session_branch_limits, child_branch_limits = _allocate_task_topology_branch_limits(query)

        def read_branch_candidates(
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
            branch_queries: list[str] = []
            branch_params: list[object] = []
            for branch_order, (branch_id, branch_limit) in enumerate(
                zip(branch_ids, branch_limits, strict=True)
            ):
                cursor = cursors.get(branch_id)
                if cursor is None:
                    cursor_clause = ""
                    cursor_params: list[object] = []
                else:
                    cursor_created_at, cursor_id = decode_task_topology_cursor(
                        cursor,
                        scope_kind=scope_kind,
                        scope_id=branch_id,
                    )
                    cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
                    formatted = sqlite_records.format_datetime(cursor_created_at)
                    cursor_params = [formatted, formatted, cursor_id]
                branch_queries.append(
                    f"""
                    SELECT branch_order, candidate.*
                    FROM (
                        SELECT ? AS branch_order, {sqlite_records.TASK_TOPOLOGY_COLUMNS}
                        FROM cayu_tasks
                        WHERE {scope_column} = ?
                          {cursor_clause}
                        ORDER BY created_at ASC, id ASC
                        LIMIT ?
                    ) AS candidate
                    """
                )
                branch_params.extend(
                    [
                        branch_order,
                        branch_id,
                        *cursor_params,
                        branch_limit + 1,
                    ]
                )
            rows = self._connection.execute(
                f"""
                {" UNION ALL ".join(branch_queries)}
                ORDER BY branch_order ASC, topology_created_at ASC, topology_id ASC
                """,
                branch_params,
            ).fetchall()
            for row in rows:
                candidates[row["branch_order"]].append(
                    sqlite_records.task_topology_node_from_row(row)
                )
            return candidates

        async with self._lock:
            self._connection.execute("BEGIN")
            try:
                observed_row = self._connection.execute(
                    "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
                ).fetchone()
                if observed_row is None:
                    raise RuntimeError("SQLite did not return a topology snapshot timestamp.")
                observed_at = sqlite_records.parse_datetime(observed_row[0])

                expanded_parents: list[TaskTopologyNode] = []
                if query.expanded_parent_ids:
                    placeholders = ", ".join("?" for _ in query.expanded_parent_ids)
                    rows = self._connection.execute(
                        f"""
                        SELECT {sqlite_records.TASK_TOPOLOGY_COLUMNS}
                        FROM cayu_tasks
                        WHERE id IN ({placeholders})
                        """,
                        query.expanded_parent_ids,
                    ).fetchall()
                    parents_by_id = {
                        row["topology_id"]: sqlite_records.task_topology_node_from_row(row)
                        for row in rows
                    }
                    for parent_id in query.expanded_parent_ids:
                        parent = parents_by_id.get(parent_id)
                        if parent is None:
                            raise KeyError(f"Task not found: {parent_id}")
                        expanded_parents.append(parent)

                session_candidates = read_branch_candidates(
                    branch_ids=query.linked_session_ids,
                    cursors=query.session_cursors,
                    scope_kind="session",
                    scope_column="session_id",
                    branch_limits=session_branch_limits,
                )
                child_candidates = read_branch_candidates(
                    branch_ids=query.expanded_parent_ids,
                    cursors=query.child_cursors,
                    scope_kind="parent_task",
                    scope_column="parent_task_id",
                    branch_limits=child_branch_limits,
                )

                async def load_parent_links(
                    task_ids: tuple[str, ...],
                ) -> dict[str, str | None]:
                    links: dict[str, str | None] = {}
                    for index in range(0, len(task_ids), 500):
                        batch = task_ids[index : index + 500]
                        placeholders = ", ".join("?" for _ in batch)
                        rows = self._connection.execute(
                            f"""
                            SELECT
                                id,
                                CASE
                                    WHEN parent_task_id IS NULL
                                      OR length(CAST(parent_task_id AS BLOB))
                                         <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                    THEN parent_task_id
                                END AS topology_parent_task_id,
                                parent_task_id IS NOT NULL
                                  AND length(CAST(parent_task_id AS BLOB))
                                      > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                    AS topology_parent_task_id_oversized
                            FROM cayu_tasks
                            WHERE id IN ({placeholders})
                            """,
                            batch,
                        ).fetchall()
                        for row in rows:
                            if row["topology_parent_task_id_oversized"]:
                                raise TaskTopologyInconsistent(
                                    "A task topology ancestor contains an oversized "
                                    "parent identifier."
                                )
                            links[row["id"]] = _bounded_optional_task_topology_parent_id(
                                row["topology_parent_task_id"]
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
                    observed_at=observed_at,
                    linked_session_ids=query.linked_session_ids,
                    session_branch_candidates=session_candidates,
                    session_branch_limits=session_branch_limits,
                    expanded_parents=expanded_parents,
                    child_branch_candidates=child_candidates,
                    child_branch_limits=child_branch_limits,
                    session_task_limit=query.session_task_limit,
                    child_limit=query.child_limit,
                )
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        filters = copy_task_aggregate_filter(filters)
        clauses, params = self._task_filter_clauses(task_query_from_aggregate_filter(filters))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        def query_snapshot(connection: sqlite3.Connection) -> TaskOperationalSnapshot:
            snapshot_as_of = sqlite_records.format_datetime(self._clock())
            rows = connection.execute(
                f"""
                WITH
                snapshot(as_of) AS (
                    SELECT ?
                ),
                matching_tasks AS (
                    SELECT id, status, session_id, available_at, retry_series_json
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
                        COALESCE(SUM(
                            CASE
                                 WHEN status = 'pending'
                                 AND session_id IS NULL
                                 AND (available_at IS NULL OR available_at <= snapshot.as_of)
                                 AND NOT EXISTS (
                                     SELECT 1
                                     FROM cayu_local_execution_attempts AS attempt
                                     WHERE attempt.retry_admissible = 0
                                       AND (
                                           attempt.task_id = matching_tasks.id
                                           OR (
                                               matching_tasks.retry_series_json IS NOT NULL
                                               AND attempt.retry_series_id = json_extract(
                                                   matching_tasks.retry_series_json,
                                                   '$.series_id'
                                               )
                                           )
                                       )
                                 )
                                THEN 1 ELSE 0
                            END
                        ), 0) AS claimable_pending_count,
                        COALESCE(SUM(
                            CASE
                                WHEN status = 'pending'
                                 AND available_at > snapshot.as_of
                                THEN 1 ELSE 0
                            END
                        ), 0) AS scheduled_pending_count
                    FROM matching_tasks
                    CROSS JOIN snapshot
                )
                SELECT
                    snapshot.as_of,
                    status_counts.status,
                    status_counts.status_count,
                    pending_counts.claimable_pending_count,
                    pending_counts.scheduled_pending_count
                FROM snapshot
                CROSS JOIN pending_counts
                LEFT JOIN status_counts ON TRUE
                """,
                (snapshot_as_of, *params),
            ).fetchall()
            counts = {status: 0 for status in TaskStatus}
            for row in rows:
                if row["status"] is not None:
                    status = TaskStatus(row["status"])
                    counts[status] = row["status_count"]
            return TaskOperationalSnapshot(
                as_of=sqlite_records.parse_datetime(rows[0]["as_of"]),
                total_count=sum(counts.values()),
                counts_by_status=TaskStatusCounts.model_validate(counts),
                claimable_pending_count=rows[0]["claimable_pending_count"],
                scheduled_pending_count=rows[0]["scheduled_pending_count"],
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

        return await _run_off_thread_with_connection_ownership(
            self._lock,
            self._connection,
            query_snapshot,
            interrupt_on_cancellation=True,
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
        async with self._lock:
            with self._verified_transaction_unlocked():
                task = self._require_task_unlocked(task_id)
                _ensure_retry_series_queue_attempt(task.retry_series)
                _ensure_can_transition(task, TaskStatus.RUNNING)
                effective_session_id = _task_session_id_for_start(
                    task_id=task_id,
                    stored_session_id=task.session_id,
                    requested_session_id=session_id,
                )
                if task.work_contract is not None:
                    self._require_task_contract_unlocked(task, task.work_contract)
                    if effective_session_id is None:
                        raise WorkCompletionConflict(
                            "Contracted tasks require a session binding before starting."
                        )
                    self._ensure_session_execution_authority_unlocked(
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
                now = self._ownership_clock()
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.RUNNING,
                        "session_id": effective_session_id,
                        "session_instance_id": session_instance_id,
                        "started_at": task.started_at or now,
                        "updated_at": now,
                    }
                )
                self._update_task_snapshot_unlocked(updated)
                self._record_task_transition_unlocked(task, updated)
                return updated.model_copy(deep=True)

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
        async with self._lock:
            with self._verified_transaction_unlocked():
                # Lease authority must be sampled only after BEGIN IMMEDIATE
                # has acquired SQLite's cross-process writer lock. A timestamp
                # captured before that wait could outlive the worker lease.
                now = self._ownership_clock()
                task = self._require_task_unlocked(task_id)
                _ensure_retry_series_queue_attempt(task.retry_series)
                if not _can_attach_claimed_task_state(
                    status=task.status,
                    session_id=task.session_id,
                    worker_id=task.worker_id,
                    lease_expires_at=task.lease_expires_at,
                    expected_worker_id=worker_id,
                    now=now,
                ):
                    self._raise_task_claim_attach_error(task_id, worker_id, now=now)
                if expected_lease is None:
                    raise TaskClaimLost("Task attachment requires its exact worker lease.")
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease,
                    now=now,
                )
                if task.work_contract is not None:
                    self._require_task_contract_unlocked(task, task.work_contract)
                    self._ensure_session_execution_authority_unlocked(
                        session_id,
                        "contracted",
                    )
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
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.RUNNING,
                        "session_id": session_id,
                        "session_instance_id": session_instance_id,
                        "started_at": task.started_at or now,
                        "updated_at": now,
                    }
                )
                self._update_task_snapshot_unlocked(updated)
                self._record_task_transition_unlocked(task, updated)
                return updated.model_copy(deep=True)

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
        async with (
            managed_task_lease_mutation(
                task_id=task_id,
                worker_id=worker_id,
                handoff_id=handoff_id,
                presented_lease_expires_at=lease_expires_at,
            ) as effective_lease,
            self._lock,
        ):
            return self._finish_task_unlocked(
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
        async with (
            managed_task_lease_mutation(
                task_id=task_id,
                worker_id=worker_id,
                handoff_id=handoff_id,
                presented_lease_expires_at=lease_expires_at,
            ) as effective_lease,
            self._lock,
        ):
            return self._finish_task_unlocked(
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
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                receipt_row = self._connection.execute(
                    "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                    "FROM cayu_task_terminalization_receipts "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.idempotency_key),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _sqlite_task_terminalization_receipt(
                        task_id=request.task_id,
                        idempotency_key=request.idempotency_key,
                        row=receipt_row,
                    )
                    replayed = _replay_task_terminalization_receipt(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    self._connection.commit()
                    return replayed

                task = self._require_task_unlocked(request.task_id)
                self._raise_if_governed_work_attempt_admission(
                    request.task_id,
                    "Admitted work attempts cannot use ordinary terminalization.",
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
                now = self._ownership_clock()
                _ensure_task_terminalization_lease_authority(task, request, now=now)
                _ensure_task_handoff_authority(task, request.handoff_id)

                _validate_ordinary_task_terminalization_against_cancellation(task, request)
                status = TaskStatus(request.kind.value)
                verified_work_support.require_contracted_completion_authority(
                    task,
                    status,
                )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        status_reason = NULL,
                        status_payload_json = NULL,
                        result_json = ?,
                        error_json = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        interrupted_handoff_id = NULL,
                        started_at = COALESCE(started_at, ?),
                        completed_at = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND status IN (?, ?)
                      AND worker_id = ?
                      AND interrupted_handoff_id IS ?
                      AND lease_expires_at IS NOT NULL
                      AND lease_expires_at > ?
                    """,
                    (
                        str(status),
                        (
                            None
                            if request.result is None
                            else sqlite_records.json_dumps(request.result)
                        ),
                        (
                            None
                            if request.error is None
                            else sqlite_records.json_dumps(request.error)
                        ),
                        sqlite_records.format_datetime(now),
                        sqlite_records.format_datetime(now),
                        sqlite_records.format_datetime(now),
                        request.task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        request.worker_id,
                        request.handoff_id,
                        sqlite_records.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_task_active_lease_error(
                        request.task_id,
                        request.worker_id,
                        now=now,
                    )
                terminal_task = self._require_task_unlocked(request.task_id)
                self._record_task_transition_unlocked(task, terminal_task)
                self._connection.execute(
                    "INSERT INTO cayu_task_terminalization_receipts "
                    "(task_id, idempotency_key, request_sha256, worker_id, "
                    "terminal_kind, task_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.idempotency_key,
                        request_sha256,
                        request.worker_id,
                        request.kind.value,
                        sqlite_records.json_dumps(terminal_task.model_dump(mode="json")),
                        sqlite_records.format_datetime(now),
                    ),
                )
                self._connection.commit()
                return terminal_task.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
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
        async with self._lock:
            row = self._connection.execute(
                "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                "FROM cayu_task_terminalization_receipts "
                "WHERE task_id = ? AND idempotency_key = ?",
                (task_id, idempotency_key),
            ).fetchone()
            if row is None:
                return None
            return _sqlite_task_terminalization_receipt(
                task_id=task_id,
                idempotency_key=idempotency_key,
                row=row,
            )

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
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                receipt_row = self._connection.execute(
                    "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                    "FROM cayu_task_terminalization_receipts "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.idempotency_key),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _sqlite_task_terminalization_receipt(
                        task_id=request.task_id,
                        idempotency_key=request.idempotency_key,
                        row=receipt_row,
                    )
                    replayed = _replay_task_terminalization_receipt(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    _ensure_recovered_attached_task_session(
                        replayed,
                        session_id=session_id,
                        session_instance_id=session_instance_id,
                    )
                    self._connection.commit()
                    return replayed

                task = self._require_task_unlocked(request.task_id)
                self._raise_if_governed_work_attempt_admission(
                    request.task_id,
                    "Admitted work attempts cannot use attached-task recovery terminalization.",
                )
                now = self._ownership_clock()
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
                terminal_task = self._finish_task_in_transaction_unlocked(
                    request.task_id,
                    TaskStatus.FAILED,
                    result=None,
                    error=request.error,
                    worker_id=None,
                )
                self._record_task_transition_unlocked(task, terminal_task)
                self._connection.execute(
                    "INSERT INTO cayu_task_terminalization_receipts "
                    "(task_id, idempotency_key, request_sha256, worker_id, "
                    "terminal_kind, task_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.idempotency_key,
                        request_sha256,
                        request.worker_id,
                        request.kind.value,
                        sqlite_records.json_dumps(terminal_task.model_dump(mode="json")),
                        sqlite_records.format_datetime(now),
                    ),
                )
                self._connection.commit()
                return terminal_task.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

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
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                receipt_row = self._connection.execute(
                    "SELECT request_sha256, request_json, task_json, committed_at "
                    "FROM cayu_task_interrupted_handoff_receipts "
                    "WHERE task_id = ? AND handoff_id = ?",
                    (request.task_id, request.handoff_id),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _sqlite_interrupted_task_handoff_receipt(
                        task_id=request.task_id,
                        handoff_id=request.handoff_id,
                        row=receipt_row,
                    )
                    replayed = _replay_interrupted_task_handoff_receipt(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                    )
                    self._connection.commit()
                    return replayed

                task = self._require_task_unlocked(request.task_id)
                self._raise_if_governed_work_attempt_admission(
                    request.task_id,
                    "Admitted work attempts do not use interrupted-task handoff release.",
                )
                now = self._ownership_clock()
                _require_interrupted_task_handoff_authority(
                    task,
                    request,
                    now=now,
                    recover_expired=recover_expired,
                )
                lease_comparison = "<=" if recover_expired else ">"
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET worker_id = NULL, lease_expires_at = NULL,
                        interrupted_handoff_id = ?, updated_at = ?
                    WHERE id = ?
                      AND status = ?
                      AND session_id = ?
                      AND session_instance_id = ?
                      AND worker_id = ?
                      AND lease_expires_at = ?
                      AND lease_expires_at {lease_comparison} ?
                    """,
                    (
                        request.handoff_id,
                        sqlite_records.format_datetime(now),
                        request.task_id,
                        str(TaskStatus.RUNNING),
                        request.session_id,
                        request.session_instance_id,
                        request.worker_id,
                        sqlite_records.format_datetime(request.lease_expires_at),
                        sqlite_records.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    raise TaskInterruptedHandoffConflict(
                        "Interrupted-task handoff lost its exact durable authority."
                    )
                released = self._require_task_unlocked(request.task_id)
                receipt = TaskInterruptedHandoffReceipt(
                    request=request,
                    request_sha256=request_sha256,
                    task=released,
                    committed_at=now,
                )
                self._connection.execute(
                    "INSERT INTO cayu_task_interrupted_handoff_receipts "
                    "(task_id, handoff_id, request_sha256, request_json, "
                    "task_json, committed_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.handoff_id,
                        request_sha256,
                        sqlite_records.json_dumps(request.model_dump(mode="json")),
                        sqlite_records.json_dumps(released.model_dump(mode="json")),
                        sqlite_records.format_datetime(now),
                    ),
                )
                self._connection.commit()
                return receipt
            except BaseException:
                self._connection.rollback()
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
        async with self._lock:
            row = self._connection.execute(
                "SELECT request_sha256, request_json, task_json, committed_at "
                "FROM cayu_task_interrupted_handoff_receipts "
                "WHERE task_id = ? AND handoff_id = ?",
                (task_id, handoff_id),
            ).fetchone()
            return (
                None
                if row is None
                else _sqlite_interrupted_task_handoff_receipt(
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
        after_params: tuple[str, ...] = ()
        if after is not None:
            after_timestamp = sqlite_records.format_datetime(after[0])
            after_clause = "AND (lease_expires_at > ? OR (lease_expires_at = ? AND id > ?)) "
            after_params = (after_timestamp, after_timestamp, after[1])
        async with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM cayu_tasks WHERE status = ? "
                "AND session_id IS NOT NULL AND session_instance_id IS NOT NULL "
                "AND worker_id IS NOT NULL AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at <= ? AND status_reason IS NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM cayu_work_attempt_admissions "
                "WHERE cayu_work_attempt_admissions.task_id = cayu_tasks.id"
                ") "
                f"{after_clause}"
                "ORDER BY lease_expires_at ASC, id ASC LIMIT ?",
                (
                    str(TaskStatus.RUNNING),
                    sqlite_records.format_datetime(self._ownership_clock()),
                    *after_params,
                    limit,
                ),
            ).fetchall()
            return [sqlite_records.task_from_row(row) for row in rows]

    async def load_expired_interrupted_task_handoff_candidate(
        self,
        task_id: str,
    ) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            row = self._connection.execute(
                "SELECT * FROM cayu_tasks WHERE id = ? AND status = ? "
                "AND session_id IS NOT NULL AND session_instance_id IS NOT NULL "
                "AND worker_id IS NOT NULL AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at <= ? AND status_reason IS NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM cayu_work_attempt_admissions "
                "WHERE cayu_work_attempt_admissions.task_id = cayu_tasks.id"
                ")",
                (
                    task_id,
                    str(TaskStatus.RUNNING),
                    sqlite_records.format_datetime(self._ownership_clock()),
                ),
            ).fetchone()
            return None if row is None else sqlite_records.task_from_row(row)

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
        handoff_id_sha256 = _interrupted_task_continuation_handoff_id_sha256(handoff_id)
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        after, scan_limit = prepare_interrupted_task_continuation_claim_page(
            after=after,
            limit=scan_limit,
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                lease_expires_at = now + timedelta(seconds=lease_seconds)
                prior_claim_row = self._connection.execute(
                    "SELECT task_id, worker_id "
                    "FROM cayu_task_interrupted_continuation_claims "
                    "WHERE handoff_id_sha256 = ?",
                    (handoff_id_sha256,),
                ).fetchone()
                if prior_claim_row is not None:
                    existing_row = self._connection.execute(
                        "SELECT * FROM cayu_tasks WHERE id = ?",
                        (prior_claim_row["task_id"],),
                    ).fetchone()
                    existing = (
                        None if existing_row is None else sqlite_records.task_from_row(existing_row)
                    )
                    if (
                        prior_claim_row["worker_id"] != worker_id
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
                    result = InterruptedTaskContinuationClaimPage(
                        task=existing,
                        next_after=(existing.created_at, existing.id),
                        scanned_candidates=0,
                        rejected_candidates=0,
                        replayed=True,
                        exhausted=False,
                    )
                    self._connection.commit()
                    return result
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_tasks WHERE interrupted_handoff_id = ? LIMIT 1",
                        (handoff_id,),
                    ).fetchone()
                    is not None
                ):
                    raise TaskClaimLost(
                        "Interrupted-task continuation claim generation is already in use."
                    )
                if query.status is not None and query.status is not TaskStatus.RUNNING:
                    result = InterruptedTaskContinuationClaimPage(
                        scanned_candidates=0,
                        rejected_candidates=0,
                        exhausted=True,
                    )
                    self._connection.commit()
                    return result
                cursor = (
                    None if after is None else (sqlite_records.format_datetime(after[0]), after[1])
                )
                after_sql = ""
                after_params: tuple[str, ...] = ()
                if cursor is not None:
                    after_sql = "AND (created_at > ? OR (created_at = ? AND id > ?)) "
                    after_params = (cursor[0], cursor[0], cursor[1])
                task_id_sql = "" if task_id is None else "AND id = ? "
                task_id_params: tuple[str, ...] = () if task_id is None else (task_id,)
                rows = self._connection.execute(
                    "SELECT * FROM cayu_tasks WHERE status = ? "
                    "AND session_id IS NOT NULL AND session_instance_id IS NOT NULL "
                    "AND status_reason IS NULL "
                    "AND worker_id IS NULL AND lease_expires_at IS NULL "
                    "AND interrupted_handoff_id IS NOT NULL "
                    f"{task_id_sql}"
                    f"{after_sql}"
                    "ORDER BY created_at ASC, id ASC LIMIT ?",
                    (
                        str(TaskStatus.RUNNING),
                        *task_id_params,
                        *after_params,
                        scan_limit,
                    ),
                ).fetchall()
                rejected = 0
                filtered = 0
                last_observed: Task | None = None
                for index, row in enumerate(rows):
                    observed = sqlite_records.task_from_row(row)
                    last_observed = observed
                    if not _task_matches_claim_filter(observed, query):
                        filtered += 1
                        continue
                    candidate_handoff_id = observed.interrupted_handoff_id
                    if candidate_handoff_id is None:
                        raise AssertionError("Continuation candidate lost its handoff generation.")
                    if (
                        self._connection.execute(
                            "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                            (observed.id,),
                        ).fetchone()
                        is not None
                    ):
                        rejected += 1
                        continue
                    receipt_row = self._connection.execute(
                        "SELECT request_sha256, request_json, task_json, committed_at "
                        "FROM cayu_task_interrupted_handoff_receipts "
                        "WHERE task_id = ? AND handoff_id = ?",
                        (observed.id, candidate_handoff_id),
                    ).fetchone()
                    try:
                        receipt = (
                            None
                            if receipt_row is None
                            else _sqlite_interrupted_task_handoff_receipt(
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
                    self._connection.execute(
                        "INSERT INTO cayu_task_interrupted_continuation_claims ("
                        "handoff_id_sha256, task_id, worker_id, claimed_at"
                        ") VALUES (?, ?, ?, ?)",
                        (
                            handoff_id_sha256,
                            observed.id,
                            worker_id,
                            sqlite_records.format_datetime(now),
                        ),
                    )
                    update_cursor = self._connection.execute(
                        "UPDATE cayu_tasks SET worker_id = ?, lease_expires_at = ?, "
                        "interrupted_handoff_id = ?, updated_at = ? "
                        "WHERE id = ? AND status = ? AND worker_id IS NULL "
                        "AND lease_expires_at IS NULL AND interrupted_handoff_id = ?",
                        (
                            worker_id,
                            sqlite_records.format_datetime(lease_expires_at),
                            handoff_id,
                            sqlite_records.format_datetime(now),
                            observed.id,
                            str(TaskStatus.RUNNING),
                            observed.interrupted_handoff_id,
                        ),
                    )
                    if update_cursor.rowcount != 1:
                        raise RuntimeError("SQLite continuation claim lost its locked candidate.")
                    claimed = self._require_task_unlocked(observed.id)
                    result = InterruptedTaskContinuationClaimPage(
                        task=claimed,
                        next_after=(observed.created_at, observed.id),
                        scanned_candidates=index + 1,
                        rejected_candidates=rejected,
                        filtered_candidates=filtered,
                        exhausted=index == len(rows) - 1 and len(rows) < scan_limit,
                    )
                    self._connection.commit()
                    return result
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
                self._connection.commit()
                return result
            except BaseException:
                self._connection.rollback()
                raise

    async def reconcile_task_cancellation(
        self,
        request: TaskCancellationReconciliationRequest,
    ) -> TaskCancellationReconciliationResult:
        request, request_sha256 = prepare_task_cancellation_reconciliation(request)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                rejection_row = self._connection.execute(
                    "SELECT request_sha256, record_json "
                    "FROM cayu_task_retry_reconciliation_rejections "
                    "WHERE task_id = ? AND reconciliation_idempotency_key = ?",
                    (request.task_id, request.reconciliation_idempotency_key),
                ).fetchone()
                if rejection_row is not None:
                    rejection = _TaskCancellationReconciliationRejectionRecord.model_validate(
                        json.loads(rejection_row["record_json"])
                    )
                    if rejection.request_sha256 != rejection_row["request_sha256"]:
                        raise RuntimeError(
                            "SQLite cancellation reconciliation rejection contains invalid "
                            "durable material."
                        )
                    raise _replay_task_cancellation_reconciliation_rejection(
                        request,
                        request_sha256=request_sha256,
                        record=rejection,
                    )

                receipt_row = self._connection.execute(
                    "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                    "FROM cayu_task_terminalization_receipts "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.cancellation_idempotency_key),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _sqlite_task_terminalization_receipt(
                        task_id=request.task_id,
                        idempotency_key=request.cancellation_idempotency_key,
                        row=receipt_row,
                    )
                    replayed = _replay_task_cancellation_reconciliation(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    self._connection.commit()
                    return replayed

                task = self._load_task_unlocked(request.task_id)
                if task is None:
                    raise _task_cancellation_reconciliation_conflict(
                        request,
                        "Task cancellation reconciliation task was not found.",
                    )
                self._raise_if_governed_work_attempt_admission(
                    request.task_id,
                    "Admitted work attempts cannot use ordinary cancellation reconciliation.",
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
                    self._connection.execute(
                        "INSERT INTO cayu_task_retry_reconciliation_rejections "
                        "(task_id, reconciliation_idempotency_key, request_sha256, "
                        "record_json, recorded_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            rejection.task_id,
                            rejection.reconciliation_idempotency_key,
                            rejection.request_sha256,
                            sqlite_records.json_dumps(rejection.model_dump(mode="json")),
                            sqlite_records.format_datetime(rejection.recorded_at),
                        ),
                    )
                    self._connection.commit()
                    raise _rejected_task_cancellation_reconciliation(rejection)

                result = _reconciled_task_cancellation(
                    task,
                    request,
                    request_sha256=request_sha256,
                    committed_at=now,
                )
                settled = result.task
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?, status_reason = NULL, status_payload_json = ?,
                        result_json = NULL, error_json = ?, worker_id = NULL,
                        lease_expires_at = NULL, interrupted_handoff_id = NULL,
                        started_at = ?, completed_at = ?, updated_at = ?
                    WHERE id = ? AND status IN (?, ?) AND status_reason = ?
                      AND worker_id = ? AND lease_expires_at = ?
                      AND lease_expires_at <= ? AND retry_series_json IS NULL
                    """,
                    (
                        str(settled.status),
                        sqlite_records.json_dumps(settled.status_payload),
                        sqlite_records.json_dumps(settled.error),
                        sqlite_records.format_optional_datetime(settled.started_at),
                        sqlite_records.format_optional_datetime(settled.completed_at),
                        sqlite_records.format_datetime(settled.updated_at),
                        request.task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        request.expected_status_reason,
                        request.original_worker_id,
                        sqlite_records.format_datetime(request.original_lease_expires_at),
                        sqlite_records.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    raise _task_cancellation_reconciliation_conflict(
                        request,
                        "Task cancellation reconciliation lost its fenced transition.",
                    )
                durable_task = self._require_task_unlocked(request.task_id)
                self._record_task_transition_unlocked(
                    task,
                    durable_task,
                    settled_execution=(task.id, task.worker_id, task.started_at)
                    if task.worker_id is not None and task.started_at is not None
                    else None,
                )
                receipt = result.terminalization_receipt.model_copy(
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
                self._connection.execute(
                    "INSERT INTO cayu_task_terminalization_receipts "
                    "(task_id, idempotency_key, request_sha256, worker_id, "
                    "terminal_kind, task_json, committed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        receipt.task_id,
                        receipt.idempotency_key,
                        receipt.request_sha256,
                        receipt.worker_id,
                        receipt.kind.value,
                        sqlite_records.json_dumps(receipt.task.model_dump(mode="json")),
                        sqlite_records.format_datetime(receipt.committed_at),
                    ),
                )
                self._connection.commit()
                return _copy_task_cancellation_reconciliation_result(durable_result)
            except BaseException:
                self._connection.rollback()
                raise

    async def settle_task_retry_attempt(
        self,
        request: TaskRetrySettlementRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_settlement(request)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT request_sha256, receipt_json "
                    "FROM cayu_task_retry_settlements "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.idempotency_key),
                ).fetchone()
                if row is not None:
                    receipt = TaskRetrySettlementResult.model_validate(
                        json.loads(row["receipt_json"])
                    )
                    replayed = _replay_task_retry_settlement(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    self._connection.commit()
                    return replayed

                task = self._require_task_unlocked(request.task_id)
                now = self._ownership_clock()
                if request.lease_expires_at is None:
                    raise TaskClaimLost("Task retry settlement requires its exact worker lease.")
                _ensure_exact_owned_active_task_lease(
                    task,
                    request.worker_id,
                    request.lease_expires_at,
                    now=now,
                )
                series_now = self._clock()
                settled, successor = _settled_task_retry_attempt(
                    task,
                    request,
                    now=now,
                    series_now=series_now,
                )
                assert settled.retry_series is not None
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?, status_reason = ?, status_payload_json = ?,
                        result_json = ?, error_json = ?, worker_id = NULL,
                        lease_expires_at = NULL, started_at = ?, completed_at = ?,
                        updated_at = ?, retry_series_json = ?
                    WHERE id = ? AND status IN (?, ?) AND worker_id = ?
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                    """,
                    (
                        str(settled.status),
                        settled.status_reason,
                        sqlite_records.json_dumps(settled.status_payload),
                        (
                            None
                            if settled.result is None
                            else sqlite_records.json_dumps(settled.result)
                        ),
                        (
                            None
                            if settled.error is None
                            else sqlite_records.json_dumps(settled.error)
                        ),
                        sqlite_records.format_optional_datetime(settled.started_at),
                        sqlite_records.format_optional_datetime(settled.completed_at),
                        sqlite_records.format_datetime(settled.updated_at),
                        sqlite_records.json_dumps(settled.retry_series.model_dump(mode="json")),
                        request.task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        request.worker_id,
                        sqlite_records.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_task_active_lease_error(
                        request.task_id,
                        request.worker_id,
                        now=now,
                    )
                if successor is not None:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_tasks (
                            id, type, title, description, status, session_id,
                            session_instance_id, parent_task_id, assigned_agent_name, available_at, worker_id,
                            lease_expires_at, interrupted_handoff_id, status_reason,
                            status_payload_json, input_json,
                            result_json, error_json, metadata_json, created_at, updated_at,
                            started_at, completed_at, invocation_json, retry_series_json,
                            work_contract_json, schedule_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        sqlite_records.task_to_row_values(successor),
                    )
                receipt = TaskRetrySettlementResult(
                    task_id=request.task_id,
                    idempotency_key=request.idempotency_key,
                    request_sha256=request_sha256,
                    task=settled,
                    successor=successor,
                    events=_task_retry_events(settled, occurred_at=now),
                    committed_at=now,
                )
                if successor is not None:
                    from cayu.storage._sqlite_task_graphs import register_retry_successor

                    register_retry_successor(self, settled, successor)
                self._record_task_transition_unlocked(task, settled)
                if successor is not None:
                    self._record_task_transition_unlocked(
                        successor, self._require_task_unlocked(successor.id)
                    )
                self._connection.execute(
                    "INSERT INTO cayu_task_retry_settlements "
                    "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.idempotency_key,
                        request_sha256,
                        sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                        sqlite_records.format_datetime(receipt.committed_at),
                    ),
                )
                self._connection.commit()
                committed = receipt.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
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
        async with self._lock:
            row = self._connection.execute(
                "SELECT receipt_json FROM cayu_task_retry_settlements "
                "WHERE task_id = ? AND idempotency_key = ?",
                (task_id, idempotency_key),
            ).fetchone()
            if row is None:
                return None
            return TaskRetrySettlementResult.model_validate(json.loads(row["receipt_json"]))

    async def reconcile_task_retry_cancellation(
        self,
        request: TaskRetryCancellationReconciliationRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_cancellation_reconciliation(request)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                rejection_row = self._connection.execute(
                    "SELECT request_sha256, record_json "
                    "FROM cayu_task_retry_reconciliation_rejections "
                    "WHERE task_id = ? AND reconciliation_idempotency_key = ?",
                    (request.task_id, request.reconciliation_idempotency_key),
                ).fetchone()
                if rejection_row is not None:
                    rejection = _TaskRetryCancellationReconciliationRejectionRecord.model_validate(
                        json.loads(rejection_row["record_json"])
                    )
                    if rejection.request_sha256 != rejection_row["request_sha256"]:
                        raise RuntimeError(
                            "SQLite retry reconciliation rejection contains invalid "
                            "durable material."
                        )
                    raise _replay_task_retry_cancellation_reconciliation_rejection(
                        request,
                        request_sha256=request_sha256,
                        record=rejection,
                    )
                row = self._connection.execute(
                    "SELECT request_sha256, receipt_json "
                    "FROM cayu_task_retry_settlements "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.cancellation_idempotency_key),
                ).fetchone()
                if row is not None:
                    receipt = TaskRetrySettlementResult.model_validate(
                        json.loads(row["receipt_json"])
                    )
                    replayed = _replay_task_retry_cancellation_reconciliation(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    self._connection.commit()
                    return replayed

                task = self._load_task_unlocked(request.task_id)
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
                    self._connection.execute(
                        "INSERT INTO cayu_task_retry_reconciliation_rejections "
                        "(task_id, reconciliation_idempotency_key, request_sha256, "
                        "record_json, recorded_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            rejection.task_id,
                            rejection.reconciliation_idempotency_key,
                            rejection.request_sha256,
                            sqlite_records.json_dumps(rejection.model_dump(mode="json")),
                            sqlite_records.format_datetime(rejection.recorded_at),
                        ),
                    )
                    self._connection.commit()
                    raise _rejected_task_retry_cancellation_reconciliation(rejection)
                receipt = _reconciled_task_retry_cancellation(
                    task,
                    request,
                    request_sha256=request_sha256,
                    committed_at=now,
                )
                settled = receipt.task
                assert settled.retry_series is not None
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?, status_reason = ?, status_payload_json = ?,
                        result_json = NULL, error_json = ?, worker_id = NULL,
                        lease_expires_at = NULL, started_at = ?, completed_at = ?,
                        updated_at = ?, retry_series_json = ?
                    WHERE id = ? AND status IN (?, ?) AND status_reason = ?
                      AND worker_id = ? AND lease_expires_at = ?
                      AND lease_expires_at <= ?
                    """,
                    (
                        str(settled.status),
                        settled.status_reason,
                        sqlite_records.json_dumps(settled.status_payload),
                        sqlite_records.json_dumps(settled.error),
                        sqlite_records.format_optional_datetime(settled.started_at),
                        sqlite_records.format_optional_datetime(settled.completed_at),
                        sqlite_records.format_datetime(settled.updated_at),
                        sqlite_records.json_dumps(settled.retry_series.model_dump(mode="json")),
                        request.task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        request.expected_status_reason,
                        request.original_worker_id,
                        sqlite_records.format_datetime(request.original_lease_expires_at),
                        sqlite_records.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    raise _task_retry_cancellation_reconciliation_conflict(
                        request,
                        "Task retry cancellation reconciliation lost its fenced transition.",
                    )
                self._record_task_transition_unlocked(
                    task,
                    settled,
                    settled_execution=(task.id, task.worker_id, task.started_at)
                    if task.worker_id is not None and task.started_at is not None
                    else None,
                )
                self._connection.execute(
                    "INSERT INTO cayu_task_retry_settlements "
                    "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.cancellation_idempotency_key,
                        request_sha256,
                        sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                        sqlite_records.format_datetime(receipt.committed_at),
                    ),
                )
                self._connection.commit()
                return receipt.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
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
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                task = self._require_task_unlocked(task_id)
                lease_now = self._ownership_clock()
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease,
                    now=lease_now,
                )
                if not _claimed_task_retry_attempt_elapsed(task, series_now=self._clock()):
                    self._connection.commit()
                    return None
                receipt = _elapsed_claimed_task_retry_settlement(
                    task,
                    committed_at=lease_now,
                    token_count=token_count,
                    estimated_cost=estimated_cost,
                )
                settled = receipt.task
                assert settled.retry_series is not None
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?, status_reason = ?, status_payload_json = ?,
                        result_json = NULL, error_json = ?, worker_id = NULL,
                        lease_expires_at = NULL, started_at = ?, completed_at = ?,
                        updated_at = ?, retry_series_json = ?
                    WHERE id = ? AND status IN (?, ?) AND worker_id = ?
                      AND lease_expires_at = ? AND lease_expires_at > ?
                    """,
                    (
                        str(settled.status),
                        settled.status_reason,
                        sqlite_records.json_dumps(settled.status_payload),
                        sqlite_records.json_dumps(settled.error),
                        sqlite_records.format_optional_datetime(settled.started_at),
                        sqlite_records.format_optional_datetime(settled.completed_at),
                        sqlite_records.format_datetime(settled.updated_at),
                        sqlite_records.json_dumps(settled.retry_series.model_dump(mode="json")),
                        task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        worker_id,
                        sqlite_records.format_datetime(expected_lease),
                        sqlite_records.format_datetime(lease_now),
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_task_active_lease_error(task_id, worker_id, now=lease_now)
                self._record_task_transition_unlocked(task, settled)
                self._connection.execute(
                    "INSERT INTO cayu_task_retry_settlements "
                    "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        receipt.task_id,
                        receipt.idempotency_key,
                        receipt.request_sha256,
                        sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                        sqlite_records.format_datetime(receipt.committed_at),
                    ),
                )
                self._connection.commit()
                return receipt.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
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
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                task = self._require_task_unlocked(task_id)
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease,
                    now=self._ownership_clock(),
                )
                elapsed = _claimed_task_retry_attempt_elapsed(
                    task,
                    series_now=self._clock(),
                )
                self._connection.commit()
                return elapsed
            except BaseException:
                self._connection.rollback()
                raise

    @runtime_task_mutation
    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        async with self._lock:
            with self._verified_transaction_unlocked():
                if self._require_task_unlocked(task_id).schedule is not None:
                    raise TaskScheduleConflict(
                        "Managed schedule cancellation requires its revision."
                    )
                prior = self._require_task_unlocked(task_id)
                updated = self._finish_task_in_transaction_unlocked(
                    task_id, TaskStatus.CANCELLED, result=None, error=copied_error
                )
                self._record_task_transition_unlocked(prior, updated)
                return updated

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
        async with self._lock:
            with self._verified_transaction_unlocked():
                current = self._require_task_unlocked(task_id)
                if current.worker_id != worker_id or current.lease_expires_at != expected_lease:
                    raise TaskClaimLost(
                        "Claimed-task cancellation no longer owns the expected worker lease."
                    )
                if _task_cancellation_requested(current) or (
                    current.status_reason == _TASK_RETRY_CANCELLATION_REQUESTED_REASON
                ):
                    return current.model_copy(deep=True)
                if current.started_at is not None:
                    requested = _expired_dispatched_task_cancellation(
                        current,
                        updated_at=self._ownership_clock(),
                        error=copied_error,
                    )
                    self._update_task_snapshot_unlocked(requested)
                    self._record_task_transition_unlocked(current, requested)
                    return requested.model_copy(deep=True)
                updated = self._finish_task_in_transaction_unlocked(
                    task_id,
                    TaskStatus.CANCELLED,
                    result=None,
                    error=copied_error,
                    worker_id=worker_id,
                    handoff_id=current.interrupted_handoff_id,
                    expected_lease_expires_at=expected_lease,
                )
                self._record_task_transition_unlocked(current, updated)
                return updated

    async def mark_claimed_task_execution_started(
        self,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                current = self._require_task_unlocked(task_id)
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
                self._update_task_snapshot_unlocked(started)
                self._record_task_transition_unlocked(current, started)
                return self._require_task_unlocked(task_id).model_copy(deep=True)

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
        async with self._lock:
            with self._verified_transaction_unlocked():
                prior = self._require_task_unlocked(task_id)
                now = self._ownership_clock()
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts cannot use ordinary task resumption."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        status_reason = NULL,
                        status_payload_json = NULL,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                      AND status IN (?, ?, ?)
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        str(TaskStatus.PENDING),
                        sqlite_records.format_datetime(now),
                        task_id,
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                        task_id,
                    ),
                )
                if cursor.rowcount == 1:
                    updated = self._require_task_unlocked(task_id)
                    self._record_task_transition_unlocked(prior, updated)
                    return self._require_task_unlocked(task_id).model_copy(deep=True)
            if cursor.rowcount != 1:
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts cannot use ordinary task resumption."
                    )
                task = self._require_task_unlocked(task_id)
                _ensure_can_resume_task(task)
                raise ValueError(f"Task {task.id} cannot resume from {task.status}")
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        retry_worker_id_is_bounded = _task_retry_reconciliation_identity_is_bounded(worker_id)
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        clauses, params = self._task_filter_clauses(query)
        retry_deadline_clause = (
            (
                "(retry_series_json IS NULL "
                "OR json_extract(retry_series_json, '$.elapsed_deadline') IS NULL "
                "OR julianday(json_extract(retry_series_json, '$.elapsed_deadline')) "
                "> julianday(?))"
            )
            if retry_worker_id_is_bounded
            else "retry_series_json IS NULL"
        )
        where_sql = " AND ".join(
            [
                "status = ?",
                "session_id IS NULL",
                "(available_at IS NULL OR available_at <= ?)",
                retry_deadline_clause,
                "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                "WHERE attempt.retry_admissible = 0 AND ("
                "attempt.task_id = cayu_tasks.id OR ("
                "cayu_tasks.retry_series_json IS NOT NULL AND "
                "attempt.retry_series_id = json_extract("
                "cayu_tasks.retry_series_json, '$.series_id'))))",
                *clauses,
            ]
        )
        # Claiming is always FIFO by creation time, independent of the query's
        # display ordering, so the oldest pending task is dispatched first.
        order_sql = sqlite_records.task_order_sql(TaskOrder.CREATED_AT_ASC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                availability_now = self._clock()
                now = self._ownership_clock()
                lease_expires_at = now + timedelta(seconds=lease_seconds)
                self._expire_held_schedules_unlocked(query, as_of=availability_now, now=now)
                expired_rows = self._connection.execute(
                    """
                    SELECT id
                    FROM cayu_tasks
                    WHERE status = ?
                      AND session_id IS NULL
                      AND retry_series_json IS NOT NULL
                      AND json_extract(retry_series_json, '$.disposition') = ?
                      AND json_extract(retry_series_json, '$.elapsed_deadline') IS NOT NULL
                      AND julianday(json_extract(retry_series_json, '$.elapsed_deadline'))
                          <= julianday(?)
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_local_execution_attempts AS attempt
                          WHERE attempt.retry_admissible = 0 AND (
                              attempt.task_id = cayu_tasks.id OR
                              attempt.retry_series_id = json_extract(
                                  cayu_tasks.retry_series_json, '$.series_id'
                              )
                          )
                      )
                    ORDER BY created_at ASC, id ASC
                    LIMIT 100
                    """,
                    (
                        str(TaskStatus.PENDING),
                        str(TaskRetrySeriesDisposition.ACTIVE),
                        sqlite_records.format_datetime(availability_now),
                    ),
                ).fetchall()
                for expired_row in expired_rows:
                    expired_task = self._require_task_unlocked(expired_row["id"])
                    expiration = _expired_task_retry_settlement(
                        expired_task,
                        committed_at=now,
                        series_now=availability_now,
                    )
                    assert expiration.task.retry_series is not None
                    cursor = self._connection.execute(
                        """
                        UPDATE cayu_tasks
                        SET status = ?, status_reason = ?, status_payload_json = ?,
                            result_json = NULL, error_json = ?, worker_id = NULL,
                            lease_expires_at = NULL, started_at = ?, completed_at = ?,
                            updated_at = ?, retry_series_json = ?
                        WHERE id = ? AND status = ? AND session_id IS NULL
                        """,
                        (
                            str(expiration.task.status),
                            expiration.task.status_reason,
                            sqlite_records.json_dumps(expiration.task.status_payload),
                            sqlite_records.json_dumps(expiration.task.error),
                            sqlite_records.format_optional_datetime(expiration.task.started_at),
                            sqlite_records.format_optional_datetime(expiration.task.completed_at),
                            sqlite_records.format_datetime(expiration.task.updated_at),
                            sqlite_records.json_dumps(
                                expiration.task.retry_series.model_dump(mode="json")
                            ),
                            expiration.task_id,
                            str(TaskStatus.PENDING),
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise TaskTerminalizationConflict(
                            "Elapsed task retry attempt changed during claim admission."
                        )
                    self._connection.execute(
                        "INSERT INTO cayu_task_retry_settlements "
                        "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            expiration.task_id,
                            expiration.idempotency_key,
                            expiration.request_sha256,
                            sqlite_records.json_dumps(expiration.model_dump(mode="json")),
                            sqlite_records.format_datetime(expiration.committed_at),
                        ),
                    )
                    self._record_task_transition_unlocked(expired_task, expiration.task)
                rows = self._connection.execute(
                    f"""
                    SELECT id
                    FROM cayu_tasks
                    WHERE {where_sql}
                    ORDER BY {order_sql}, id ASC
                    LIMIT 100
                    """,
                    [
                        str(TaskStatus.PENDING),
                        sqlite_records.format_datetime(availability_now),
                        *(
                            [sqlite_records.format_datetime(availability_now)]
                            if retry_worker_id_is_bounded
                            else []
                        ),
                        *params,
                    ],
                ).fetchall()
                prior: Task | None = None
                for row in rows:
                    candidate = self._require_task_unlocked(row["id"])
                    if candidate.schedule is not None and candidate.schedule.admitted_at is None:
                        assert candidate.available_at is not None
                        eligibility = task_schedule_eligibility(
                            available_at=candidate.available_at,
                            policy=candidate.schedule.policy,
                            as_of=availability_now,
                        )
                        if eligibility in {
                            TaskScheduleEligibility.EXPIRED,
                            TaskScheduleEligibility.SKIPPED,
                        }:
                            self._settle_schedule_nonexecution_unlocked(
                                candidate, eligibility, now=now
                            )
                            continue
                    prior = candidate
                    break
                if prior is None:
                    self._connection.commit()
                    return None
                task_id = prior.id
                schedule = admitted_schedule(prior, now=availability_now)
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        worker_id = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        str(TaskStatus.CLAIMED),
                        worker_id,
                        sqlite_records.format_datetime(lease_expires_at),
                        sqlite_records.format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                    ),
                )
                if cursor.rowcount != 1:
                    self._connection.rollback()
                    return None
                updated = self._require_task_unlocked(task_id)
                if schedule is not None:
                    updated = updated.model_copy(update={"schedule": schedule})
                    self._update_task_snapshot_unlocked(updated)
                self._record_task_transition_unlocked(prior, updated)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    def _settle_schedule_nonexecution_unlocked(
        self, task: Task, eligibility: TaskScheduleEligibility, *, now: datetime
    ) -> None:
        terminal, settlement = _scheduled_task_nonexecution(task, eligibility=eligibility, now=now)
        self._update_task_snapshot_unlocked(terminal)
        self._record_task_transition_unlocked(task, terminal)
        if settlement is not None:
            self._connection.execute(
                "INSERT INTO cayu_task_retry_settlements "
                "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    settlement.task_id,
                    settlement.idempotency_key,
                    settlement.request_sha256,
                    sqlite_records.json_dumps(settlement.model_dump(mode="json")),
                    sqlite_records.format_datetime(settlement.committed_at),
                ),
            )

    def _expire_held_schedules_unlocked(
        self, query: TaskQuery, *, as_of: datetime, now: datetime
    ) -> None:
        clauses, params = self._task_filter_clauses(query.model_copy(update={"status": None}))
        scope = " AND ".join(["session_id IS NULL", *clauses])
        stamp = sqlite_records.format_datetime(as_of)
        rows = self._connection.execute(
            f"SELECT id FROM cayu_tasks WHERE {scope} "
            "AND status IN ('paused', 'blocked', 'needs_attention', 'waiting_dependencies', 'waiting_group') "
            "AND schedule_json IS NOT NULL AND json_extract(schedule_json, '$.admitted_at') IS NULL "
            "AND (replace(json_extract(schedule_json, '$.policy.expires_at'), 'Z', '+00:00') <= ? OR "
            "(json_extract(schedule_json, '$.policy.misfire_policy') = 'skip' AND "
            "(julianday(?) - julianday(available_at)) * 86400 >= "
            "json_extract(schedule_json, '$.policy.misfire_grace_seconds'))) "
            "AND NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
            "WHERE attempt.retry_admissible = 0 AND (attempt.task_id = cayu_tasks.id OR "
            "(cayu_tasks.retry_series_json IS NOT NULL AND attempt.retry_series_id = "
            "json_extract(cayu_tasks.retry_series_json, '$.series_id')))) "
            "ORDER BY available_at, id LIMIT 100",
            [*params, stamp, stamp],
        ).fetchall()
        for row in rows:
            task = self._require_task_unlocked(row["id"])
            if task.status not in {
                TaskStatus.PAUSED,
                TaskStatus.BLOCKED,
                TaskStatus.NEEDS_ATTENTION,
                TaskStatus.WAITING_DEPENDENCIES,
                TaskStatus.WAITING_GROUP,
            }:
                continue
            assert task.schedule is not None and task.available_at is not None
            eligibility = task_schedule_eligibility(
                available_at=task.available_at, policy=task.schedule.policy, as_of=as_of
            )
            if eligibility in {TaskScheduleEligibility.EXPIRED, TaskScheduleEligibility.SKIPPED}:
                self._settle_schedule_nonexecution_unlocked(task, eligibility, now=now)

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
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                lease_expires_at = now + timedelta(seconds=extend_seconds)
                task = self._require_task_unlocked(task_id)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost("Task heartbeat no longer owns the expected worker lease.")
                _ensure_task_handoff_authority(task, handoff_id)
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts require claim-fenced lease renewal."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND interrupted_handoff_id IS ?
                      AND status IN (?, ?)
                      AND lease_expires_at = ? AND lease_expires_at > ?
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        sqlite_records.format_datetime(lease_expires_at),
                        sqlite_records.format_datetime(now),
                        task_id,
                        worker_id,
                        handoff_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        sqlite_records.format_datetime(expected_lease),
                        sqlite_records.format_datetime(now),
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_if_governed_work_attempt_admission(
                        task_id,
                        "Admitted work attempts require claim-fenced lease renewal.",
                    )
                    self._raise_task_active_lease_error(task_id, worker_id, now=now)
                updated = self._require_task_unlocked(task_id)
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
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                task = self._require_task_unlocked(task_id)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost("Task release no longer owns the expected worker lease.")
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts release ownership through proposal publication."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        started_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status = ?
                      AND session_id IS NULL
                      AND (status_reason IS NULL OR status_reason NOT IN (?, ?))
                      AND lease_expires_at = ? AND lease_expires_at > ?
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        str(TaskStatus.PENDING),
                        sqlite_records.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        sqlite_records.format_datetime(expected_lease),
                        sqlite_records.format_datetime(now),
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_if_governed_work_attempt_admission(
                        task_id,
                        "Admitted work attempts release ownership through proposal publication.",
                    )
                    self._raise_task_release_error(task_id, worker_id, now=now)
                updated = self._require_task_unlocked(task_id)
                self._record_task_transition_unlocked(task, updated)
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
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                task = self._require_task_unlocked(task_id)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost(
                        "Attached-task release no longer owns the expected worker lease."
                    )
                if task.interrupted_handoff_id is not None:
                    raise TaskInterruptedHandoffConflict(
                        "Recovery-owned attached tasks must publish an interrupted handoff."
                    )
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts release ownership through proposal publication."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status = ?
                      AND session_id IS NOT NULL
                      AND (status_reason IS NULL OR status_reason != ?)
                      AND lease_expires_at = ? AND lease_expires_at > ?
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        sqlite_records.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.RUNNING),
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        sqlite_records.format_datetime(expected_lease),
                        sqlite_records.format_datetime(now),
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_if_governed_work_attempt_admission(
                        task_id,
                        "Admitted work attempts release ownership through proposal publication.",
                    )
                    self._raise_attached_task_worker_release_error(
                        task_id,
                        worker_id,
                        now=now,
                    )
                updated = self._require_task_unlocked(task_id)
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
        clauses, params = self._task_filter_clauses(query)
        where_sql = " AND ".join(
            [
                "status = ?",
                "session_id IS NULL",
                "lease_expires_at IS NOT NULL",
                "lease_expires_at <= ?",
                "(status_reason IS NULL OR status_reason NOT IN (?, ?))",
                "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                "WHERE attempt.retry_admissible = 0 AND ("
                "attempt.task_id = cayu_tasks.id OR ("
                "cayu_tasks.retry_series_json IS NOT NULL AND "
                "attempt.retry_series_id = json_extract("
                "cayu_tasks.retry_series_json, '$.series_id'))))",
                *clauses,
            ]
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                rows = self._connection.execute(
                    f"""
                    SELECT id
                    FROM cayu_tasks
                    WHERE {where_sql}
                    ORDER BY lease_expires_at ASC, id ASC
                    LIMIT ?
                    """,
                    [
                        str(TaskStatus.CLAIMED),
                        sqlite_records.format_datetime(now),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        *params,
                        max_reclaims,
                    ],
                ).fetchall()
                task_ids = [row["id"] for row in rows]
                reclaimed: list[Task] = []
                for task_id in task_ids:
                    task = self._require_task_unlocked(task_id)
                    if task.started_at is not None:
                        requested = _expired_dispatched_task_cancellation(
                            task,
                            updated_at=now,
                        )
                        self._update_task_snapshot_unlocked(requested)
                        self._record_task_transition_unlocked(task, requested)
                        continue
                    updated = task.model_copy(
                        update={
                            "status": TaskStatus.PENDING,
                            "worker_id": None,
                            "lease_expires_at": None,
                            "updated_at": now,
                        }
                    )
                    self._update_task_snapshot_unlocked(updated)
                    self._record_task_transition_unlocked(task, updated)
                    reclaimed.append(updated)
                self._connection.commit()
                return [task.model_copy(deep=True) for task in reclaimed]
            except Exception:
                self._connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        return sqlite_connection.connect(path)

    def _initialize_schema(self) -> None:
        sqlite_support.reconcile_schema(
            self._connection,
            self._schema_mode,
            app_min_supported=_SQLITE_TASK_MIN_REQUIRED_REVISION,
        )

    def _load_task_unlocked(self, task_id: str) -> Task | None:
        row = self._connection.execute(
            "SELECT * FROM cayu_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return sqlite_records.task_from_row(row)

    def _task_parent_for_create_unlocked(
        self,
        request: TaskCreate,
        *,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        parent_task_id = request.parent_task_id
        if parent_task_id is None:
            return None
        if parent_task_id == task_id:
            raise ValueError("Task cannot be its own parent.")
        row = self._connection.execute(
            "SELECT id, session_id, session_instance_id, invocation_json "
            "FROM cayu_tasks WHERE id = ?",
            (parent_task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Parent task not found: {parent_task_id}")
        return TaskInvocationSnapshot(
            id=row["id"],
            session_id=row["session_id"],
            session_instance_id=row["session_instance_id"],
            invocation=TaskInvocation.model_validate(json.loads(row["invocation_json"])),
        )

    def _require_task_unlocked(self, task_id: str) -> Task:
        task = self._load_task_unlocked(task_id)
        from cayu.tasks.access import require_mutation

        require_mutation(task)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _task_exists_unlocked(self, task_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM cayu_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        return row is not None

    def _raise_if_governed_work_attempt_admission(
        self,
        task_id: str,
        message: str,
    ) -> None:
        if (
            self._connection.execute(
                "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                (task_id,),
            ).fetchone()
            is not None
        ):
            raise WorkAttemptExecutionClaimLost(message)

    def _finish_task_unlocked(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
        handoff_id: str | None = None,
        expected_lease_expires_at: datetime | None = None,
    ) -> Task:
        with self._verified_transaction_unlocked():
            prior = self._require_task_unlocked(task_id)
            updated = self._finish_task_in_transaction_unlocked(
                task_id,
                status,
                result=result,
                error=error,
                worker_id=worker_id,
                handoff_id=handoff_id,
                expected_lease_expires_at=expected_lease_expires_at,
            )
            self._record_task_transition_unlocked(prior, updated)
            return updated

    def _finish_task_in_transaction_unlocked(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
        handoff_id: str | None = None,
        expected_lease_expires_at: datetime | None = None,
    ) -> Task:
        now = self._ownership_clock()
        task = self._require_task_unlocked(task_id)
        if worker_id is not None:
            if expected_lease_expires_at is not None:
                expected_lease_expires_at = normalize_utc_datetime(
                    expected_lease_expires_at,
                    "lease_expires_at",
                )
                if task.lease_expires_at != expected_lease_expires_at:
                    raise TaskClaimLost(
                        "Task terminalization no longer owns the expected worker lease."
                    )
            _ensure_owned_active_task_lease(task, worker_id, now=now)
            _ensure_task_handoff_authority(task, handoff_id)
        cancellation_owner_clause = ""
        cancellation_owner_params: list[str | None] = []
        if worker_id is not None and expected_lease_expires_at is not None:
            cancellation_owner_clause = (
                "\n                          AND worker_id = ?"
                "\n                          AND lease_expires_at = ?"
                "\n                          AND interrupted_handoff_id IS ?"
            )
            cancellation_owner_params = [
                worker_id,
                sqlite_records.format_datetime(expected_lease_expires_at),
                handoff_id,
            ]
        if (
            self._connection.execute(
                "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                (task_id,),
            ).fetchone()
            is not None
        ):
            raise WorkAttemptExecutionClaimLost(
                "Admitted work attempts cannot use ordinary terminalization."
            )
        verified_work_support.require_contracted_completion_authority(task, status)
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
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status_reason = ?, status_payload_json = ?, updated_at = ?
                    WHERE id = ? AND status IN (?, ?){cancellation_owner_clause}
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        cancellation_requested.status_reason,
                        sqlite_records.json_dumps(cancellation_requested.status_payload),
                        sqlite_records.format_datetime(cancellation_requested.updated_at),
                        task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        *cancellation_owner_params,
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_if_governed_work_attempt_admission(
                        task_id,
                        "Admitted work attempts cannot use ordinary terminalization.",
                    )
                    raise TaskTerminalizationConflict(
                        "Task retry cancellation lost active ownership."
                    )
                return self._require_task_unlocked(task_id).model_copy(deep=True)
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
            cursor = self._connection.execute(
                f"""
                UPDATE cayu_tasks
                SET status_reason = ?, status_payload_json = ?, updated_at = ?
                WHERE id = ? AND status IN (?, ?){cancellation_owner_clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM cayu_work_attempt_admissions
                      WHERE task_id = ?
                  )
                """,
                (
                    cancellation_requested.status_reason,
                    sqlite_records.json_dumps(cancellation_requested.status_payload),
                    sqlite_records.format_datetime(cancellation_requested.updated_at),
                    task_id,
                    str(TaskStatus.CLAIMED),
                    str(TaskStatus.RUNNING),
                    *cancellation_owner_params,
                    task_id,
                ),
            )
            if cursor.rowcount != 1:
                self._raise_if_governed_work_attempt_admission(
                    task_id,
                    "Admitted work attempts cannot use ordinary terminalization.",
                )
                raise TaskTerminalizationConflict("Task cancellation lost active ownership.")
            return self._require_task_unlocked(task_id).model_copy(deep=True)
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
        # When a worker_id is given, only terminalize if that worker still owns an active
        # lease — a worker that lost its lease must not clobber a task another has reclaimed.
        owner_clause = ""
        owner_params: list[str | None] = []
        if worker_id is not None:
            owner_clause = (
                "\n                  AND worker_id = ?"
                "\n                  AND lease_expires_at IS NOT NULL AND lease_expires_at > ?"
                "\n                  AND interrupted_handoff_id IS ?"
            )
            owner_params = [worker_id, sqlite_records.format_datetime(now), handoff_id]
        cursor = self._connection.execute(
            f"""
            UPDATE cayu_tasks
            SET status = ?,
                status_reason = ?,
                status_payload_json = ?,
                result_json = ?,
                error_json = ?,
                worker_id = NULL,
                lease_expires_at = NULL,
                interrupted_handoff_id = NULL,
                started_at = COALESCE(started_at, ?),
                completed_at = ?,
                updated_at = ?,
                retry_series_json = ?
            WHERE id = ?
              AND status NOT IN (?, ?, ?)
              AND NOT EXISTS (
                  SELECT 1 FROM cayu_work_attempt_admissions
                  WHERE task_id = ?
              ){owner_clause}
            """,
            (
                str(status),
                terminal_task.status_reason,
                (
                    None
                    if terminal_task.status_payload is None
                    else sqlite_records.json_dumps(terminal_task.status_payload)
                ),
                (
                    None
                    if terminal_task.result is None
                    else sqlite_records.json_dumps(terminal_task.result)
                ),
                (
                    None
                    if terminal_task.error is None
                    else sqlite_records.json_dumps(terminal_task.error)
                ),
                sqlite_records.format_optional_datetime(terminal_task.started_at),
                sqlite_records.format_optional_datetime(terminal_task.completed_at),
                sqlite_records.format_datetime(terminal_task.updated_at),
                (
                    None
                    if terminal_task.retry_series is None
                    else sqlite_records.json_dumps(
                        terminal_task.retry_series.model_dump(mode="json")
                    )
                ),
                task_id,
                str(TaskStatus.COMPLETED),
                str(TaskStatus.FAILED),
                str(TaskStatus.CANCELLED),
                task_id,
                *owner_params,
            ),
        )
        if cursor.rowcount == 1 and cancellation is not None:
            self._connection.execute(
                "INSERT INTO cayu_task_retry_settlements "
                "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cancellation.task_id,
                    cancellation.idempotency_key,
                    cancellation.request_sha256,
                    sqlite_records.json_dumps(cancellation.model_dump(mode="json")),
                    sqlite_records.format_datetime(cancellation.committed_at),
                ),
            )
        if cursor.rowcount != 1:
            if worker_id is not None:
                current = self._require_task_unlocked(task_id)
                _ensure_owned_active_task_lease(current, worker_id, now=now)
                _ensure_task_handoff_authority(current, handoff_id)
            self._raise_if_governed_work_attempt_admission(
                task_id,
                "Admitted work attempts cannot use ordinary terminalization.",
            )
            if worker_id is not None:
                raise RuntimeError(f"Task {task_id} active-lease mutation did not update a row.")
            task = self._require_task_unlocked(task_id)
            _ensure_can_transition(task, status)
            raise ValueError(f"Task {task.id} cannot transition from {task.status}")
        updated = self._require_task_unlocked(task_id)
        return updated.model_copy(deep=True)

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
        async with self._lock:
            with self._verified_transaction_unlocked():
                prior = self._require_task_unlocked(task_id)
                now = self._ownership_clock()
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts cannot use ordinary task holds."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        status_reason = ?,
                        status_payload_json = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                      AND (
                        status = ?
                        OR status IN ('waiting_dependencies', 'waiting_group')
                        OR status = ?
                        OR status = ?
                        OR status = ?
                        OR status = ?
                        OR (status = ? AND session_id IS NULL)
                      )
                      AND (status_reason IS NULL OR status_reason NOT IN (?, ?))
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        str(status),
                        reason,
                        None if payload is None else sqlite_records.json_dumps(payload),
                        sqlite_records.format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                        str(TaskStatus.RUNNING),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    if (
                        self._connection.execute(
                            "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                            (task_id,),
                        ).fetchone()
                        is not None
                    ):
                        raise WorkAttemptExecutionClaimLost(
                            "Admitted work attempts cannot use ordinary task holds."
                        )
                    task = self._require_task_unlocked(task_id)
                    _ensure_can_hold_task(task, status)
                    raise ValueError(f"Task {task.id} cannot transition to {status}")
                updated = self._require_task_unlocked(task_id)
                self._record_task_transition_unlocked(prior, updated)
                return updated.model_copy(deep=True)

    def _task_filter_clauses(self, query: TaskQuery) -> tuple[list[str], list[object]]:
        clauses: list[str] = []
        params: list[object] = []
        if query.has_work_contract is not None:
            clauses.append(
                "work_contract_json IS NOT NULL"
                if query.has_work_contract
                else "work_contract_json IS NULL"
            )
        if query.type is not None:
            clauses.append("type = ?")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = ?")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = ?")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = ?")
            params.append(query.assigned_agent_name)
        return clauses, params

    def _raise_task_active_lease_error(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = self._require_task_unlocked(task_id)
        _ensure_owned_active_task_lease(task, worker_id, now=now)
        raise RuntimeError(f"Task {task.id} active-lease mutation did not update a row.")

    def _raise_task_release_error(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = self._require_task_unlocked(task_id)
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

    def _raise_attached_task_worker_release_error(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = self._require_task_unlocked(task_id)
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

    def _raise_task_claim_attach_error(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = self._require_task_unlocked(task_id)
        _raise_task_claim_attach_error(task, worker_id, now=now)


def _like_contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _sqlite_task_terminalization_receipt(
    *,
    task_id: str,
    idempotency_key: str,
    row: sqlite3.Row,
) -> TaskTerminalizationReceipt:
    try:
        return TaskTerminalizationReceipt(
            task_id=task_id,
            idempotency_key=idempotency_key,
            worker_id=row["worker_id"],
            kind=row["terminal_kind"],
            request_sha256=row["request_sha256"],
            task=Task.model_validate(json.loads(row["task_json"])),
            committed_at=sqlite_records.parse_datetime(row["committed_at"]),
        )
    except Exception as exc:
        raise TaskTerminalizationConflict("Task terminalization receipt is malformed.") from exc


def _sqlite_interrupted_task_handoff_receipt(
    *,
    task_id: str,
    handoff_id: str,
    row: sqlite3.Row,
) -> TaskInterruptedHandoffReceipt:
    try:
        receipt = TaskInterruptedHandoffReceipt(
            request=TaskInterruptedHandoffRequest.model_validate(json.loads(row["request_json"])),
            request_sha256=row["request_sha256"],
            task=Task.model_validate(json.loads(row["task_json"])),
            committed_at=sqlite_records.parse_datetime(row["committed_at"]),
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
