from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right, insort
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from itertools import islice
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, cast
from uuid import uuid4

from cayu._resource_store_surface import model_store_surface
from cayu.tasks.cancellation import (
    _TASK_CANCELLATION_REQUESTED_REASON,
    _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
    _copy_task_cancellation_reconciliation_result,
    _rejected_task_cancellation_reconciliation,
    _rejected_task_retry_cancellation_reconciliation,
    _replay_task_cancellation_reconciliation,
    _replay_task_cancellation_reconciliation_rejection,
    _replay_task_retry_cancellation_reconciliation_rejection,
    _task_cancellation_reconciliation_conflict,
    _task_cancellation_reconciliation_event,
    _task_cancellation_reconciliation_rejection_record,
    _task_cancellation_requested,
    _task_cancellation_requested_event,
    _task_retry_cancellation_reconciliation_conflict,
    _task_retry_cancellation_reconciliation_event,
    _task_retry_cancellation_reconciliation_rejection_record,
    _task_retry_cancellation_requested,
    _task_retry_reconciliation_identity_is_bounded,
    _TaskCancellationReconciliationRejectionRecord,
    _TaskRetryCancellationReconciliationRejectionRecord,
)
from cayu.tasks.cancellation import TaskCancellationReconciliation as TaskCancellationReconciliation
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationConflict as TaskCancellationReconciliationConflict,
)
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationEvent as TaskCancellationReconciliationEvent,
)
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationEventType as TaskCancellationReconciliationEventType,
)
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationEvidence as TaskCancellationReconciliationEvidence,
)
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationOutcome as TaskCancellationReconciliationOutcome,
)
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationRejected as TaskCancellationReconciliationRejected,
)
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationRequest as TaskCancellationReconciliationRequest,
)
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationResult as TaskCancellationReconciliationResult,
)
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliation as TaskRetryCancellationReconciliation,
)
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationConflict as TaskRetryCancellationReconciliationConflict,
)
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationEvent as TaskRetryCancellationReconciliationEvent,
)
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationEventType as TaskRetryCancellationReconciliationEventType,
)
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationEvidence as TaskRetryCancellationReconciliationEvidence,
)
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationOutcome as TaskRetryCancellationReconciliationOutcome,
)
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationRejected as TaskRetryCancellationReconciliationRejected,
)
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationRequest as TaskRetryCancellationReconciliationRequest,
)
from cayu.tasks.cancellation import (
    prepare_task_cancellation_reconciliation as prepare_task_cancellation_reconciliation,
)
from cayu.tasks.cancellation import (
    prepare_task_retry_cancellation_reconciliation as prepare_task_retry_cancellation_reconciliation,
)
from cayu.tasks.creation import TaskCreate as TaskCreate
from cayu.tasks.creation import TaskInvocationSnapshot as TaskInvocationSnapshot
from cayu.tasks.creation import (
    _copy_optional_session_binding,
    _copy_required_session_binding,
    _running_task_from_create,
    _task_from_create,
    _task_invocation_for_attachment,
    _task_session_id_for_start,
    _task_session_instance_for_attachment,
)
from cayu.tasks.creation import copy_task_create as copy_task_create
from cayu.tasks.creation import (
    preflight_contract_bound_task_creation as preflight_contract_bound_task_creation,
)
from cayu.tasks.creation import (
    require_contract_bound_task_creation_snapshot as require_contract_bound_task_creation_snapshot,
)
from cayu.tasks.creation import (
    task_create_with_execution_source as task_create_with_execution_source,
)
from cayu.tasks.creation import (
    task_create_with_runtime_invocation as task_create_with_runtime_invocation,
)
from cayu.tasks.creation import task_invocation_for_create as task_invocation_for_create
from cayu.tasks.handoff import (
    _TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    _copy_interrupted_task_handoff_receipt,
    _interrupted_task_continuation_handoff_id_sha256,
    _replay_interrupted_task_handoff_receipt,
)
from cayu.tasks.handoff import (
    InterruptedTaskContinuationClaimPage as InterruptedTaskContinuationClaimPage,
)
from cayu.tasks.handoff import TaskInterruptedHandoffConflict as TaskInterruptedHandoffConflict
from cayu.tasks.handoff import TaskInterruptedHandoffReceipt as TaskInterruptedHandoffReceipt
from cayu.tasks.handoff import TaskInterruptedHandoffRequest as TaskInterruptedHandoffRequest
from cayu.tasks.handoff import interrupted_task_handoff_request as interrupted_task_handoff_request
from cayu.tasks.handoff import (
    new_interrupted_task_continuation_handoff_id as new_interrupted_task_continuation_handoff_id,
)
from cayu.tasks.handoff import (
    prepare_interrupted_task_continuation_claim_page as prepare_interrupted_task_continuation_claim_page,
)
from cayu.tasks.handoff import prepare_interrupted_task_handoff as prepare_interrupted_task_handoff
from cayu.tasks.handoff import (
    prepare_interrupted_task_handoff_candidate_page as prepare_interrupted_task_handoff_candidate_page,
)
from cayu.tasks.handoff import (
    prepare_interrupted_task_handoff_receipt_lookup as prepare_interrupted_task_handoff_receipt_lookup,
)
from cayu.tasks.queries import TaskAggregateFilter as TaskAggregateFilter
from cayu.tasks.queries import TaskOperationalSnapshot as TaskOperationalSnapshot
from cayu.tasks.queries import TaskOrder as TaskOrder
from cayu.tasks.queries import TaskQuery as TaskQuery
from cayu.tasks.queries import TaskStatusCounts as TaskStatusCounts
from cayu.tasks.queries import (
    _ensure_claim_query_supported,
    _sort_tasks,
    _task_matches,
    _task_matches_claim_filter,
    _work_attempt_discovery_query,
)
from cayu.tasks.queries import copy_task_aggregate_filter as copy_task_aggregate_filter
from cayu.tasks.queries import copy_task_query as copy_task_query
from cayu.tasks.queries import task_query_from_aggregate_filter as task_query_from_aggregate_filter
from cayu.tasks.records import _HELD_TASK_STATUSES, _TERMINAL_TASK_STATUSES, _validate_positive_int
from cayu.tasks.records import TaskClaimLost as TaskClaimLost
from cayu.tasks.records import TaskSessionClosureClaim as TaskSessionClosureClaim
from cayu.tasks.records import copy_task_session_closure_claim as copy_task_session_closure_claim
from cayu.tasks.retry import TaskRetryAttemptDisposition as TaskRetryAttemptDisposition
from cayu.tasks.retry import TaskRetryAttemptReport as TaskRetryAttemptReport
from cayu.tasks.retry import TaskRetryEvent as TaskRetryEvent
from cayu.tasks.retry import TaskRetryEventType as TaskRetryEventType
from cayu.tasks.retry import TaskRetrySettlementRequest as TaskRetrySettlementRequest
from cayu.tasks.retry import TaskRetrySettlementResult as TaskRetrySettlementResult
from cayu.tasks.retry import (
    _cancelled_task_retry_settlement,
    _claimed_task_retry_attempt_elapsed,
    _elapsed_claimed_task_retry_settlement,
    _expired_task_retry_settlement,
    _replay_task_retry_cancellation_reconciliation,
    _replay_task_retry_settlement,
    _scheduled_task_nonexecution,
    _settled_task_retry_attempt,
    _task_retry_attempt_elapsed,
    _task_retry_cancellation_requested_event,
    _task_retry_events,
    _task_retry_runtime_idempotency_key,
    _validate_task_retry_settlement_receipt_identity,
    _validated_task_retry_terminal_accounting,
)
from cayu.tasks.retry import prepare_task_retry_settlement as prepare_task_retry_settlement
from cayu.tasks.terminalization import (
    TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES as TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES,
)
from cayu.tasks.terminalization import TaskTerminalizationConflict as TaskTerminalizationConflict
from cayu.tasks.terminalization import TaskTerminalizationReceipt as TaskTerminalizationReceipt
from cayu.tasks.terminalization import TaskTerminalizationRequest as TaskTerminalizationRequest
from cayu.tasks.terminalization import (
    TaskTerminalizationRetryPolicy as TaskTerminalizationRetryPolicy,
)
from cayu.tasks.terminalization import (
    TaskTerminalizationRetryResult as TaskTerminalizationRetryResult,
)
from cayu.tasks.terminalization import TaskTerminalizationUncertain as TaskTerminalizationUncertain
from cayu.tasks.terminalization import TaskTerminalKind as TaskTerminalKind
from cayu.tasks.terminalization import (
    _replay_task_terminalization_receipt,
    _task_terminalization_request_matches_sha256,
)
from cayu.tasks.terminalization import prepare_task_terminalization as prepare_task_terminalization
from cayu.tasks.terminalization import (
    prepare_task_terminalization_receipt_lookup as prepare_task_terminalization_receipt_lookup,
)
from cayu.tasks.work_receipts import (
    CompletionDecisionApplicationReceipt as CompletionDecisionApplicationReceipt,
)
from cayu.tasks.work_receipts import WorkAttemptLifecycleReceipt as WorkAttemptLifecycleReceipt
from cayu.tasks.work_receipts import (
    WorkAttemptPreparationHoldReceipt as WorkAttemptPreparationHoldReceipt,
)

if TYPE_CHECKING:
    from cayu.tasks._group_maintenance import TaskGroupMaintenance
    from cayu.tasks.graphs import (
        TaskGraphCreate,
        TaskGraphCreationReceipt,
        TaskGraphEvent,
        TaskGraphMember,
        TaskGraphSnapshot,
    )
    from cayu.tasks.groups import (
        TaskGroupCreate,
        TaskGroupCreationReceipt,
        TaskGroupEvent,
        TaskGroupInvocationObligation,
        TaskGroupQuiescenceResolution,
        TaskGroupSnapshot,
    )


from cayu._clock import normalize_utc_datetime, utc_clock
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    revalidate_model_input,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import (
    require_durable_nonblank as require_nonblank,
)
from cayu.approvals.tools import (
    ResolutionActor,
    copy_resolution_actor,
)
from cayu.budgets.aggregates import (
    _IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL,
    EXACT_AGGREGATE,
    _cooperate_with_in_memory_aggregate_cancellation,
)
from cayu.runtime._durable_worker_loop import (
    DurableWorkerDemandPolicy,
    DurableWorkerPoller,
    DurableWorkerPollerGroup,
)
from cayu.runtime._task_admission_wakeup import (
    TaskAdmissionWakeup,
    TaskAdmissionWakeupBroker,
)
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
    local_execution_effect_scope,
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
from cayu.sessions.invocation import (
    SessionInvocationBinding,
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
from cayu.tasks.completion_evaluations import (
    CompletionEvaluationRun,
    CompletionEvaluationRunRequest,
    CompletionEvaluationSettlementRequest,
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
    validate_work_completion_linked_id,
    work_attempt_request_sha256,
)
from cayu.tasks.records import _CONTRACT_TASK_JSON_FIELDS as _CONTRACT_TASK_JSON_FIELDS
from cayu.tasks.records import (
    _TASK_RETRY_COST_MAX_DECIMAL_PLACES as _TASK_RETRY_COST_MAX_DECIMAL_PLACES,
)
from cayu.tasks.records import _TASK_RETRY_COST_MAX_DIGITS as _TASK_RETRY_COST_MAX_DIGITS
from cayu.tasks.records import (
    _TASK_RETRY_RECONCILIATION_IDENTITY_MAX_BYTES as _TASK_RETRY_RECONCILIATION_IDENTITY_MAX_BYTES,
)
from cayu.tasks.records import (
    _TASK_RETRY_TOTAL_COST_MAX_DIGITS as _TASK_RETRY_TOTAL_COST_MAX_DIGITS,
)
from cayu.tasks.records import Task as Task
from cayu.tasks.records import TaskRetryPolicy as TaskRetryPolicy
from cayu.tasks.records import TaskRetrySeriesDisposition as TaskRetrySeriesDisposition
from cayu.tasks.records import TaskRetrySeriesSnapshot as TaskRetrySeriesSnapshot
from cayu.tasks.records import TaskStatus as TaskStatus
from cayu.tasks.records import _bounded_task_retry_decimal as _bounded_task_retry_decimal
from cayu.tasks.records import _copy_task_retry_policy as _copy_task_retry_policy
from cayu.tasks.records import _copy_task_retry_series_snapshot as _copy_task_retry_series_snapshot
from cayu.tasks.records import _preflight_bounded_task_payloads as _preflight_bounded_task_payloads
from cayu.tasks.records import (
    _task_retry_attempt_authority_sha256 as _task_retry_attempt_authority_sha256,
)
from cayu.tasks.records import (
    _validate_task_retry_cost_currency as _validate_task_retry_cost_currency,
)
from cayu.tasks.records import (
    _validate_task_retry_reconciliation_identity as _validate_task_retry_reconciliation_identity,
)
from cayu.tasks.records import copy_task as copy_task
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
from cayu.tasks.topology import (
    TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT as TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT,
)
from cayu.tasks.topology import TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH as TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH
from cayu.tasks.topology import TASK_TOPOLOGY_MAX_BRANCH_LIMIT as TASK_TOPOLOGY_MAX_BRANCH_LIMIT
from cayu.tasks.topology import TASK_TOPOLOGY_MAX_CURSOR_BYTES as TASK_TOPOLOGY_MAX_CURSOR_BYTES
from cayu.tasks.topology import (
    TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES as TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
)
from cayu.tasks.topology import (
    TASK_TOPOLOGY_MAX_EXPANDED_PARENTS as TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
)
from cayu.tasks.topology import (
    TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS as TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
)
from cayu.tasks.topology import (
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES as TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
)
from cayu.tasks.topology import TASK_TOPOLOGY_MAX_NODES as TASK_TOPOLOGY_MAX_NODES
from cayu.tasks.topology import (
    TASK_TOPOLOGY_MAX_VALIDATION_NODES as TASK_TOPOLOGY_MAX_VALIDATION_NODES,
)
from cayu.tasks.topology import TaskTopologyChildBranch as TaskTopologyChildBranch
from cayu.tasks.topology import TaskTopologyCycle as TaskTopologyCycle
from cayu.tasks.topology import TaskTopologyInconsistent as TaskTopologyInconsistent
from cayu.tasks.topology import TaskTopologyNode as TaskTopologyNode
from cayu.tasks.topology import TaskTopologyQuery as TaskTopologyQuery
from cayu.tasks.topology import TaskTopologySessionBranch as TaskTopologySessionBranch
from cayu.tasks.topology import TaskTopologyStoreResult as TaskTopologyStoreResult
from cayu.tasks.topology import (
    TaskTopologyTraversalLimitExceeded as TaskTopologyTraversalLimitExceeded,
)
from cayu.tasks.topology import TaskTopologyTruncatedField as TaskTopologyTruncatedField
from cayu.tasks.topology import (
    _allocate_task_topology_branch_limits as _allocate_task_topology_branch_limits,
)
from cayu.tasks.topology import (
    _bounded_optional_task_topology_parent_id as _bounded_optional_task_topology_parent_id,
)
from cayu.tasks.topology import _bounded_task_topology_display as _bounded_task_topology_display
from cayu.tasks.topology import _bounded_task_topology_text as _bounded_task_topology_text
from cayu.tasks.topology import (
    _copy_task_topology_branch_limits as _copy_task_topology_branch_limits,
)
from cayu.tasks.topology import (
    _reject_loaded_task_topology_cycles as _reject_loaded_task_topology_cycles,
)
from cayu.tasks.topology import _reject_task_parent_link_cycles as _reject_task_parent_link_cycles
from cayu.tasks.topology import _retain_task_topology_page as _retain_task_topology_page
from cayu.tasks.topology import _validate_task_topology_ancestry as _validate_task_topology_ancestry
from cayu.tasks.topology import _validate_task_topology_page as _validate_task_topology_page
from cayu.tasks.topology import build_task_topology_result as build_task_topology_result
from cayu.tasks.topology import decode_task_topology_cursor as decode_task_topology_cursor
from cayu.tasks.topology import encode_task_topology_cursor as encode_task_topology_cursor

_DURABLE_WORKER_POLLER_REGISTRY_LOCK = Lock()


class TaskStore(ABC):
    """Persistent store for durable work items.

    Worker-lease and recovery-owner transitions use store-authoritative time.
    Implementations sample, compare, and stamp that time inside the atomic
    ownership mutation; worker-provided timestamps or cutoffs are not lease
    authority.
    """

    _task_group_maintenance: TaskGroupMaintenance

    supports_delayed_availability: ClassVar[bool] = False
    supports_task_graphs: ClassVar[bool] = False
    supports_task_groups: ClassVar[bool] = False
    supports_task_group_quiescence: ClassVar[bool] = False

    async def list_task_group_reconciliation_candidates(
        self,
        *,
        after_group_id: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """Read an ordered bounded page of draining groups, without changing them."""
        raise NotImplementedError("This TaskStore does not support group quiescence.")

    async def reconcile_task_group(self, group_id: str) -> TaskGroupSnapshot:
        raise NotImplementedError("This TaskStore does not support group quiescence.")

    async def _settle_task_group_execution(self, task: Task) -> None:
        """Internal execution-owner boundary; terminal task status is not this proof."""
        if self.supports_task_group_quiescence:
            raise NotImplementedError("Group-capable stores must settle execution ownership.")

    async def _task_group_cancellation_requested(self, task_id: str) -> bool:
        """Internal store-owned stop observation, including exact retry descendants."""
        if self.supports_task_group_quiescence:
            raise NotImplementedError("Group-capable stores must expose cancellation intent.")
        return False

    async def _task_group_retains_execution(self, task_id: str) -> bool:
        """Resolve selected membership or exact retry lineage from store authority."""
        if self.supports_task_group_quiescence:
            raise NotImplementedError("Group-capable stores must resolve execution ownership.")
        return False

    async def _observe_task_group_invocation(
        self, invocation: TaskGroupInvocationObligation
    ) -> None:
        """Internal runtime binding/release, never a public quiescence assertion."""
        if self.supports_task_group_quiescence:
            raise NotImplementedError(
                "Group-capable stores must observe exact invocation ownership."
            )

    async def _observe_task_group_result_resolution(
        self, task_id: str, decision_id: str, owner_id: str, *, settled: bool
    ) -> None:
        """Internal retained resolver dispatch/settlement; not caller proof."""
        if self.supports_task_group_quiescence:
            raise NotImplementedError("Group-capable stores must own result resolution.")

    async def resolve_task_group_quiescence(
        self,
        request: TaskGroupQuiescenceResolution,
    ) -> TaskGroupSnapshot:
        raise NotImplementedError("This TaskStore does not support group quiescence.")

    async def create_task_group(self, request: TaskGroupCreate) -> TaskGroupCreationReceipt:
        """Atomically admit a new graph and immutable group, or replay its receipt."""
        raise NotImplementedError("This TaskStore does not support task groups.")

    async def load_task_group(self, group_id: str) -> TaskGroupSnapshot | None:
        """Read the group decision and current selected-member states consistently."""
        raise NotImplementedError("This TaskStore does not support task groups.")

    async def list_task_group_events(
        self, group_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskGroupEvent]:
        """Read bounded, ordered, task-store-owned group evidence."""
        raise NotImplementedError("This TaskStore does not support task groups.")

    async def create_task_graph(self, request: TaskGraphCreate) -> TaskGraphCreationReceipt:
        """Atomically admit a bounded graph, or replay its exact creation receipt.

        Preserve runtime-prepared invocation authority when copying requests.
        Expected external parents must match inside the admission transaction;
        receipts bind both the submitted request and resolved admission digests.
        """
        raise NotImplementedError("This TaskStore does not support task graphs.")

    async def load_task_graph(self, graph_id: str) -> TaskGraphSnapshot | None:
        """Read current graph members and retained terminal evidence atomically."""
        raise NotImplementedError("This TaskStore does not support task graphs.")

    async def list_task_graph_events(
        self, graph_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskGraphEvent]:
        """Read one bounded page of durable, task-store-owned graph events."""
        raise NotImplementedError("This TaskStore does not support task graphs.")

    supports_task_scheduling: ClassVar[bool] = False
    supports_task_topology: ClassVar[bool] = False
    supports_idempotent_terminalization: ClassVar[bool] = False
    supports_attached_task_recovery_terminalization: ClassVar[bool] = False
    supports_interrupted_task_handoffs: ClassVar[bool] = False
    supports_exact_interrupted_task_handoffs: ClassVar[bool] = False
    supports_task_cancellation_reconciliation: ClassVar[bool] = False
    supports_task_retry_series: ClassVar[bool] = False
    supports_verified_work_contracts: ClassVar[bool] = False
    supports_completion_verifier_dispatches: ClassVar[bool] = False
    supports_completion_evaluations: ClassVar[bool] = False
    supports_work_attempt_admission: ClassVar[bool] = False
    supports_verified_task_worker: ClassVar[bool] = False
    supports_local_execution_attempts: ClassVar[bool] = False
    supports_session_closure_deletion: ClassVar[bool] = False
    supports_session_closure_claims: ClassVar[bool] = False
    verified_work_mutations_are_cancellation_quiescent: ClassVar[bool] = False
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.UNVERIFIED

    # ``supports_verified_work_contracts`` alone is not settlement authority.
    # A class may set this flag to exactly ``True`` only when each verified-work
    # mutation implementation it owns has stopped mutating before its awaitable
    # returns or raises, including after caller cancellation. Every concrete
    # subclass must redeclare the proof even when the public mutation is
    # inherited, because subclass helpers and wrappers can change its behavior.

    def _enable_task_admission_wakeups(self) -> None:
        """Enable the built-in process-local, content-free wakeup optimization."""

        self._task_admission_wakeup_broker = TaskAdmissionWakeupBroker()

    def _durable_worker_poller(
        self,
        queries: Iterable[TaskQuery | None],
        policy: DurableWorkerDemandPolicy,
        *,
        clock: Callable[[], float],
    ) -> DurableWorkerPoller:
        """Join the fair active-poller group for one exact claim-filter cohort."""

        copied_queries = tuple(copy_task_query(query) for query in queries)
        if not copied_queries:
            raise ValueError("Durable worker pollers require at least one claim query.")
        for query in copied_queries:
            _ensure_claim_query_supported(query)
        cohort_key = (
            policy,
            tuple(
                sorted(
                    {
                        (
                            "" if query.status is None else query.status.value,
                            "" if query.type is None else query.type,
                            "" if query.parent_task_id is None else query.parent_task_id,
                            (
                                "all"
                                if query.has_work_contract is None
                                else "contracted"
                                if query.has_work_contract
                                else "ordinary"
                            ),
                            (
                                ""
                                if query.assigned_agent_name is None
                                else query.assigned_agent_name
                            ),
                        )
                        for query in copied_queries
                    }
                )
            ),
        )
        with _DURABLE_WORKER_POLLER_REGISTRY_LOCK:
            groups = getattr(self, "_durable_worker_poller_groups", None)
            if groups is None:
                groups = {}
                self._durable_worker_poller_groups = groups
            if type(groups) is not dict:
                raise RuntimeError("TaskStore durable worker poller state is invalid.")
            group = groups.get(cohort_key)
            if group is None:

                def remove_empty_group(empty_group: DurableWorkerPollerGroup) -> None:
                    with _DURABLE_WORKER_POLLER_REGISTRY_LOCK:
                        if (
                            groups.get(cohort_key) is empty_group
                            and empty_group.subscriber_count == 0
                        ):
                            del groups[cohort_key]

                group = DurableWorkerPollerGroup(on_empty=remove_empty_group)
                groups[cohort_key] = group
            if not isinstance(group, DurableWorkerPollerGroup):
                raise RuntimeError("TaskStore durable worker poller cohort is invalid.")
        return group.subscribe(policy, clock=clock)

    async def _task_admission_wakeup(
        self,
        queries: Iterable[TaskQuery | None],
    ) -> TaskAdmissionWakeup | None:
        """Subscribe one worker to a conservative union of claim filters.

        Custom stores inherit a polling-only fallback unless they explicitly
        enable the built-in broker. The returned edge is never claim authority.
        """

        broker = getattr(self, "_task_admission_wakeup_broker", None)
        if not isinstance(broker, TaskAdmissionWakeupBroker):
            return None
        copied_queries = tuple(copy_task_query(query) for query in queries)
        if not copied_queries:
            raise ValueError("Task admission wakeups require at least one claim query.")
        for query in copied_queries:
            _ensure_claim_query_supported(query)

        def matches(admitted: object) -> bool:
            if type(admitted) is not Task:
                return False
            task = admitted
            if task.status is not TaskStatus.PENDING or task.session_id is not None:
                return False
            return any(
                (query.status is None or query.status is TaskStatus.PENDING)
                and _task_matches_claim_filter(task, query)
                for query in copied_queries
            )

        return broker.subscribe(matches)

    def _publish_task_admission_wakeup(
        self,
        task: Task,
        *,
        now: datetime,
    ) -> None:
        """Publish an edge after one immediately claimable task is committed."""

        if type(task) is not Task:
            return
        try:
            now = normalize_utc_datetime(now, "now")
        except (TypeError, ValueError):
            return
        if (
            task.status is not TaskStatus.PENDING
            or task.session_id is not None
            or (task.available_at is not None and task.available_at > now)
        ):
            return
        broker = getattr(self, "_task_admission_wakeup_broker", None)
        if isinstance(broker, TaskAdmissionWakeupBroker):
            try:
                broker.publish(task)
            except Exception:
                # A process-local optimization cannot turn a committed task
                # creation into an apparent publication failure.
                return

    def _publish_task_admission_broadcast(self) -> None:
        """Publish one content-free conservative edge from a backend notifier."""

        broker = getattr(self, "_task_admission_wakeup_broker", None)
        if isinstance(broker, TaskAdmissionWakeupBroker):
            try:
                broker.publish()
            except Exception:
                return

    async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
        """Publish one immutable version or replay its exact canonical content."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
        """Load the exact contract named by ``reference`` or reject a fingerprint conflict."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_active_work_contract_task_for_session(
        self,
        session_id: str,
    ) -> Task | None:
        """Load a contracted task whose binding retains authority over a session.

        A terminal task does not implicitly release pending session work into the
        ordinary runtime. Only work-attempt lifecycle settlement retires a binding
        (``retired_contract_binding`` on its receipt): when the task completes
        through an accepted decision or is cancelled by group cancellation. Until
        then the durable session binding remains authoritative and callers must
        start a new ordinary session.
        """
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def admit_ordinary_session_execution(self, session_id: str) -> None:
        """Atomically admit a session to the ordinary, non-verifier runtime.

        Supporting stores must reject admission while a contracted task binding
        retains authority over the session and must durably prevent later contract
        attachment to an admitted session. Task terminalization alone is not a
        release; only the lifecycle settlement that retires the contract binding
        is. Repeated admission of the same session is idempotent.
        """
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def hold_claimed_work_contract_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        lease_expires_at: datetime | None = None,
        contract: WorkContractRef,
    ) -> Task:
        """Claim-fence an unsupported contracted task into operator attention."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def begin_work_attempt(self, request: WorkAttemptCreate) -> WorkAttempt:
        """Create or replay one bounded execution attempt under a task's frozen contract."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def enter_work_attempt_execution(
        self, request: WorkAttemptExecutionEntryRequest
    ) -> WorkAttemptExecutionEntryResult:
        """Elect one dispatch; an existing entry permits reconciliation only."""
        raise NotImplementedError("This TaskStore does not support verified task workers.")

    async def hold_work_attempt_preparation(
        self, request: WorkAttemptPreparationHold
    ) -> WorkAttemptPreparationHoldReceipt:
        """Atomically park an exact unattached claim and publish its failure receipt."""
        raise NotImplementedError("This TaskStore does not support verified task workers.")

    async def record_work_attempt_execution_stop(
        self, request: WorkAttemptExecutionStopRequest
    ) -> WorkAttemptAdmission:
        """Record immutable stop intent; exact replay never authorizes dispatch."""
        raise NotImplementedError("This TaskStore does not support verified task workers.")

    async def load_work_attempt_preparation_hold_receipt(
        self, hold_id: str
    ) -> WorkAttemptPreparationHoldReceipt | None:
        """Read an advisory original result; exact replay must compare the request."""
        raise NotImplementedError("This TaskStore does not support verified task workers.")

    async def prepare_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptAdmission:
        """Reserve one exact initial or continuation attempt before session mutation."""
        raise NotImplementedError("This TaskStore does not support work-attempt admission.")

    async def activate_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionActivate,
    ) -> WorkAttemptAdmission:
        """Publish a prepared attempt after exact session admission evidence exists."""
        raise NotImplementedError("This TaskStore does not support work-attempt admission.")

    async def load_work_attempt_admission(
        self,
        admission_id: str,
    ) -> WorkAttemptAdmission | None:
        """Load one durable admission intent or receipt by stable identity."""
        raise NotImplementedError("This TaskStore does not support work-attempt admission.")

    async def load_work_attempt_execution_claim(
        self,
        claim_id: str,
    ) -> WorkAttemptExecutionClaim | None:
        """Load one immutable execution-claim generation by stable identity."""
        raise NotImplementedError("This TaskStore does not support work-attempt admission.")

    async def load_latest_work_attempt_admission(
        self,
        task_id: str,
    ) -> WorkAttemptAdmission | None:
        """Read the admission with no successor, including pre-dispatch preparation.

        This is discovery, not acquisition of execution authority. The caller
        must still use exact admission/claim operations before dispatch.
        """
        raise NotImplementedError("This TaskStore does not support verified-task discovery.")

    async def settle_work_attempt_lifecycle(
        self, request: WorkAttemptLifecycleSettlement
    ) -> WorkAttemptLifecycleReceipt:
        """Atomically publish or exactly replay final task and release evidence."""
        raise NotImplementedError("This TaskStore does not support verified-task settlement.")

    async def list_unsettled_work_attempt_admissions(
        self,
        *,
        task_filter: TaskAggregateFilter | None = None,
        limit: int = 100,
        after: str | None = None,
    ) -> list[WorkAttemptAdmission]:
        """Discover latest unfinished admissions in admission-ID keyset order.

        Includes final decision application awaiting lifecycle retirement, even
        when the task is already completed. Results never acquire authority.
        """
        raise NotImplementedError("This TaskStore does not support verified-task discovery.")

    async def load_work_attempt_lifecycle_receipt(
        self, admission_id: str
    ) -> WorkAttemptLifecycleReceipt | None:
        """Read the original final outcome, including after acknowledgement loss."""
        raise NotImplementedError("This TaskStore does not support verified-task settlement.")

    async def prepare_local_execution_attempt(
        self,
        authority: LocalExecutionAttemptAuthority,
    ) -> LocalExecutionAttemptRecord:
        """Prepare or exactly replay one task-claim-bound local execution attempt."""

        raise NotImplementedError("This TaskStore does not support local execution attempts.")

    async def start_local_execution_attempt(
        self,
        start: LocalExecutionAttemptStart,
    ) -> LocalExecutionAttemptRecord:
        """Publish exact supervisor or root launch authority before further work."""

        raise NotImplementedError("This TaskStore does not support local execution attempts.")

    async def settle_local_execution_attempt(
        self,
        settlement: LocalExecutionAttemptSettlement,
    ) -> LocalExecutionAttemptRecord:
        """Commit or replay one exact terminal/quiescence receipt."""

        raise NotImplementedError("This TaskStore does not support local execution attempts.")

    async def load_local_execution_attempt(
        self,
        attempt_id: str,
    ) -> LocalExecutionAttemptRecord | None:
        """Load one local execution attempt by immutable identity."""

        raise NotImplementedError("This TaskStore does not support local execution attempts.")

    async def list_unsettled_local_execution_attempts(
        self,
        *,
        limit: int = 100,
        after: LocalExecutionAttemptListCursor | None = None,
    ) -> tuple[LocalExecutionAttemptRecord, ...]:
        """List one keyset page that still lacks positive containment settlement."""

        raise NotImplementedError("This TaskStore does not support local execution attempts.")

    async def claim_local_execution_attempt_recovery(
        self,
        claim: LocalExecutionAttemptRecoveryClaim,
    ) -> LocalExecutionAttemptRecord:
        """Claim one exact recovery generation without releasing task ownership."""

        raise NotImplementedError("This TaskStore does not support local execution attempts.")

    async def renew_work_attempt_execution_claim(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        """Renew an exact live preparing, active, or recovering claim in place."""
        raise NotImplementedError("This TaskStore does not support work-attempt admission.")

    async def claim_work_attempt_recovery(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        """Claim recovery after the prior execution generation has expired."""
        raise NotImplementedError("This TaskStore does not support work-attempt admission.")

    async def activate_work_attempt_recovery(
        self,
        request: WorkAttemptRecoveryActivate,
    ) -> WorkAttemptAdmission:
        """Activate a replacement only after positive session-settlement evidence."""
        raise NotImplementedError("This TaskStore does not support work-attempt admission.")

    async def submit_admitted_completion_proposal(
        self,
        request: AdmittedCompletionProposalRequest,
    ) -> CompletionProposal:
        """Publish one proposal under the exact current execution claim."""
        raise NotImplementedError("This TaskStore does not support work-attempt admission.")

    async def load_work_attempt(self, attempt_id: str) -> WorkAttempt | None:
        """Load one work attempt by stable identity."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def submit_completion_proposal(
        self,
        request: CompletionProposalCreate,
    ) -> CompletionProposal:
        """Persist a worker proposal without granting completion authority."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_proposal(self, proposal_id: str) -> CompletionProposal | None:
        """Load one completion proposal by stable identity."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_proposal_for_attempt(
        self, attempt_id: str
    ) -> CompletionProposal | None:
        """Load the unique published proposal for an attempt, without guessing its ID."""
        raise NotImplementedError("This TaskStore does not support verified task workers.")

    async def prepare_completion_verifier_profile(
        self,
        request: CompletionVerifierProfilePreparationRequest,
    ) -> CompletionVerifierProfileRecord:
        """Insert or exactly replay immutable verifier-profile authority.

        Implementations must atomically bind an adoption idempotency key to at
        most one proposal within its task. A later proposal cannot reuse that
        task-scoped transition identity.
        """
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        """Load the immutable verifier profile prepared for one proposal."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_prior_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        """Load the immediately preceding task attempt's verifier profile."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def claim_completion_verification(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        """Claim bounded exclusive authority to verify one undecided proposal."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_verification_claim(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        """Load the latest verification claim, including an expired claim for recovery."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def renew_completion_verification_claim(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        """Extend one exact live verification claim without changing its owner."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def record_completion_decision(
        self,
        request: CompletionDecisionCreate,
    ) -> CompletionDecision:
        """Persist the one authoritative verifier decision for a proposal."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_decision(self, decision_id: str) -> CompletionDecision | None:
        """Load one completion decision by stable identity."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_decision_for_proposal(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        """Load the one authoritative decision published for a proposal."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def apply_completion_decision(
        self,
        request: CompletionDecisionApplicationRequest,
    ) -> Task:
        """Apply or exactly replay a decision-bound task transition."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_decision_application_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        """Load exact durable evidence for decision application reconciliation."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def record_completion_verifier_dispatch(
        self,
        request: CompletionVerifierDispatchRequest,
    ) -> CompletionVerifierDispatch:
        """Persist one provider-verifier attempt intent before the provider is entered.

        Implementations must atomically read the current verification claim,
        prepared verifier profile, decision index and prior dispatches for the
        proposal, call ``require_completion_verifier_dispatch_admission`` and
        insert the record, or exactly replay an identical intent.
        """
        raise NotImplementedError("This TaskStore does not support provider verifier dispatches.")

    async def settle_completion_verifier_dispatch(
        self,
        request: CompletionVerifierDispatchSettlementRequest,
    ) -> CompletionVerifierDispatch:
        """Write the one terminal settlement for a recorded provider attempt."""
        raise NotImplementedError("This TaskStore does not support provider verifier dispatches.")

    async def list_completion_verifier_dispatches(
        self,
        proposal_id: str,
    ) -> tuple[CompletionVerifierDispatch, ...]:
        """List one proposal's provider attempts in their durable order."""
        raise NotImplementedError("This TaskStore does not support provider verifier dispatches.")

    async def record_completion_evaluation_run(
        self,
        request: CompletionEvaluationRunRequest,
    ) -> CompletionEvaluationRun:
        """Persist one evaluator-run intent before its external effect.

        Implementations must atomically read the current verification claim,
        contract evaluation policy, decision index and prior runs for the
        proposal, call ``require_completion_evaluation_admission`` and insert the
        record, or exactly replay an identical intent.
        """
        raise NotImplementedError("This TaskStore does not support completion evaluations.")

    async def settle_completion_evaluation_run(
        self,
        request: CompletionEvaluationSettlementRequest,
    ) -> CompletionEvaluationRun:
        """Write the one terminal settlement for a recorded evaluator run."""
        raise NotImplementedError("This TaskStore does not support completion evaluations.")

    async def list_completion_evaluation_runs(
        self,
        proposal_id: str,
    ) -> tuple[CompletionEvaluationRun, ...]:
        """List one proposal's evaluator runs in their durable order."""
        raise NotImplementedError("This TaskStore does not support completion evaluations.")

    @abstractmethod
    async def create_task(self, request: TaskCreate) -> Task:
        """Create a task and its immutable invocation provenance atomically.

        Implementations must mint the final task ID, load any requested parent,
        and call ``task_invocation_for_create`` inside the same create boundary.
        """

    async def reschedule_task(self, request: TaskRescheduleRequest) -> TaskScheduleReceipt:
        """Atomically replace an exact unadmitted schedule, or replay its receipt."""
        raise NotImplementedError("This TaskStore does not support task scheduling.")

    async def cancel_scheduled_task(
        self, request: TaskScheduleCancelRequest
    ) -> TaskScheduleReceipt:
        """Cancel an exact schedule without releasing live execution ownership."""
        raise NotImplementedError("This TaskStore does not support task scheduling.")

    async def list_task_schedule_events(
        self, task_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskScheduleEvent]:
        """Read bounded task-owned scheduling evidence in durable sequence order."""
        raise NotImplementedError("This TaskStore does not support task scheduling.")

    async def next_task_schedule_wakeup(self, query: TaskQuery | None = None) -> TaskScheduleWakeup:
        """Observe the next matching deadline; the result grants no claim authority."""
        raise NotImplementedError("This TaskStore does not support task scheduling.")

    @abstractmethod
    async def create_running_task(
        self,
        request: TaskCreate,
        *,
        session_invocation: SessionInvocationBinding,
    ) -> Task:
        """Atomically create a running task already attached to its session.

        ``request.session_id`` is required. This avoids leaving an attached,
        unclaimable pending task if a process stops between separate create and
        start operations. ``session_invocation`` must describe that exact
        session so the atomic insert cannot create a contradictory structural
        attachment and invocation record.
        """

    @abstractmethod
    async def load_task(self, task_id: str, *, _access_bounds=None) -> Task | None:
        """Load a task by id."""

    async def load_active_attached_task_worker(
        self,
        task_id: str,
        worker_id: str,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        """Load exact active attached-worker authority using store time.

        Stores that support worker-owned session resume must override this
        projection. A stale or mismatched owner raises ``TaskClaimLost`` before
        the session can admit provider or tool work.
        """

        raise NotImplementedError(
            "This TaskStore does not support active attached-worker authority reads."
        )

    async def load_direct_attached_task_resume(
        self,
        task_id: str,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        """Load a direct workerless attachment with no recovery authority.

        A committed interrupted-handoff generation is reserved for an elected
        recovery owner and must reject an ordinary ownerless resume.
        """

        if self.supports_interrupted_task_handoffs:
            raise NotImplementedError(
                "Recovery-capable TaskStores must implement an atomic direct "
                "attached-task resume read."
            )
        task = await self.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        _require_direct_attached_task_resume(
            task,
            session_id=session_id,
            session_instance_id=session_instance_id,
        )
        return task

    @abstractmethod
    async def load_invocation_snapshot(
        self,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        """Load bounded immutable provenance without task payloads or metadata."""

    @abstractmethod
    async def list_tasks(
        self, query: TaskQuery | None = None, *, _access_bounds=None
    ) -> list[Task]:
        """List tasks for dashboards, queues, and orchestration."""

    async def load_session_closure_claim(self, session_id: str) -> TaskSessionClosureClaim | None:
        """Read the retained task-set authority without reopening admission."""
        raise NotImplementedError("Task store does not support session closure claims.")

    async def claim_session_closure(
        self, claim: TaskSessionClosureClaim
    ) -> TaskSessionClosureClaim:
        """Reserve an exact, complete quiescent task set before independent deletion.

        Retain the claim after deletion so future creation/attachment cannot
        revive the retired session's task namespace. Exact retries are read-only.
        """
        raise NotImplementedError("This TaskStore does not support session closure claims.")

    async def delete_session_tasks(
        self,
        session_id: str,
        *,
        task_ids: tuple[str, ...],
        policy: Any,
    ) -> None:
        """Delete a quiescent session task set owned by this store."""

        raise NotImplementedError("This TaskStore does not support session task closure deletion.")

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        """Count current task states in one store-local read snapshot.

        Default raises ``NotImplementedError`` so existing out-of-tree stores
        remain instantiable when they do not expose this control-plane read model.
        """
        raise NotImplementedError(
            "This TaskStore does not support operational aggregate snapshots."
        )

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        """Read bounded task/session and direct-child branches.

        Default raises ``NotImplementedError`` so existing out-of-tree stores
        remain instantiable while advertising the operation as unsupported.
        """
        raise NotImplementedError("This TaskStore does not support topology queries.")

    @abstractmethod
    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
        session_invocation: SessionInvocationBinding | None = None,
    ) -> Task:
        """Mark a pending task as running, optionally attached to a session.

        A session provenance binding is required for attachment. Stores must
        validate it before changing lifecycle state.
        """

    @abstractmethod
    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        session_invocation: SessionInvocationBinding,
        worker_id: str,
        lease_expires_at: datetime | None = None,
    ) -> Task:
        """Attach a live worker-claimed task to a session and mark it running.

        Raise ``TaskClaimLost`` unless the worker and exact lease generation
        still own a live claim.
        """

    @abstractmethod
    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any],
        *,
        worker_id: str | None = None,
        lease_expires_at: datetime | None = None,
        handoff_id: str | None = None,
    ) -> Task:
        """Mark a pending or running task as completed.

        If ``worker_id`` is given, ``lease_expires_at`` must identify the exact
        active lease generation. The update also requires the exact continuation
        generation, so a stale worker cannot clobber a task after renewal or
        reclaim.
        """

    @abstractmethod
    async def fail_task(
        self,
        task_id: str,
        error: dict[str, Any],
        *,
        worker_id: str | None = None,
        lease_expires_at: datetime | None = None,
        handoff_id: str | None = None,
    ) -> Task:
        """Mark a pending or running task as failed.

        If ``worker_id`` is given, ``lease_expires_at`` must identify the exact
        active lease generation. The update also requires the exact continuation
        generation.
        """

    async def terminalize_task(self, request: TaskTerminalizationRequest) -> Task:
        """Atomically terminalize one live claim or replay its exact receipt.

        Custom stores opt into this operation by overriding it. The default keeps
        existing out-of-tree ``TaskStore`` implementations instantiable without
        claiming acknowledgement-loss safety they do not provide.
        """
        raise NotImplementedError(
            "This TaskStore does not support idempotent task terminalization."
        )

    async def load_task_terminalization_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskTerminalizationReceipt | None:
        """Load exact durable commit evidence for receipt reconciliation."""
        raise NotImplementedError("This TaskStore does not support task terminalization receipts.")

    async def recover_attached_task_failure(
        self,
        request: TaskTerminalizationRequest,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        """Fail one exact expired task owner after its attached session is fenced.

        Custom stores opt into this operation explicitly. Implementations must
        atomically validate the task's session incarnation, worker, lease, and
        handoff generation before bypassing the ordinary active-lease rule.
        """

        raise NotImplementedError(
            "This TaskStore does not support attached-task recovery terminalization."
        )

    async def release_interrupted_task_worker(
        self,
        request: TaskInterruptedHandoffRequest,
    ) -> TaskInterruptedHandoffReceipt:
        """Release a live exact owner and atomically publish replay evidence."""

        raise NotImplementedError(
            "This TaskStore does not support idempotent interrupted-task handoffs."
        )

    async def recover_interrupted_task_worker(
        self,
        request: TaskInterruptedHandoffRequest,
    ) -> TaskInterruptedHandoffReceipt:
        """Release the exact expired owner from reconstructed interruption authority."""

        raise NotImplementedError(
            "This TaskStore does not support interrupted-task handoff recovery."
        )

    async def load_interrupted_task_handoff_receipt(
        self,
        task_id: str,
        handoff_id: str,
    ) -> TaskInterruptedHandoffReceipt | None:
        """Load immutable commit evidence for one exact handoff operation."""

        raise NotImplementedError(
            "This TaskStore does not support interrupted-task handoff receipts."
        )

    async def list_expired_interrupted_task_handoff_candidates(
        self,
        *,
        after: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Task]:
        """List one stable page of expired owners using store-authoritative time."""

        raise NotImplementedError(
            "This TaskStore does not support interrupted-task handoff recovery."
        )

    async def load_expired_interrupted_task_handoff_candidate(
        self,
        task_id: str,
    ) -> Task | None:
        """Load one exact expired attached owner using store-authoritative time."""

        raise NotImplementedError(
            "This TaskStore does not support exact interrupted-task handoff recovery."
        )

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
        """Scan one bounded page and lease its first authentic continuation.

        ``handoff_id`` is a caller-generated, one-use claim generation. ``task_id``
        is an optional exact selector available only from stores that advertise
        ``supports_exact_interrupted_task_handoffs``. Supporting
        stores must first replay an exact still-live claim with the same worker and
        generation, making commit-before-ack loss recoverable without transferring
        authority. Otherwise they select only running attached tasks whose current
        one-use handoff generation names a fully validated committed receipt and
        that have no live worker lease. Generation consumption and lease publication
        are one atomic mutation so competing recovery owners cannot invoke the same
        continuation concurrently. ``after`` and ``scan_limit`` bound a stable
        creation-order scan. The result advances past rejected authority without
        allowing one malformed candidate to block later independent work.
        """

        raise NotImplementedError(
            "This TaskStore does not support interrupted-task continuation claims."
        )

    async def reconcile_task_cancellation(
        self,
        request: TaskCancellationReconciliationRequest,
    ) -> TaskCancellationReconciliationResult:
        """Settle an owner-lost ordinary cancellation from positive evidence.

        Lease expiry establishes only that the original owner is lost. The
        application must supply positive quiescence or external-effect evidence.
        Supporting stores commit that evidence with the ordinary cancellation
        terminalization receipt in one atomic transition.
        """

        raise NotImplementedError(
            "This TaskStore does not support ordinary task cancellation reconciliation."
        )

    async def settle_task_retry_attempt(
        self,
        request: TaskRetrySettlementRequest,
    ) -> TaskRetrySettlementResult:
        """Atomically record one attempt and optionally create its delayed successor."""

        raise NotImplementedError("This TaskStore does not support task retry series.")

    async def load_task_retry_settlement(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskRetrySettlementResult | None:
        """Load an exact retry-attempt settlement receipt for acknowledgement recovery."""

        raise NotImplementedError("This TaskStore does not support task retry series.")

    async def reconcile_task_retry_cancellation(
        self,
        request: TaskRetryCancellationReconciliationRequest,
    ) -> TaskRetrySettlementResult:
        """Settle an exact owner-lost cancellation from positive application evidence.

        Lease expiry establishes only that the original owner is lost. The request's
        typed evidence remains the application's positive quiescence/effect proof.
        Custom retry-capable stores must implement the transition atomically with the
        ordinary retry settlement receipt.
        """

        raise NotImplementedError(
            "This TaskStore does not support task retry cancellation reconciliation."
        )

    async def enforce_task_retry_deadline(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
        token_count: int = 0,
        estimated_cost: Decimal = Decimal(0),
    ) -> TaskRetrySettlementResult | None:
        """Atomically terminalize an owned attempt only when store time exhausted it.

        Returning ``None`` is positive store evidence that the cumulative deadline
        had not elapsed at this check. Retry-capable custom stores must override
        this operation so workers never substitute their own wall clock.
        """

        raise NotImplementedError("This TaskStore does not support task retry deadlines.")

    async def task_retry_deadline_elapsed(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> bool:
        """Return store-authoritative elapsed evidence without releasing ownership."""

        raise NotImplementedError("This TaskStore does not support task retry deadlines.")

    @abstractmethod
    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        """Cancel idle work or durably request cancellation from its live owner.

        Tasks with a live worker remain fenced until that worker proves its
        dispatched work quiescent and commits the cancellation receipt.
        """

    async def request_claimed_task_cancellation(
        self,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
        error: dict[str, Any] | None = None,
    ) -> Task:
        """Request cancellation only for one exact live worker lease.

        A delayed stale owner must not place a cancellation marker on a claim
        acquired by a replacement worker.
        """

        raise NotImplementedError(
            "This TaskStore does not support exact claimed-task cancellation."
        )

    async def mark_claimed_task_execution_started(
        self,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
    ) -> Task:
        """Fence an exact claim before its worker can dispatch opaque work.

        Supporting stores persist ``started_at`` while the exact worker lease is
        still live.  Exact replay is idempotent.  Once this marker exists, expiry
        reclamation must enter cancellation reconciliation instead of making the
        task claimable while the original effect may still be running.
        """

        raise NotImplementedError(
            "This TaskStore does not support exact claimed-task execution starts."
        )

    @abstractmethod
    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Pause a pending or unattached running task until app code resumes it."""

    @abstractmethod
    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a pending or unattached running task as blocked on an external dependency."""

    @abstractmethod
    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a pending or unattached running task as waiting for human/operator input."""

    @abstractmethod
    async def resume_task(self, task_id: str) -> Task:
        """Return a paused, blocked, or attention-needed task to the pending queue."""

    @abstractmethod
    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        """Atomically claim the next pending task matching ``query``."""

    @abstractmethod
    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
        handoff_id: str | None = None,
        extend_seconds: int = 300,
    ) -> Task:
        """Extend the exact worker-owned lease identified by its expiry.

        Raise ``TaskClaimLost`` if the worker and lease expiry no longer identify
        the live claim.
        """

    @abstractmethod
    async def release_task(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> Task:
        """Release the exact claimed task lease back to pending.

        Raise ``TaskClaimLost`` if the worker and lease expiry no longer identify
        the live claim.
        """

    @abstractmethod
    async def release_attached_task_worker(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> Task:
        """Release an exact lease while preserving a running task's session link.

        Raise ``TaskClaimLost`` if the worker and lease expiry no longer identify
        the live claim.
        """

    @abstractmethod
    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        """Return expired claimed task leases to pending."""


class _PreparedMemoryTaskWrite(NamedTuple):
    task: Task
    prior: Task | None
    events: tuple[TaskScheduleEvent, ...]


def runtime_task_creation(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.tasks.access import runtime_task_creation as wrap

        return await wrap(operation)(self, *args, **kwargs)

    return guarded


def runtime_collection_read(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.tasks.access import runtime_collection_read as wrap

        return await wrap(operation)(self, *args, **kwargs)

    return guarded


def runtime_task_mutation(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.tasks.access import runtime_task_mutation as wrap

        return await wrap(operation)(self, *args, **kwargs)

    return guarded


@model_store_surface("tasks")
class InMemoryTaskStore(TaskStore):
    """In-process task store for tests, local development, and examples."""

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
        from cayu.tasks._memory_groups import reconciliation_candidates

        return await reconciliation_candidates(self, after_group_id=after_group_id, limit=limit)

    async def _settle_task_group_execution(self, task: Task) -> None:
        from cayu.tasks._memory_groups import settle_execution

        await settle_execution(self, task)

    async def _task_group_cancellation_requested(self, task_id: str) -> bool:
        from cayu.tasks._memory_groups import cancellation_requested

        return await cancellation_requested(self, task_id)

    async def _task_group_retains_execution(self, task_id: str) -> bool:
        from cayu.tasks._memory_groups import retains_execution

        return await retains_execution(self, task_id)

    async def _observe_task_group_invocation(
        self, invocation: TaskGroupInvocationObligation
    ) -> None:
        from cayu.tasks._memory_groups import observe_invocation

        await observe_invocation(self, invocation)

    async def _observe_task_group_result_resolution(
        self, task_id: str, decision_id: str, owner_id: str, *, settled: bool
    ) -> None:
        from cayu.tasks._memory_groups import observe_result_resolution

        await observe_result_resolution(self, task_id, decision_id, owner_id, settled=settled)

    async def reconcile_task_group(self, group_id: str) -> TaskGroupSnapshot:
        from cayu.tasks._memory_groups import reconcile

        return await reconcile(self, group_id)

    async def resolve_task_group_quiescence(
        self,
        request: TaskGroupQuiescenceResolution,
    ) -> TaskGroupSnapshot:
        from cayu.tasks._memory_groups import reconcile

        return await reconcile(self, request.group_id, resolution=request)

    @runtime_task_creation
    async def create_task_group(self, request: TaskGroupCreate) -> TaskGroupCreationReceipt:
        from cayu.tasks._memory_groups import create_group

        return await create_group(self, request, submitted_digest=request._submitted_request_sha256)

    @runtime_collection_read
    async def load_task_group(self, group_id: str) -> TaskGroupSnapshot | None:
        from cayu.tasks._memory_groups import load_group

        return await load_group(self, group_id)

    @runtime_collection_read
    async def list_task_group_events(
        self, group_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskGroupEvent]:
        from cayu.tasks._memory_groups import list_events

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
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.DEVELOPMENT

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        ownership_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._enable_task_admission_wakeups()
        self._lock = asyncio.Lock()
        self._clock = utc_clock(clock)
        self._ownership_clock = utc_clock(ownership_clock)
        self._tasks: dict[str, Task] = {}
        self._task_groups: dict[str, TaskGroupSnapshot] = {}
        self._task_group_by_graph: dict[str, str] = {}
        self._task_group_events: dict[str, list[TaskGroupEvent]] = {}
        self._task_group_resolutions: dict[tuple[str, str], tuple[str, TaskGroupSnapshot]] = {}
        self._task_graph_receipts: dict[str, TaskGraphCreationReceipt] = {}
        self._task_graph_members: dict[str, dict[str, tuple[str, ...]]] = {}
        self._task_graph_events: dict[str, list[TaskGraphEvent]] = {}
        self._task_graph_by_task: dict[str, str] = {}
        self._task_group_retry_lineage: dict[str, tuple[str, str]] = {}
        self._task_graph_terminal_members: dict[str, TaskGraphMember] = {}
        self._schedule_receipts: dict[tuple[str, str], TaskScheduleReceipt] = {}
        self._schedule_events: dict[str, list[TaskScheduleEvent]] = {}
        self._session_closure_claims: dict[str, TaskSessionClosureClaim] = {}
        self._task_id_by_interrupted_handoff_id: dict[str, str] = {}
        self._interrupted_continuation_claims: dict[str, tuple[str, str]] = {}
        self._terminalization_receipts: dict[tuple[str, str], TaskTerminalizationReceipt] = {}
        self._interrupted_handoff_receipts: dict[
            tuple[str, str], TaskInterruptedHandoffReceipt
        ] = {}
        self._retry_settlements: dict[tuple[str, str], TaskRetrySettlementResult] = {}
        self._retry_reconciliation_rejections: dict[
            tuple[str, str],
            _TaskRetryCancellationReconciliationRejectionRecord,
        ] = {}
        self._cancellation_reconciliation_rejections: dict[
            tuple[str, str],
            _TaskCancellationReconciliationRejectionRecord,
        ] = {}
        self._work_contracts: dict[tuple[str, int], WorkContract] = {}
        self._work_attempts: dict[str, WorkAttempt] = {}
        self._attempt_ids_by_task: dict[str, list[str]] = {}
        self._work_attempt_admissions: dict[str, WorkAttemptAdmission] = {}
        self._admission_id_by_attempt: dict[str, str] = {}
        self._admission_id_by_session_interaction: dict[tuple[str, str], str] = {}
        self._unreleased_admission_id_by_session: dict[str, str] = {}
        self._latest_admission_id_by_task: dict[str, str] = {}
        self._work_attempt_lifecycle_receipts: dict[str, WorkAttemptLifecycleReceipt] = {}
        self._lifecycle_admission_by_settlement_id: dict[str, str] = {}
        self._work_attempt_preparation_holds: dict[str, WorkAttemptPreparationHoldReceipt] = {}
        self._work_attempt_execution_claims: dict[str, WorkAttemptExecutionClaim] = {}
        self._local_execution_attempts: dict[str, LocalExecutionAttemptRecord] = {}
        self._local_execution_attempt_by_lineage: dict[tuple[str, str], str] = {}
        self._completion_proposals: dict[str, CompletionProposal] = {}
        self._proposal_id_by_attempt: dict[str, str] = {}
        self._completion_verifier_profiles: dict[str, CompletionVerifierProfileRecord] = {}
        self._completion_verifier_dispatches: dict[str, list[CompletionVerifierDispatch]] = {}
        self._completion_verifier_dispatch_proposals: dict[str, str] = {}
        self._completion_evaluation_runs: dict[str, list[CompletionEvaluationRun]] = {}
        self._completion_evaluation_run_proposals: dict[str, str] = {}
        self._completion_verification_claims: dict[str, CompletionVerificationClaim] = {}
        self._verification_claims_by_id: dict[str, CompletionVerificationClaim] = {}
        self._completion_decisions: dict[str, CompletionDecision] = {}
        self._decision_id_by_proposal: dict[str, str] = {}
        self._decision_application_receipts: dict[
            tuple[str, str], CompletionDecisionApplicationReceipt
        ] = {}
        self._decision_application_key_by_decision: dict[str, tuple[str, str]] = {}
        self._ordinary_execution_session_ids: set[str] = set()
        self._contracted_task_ids_by_session: dict[str, dict[str, None]] = {}
        self._task_keys_by_session: dict[str, list[tuple[datetime, str]]] = {}
        self._task_keys_by_parent: dict[str, list[tuple[datetime, str]]] = {}

    @runtime_task_creation
    async def create_task_graph(self, request: TaskGraphCreate) -> TaskGraphCreationReceipt:
        from cayu.tasks._memory_graphs import create_graph

        return await create_graph(self, request)

    @runtime_collection_read
    async def load_task_graph(self, graph_id: str) -> TaskGraphSnapshot | None:
        from cayu.tasks._memory_graphs import load_graph

        return await load_graph(self, graph_id)

    @runtime_collection_read
    async def list_task_graph_events(
        self, graph_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskGraphEvent]:
        from cayu.tasks._memory_graphs import list_graph_events

        return await list_graph_events(self, graph_id, after_sequence=after_sequence, limit=limit)

    async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
        contract = copy_work_contract(contract)
        key = (contract.contract_id, contract.version)
        async with self._lock:
            existing = self._work_contracts.get(key)
            if existing is not None:
                if existing != contract:
                    raise WorkContractConflict(
                        "Work-contract identity is already bound to different content."
                    )
                return copy_work_contract(existing)
            if contract.supersedes is not None:
                predecessor = self._work_contracts.get(
                    (contract.supersedes.contract_id, contract.supersedes.version)
                )
                if predecessor is None:
                    raise WorkContractConflict("Work-contract predecessor has not been published.")
                if predecessor.reference() != contract.supersedes:
                    raise WorkContractConflict(
                        "Work-contract predecessor fingerprint conflicts with durable history."
                    )
            self._work_contracts[key] = contract
            return copy_work_contract(contract)

    async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
        copied_reference = copy_work_contract_ref(reference)
        if copied_reference is None:  # pragma: no cover - excluded by the public type
            raise TypeError("reference must be a WorkContractRef.")
        async with self._lock:
            contract = self._work_contracts.get(
                (copied_reference.contract_id, copied_reference.version)
            )
            if contract is None:
                return None
            if contract.fingerprint != copied_reference.fingerprint:
                raise WorkContractConflict(
                    "Work-contract reference conflicts with the published fingerprint."
                )
            return copy_work_contract(contract)

    async def load_active_work_contract_task_for_session(
        self,
        session_id: str,
    ) -> Task | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            task = self._active_work_contract_task_for_session(session_id)
            return None if task is None else task.model_copy(deep=True)

    async def admit_ordinary_session_execution(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            contracted_task = self._active_work_contract_task_for_session(session_id)
            requires_completion_decision = contracted_task is not None
            del contracted_task
            if requires_completion_decision:
                raise TaskCompletionDecisionRequired(
                    "Contracted tasks require the verifier-aware execution entrance."
                ) from None
            self._ordinary_execution_session_ids.add(session_id)

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
        if copied_contract is None:  # pragma: no cover - excluded by the public type
            raise TypeError("contract must be a WorkContractRef.")
        async with self._lock:
            task = self._require_task(task_id)
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
                raise TaskClaimLost("Only the current worker may park its unattached claimed task.")
            self._ensure_task_contract_matches(task, copied_contract)
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
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def begin_work_attempt(self, request: WorkAttemptCreate) -> WorkAttempt:
        request = copy_work_attempt_create(request)
        request_sha256 = work_attempt_request_sha256(request)
        async with self._lock:
            if request.attempt_id in self._admission_id_by_attempt:
                raise WorkAttemptAdmissionConflict(
                    "Admitted work attempts must be created through their exact admission."
                )
            existing = self._work_attempts.get(request.attempt_id)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Work-attempt identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            task = self._require_task(request.task_id)
            if task.id in self._latest_admission_id_by_task:
                raise WorkAttemptAdmissionConflict(
                    "Task is permanently governed by runtime-owned work-attempt admission."
                )
            contract = self._ensure_task_contract_matches(task, request.contract)
            if task.status is not TaskStatus.RUNNING:
                raise ValueError("Work attempts require a running contracted task.")
            if task.session_id != request.session_id:
                raise WorkCompletionConflict("Work attempt is bound to a different task session.")
            self._ensure_attempt_worker_matches(task, request.worker_id)
            attempt_ids = self._attempt_ids_by_task.get(task.id, [])
            if len(attempt_ids) >= contract.continuation_policy.max_attempts:
                raise WorkCompletionConflict(
                    "Work-contract attempt limit forbids another work attempt."
                )
            if attempt_ids:
                prior_attempt_id = attempt_ids[-1]
                prior_proposal_id = self._proposal_id_by_attempt.get(prior_attempt_id)
                prior_decision_id = (
                    None
                    if prior_proposal_id is None
                    else self._decision_id_by_proposal.get(prior_proposal_id)
                )
                if prior_decision_id is None:
                    raise WorkCompletionConflict(
                        "A prior work attempt has not reached a durable decision."
                    )
                if prior_decision_id not in self._decision_application_key_by_decision:
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
                ordinal=len(attempt_ids) + 1,
                request_sha256=request_sha256,
                started_at=self._clock(),
            )
            self._work_attempts[attempt.attempt_id] = attempt
            self._attempt_ids_by_task.setdefault(task.id, []).append(attempt.attempt_id)
            return attempt.model_copy(deep=True)

    async def prepare_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptAdmission:
        from cayu.tasks._memory_groups import cancellation_requested_unlocked

        request = copy_work_attempt_admission_prepare(request)
        if request.generation != 1:
            raise WorkAttemptAdmissionConflict(
                "A new work-attempt admission must start at execution generation 1."
            )
        request_sha256 = work_attempt_admission_prepare_sha256(request)
        async with self._lock:
            existing = self._work_attempt_admissions.get(request.admission_id)
            if existing is not None:
                if not work_attempt_admission_prepare_matches_sha256(
                    request,
                    existing.prepare_request_sha256,
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Work-attempt admission identity is bound to another request."
                    )
                return self._copy_work_attempt_admission(existing)
            prior_admission_id = self._admission_id_by_attempt.get(request.attempt_id)
            if prior_admission_id is not None:
                raise WorkAttemptAdmissionConflict(
                    "Work-attempt identity is already bound to another admission."
                )
            if request.attempt_id in self._work_attempts:
                raise WorkAttemptAdmissionConflict(
                    "Work-attempt identity already exists without this admission."
                )
            if request.claim_id in self._work_attempt_execution_claims:
                raise WorkAttemptAdmissionConflict(
                    "Execution-claim identity is bound to another admission."
                )
            if (
                request.session_id,
                request.interaction_id,
            ) in self._admission_id_by_session_interaction:
                raise WorkAttemptAdmissionConflict(
                    "Session interaction is already bound to another admission."
                )
            if request.session_id in self._unreleased_admission_id_by_session:
                raise WorkAttemptAdmissionConflict(
                    "Session already has an unreleased work-attempt admission."
                )
            latest_admission_id = self._latest_admission_id_by_task.get(request.task_id)
            if latest_admission_id is not None:
                latest_admission = self._work_attempt_admissions[latest_admission_id]
                if latest_admission.state is not WorkAttemptAdmissionState.RELEASED:
                    raise WorkAttemptAdmissionConflict(
                        "Task already has an unreleased work-attempt admission."
                    )

            task = self._require_task(request.task_id)
            contract = self._ensure_task_contract_matches(task, request.contract)
            lease_now = self._ownership_clock()
            availability_now = self._clock()
            if cancellation_requested_unlocked(self, task.id):
                raise WorkAttemptAdmissionConflict(
                    "A decided group loser cannot admit another execution."
                )
            continuation = self._work_attempt_continuation_context(task, contract, request)
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

            self._ensure_contract_session_accepts_attachment(
                request.contract,
                request.session_id,
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
                source_execution_profile_fingerprint=(request.source_execution_profile_fingerprint),
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
            self._store_task(updated_task)
            self._work_attempt_admissions[admission.admission_id] = admission
            self._admission_id_by_attempt[admission.attempt_id] = admission.admission_id
            self._admission_id_by_session_interaction[
                (admission.session_id, admission.interaction_id)
            ] = admission.admission_id
            self._unreleased_admission_id_by_session[admission.session_id] = admission.admission_id
            self._latest_admission_id_by_task[admission.task_id] = admission.admission_id
            self._work_attempt_execution_claims[claim.claim_id] = claim
            return self._copy_work_attempt_admission(admission)

    async def activate_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionActivate,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_admission_activate(request)
        async with self._lock:
            admission = self._require_work_attempt_admission(request.admission_id)
            if admission.prepare_request_sha256 != request.prepare_request_sha256:
                raise WorkAttemptAdmissionConflict(
                    "Admission activation conflicts with its prepared request."
                )
            if admission.claim.claim_id != request.claim_id:
                raise WorkAttemptExecutionClaimLost(
                    "Admission activation no longer owns the prepared execution claim."
                )
            task = self._require_task(admission.task_id)
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
                return self._copy_work_attempt_admission(admission)
            if admission.state is not WorkAttemptAdmissionState.PREPARING:
                raise WorkAttemptAdmissionConflict(
                    "Only a prepared admission can publish its work attempt."
                )
            lease_now = self._ownership_clock()
            self._ensure_live_work_attempt_claim(admission, now=lease_now)
            if (
                task.status is not TaskStatus.RUNNING
                or task.worker_id != admission.claim.worker_id
                or task.lease_expires_at != admission.claim.lease_expires_at
            ):
                raise WorkAttemptExecutionClaimLost(
                    "Prepared admission conflicts with current task ownership."
                )
            attempt_ids = self._attempt_ids_by_task.get(task.id, [])
            if (
                len(attempt_ids)
                >= self._require_work_contract(admission.contract).continuation_policy.max_attempts
            ):
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
                ordinal=len(attempt_ids) + 1,
                request_sha256=work_attempt_request_sha256(attempt_request),
                started_at=(evidence_now := self._clock()),
            )
            activated = admission.model_copy(
                update={
                    "state": WorkAttemptAdmissionState.ACTIVE,
                    "attempt": attempt,
                    "session_evidence_sha256": request.session_evidence_sha256,
                    "activated_at": evidence_now,
                }
            )
            activated = self._copy_work_attempt_admission(activated)
            self._work_attempts[attempt.attempt_id] = attempt
            self._attempt_ids_by_task.setdefault(task.id, []).append(attempt.attempt_id)
            self._work_attempt_admissions[activated.admission_id] = activated
            return self._copy_work_attempt_admission(activated)

    async def load_work_attempt_admission(
        self,
        admission_id: str,
    ) -> WorkAttemptAdmission | None:
        admission_id = require_clean_nonblank(admission_id, "admission_id")
        async with self._lock:
            admission = self._work_attempt_admissions.get(admission_id)
            return None if admission is None else self._copy_work_attempt_admission(admission)

    async def load_latest_work_attempt_admission(
        self,
        task_id: str,
    ) -> WorkAttemptAdmission | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            admission_id = self._latest_admission_id_by_task.get(task_id)
            if admission_id is None:
                return None
            admission = self._require_work_attempt_admission(admission_id)
            if admission.task_id != task_id:
                raise WorkAttemptAdmissionConflict("Latest admission conflicts with its task.")
            return self._copy_work_attempt_admission(admission)

    async def load_work_attempt_lifecycle_receipt(
        self, admission_id: str
    ) -> WorkAttemptLifecycleReceipt | None:
        admission_id = require_clean_nonblank(admission_id, "admission_id")
        async with self._lock:
            receipt = self._work_attempt_lifecycle_receipts.get(admission_id)
            return None if receipt is None else receipt.model_copy(deep=True)

    async def list_unsettled_work_attempt_admissions(
        self,
        *,
        task_filter: TaskAggregateFilter | None = None,
        limit: int = 100,
        after: str | None = None,
    ) -> list[WorkAttemptAdmission]:
        query, after = _work_attempt_discovery_query(task_filter, limit=limit, after=after)
        async with self._lock:
            selected: list[str] = []
            for index, (task_id, admission_id) in enumerate(
                self._latest_admission_id_by_task.items()
            ):
                if index % 128 == 0:
                    await asyncio.sleep(0)
                if (after is not None and admission_id <= after) or (
                    admission_id in self._work_attempt_lifecycle_receipts
                ):
                    continue
                task = self._require_task(task_id)
                if not _task_matches(task, query):
                    continue
                insort(selected, admission_id)
                if len(selected) > limit:
                    selected.pop()
            return [
                self._copy_work_attempt_admission(self._require_work_attempt_admission(identity))
                for identity in selected
            ]

    async def enter_work_attempt_execution(
        self, request: WorkAttemptExecutionEntryRequest
    ) -> WorkAttemptExecutionEntryResult:
        from cayu.runtime._work_attempt_lifecycle_policy import plan_work_attempt_execution_entry
        from cayu.tasks._memory_groups import cancellation_requested_unlocked

        request = copy_work_attempt_execution_entry_request(request)
        async with self._lock:
            admission = self._require_work_attempt_admission(request.admission_id)
            result = plan_work_attempt_execution_entry(
                request,
                admission=admission,
                task=self._require_task(admission.task_id),
                now=self._ownership_clock(),
                group_cancelled=cancellation_requested_unlocked(self, admission.task_id),
            )
            if result.disposition is WorkAttemptExecutionEntryDisposition.ENTERED:
                self._work_attempt_admissions[admission.admission_id] = (
                    self._copy_work_attempt_admission(result.admission)
                )
            return result.model_copy(deep=True)

    async def record_work_attempt_execution_stop(
        self, request: WorkAttemptExecutionStopRequest
    ) -> WorkAttemptAdmission:
        from cayu.runtime._work_attempt_lifecycle_policy import plan_work_attempt_execution_stop

        request = copy_work_attempt_execution_stop_request(request)
        async with self._lock:
            admission = self._require_work_attempt_admission(request.admission_id)
            result = plan_work_attempt_execution_stop(
                request,
                admission=admission,
                task=self._require_task(admission.task_id),
                now=self._ownership_clock(),
            )
            if admission.execution_stop is None:
                self._work_attempt_admissions[admission.admission_id] = (
                    self._copy_work_attempt_admission(result)
                )
            return self._copy_work_attempt_admission(result)

    async def load_work_attempt_preparation_hold_receipt(
        self, hold_id: str
    ) -> WorkAttemptPreparationHoldReceipt | None:
        hold_id = validate_work_completion_idempotency_key(hold_id)
        async with self._lock:
            receipt = self._work_attempt_preparation_holds.get(hold_id)
            return None if receipt is None else receipt.model_copy(deep=True)

    async def hold_work_attempt_preparation(
        self, request: WorkAttemptPreparationHold
    ) -> WorkAttemptPreparationHoldReceipt:
        from cayu.runtime._work_attempt_lifecycle_policy import plan_work_attempt_preparation_hold
        from cayu.tasks._memory_groups import cancellation_requested_unlocked

        request = copy_work_attempt_preparation_hold(request)
        digest = work_attempt_preparation_hold_sha256(request)
        async with self._lock:
            existing = self._work_attempt_preparation_holds.get(request.hold_id)
            if existing is not None:
                if existing.request_sha256 != digest:
                    raise WorkAttemptAdmissionConflict(
                        "Preparation hold conflicts with its receipt."
                    )
                return existing.model_copy(deep=True)
            task = self._require_task(request.task_id)
            self._ensure_task_contract_matches(task, request.contract)
            updated, receipt = plan_work_attempt_preparation_hold(
                request,
                task=task,
                has_attempt=bool(self._attempt_ids_by_task.get(task.id)),
                now=self._ownership_clock(),
                group_cancelled=cancellation_requested_unlocked(self, task.id),
            )
            self._store_task(
                updated,
                settled_execution=(task.id, request.worker_id, task.started_at)
                if task.started_at is not None
                else None,
            )
            self._work_attempt_preparation_holds[request.hold_id] = receipt
            return receipt.model_copy(deep=True)

    async def settle_work_attempt_lifecycle(
        self, request: WorkAttemptLifecycleSettlement
    ) -> WorkAttemptLifecycleReceipt:
        from cayu.runtime._work_attempt_lifecycle_policy import (
            plan_work_attempt_lifecycle_settlement,
        )

        request = copy_work_attempt_lifecycle_settlement(request)
        request_sha256 = work_attempt_lifecycle_settlement_sha256(request)
        async with self._lock:
            existing = self._work_attempt_lifecycle_receipts.get(request.admission_id)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkAttemptAdmissionConflict(
                        "Lifecycle settlement conflicts with its receipt."
                    )
                return existing.model_copy(deep=True)
            if request.settlement_id in self._lifecycle_admission_by_settlement_id:
                raise WorkAttemptAdmissionConflict(
                    "Lifecycle settlement identity is already bound."
                )
            admission = self._require_work_attempt_admission(request.admission_id)
            task = self._require_task(request.task_id)
            proposal_id = self._proposal_id_by_attempt.get(admission.attempt_id)
            proposal = None if proposal_id is None else self._completion_proposals.get(proposal_id)
            decision_id = (
                None if proposal_id is None else self._decision_id_by_proposal.get(proposal_id)
            )
            decision = None if decision_id is None else self._completion_decisions.get(decision_id)
            application = self._decision_application_receipts.get(
                (task.id, request.application_idempotency_key or "")
            )
            from cayu.tasks._memory_groups import cancellation_requested_unlocked

            updated, settled_admission, receipt = plan_work_attempt_lifecycle_settlement(
                request,
                task=task,
                admission=admission,
                latest_admission_id=self._latest_admission_id_by_task.get(task.id, ""),
                proposal=proposal,
                decision=decision,
                application=application,
                now=self._ownership_clock(),
                group_cancelled=cancellation_requested_unlocked(self, task.id),
                verification_claim=None
                if proposal_id is None
                else self._completion_verification_claims.get(proposal_id),
            )
            settled_admission = self._copy_work_attempt_admission(settled_admission)
            self._store_task(
                updated,
                settled_execution=(task.id, admission.claim.worker_id, task.started_at)
                if task.started_at is not None
                else None,
            )
            if receipt.retired_contract_binding:
                self._remove_contracted_session_index_entry(updated)
            self._work_attempt_admissions[admission.admission_id] = settled_admission
            if (
                self._unreleased_admission_id_by_session.get(admission.session_id)
                == admission.admission_id
            ):
                del self._unreleased_admission_id_by_session[admission.session_id]
            self._work_attempt_lifecycle_receipts[admission.admission_id] = receipt
            self._lifecycle_admission_by_settlement_id[request.settlement_id] = (
                admission.admission_id
            )
            return receipt.model_copy(deep=True)

    async def load_work_attempt_execution_claim(
        self,
        claim_id: str,
    ) -> WorkAttemptExecutionClaim | None:
        claim_id = require_clean_nonblank(claim_id, "claim_id")
        async with self._lock:
            claim = self._work_attempt_execution_claims.get(claim_id)
            return (
                None
                if claim is None
                else WorkAttemptExecutionClaim.model_validate(
                    claim.model_dump(mode="python", warnings=False)
                )
            )

    async def renew_work_attempt_execution_claim(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_execution_claim_request(request)
        async with self._lock:
            admission = self._require_work_attempt_admission(request.admission_id)
            if admission.state not in WORK_ATTEMPT_RENEWABLE_STATES:
                raise WorkAttemptExecutionClaimLost(
                    "A released admission cannot renew execution authority."
                )
            claim = admission.claim
            if (
                claim.claim_id != request.claim_id
                or claim.worker_id != request.worker_id
                or claim.execution_owner_id != request.execution_owner_id
                or claim.generation != request.generation
            ):
                raise WorkAttemptExecutionClaimLost(
                    "Execution-claim renewal conflicts with current authority."
                )
            lease_now = self._ownership_clock()
            self._ensure_live_work_attempt_claim(admission, now=lease_now)
            if admission.attempt_id in self._proposal_id_by_attempt:
                raise WorkAttemptExecutionClaimLost(
                    "Completion proposal has already closed execution authority."
                )
            renewed_claim = renewed_work_attempt_execution_claim(claim, request, now=lease_now)
            renewed = self._copy_work_attempt_admission(
                admission.model_copy(update={"claim": renewed_claim})
            )
            task = self._require_task(admission.task_id)
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
            self._store_task(
                task.model_copy(
                    update={
                        "lease_expires_at": renewed_claim.lease_expires_at,
                        "updated_at": lease_now,
                    }
                )
            )
            self._work_attempt_admissions[renewed.admission_id] = renewed
            self._work_attempt_execution_claims[renewed_claim.claim_id] = renewed_claim
            return self._copy_work_attempt_admission(renewed)

    async def claim_work_attempt_recovery(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_execution_claim_request(request)
        request_sha256 = work_attempt_execution_claim_request_sha256(request)
        async with self._lock:
            admission = self._require_work_attempt_admission(request.admission_id)
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
            task = self._require_task(admission.task_id)
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
                return self._copy_work_attempt_admission(admission)
            if admission.state is WorkAttemptAdmissionState.ACTIVE and exact_current_request:
                if current.lease_expires_at <= now:
                    raise WorkAttemptExecutionClaimLost(
                        "The active execution claim expired and must be replaced."
                    )
                return self._copy_work_attempt_admission(admission)
            if admission.state is WorkAttemptAdmissionState.RECOVERING:
                if exact_current_request:
                    if current.lease_expires_at <= now:
                        raise WorkAttemptExecutionClaimLost(
                            "The recovery claim expired and must be replaced."
                        )
                    return self._copy_work_attempt_admission(admission)
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
            from cayu.tasks._memory_groups import cancellation_requested_unlocked

            # Exact live replay above does not acquire new authority. A losing
            # execution must retain its original owner until positive settlement.
            if cancellation_requested_unlocked(self, admission.task_id):
                raise WorkAttemptAdmissionConflict(
                    "Task-group cancellation forbids replacement execution authority."
                )
            if admission.attempt_id in self._proposal_id_by_attempt:
                raise WorkAttemptAdmissionConflict(
                    "A proposed attempt cannot acquire replacement execution authority."
                )
            if request.claim_id in self._work_attempt_execution_claims:
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
            replacement_state = (
                WorkAttemptAdmissionState.PREPARING
                if preparing
                else WorkAttemptAdmissionState.RECOVERING
            )
            recovering = self._copy_work_attempt_admission(
                admission.model_copy(
                    update={
                        "state": replacement_state,
                        "claim": replacement,
                        "recovery_evidence_sha256": None,
                    }
                )
            )
            self._store_task(
                task.model_copy(
                    update={
                        "worker_id": request.worker_id,
                        "lease_expires_at": replacement.lease_expires_at,
                        "updated_at": now,
                    }
                )
            )
            self._work_attempt_admissions[recovering.admission_id] = recovering
            self._work_attempt_execution_claims[replacement.claim_id] = replacement
            return self._copy_work_attempt_admission(recovering)

    async def activate_work_attempt_recovery(
        self,
        request: WorkAttemptRecoveryActivate,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_recovery_activate(request)
        async with self._lock:
            admission = self._require_work_attempt_admission(request.admission_id)
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
            task = self._require_task(admission.task_id)
            if (
                task.session_id != admission.session_id
                or task.session_instance_id != admission.session_invocation.session_instance_id
            ):
                raise WorkAttemptExecutionClaimLost(
                    "Recovery activation conflicts with exact task-session authority."
                )
            if admission.state is WorkAttemptAdmissionState.ACTIVE:
                return self._copy_work_attempt_admission(admission)
            lease_now = self._ownership_clock()
            self._ensure_live_work_attempt_claim(admission, now=lease_now)
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
            self._ensure_attempt_state_is_current(task, admission.attempt)
            active = self._copy_work_attempt_admission(
                admission.model_copy(
                    update={
                        "state": WorkAttemptAdmissionState.ACTIVE,
                        "recovery_evidence_sha256": request.recovery_evidence_sha256,
                    }
                )
            )
            self._work_attempt_admissions[active.admission_id] = active
            return self._copy_work_attempt_admission(active)

    async def load_work_attempt(self, attempt_id: str) -> WorkAttempt | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            attempt = self._work_attempts.get(attempt_id)
            return None if attempt is None else attempt.model_copy(deep=True)

    async def submit_completion_proposal(
        self,
        request: CompletionProposalCreate,
    ) -> CompletionProposal:
        request = copy_completion_proposal_create(request)
        request_sha256 = completion_proposal_request_sha256(request)
        async with self._lock:
            admission_id = self._admission_id_by_attempt.get(request.attempt_id)
            if admission_id is not None:
                raise WorkAttemptAdmissionConflict(
                    "Admitted work attempts require claim-fenced proposal publication."
                )
            existing = self._completion_proposals.get(request.proposal_id)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Completion-proposal identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            prior_proposal_id = self._proposal_id_by_attempt.get(request.attempt_id)
            if prior_proposal_id is not None:
                raise WorkCompletionConflict(
                    "Work attempt already has a different completion proposal."
                )
            attempt = self._require_work_attempt(request.attempt_id)
            task = self._require_task(attempt.task_id)
            self._ensure_attempt_is_current(task, attempt)
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
            self._completion_proposals[proposal.proposal_id] = proposal
            self._proposal_id_by_attempt[attempt.attempt_id] = proposal.proposal_id
            return proposal.model_copy(deep=True)

    async def submit_admitted_completion_proposal(
        self,
        request: AdmittedCompletionProposalRequest,
    ) -> CompletionProposal:
        from cayu.tasks._memory_groups import cancellation_requested_unlocked

        request = copy_admitted_completion_proposal_request(request)
        proposal_request = request.proposal
        proposal_sha256 = completion_proposal_request_sha256(proposal_request)
        async with self._lock:
            admission = self._require_work_attempt_admission(request.admission_id)
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
            existing = self._completion_proposals.get(proposal_request.proposal_id)
            if admission.state is WorkAttemptAdmissionState.RELEASED:
                if (
                    existing is None
                    or existing.request_sha256 != proposal_sha256
                    or self._proposal_id_by_attempt.get(admission.attempt_id)
                    != proposal_request.proposal_id
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
            self._ensure_live_work_attempt_claim(admission, now=lease_now)
            if existing is not None:
                if existing.request_sha256 != proposal_sha256:
                    raise WorkCompletionConflict(
                        "Completion-proposal identity is bound to another request."
                    )
                return existing.model_copy(deep=True)
            prior_proposal_id = self._proposal_id_by_attempt.get(admission.attempt_id)
            if prior_proposal_id is not None:
                raise WorkCompletionConflict(
                    "Work attempt already has a different completion proposal."
                )
            task = self._require_task(admission.task_id)
            if cancellation_requested_unlocked(self, task.id):
                raise WorkAttemptAdmissionConflict(
                    "A decided group loser cannot submit a new proposal."
                )
            self._ensure_attempt_state_is_current(task, admission.attempt)
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
            released = self._copy_work_attempt_admission(
                admission.model_copy(update={"state": WorkAttemptAdmissionState.RELEASED})
            )
            self._store_task(
                task.model_copy(
                    update={
                        "worker_id": None,
                        "lease_expires_at": None,
                        "updated_at": lease_now,
                    }
                )
            )
            self._completion_proposals[proposal.proposal_id] = proposal
            self._proposal_id_by_attempt[admission.attempt_id] = proposal.proposal_id
            self._work_attempt_admissions[released.admission_id] = released
            if (
                self._unreleased_admission_id_by_session.get(released.session_id)
                == released.admission_id
            ):
                del self._unreleased_admission_id_by_session[released.session_id]
            return proposal.model_copy(deep=True)

    async def load_completion_proposal(self, proposal_id: str) -> CompletionProposal | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            proposal = self._completion_proposals.get(proposal_id)
            return None if proposal is None else proposal.model_copy(deep=True)

    async def load_completion_proposal_for_attempt(
        self, attempt_id: str
    ) -> CompletionProposal | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            proposal_id = self._proposal_id_by_attempt.get(attempt_id)
            if proposal_id is None:
                return None
            proposal = self._require_completion_proposal(proposal_id)
            if proposal.attempt_id != attempt_id:
                raise WorkCompletionConflict("Proposal index conflicts with its attempt.")
            return proposal.model_copy(deep=True)

    async def prepare_completion_verifier_profile(
        self,
        request: CompletionVerifierProfilePreparationRequest,
    ) -> CompletionVerifierProfileRecord:
        request = copy_completion_verifier_profile_preparation_request(request)
        request_sha256 = completion_verifier_profile_preparation_request_sha256(request)
        async with self._lock:
            existing = self._completion_verifier_profiles.get(request.proposal_id)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Completion-verifier profile is already bound to another request."
                    )
                return copy_completion_verifier_profile_record(existing)

            proposal = self._require_completion_proposal(request.proposal_id)
            attempt = self._require_work_attempt(proposal.attempt_id)
            contract = self._require_work_contract(proposal.contract)
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

            attempt_ids = self._attempt_ids_by_task.get(request.task_id, [])
            try:
                attempt_index = attempt_ids.index(attempt.attempt_id)
            except ValueError:
                raise WorkCompletionConflict(
                    "Completion-verifier profile refers to an unindexed work attempt."
                ) from None
            prior_profile: CompletionVerifierProfileRecord | None = None
            if attempt_index > 0:
                prior_attempt_id = attempt_ids[attempt_index - 1]
                prior_proposal_id = self._proposal_id_by_attempt.get(prior_attempt_id)
                if prior_proposal_id is None:
                    raise WorkCompletionConflict(
                        "Prior work attempt has no completion proposal authority."
                    )
                prior_profile = self._completion_verifier_profiles.get(prior_proposal_id)
                if prior_profile is None:
                    raise WorkCompletionConflict(
                        "Prior work attempt has no verifier-profile authority."
                    )
            require_completion_verifier_profile_transition(request, prior_profile)

            adoption = request.adoption
            if adoption is not None and any(
                profile.task_id == request.task_id
                and profile.adoption is not None
                and profile.adoption.idempotency_key == adoption.idempotency_key
                for profile in self._completion_verifier_profiles.values()
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
            self._completion_verifier_profiles[request.proposal_id] = record
            return copy_completion_verifier_profile_record(record)

    async def load_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            profile = self._completion_verifier_profiles.get(proposal_id)
            return None if profile is None else copy_completion_verifier_profile_record(profile)

    async def load_prior_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            proposal = self._require_completion_proposal(proposal_id)
            attempt_ids = self._attempt_ids_by_task.get(proposal.task_id, [])
            try:
                attempt_index = attempt_ids.index(proposal.attempt_id)
            except ValueError:
                raise WorkCompletionConflict(
                    "Completion proposal refers to an unindexed work attempt."
                ) from None
            if attempt_index == 0:
                return None
            prior_proposal_id = self._proposal_id_by_attempt.get(attempt_ids[attempt_index - 1])
            if prior_proposal_id is None:
                raise WorkCompletionConflict(
                    "Prior work attempt has no completion proposal authority."
                )
            profile = self._completion_verifier_profiles.get(prior_proposal_id)
            if profile is None:
                raise WorkCompletionConflict(
                    "Prior work attempt has no verifier-profile authority."
                )
            return copy_completion_verifier_profile_record(profile)

    async def claim_completion_verification(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        from cayu.tasks._memory_groups import cancellation_requested_unlocked

        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        async with self._lock:
            claim_by_id = self._verification_claims_by_id.get(request.claim_id)
            if claim_by_id is not None and (
                claim_by_id.proposal_id != request.proposal_id
                or claim_by_id.request_sha256 != request_sha256
            ):
                raise WorkCompletionConflict(
                    "Verification-claim identity is already bound to another request."
                )
            proposal = self._require_completion_proposal(request.proposal_id)
            contract = self._require_work_contract(proposal.contract)
            profile = self._completion_verifier_profiles.get(request.proposal_id)
            if request.verifier != contract.verifier:
                raise WorkCompletionConflict(
                    "Verification claim uses a verifier other than the frozen contract verifier."
                )
            if (
                profile is None
                or request.verifier_profile_fingerprint != profile.profile.fingerprint
            ):
                raise WorkCompletionConflict(
                    "Verification claim requires the exact prepared verifier profile."
                )
            now = self._ownership_clock()
            current = self._completion_verification_claims.get(request.proposal_id)
            if (
                proposal.proposal_id not in self._decision_id_by_proposal
                and cancellation_requested_unlocked(self, proposal.task_id)
            ):
                raise _GroupVerificationAdmissionRefused(
                    "A decided group loser cannot dispatch verification."
                )
            if (
                current is not None
                and current.claim_id == request.claim_id
                and current.request_sha256 == request_sha256
            ):
                if (
                    current.lease_expires_at > now
                    or proposal.proposal_id in self._decision_id_by_proposal
                ):
                    return current.model_copy(deep=True)
                raise CompletionVerificationClaimLost(
                    "Verification claim expired and cannot regain authority by replay."
                )
            if proposal.proposal_id in self._decision_id_by_proposal:
                raise WorkCompletionConflict("Completion proposal already has a durable decision.")
            if current is not None and current.lease_expires_at > now:
                raise CompletionVerificationClaimLost(
                    "Completion proposal is owned by another live verifier claim."
                )
            if claim_by_id is not None:
                raise CompletionVerificationClaimLost(
                    "Verification claim expired and cannot regain authority by replay."
                )
            self._ensure_completion_proposal_is_current(proposal)
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
            self._completion_verification_claims[proposal.proposal_id] = claim
            self._verification_claims_by_id[claim.claim_id] = claim
            return claim.model_copy(deep=True)

    async def load_completion_verification_claim(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            claim = self._completion_verification_claims.get(proposal_id)
            return None if claim is None else claim.model_copy(deep=True)

    async def renew_completion_verification_claim(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        async with self._lock:
            proposal = self._require_completion_proposal(request.proposal_id)
            current = self._completion_verification_claims.get(request.proposal_id)
            lease_now = self._ownership_clock()
            if (
                current is None
                or current.claim_id != request.claim_id
                or current.proposal_id != request.proposal_id
                or current.worker_id != request.worker_id
                or current.execution_owner_id != request.execution_owner_id
                or current.execution_timeout_seconds != request.execution_timeout_seconds
                or current.verifier != request.verifier
                or current.verifier_profile_fingerprint != request.verifier_profile_fingerprint
                or current.request_sha256 != request_sha256
                or current.lease_expires_at <= lease_now
                or proposal.proposal_id in self._decision_id_by_proposal
            ):
                raise CompletionVerificationClaimLost(
                    "Verification claim cannot be renewed without exact current live authority."
                )
            self._ensure_completion_proposal_is_current(proposal)
            renewed = CompletionVerificationClaim(
                claim_id=current.claim_id,
                lease_seconds=current.lease_seconds,
                proposal_id=current.proposal_id,
                worker_id=current.worker_id,
                execution_owner_id=current.execution_owner_id,
                execution_timeout_seconds=current.execution_timeout_seconds,
                verifier=current.verifier,
                verifier_profile_fingerprint=current.verifier_profile_fingerprint,
                attempt_number=current.attempt_number,
                request_sha256=current.request_sha256,
                claimed_at=current.claimed_at,
                lease_expires_at=max(
                    current.lease_expires_at,
                    lease_now + timedelta(seconds=request.lease_seconds),
                ),
            )
            self._completion_verification_claims[proposal.proposal_id] = renewed
            self._verification_claims_by_id[renewed.claim_id] = renewed
            return renewed.model_copy(deep=True)

    async def record_completion_decision(
        self,
        request: CompletionDecisionCreate,
    ) -> CompletionDecision:
        request = copy_completion_decision_create(request)
        request_sha256 = completion_decision_request_sha256(request)
        async with self._lock:
            existing = self._completion_decisions.get(request.decision_id)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Completion-decision identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            prior_decision_id = self._decision_id_by_proposal.get(request.proposal_id)
            if prior_decision_id is not None:
                raise WorkCompletionConflict(
                    "Completion proposal already has a different durable decision."
                )
            proposal = self._require_completion_proposal(request.proposal_id)
            profile = self._completion_verifier_profiles.get(request.proposal_id)
            claim = self._completion_verification_claims.get(proposal.proposal_id)
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
            self._ensure_completion_proposal_is_current(proposal)
            contract = self._require_work_contract(proposal.contract)
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
            self._completion_decisions[decision.decision_id] = decision
            self._decision_id_by_proposal[proposal.proposal_id] = decision.decision_id
            return decision.model_copy(deep=True)

    async def load_completion_decision(self, decision_id: str) -> CompletionDecision | None:
        decision_id = require_clean_nonblank(decision_id, "decision_id")
        async with self._lock:
            decision = self._completion_decisions.get(decision_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def load_completion_decision_for_proposal(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            decision_id = self._decision_id_by_proposal.get(proposal_id)
            decision = None if decision_id is None else self._completion_decisions.get(decision_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def record_completion_verifier_dispatch(
        self,
        request: CompletionVerifierDispatchRequest,
    ) -> CompletionVerifierDispatch:
        request = copy_completion_verifier_dispatch_request(request)
        async with self._lock:
            existing_proposal_id = self._completion_verifier_dispatch_proposals.get(
                request.dispatch_id
            )
            if existing_proposal_id is not None:
                existing = next(
                    item
                    for item in self._completion_verifier_dispatches[existing_proposal_id]
                    if item.dispatch_id == request.dispatch_id
                )
                return replay_completion_verifier_dispatch(existing, request)
            proposal = self._require_completion_proposal(request.proposal_id)
            contract = self._require_work_contract(proposal.contract)
            profile = self._completion_verifier_profiles.get(proposal.proposal_id)
            prior = tuple(self._completion_verifier_dispatches.get(proposal.proposal_id, ()))
            require_completion_verifier_dispatch_admission(
                request,
                proposal=proposal,
                contract_verifier=contract.verifier,
                claim=self._completion_verification_claims.get(proposal.proposal_id),
                profile_fingerprint=None if profile is None else profile.profile.fingerprint,
                decided=proposal.proposal_id in self._decision_id_by_proposal,
                existing=prior,
                lease_now=self._ownership_clock(),
            )
            self._ensure_completion_proposal_is_current(proposal)
            record = completion_verifier_dispatch_from_request(
                request,
                proposal=proposal,
                ordinal=len(prior) + 1,
                dispatched_at=self._clock(),
            )
            self._completion_verifier_dispatches.setdefault(proposal.proposal_id, []).append(record)
            self._completion_verifier_dispatch_proposals[record.dispatch_id] = proposal.proposal_id
            return copy_completion_verifier_dispatch(record)

    async def settle_completion_verifier_dispatch(
        self,
        request: CompletionVerifierDispatchSettlementRequest,
    ) -> CompletionVerifierDispatch:
        request = copy_completion_verifier_dispatch_settlement_request(request)
        async with self._lock:
            proposal_id = self._completion_verifier_dispatch_proposals.get(request.dispatch_id)
            if proposal_id is None:
                raise KeyError(f"Completion verifier dispatch not found: {request.dispatch_id}")
            dispatches = self._completion_verifier_dispatches[proposal_id]
            index = next(
                position
                for position, item in enumerate(dispatches)
                if item.dispatch_id == request.dispatch_id
            )
            updated, changed = settle_completion_verifier_dispatch_record(
                dispatches[index],
                request,
                settled_at=self._clock(),
            )
            if changed:
                dispatches[index] = updated
            return copy_completion_verifier_dispatch(updated)

    async def list_completion_verifier_dispatches(
        self,
        proposal_id: str,
    ) -> tuple[CompletionVerifierDispatch, ...]:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            return tuple(
                copy_completion_verifier_dispatch(item)
                for item in self._completion_verifier_dispatches.get(proposal_id, ())
            )

    async def record_completion_evaluation_run(
        self,
        request: CompletionEvaluationRunRequest,
    ) -> CompletionEvaluationRun:
        request = copy_completion_evaluation_run_request(request)
        async with self._lock:
            existing_proposal_id = self._completion_evaluation_run_proposals.get(request.effect_id)
            if existing_proposal_id is not None:
                existing = next(
                    item
                    for item in self._completion_evaluation_runs[existing_proposal_id]
                    if item.effect_id == request.effect_id
                )
                return replay_completion_evaluation_run(existing, request)
            proposal = self._require_completion_proposal(request.proposal_id)
            contract = self._require_work_contract(proposal.contract)
            prior = tuple(self._completion_evaluation_runs.get(proposal.proposal_id, ()))
            require_completion_evaluation_admission(
                request,
                proposal=proposal,
                contract_evaluation=contract.evaluation,
                claim=self._completion_verification_claims.get(proposal.proposal_id),
                decided=proposal.proposal_id in self._decision_id_by_proposal,
                existing=prior,
                lease_now=self._ownership_clock(),
            )
            self._ensure_completion_proposal_is_current(proposal)
            record = completion_evaluation_run_from_request(
                request, proposal=proposal, started_at=self._clock()
            )
            self._completion_evaluation_runs.setdefault(proposal.proposal_id, []).append(record)
            self._completion_evaluation_run_proposals[record.effect_id] = proposal.proposal_id
            return copy_completion_evaluation_run(record)

    async def settle_completion_evaluation_run(
        self,
        request: CompletionEvaluationSettlementRequest,
    ) -> CompletionEvaluationRun:
        request = copy_completion_evaluation_settlement_request(request)
        async with self._lock:
            proposal_id = self._completion_evaluation_run_proposals.get(request.effect_id)
            if proposal_id is None:
                raise KeyError(f"Completion evaluation run not found: {request.effect_id}")
            runs = self._completion_evaluation_runs[proposal_id]
            index = next(
                position
                for position, item in enumerate(runs)
                if item.effect_id == request.effect_id
            )
            updated, changed = settle_completion_evaluation_run_record(
                runs[index], request, settled_at=self._clock()
            )
            if changed:
                runs[index] = updated
            return copy_completion_evaluation_run(updated)

    async def list_completion_evaluation_runs(
        self,
        proposal_id: str,
    ) -> tuple[CompletionEvaluationRun, ...]:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            return tuple(
                copy_completion_evaluation_run(item)
                for item in self._completion_evaluation_runs.get(proposal_id, ())
            )

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
        receipt_key = (request.task_id, request.idempotency_key)
        async with self._lock:
            receipt = self._decision_application_receipts.get(receipt_key)
            if receipt is not None:
                if receipt.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Decision-application identity is already bound to another request."
                    )
                return receipt.task.model_copy(deep=True)
            prior_receipt_key = self._decision_application_key_by_decision.get(request.decision_id)
            if prior_receipt_key is not None:
                raise WorkCompletionConflict(
                    "Completion decision was already applied under another identity."
                )
            task = self._require_task(request.task_id)
            decision = self._completion_decisions.get(request.decision_id)
            if decision is None:
                raise KeyError(f"Completion decision not found: {request.decision_id}")
            if decision.task_id != task.id:
                raise WorkCompletionConflict("Completion decision belongs to another task.")
            contract = self._ensure_task_contract_matches(task, decision.contract)
            attempt = self._require_work_attempt(decision.attempt_id)
            # The durable verifier decision, exact application tuple, and latest
            # attempt identity authorize this transition. The originating task
            # worker may have crashed or its lease may have expired while an
            # independent verifier was running.
            self._ensure_decision_attempt_is_current(task, attempt)
            proposal = self._require_completion_proposal(decision.proposal_id)
            profile = self._completion_verifier_profiles.get(proposal.proposal_id)
            if (
                profile is None
                or profile.profile.fingerprint != decision.verifier_profile_fingerprint
            ):
                raise WorkCompletionConflict(
                    "Completion decision has no exact verifier-profile authority."
                )
            from cayu.tasks._verified_work_policy import plan_decision_application

            updated, receipt = plan_decision_application(
                request,
                request_sha256=request_sha256,
                task=task,
                decision=decision,
                proposal=proposal,
                attempt=attempt,
                contract=contract,
                matching_gap_count=self._matching_completion_gap_count(decision),
                now=self._ownership_clock(),
            )
            if updated != task:
                self._store_task(updated)
            task = updated
            self._decision_application_receipts[receipt_key] = receipt
            self._decision_application_key_by_decision[decision.decision_id] = receipt_key
            return task.model_copy(deep=True)

    async def load_completion_decision_application_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        idempotency_key = validate_work_completion_idempotency_key(idempotency_key)
        async with self._lock:
            receipt = self._decision_application_receipts.get((task_id, idempotency_key))
            if receipt is None:
                return None
            return CompletionDecisionApplicationReceipt.model_validate(
                receipt.model_dump(mode="python", warnings=False)
            )

    @runtime_task_creation
    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task_id = request.task_id or str(uuid4())
            if request.schedule_policy is not None and task_id in self._tasks:
                existing = self._require_task(task_id)
                if (
                    existing.schedule is None
                    or existing.schedule.creation_sha256 != schedule_creation_digest(request)
                ):
                    raise TaskScheduleConflict(
                        "Task schedule creation conflicts with retained intent."
                    )
                return existing.model_copy(deep=True)
            parent = self._task_parent_for_create(request, task_id=task_id)
            if request.work_contract is not None:
                self._require_work_contract(request.work_contract)
                self._ensure_contract_session_accepts_attachment(
                    request.work_contract,
                    request.session_id,
                )
            admission_now = self._clock()
            task = _task_from_create(
                request,
                task_id=task_id,
                parent_task=parent,
                retry_started_at=admission_now,
                supports_verified_work_contracts=True,
            )
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._store_task(task)
            created = task.model_copy(deep=True)
        self._publish_task_admission_wakeup(task, now=admission_now)
        return created

    async def reschedule_task(self, request: TaskRescheduleRequest) -> TaskScheduleReceipt:
        if type(request) is not TaskRescheduleRequest:
            raise TypeError("A typed task reschedule request is required.")
        request = revalidate_model_input(request, TaskRescheduleRequest)
        digest = schedule_mutation_digest(request)
        async with self._lock:
            key = (request.task_id, request.operation_id)
            retained = self._schedule_receipts.get(key)
            if retained is not None:
                if retained.request_sha256 != digest:
                    raise TaskScheduleConflict("Schedule operation identity has different content.")
                return retained.model_copy(deep=True)
            current = self._require_task(request.task_id)
            if self._task_has_unsettled_local_execution_attempt(current.id):
                raise TaskScheduleConflict("Task has unsettled execution authority.")
            now = self._clock()
            updated = rescheduled_task(current, request, now=now)
            receipt = schedule_receipt(
                updated, request, now=now, kind=TaskScheduleEventType.RESCHEDULED
            )
            self._store_task(updated, schedule_operation_id=request.operation_id)
            self._schedule_receipts[key] = receipt
        # A changed future deadline can shorten an existing worker wait. The
        # edge is advisory and contains no task data; claims recheck the store.
        self._publish_task_admission_broadcast()
        return receipt.model_copy(deep=True)

    async def cancel_scheduled_task(
        self, request: TaskScheduleCancelRequest
    ) -> TaskScheduleReceipt:
        if type(request) is not TaskScheduleCancelRequest:
            raise TypeError("A typed task schedule cancellation is required.")
        request = revalidate_model_input(request, TaskScheduleCancelRequest)
        digest = schedule_mutation_digest(request)
        async with self._lock:
            key = (request.task_id, request.operation_id)
            retained = self._schedule_receipts.get(key)
            if retained is not None:
                if retained.request_sha256 != digest:
                    raise TaskScheduleConflict("Schedule operation identity has different content.")
                return retained.model_copy(deep=True)
            current = self._require_task(request.task_id)
            state = require_schedule_mutation(current, request.expected_revision)
            now = self._ownership_clock()
            updated, retry_settlement = self._prepare_finished_task(
                current.id,
                TaskStatus.CANCELLED,
                result=None,
                error=None,
                now=now,
                worker_id=None,
                expected_lease_expires_at=None,
                accepted_decision_id=None,
            )
            updated = updated.model_copy(
                update={
                    "schedule": state.model_copy(
                        update={"revision": schedule_revision_after(state)}
                    )
                }
            )
            kind = (
                TaskScheduleEventType.CANCELLED
                if updated.status is TaskStatus.CANCELLED
                else TaskScheduleEventType.CANCELLATION_REQUESTED
            )
            receipt = schedule_receipt(updated, request, now=now, kind=kind)
            if retry_settlement is not None:
                retry_settlement = retry_settlement.model_copy(update={"task": updated}, deep=True)
            self._store_task(updated, schedule_operation_id=request.operation_id)
            if retry_settlement is not None:
                self._retry_settlements[(updated.id, retry_settlement.idempotency_key)] = (
                    retry_settlement
                )
            self._schedule_receipts[key] = receipt
            return receipt.model_copy(deep=True)

    async def list_task_schedule_events(
        self, task_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[TaskScheduleEvent]:
        task_id = require_clean_nonblank(task_id, "task_id")
        if type(after_sequence) is not int or not 0 <= after_sequence <= 9007199254740991:
            raise ValueError("after_sequence must be a bounded nonnegative integer.")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Schedule event limit must be between 1 and 1000.")
        async with self._lock:
            events = self._schedule_events.get(task_id, ())
            return [
                event.model_copy(deep=True)
                for event in events[after_sequence : after_sequence + limit]
            ]

    async def next_task_schedule_wakeup(self, query: TaskQuery | None = None) -> TaskScheduleWakeup:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        async with self._lock:
            now = self._clock()
            if query.status is not None and query.status is not TaskStatus.PENDING:
                return TaskScheduleWakeup(as_of=now)
            due: datetime | None = None
            expiry: datetime | None = None
            maintenance = False
            for task in self._tasks.values():
                if (
                    task.session_id is not None
                    or task.status
                    not in {
                        TaskStatus.PENDING,
                        TaskStatus.PAUSED,
                        TaskStatus.BLOCKED,
                        TaskStatus.NEEDS_ATTENTION,
                    }
                    or not _task_matches_claim_filter(task, query)
                    or self._task_has_unsettled_local_execution_attempt(task.id)
                ):
                    continue
                if (
                    task.status is TaskStatus.PENDING
                    and task.available_at is not None
                    and task.available_at > now
                    and (task.schedule is None or task.schedule.admitted_at is None)
                    and not _task_retry_attempt_elapsed(task, series_now=now)
                ):
                    due = task.available_at if due is None else min(due, task.available_at)
                if task.schedule is not None and task.schedule.admitted_at is None:
                    if task.schedule.policy.expires_at is not None:
                        candidate_expiry = task.schedule.policy.expires_at
                        if candidate_expiry > now:
                            expiry = (
                                candidate_expiry
                                if expiry is None
                                else min(expiry, candidate_expiry)
                            )
                    assert task.available_at is not None
                    eligibility = task_schedule_eligibility(
                        available_at=task.available_at, policy=task.schedule.policy, as_of=now
                    )
                    maintenance |= eligibility in {
                        TaskScheduleEligibility.EXPIRED,
                        TaskScheduleEligibility.SKIPPED,
                    }
            return TaskScheduleWakeup(
                as_of=now,
                next_available_at=due,
                next_expiry_at=expiry,
                maintenance_required=maintenance,
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
            task_id = request.task_id or str(uuid4())
            parent = self._task_parent_for_create(request, task_id=task_id)
            if request.work_contract is not None:
                self._require_work_contract(request.work_contract)
                self._ensure_contract_session_accepts_attachment(
                    request.work_contract,
                    request.session_id,
                )
            task = _running_task_from_create(
                request,
                task_id=task_id,
                parent_task=parent,
                session_invocation=session_binding,
                retry_started_at=self._clock(),
                supports_verified_work_contracts=True,
            )
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._store_task(task)
            return task.model_copy(deep=True)

    async def load_task(self, task_id: str, *, _access_bounds=None) -> Task | None:
        if _access_bounds is None:
            from cayu.resource_access import current_data_bounds

            _access_bounds = await current_data_bounds("tasks")
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._tasks.get(task_id)
            if _access_bounds is not None:
                from cayu.tasks.access import require_read

                require_read(task, _access_bounds)
            if task is None:
                return None
            return task.model_copy(deep=True)

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
            task = self._require_task(task_id)
            return _require_active_attached_task_worker(
                task,
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
                self._require_task(task_id),
                session_id=session_id,
                session_instance_id=session_instance_id,
            )

    async def load_invocation_snapshot(
        self,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return TaskInvocationSnapshot(
                id=task.id,
                session_id=task.session_id,
                session_instance_id=task.session_instance_id,
                invocation=task.invocation,
            )

    async def list_tasks(
        self, query: TaskQuery | None = None, *, _access_bounds=None
    ) -> list[Task]:
        if _access_bounds is None:
            from cayu.resource_access import current_data_bounds

            _access_bounds = await current_data_bounds("tasks")
        query = copy_task_query(query)
        async with self._lock:
            from cayu.tasks.access import visible

            tasks = [
                task
                for task in self._tasks.values()
                if _task_matches(task, query)
                and (_access_bounds is None or visible(task, _access_bounds))
            ]
            tasks = _sort_tasks(tasks, query.order_by)
            page = tasks[query.offset : query.offset + query.limit]
            return [task.model_copy(deep=True) for task in page]

    async def claim_session_closure(
        self, claim: TaskSessionClosureClaim
    ) -> TaskSessionClosureClaim:
        claim = copy_task_session_closure_claim(claim)
        async with self._lock:
            from cayu.tasks._memory_graphs import require_graph_deletion_ready

            require_graph_deletion_ready(self, claim.task_ids)
            existing = self._session_closure_claims.get(claim.session_id)
            if existing is not None:
                existing = copy_task_session_closure_claim(existing)
                if existing != claim:
                    raise ValueError("Task closure claim conflicts with its retained authority.")
                return existing
            indexed = self._task_keys_by_session.get(claim.session_id, ())
            if len(indexed) != len(claim.task_ids) or {item[1] for item in indexed} != set(
                claim.task_ids
            ):
                raise ValueError("Task closure set changed before admission.")
            for task_id in claim.task_ids:
                task = self._tasks.get(task_id)
                if task is None or task.session_id != claim.session_id:
                    raise ValueError("Task closure source is unavailable.")
                if (
                    task.status
                    not in {
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                        TaskStatus.DEPENDENCY_SKIPPED,
                    }
                    or task.worker_id is not None
                    or task.lease_expires_at is not None
                ):
                    raise ValueError("Task closure requires quiescent terminal tasks.")
            self._session_closure_claims[claim.session_id] = claim
            return copy_task_session_closure_claim(claim)

    async def load_session_closure_claim(self, session_id: str) -> TaskSessionClosureClaim | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            claim = self._session_closure_claims.get(session_id)
            return None if claim is None else copy_task_session_closure_claim(claim)

    async def delete_session_tasks(
        self,
        session_id: str,
        *,
        task_ids: tuple[str, ...],
        policy: Any,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        task_id_set = set(task_ids)
        async with self._lock:
            claim = self._session_closure_claims.get(session_id)
            if claim is not None and task_id_set != set(claim.task_ids):
                raise ValueError("Task deletion conflicts with the retained closure set.")
            tasks = [self._tasks.get(task_id) for task_id in task_ids]
            from cayu.tasks._memory_graphs import require_graph_deletion_ready

            require_graph_deletion_ready(self, task_ids)
            if any(
                (task is None and claim is None)
                or (task is not None and task.session_id != session_id)
                for task in tasks
            ):
                raise ValueError("Task closure authority changed during deletion.")
            tasks = [task for task in tasks if task is not None]
            if any(
                task.status
                not in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.DEPENDENCY_SKIPPED,
                }
                or task.worker_id is not None
                or task.lease_expires_at is not None
                for task in tasks
            ):
                raise ValueError("Task closure requires quiescent terminal tasks.")
            # Resolve ownership before mutation. Equal strings in other identity
            # namespaces (sessions, contracts, workers) do not confer task ownership.
            attempt_ids = {
                key for key, value in self._work_attempts.items() if value.task_id in task_id_set
            }
            admission_ids = {
                key
                for key, value in self._work_attempt_admissions.items()
                if value.task_id in task_id_set
            }
            attempt_ids.update(
                self._work_attempt_admissions[key].attempt_id for key in admission_ids
            )
            proposal_ids = {
                key
                for key, value in self._completion_proposals.items()
                if value.attempt_id in attempt_ids
            }
            decision_ids = {
                key
                for key, value in self._completion_decisions.items()
                if value.proposal_id in proposal_ids
            }
            local_attempt_ids = {
                key
                for key, value in self._local_execution_attempts.items()
                if value.authority.task_id in task_id_set
            }
            owned_entries: tuple[tuple[dict[Any, Any], set[Any]], ...] = (
                (self._schedule_events, task_id_set),
                (
                    self._schedule_receipts,
                    {key for key in self._schedule_receipts if key[0] in task_id_set},
                ),
                (self._work_attempts, attempt_ids),
                (self._work_attempt_admissions, admission_ids),
                (self._completion_proposals, proposal_ids),
                (self._completion_decisions, decision_ids),
                (self._local_execution_attempts, local_attempt_ids),
                (self._attempt_ids_by_task, task_id_set),
                (self._latest_admission_id_by_task, task_id_set),
                (self._admission_id_by_attempt, attempt_ids),
                (self._proposal_id_by_attempt, attempt_ids),
                (self._completion_verifier_profiles, proposal_ids),
                (self._completion_verifier_dispatches, proposal_ids),
                (self._completion_evaluation_runs, proposal_ids),
                (
                    self._completion_evaluation_run_proposals,
                    {
                        key
                        for key, value in self._completion_evaluation_run_proposals.items()
                        if value in proposal_ids
                    },
                ),
                (
                    self._completion_verifier_dispatch_proposals,
                    {
                        key
                        for key, value in self._completion_verifier_dispatch_proposals.items()
                        if value in proposal_ids
                    },
                ),
                (self._completion_verification_claims, proposal_ids),
                (self._decision_id_by_proposal, proposal_ids),
                (self._decision_application_key_by_decision, decision_ids),
                (self._work_attempt_lifecycle_receipts, admission_ids),
                (
                    self._admission_id_by_session_interaction,
                    {
                        key
                        for key, value in self._admission_id_by_session_interaction.items()
                        if value in admission_ids
                    },
                ),
                (
                    self._unreleased_admission_id_by_session,
                    {
                        key
                        for key, value in self._unreleased_admission_id_by_session.items()
                        if value in admission_ids
                    },
                ),
                (
                    self._lifecycle_admission_by_settlement_id,
                    {
                        key
                        for key, value in self._lifecycle_admission_by_settlement_id.items()
                        if value in admission_ids
                    },
                ),
                (
                    self._local_execution_attempt_by_lineage,
                    {
                        key
                        for key, value in self._local_execution_attempt_by_lineage.items()
                        if value in local_attempt_ids
                    },
                ),
                (
                    self._work_attempt_preparation_holds,
                    {
                        key
                        for key, value in self._work_attempt_preparation_holds.items()
                        if value.task.id in task_id_set
                    },
                ),
                (
                    self._work_attempt_execution_claims,
                    {
                        key
                        for key, value in self._work_attempt_execution_claims.items()
                        if value.admission_id in admission_ids
                    },
                ),
                (
                    self._verification_claims_by_id,
                    {
                        key
                        for key, value in self._verification_claims_by_id.items()
                        if value.proposal_id in proposal_ids
                    },
                ),
            )
            for mapping in (
                self._terminalization_receipts,
                self._interrupted_handoff_receipts,
                self._retry_settlements,
                self._retry_reconciliation_rejections,
                self._cancellation_reconciliation_rejections,
                self._decision_application_receipts,
            ):
                # These exact-replay indexes are keyed by (task_id, operation_id).
                for key in tuple(mapping):
                    if key[0] in task_id_set:
                        mapping.pop(key, None)
            for mapping, removed_ids in owned_entries:
                for key in removed_ids:
                    cast("dict[Any, Any]", mapping).pop(key, None)
            for topology_index, scope_ids in (
                (self._task_keys_by_session, {task.session_id for task in tasks}),
                (self._task_keys_by_parent, {task.parent_task_id for task in tasks}),
            ):
                for scope_id in scope_ids:
                    if scope_id is None:
                        continue
                    entries = [
                        entry
                        for entry in topology_index.get(scope_id, ())
                        if entry[1] not in task_id_set
                    ]
                    if entries:
                        topology_index[scope_id] = entries
                    else:
                        topology_index.pop(scope_id, None)
            for scope_id in {task.session_id for task in tasks}:
                if scope_id is None:
                    continue
                contracted = self._contracted_task_ids_by_session.get(scope_id)
                if contracted is not None:
                    for task_id in task_id_set:
                        contracted.pop(task_id, None)
                    if not contracted:
                        self._contracted_task_ids_by_session.pop(scope_id, None)
            for task_id in task_id_set:
                self._tasks.pop(task_id, None)
            self._task_id_by_interrupted_handoff_id = {
                key: value
                for key, value in self._task_id_by_interrupted_handoff_id.items()
                if value not in task_id_set
            }
            self._interrupted_continuation_claims = {
                key: value
                for key, value in self._interrupted_continuation_claims.items()
                if value[0] not in task_id_set
            }

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology queries must be TaskTopologyQuery instances.")
        query = TaskTopologyQuery.model_validate(query.model_dump(mode="python"))
        async with self._lock:
            session_branch_limits, child_branch_limits = _allocate_task_topology_branch_limits(
                query
            )
            expanded_parents: list[TaskTopologyNode] = []
            for parent_id in query.expanded_parent_ids:
                parent = self._tasks.get(parent_id)
                if parent is None:
                    raise KeyError(f"Task not found: {parent_id}")
                expanded_parents.append(TaskTopologyNode.from_task(parent))

            session_candidates: list[list[TaskTopologyNode]] = []
            for session_id, branch_limit in zip(
                query.linked_session_ids,
                session_branch_limits,
                strict=True,
            ):
                session_candidates.append(
                    [
                        TaskTopologyNode.from_task(task)
                        for task in self._task_topology_candidates(
                            self._task_keys_by_session.get(session_id, ()),
                            cursor=query.session_cursors.get(session_id),
                            scope_kind="session",
                            scope_id=session_id,
                            limit=branch_limit,
                        )
                    ]
                )

            child_candidates: list[list[TaskTopologyNode]] = []
            for parent, branch_limit in zip(
                expanded_parents,
                child_branch_limits,
                strict=True,
            ):
                child_candidates.append(
                    [
                        TaskTopologyNode.from_task(task)
                        for task in self._task_topology_candidates(
                            self._task_keys_by_parent.get(parent.id, ()),
                            cursor=query.child_cursors.get(parent.id),
                            scope_kind="parent_task",
                            scope_id=parent.id,
                            limit=branch_limit,
                        )
                    ]
                )

            async def load_parent_links(task_ids: tuple[str, ...]) -> Mapping[str, str | None]:
                links: dict[str, str | None] = {}
                for task_id in task_ids:
                    task = self._tasks.get(task_id)
                    if task is None:
                        continue
                    links[task_id] = _bounded_optional_task_topology_parent_id(task.parent_task_id)
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
                observed_at=self._ownership_clock(),
                linked_session_ids=query.linked_session_ids,
                session_branch_candidates=session_candidates,
                session_branch_limits=session_branch_limits,
                expanded_parents=expanded_parents,
                child_branch_candidates=child_candidates,
                child_branch_limits=child_branch_limits,
                session_task_limit=query.session_task_limit,
                child_limit=query.child_limit,
            )
            return result

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        filters = copy_task_aggregate_filter(filters)
        task_query = task_query_from_aggregate_filter(filters)
        async with self._lock:
            as_of = self._clock()
            unsettled_task_ids: set[str] = set()
            unsettled_retry_series_ids: set[str] = set()
            for item_index, record in enumerate(
                self._local_execution_attempts.values(),
                start=1,
            ):
                if not record.retry_admissible:
                    unsettled_task_ids.add(record.authority.task_id)
                    if record.authority.retry_series_id is not None:
                        unsettled_retry_series_ids.add(record.authority.retry_series_id)
                if item_index % _IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL == 0:
                    await _cooperate_with_in_memory_aggregate_cancellation()
            counts = {status: 0 for status in TaskStatus}
            total_count = 0
            claimable_pending_count = 0
            scheduled_pending_count = 0
            for item_index, task in enumerate(self._tasks.values(), start=1):
                if _task_matches(task, task_query):
                    counts[task.status] += 1
                    total_count += 1
                    if task.status == TaskStatus.PENDING:
                        if task.available_at is not None and task.available_at > as_of:
                            scheduled_pending_count += 1
                        elif (
                            task.session_id is None
                            and task.id not in unsettled_task_ids
                            and (
                                task.retry_series is None
                                or task.retry_series.series_id not in unsettled_retry_series_ids
                            )
                        ):
                            claimable_pending_count += 1
                if item_index % _IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL == 0:
                    await _cooperate_with_in_memory_aggregate_cancellation()
            return TaskOperationalSnapshot(
                as_of=as_of,
                total_count=total_count,
                counts_by_status=TaskStatusCounts.model_validate(counts),
                claimable_pending_count=claimable_pending_count,
                scheduled_pending_count=scheduled_pending_count,
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
        async with self._lock:
            task = self._require_task(task_id)
            _ensure_retry_series_queue_attempt(task.retry_series)
            now = self._ownership_clock()
            _ensure_can_transition(task, TaskStatus.RUNNING)
            effective_session_id = _task_session_id_for_start(
                task_id=task.id,
                stored_session_id=task.session_id,
                requested_session_id=session_id,
            )
            self._ensure_contract_session_accepts_attachment(
                task.work_contract,
                effective_session_id,
                require_session=True,
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
            updated = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "session_id": effective_session_id,
                    "session_instance_id": session_instance_id,
                    "started_at": task.started_at or now,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
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
            task = self._require_task(task_id)
            _ensure_retry_series_queue_attempt(task.retry_series)
            now = self._ownership_clock()
            if not _can_attach_claimed_task(task, worker_id=worker_id, now=now):
                _raise_task_claim_attach_error(task, worker_id, now=now)
            if expected_lease is None:
                raise TaskClaimLost("Task attachment requires its exact worker lease.")
            _ensure_exact_owned_active_task_lease(
                task,
                worker_id,
                expected_lease,
                now=now,
            )
            self._ensure_contract_session_accepts_attachment(
                task.work_contract,
                session_id,
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
            self._store_task(updated)
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
            return self._finish_task(
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
            return self._finish_task(
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
        receipt_key = (request.task_id, request.idempotency_key)
        async with self._lock:
            existing = self._terminalization_receipts.get(receipt_key)
            if existing is not None:
                return _replay_task_terminalization_receipt(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=existing,
                    current_task=self._tasks.get(request.task_id),
                )

            task = self._require_task(request.task_id)
            if request.task_id in self._latest_admission_id_by_task:
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
            committed_at = self._ownership_clock()
            _ensure_task_terminalization_lease_authority(
                task,
                request,
                now=committed_at,
            )
            _ensure_task_handoff_authority(task, request.handoff_id)
            _validate_ordinary_task_terminalization_against_cancellation(task, request)
            status = TaskStatus(request.kind.value)
            terminal_task = self._finish_task(
                request.task_id,
                status,
                result=request.result,
                error=request.error,
                worker_id=request.worker_id,
                expected_lease_expires_at=request.lease_expires_at,
                handoff_id=request.handoff_id,
                now=committed_at,
            )
            self._terminalization_receipts[receipt_key] = TaskTerminalizationReceipt(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                worker_id=request.worker_id,
                kind=request.kind,
                request_sha256=request_sha256,
                task=terminal_task,
                committed_at=committed_at,
            )
            return terminal_task.model_copy(deep=True)

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
            receipt = self._terminalization_receipts.get((task_id, idempotency_key))
            if receipt is None:
                return None
            return TaskTerminalizationReceipt(
                task_id=receipt.task_id,
                idempotency_key=receipt.idempotency_key,
                worker_id=receipt.worker_id,
                kind=receipt.kind,
                request_sha256=receipt.request_sha256,
                task=receipt.task.model_copy(deep=True),
                committed_at=receipt.committed_at,
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
        receipt_key = (request.task_id, request.idempotency_key)
        async with self._lock:
            existing = self._terminalization_receipts.get(receipt_key)
            if existing is not None:
                replayed = _replay_task_terminalization_receipt(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=existing,
                    current_task=self._tasks.get(request.task_id),
                )
                _ensure_recovered_attached_task_session(
                    replayed,
                    session_id=session_id,
                    session_instance_id=session_instance_id,
                )
                return replayed

            task = self._require_task(request.task_id)
            if request.task_id in self._latest_admission_id_by_task:
                raise WorkAttemptExecutionClaimLost(
                    "Admitted work attempts cannot use attached-task recovery terminalization."
                )
            committed_at = self._ownership_clock()
            _ensure_recovered_attached_task_failure_authority(
                task,
                request,
                session_id=session_id,
                session_instance_id=session_instance_id,
                now=committed_at,
            )
            terminal_task = self._finish_task(
                request.task_id,
                TaskStatus.FAILED,
                result=None,
                error=request.error,
                worker_id=None,
                now=committed_at,
            )
            self._terminalization_receipts[receipt_key] = TaskTerminalizationReceipt(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                worker_id=request.worker_id,
                kind=request.kind,
                request_sha256=request_sha256,
                task=terminal_task,
                committed_at=committed_at,
            )
            return terminal_task.model_copy(deep=True)

    async def release_interrupted_task_worker(
        self,
        request: TaskInterruptedHandoffRequest,
    ) -> TaskInterruptedHandoffReceipt:
        return await self._settle_interrupted_task_handoff(request, recover_expired=False)

    async def recover_interrupted_task_worker(
        self,
        request: TaskInterruptedHandoffRequest,
    ) -> TaskInterruptedHandoffReceipt:
        return await self._settle_interrupted_task_handoff(request, recover_expired=True)

    async def _settle_interrupted_task_handoff(
        self,
        request: TaskInterruptedHandoffRequest,
        *,
        recover_expired: bool,
    ) -> TaskInterruptedHandoffReceipt:
        request, request_sha256 = prepare_interrupted_task_handoff(request)
        receipt_key = (request.task_id, request.handoff_id)
        async with self._lock:
            existing = self._interrupted_handoff_receipts.get(receipt_key)
            if existing is not None:
                return _replay_interrupted_task_handoff_receipt(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=existing,
                )
            task = self._require_task(request.task_id)
            if self._latest_admission_id_by_task.get(task.id) is not None:
                raise TaskInterruptedHandoffConflict(
                    "Admitted work attempts do not use interrupted-task handoff release."
                )
            committed_at = self._ownership_clock()
            _require_interrupted_task_handoff_authority(
                task,
                request,
                now=committed_at,
                recover_expired=recover_expired,
            )
            released = task.model_copy(
                update={
                    "worker_id": None,
                    "lease_expires_at": None,
                    "interrupted_handoff_id": request.handoff_id,
                    "updated_at": committed_at,
                }
            )
            receipt = TaskInterruptedHandoffReceipt(
                request=request,
                request_sha256=request_sha256,
                task=released,
                committed_at=committed_at,
            )
            self._store_task(released)
            self._interrupted_handoff_receipts[receipt_key] = receipt
            return _copy_interrupted_task_handoff_receipt(receipt)

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
            receipt = self._interrupted_handoff_receipts.get((task_id, handoff_id))
            if receipt is None:
                return None
            return _copy_interrupted_task_handoff_receipt(receipt)

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
        async with self._lock:
            now = self._ownership_clock()
            candidates = sorted(
                (
                    task
                    for task in self._tasks.values()
                    if task.status is TaskStatus.RUNNING
                    and task.session_id is not None
                    and task.session_instance_id is not None
                    and task.worker_id is not None
                    and task.lease_expires_at is not None
                    and task.lease_expires_at <= now
                    and task.status_reason is None
                    and self._latest_admission_id_by_task.get(task.id) is None
                    and (after is None or (task.lease_expires_at, task.id) > after)
                ),
                key=lambda task: (task.lease_expires_at, task.id),
            )
            return [copy_task(task) for task in islice(candidates, limit)]

    async def load_expired_interrupted_task_handoff_candidate(
        self,
        task_id: str,
    ) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = self._ownership_clock()
            if not (
                task.status is TaskStatus.RUNNING
                and task.session_id is not None
                and task.session_instance_id is not None
                and task.worker_id is not None
                and task.lease_expires_at is not None
                and task.lease_expires_at <= now
                and task.status_reason is None
                and self._latest_admission_id_by_task.get(task.id) is None
            ):
                return None
            return copy_task(task)

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
        lease_seconds = _validate_positive_int(lease_seconds, "lease_seconds")
        after, scan_limit = prepare_interrupted_task_continuation_claim_page(
            after=after,
            limit=scan_limit,
        )
        async with self._lock:
            now = self._ownership_clock()
            prior_claim = self._interrupted_continuation_claims.get(handoff_id_sha256)
            if prior_claim is not None:
                existing = self._tasks.get(prior_claim[0])
                if (
                    existing is None
                    or prior_claim[1] != worker_id
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
                return InterruptedTaskContinuationClaimPage(
                    task=existing,
                    next_after=(existing.created_at, existing.id),
                    scanned_candidates=0,
                    rejected_candidates=0,
                    replayed=True,
                    exhausted=False,
                )
            if handoff_id in self._task_id_by_interrupted_handoff_id:
                raise TaskClaimLost(
                    "Interrupted-task continuation claim generation is already in use."
                )
            if query.status is not None and query.status is not TaskStatus.RUNNING:
                return InterruptedTaskContinuationClaimPage(
                    scanned_candidates=0,
                    rejected_candidates=0,
                    exhausted=True,
                )
            candidates = [
                task
                for task in self._tasks.values()
                if task.interrupted_handoff_id is not None
                and (task_id is None or task.id == task_id)
                and task.status is TaskStatus.RUNNING
                and task.session_id is not None
                and task.session_instance_id is not None
                and task.status_reason is None
                and task.worker_id is None
                and task.lease_expires_at is None
                and (after is None or (task.created_at, task.id) > after)
            ]
            page = _sort_tasks(candidates, TaskOrder.CREATED_AT_ASC)[:scan_limit]
            rejected = 0
            filtered = 0
            for index, task in enumerate(page):
                if not _task_matches_claim_filter(task, query):
                    filtered += 1
                    continue
                if self._latest_admission_id_by_task.get(task.id) is not None:
                    rejected += 1
                    continue
                candidate_handoff_id = task.interrupted_handoff_id
                if candidate_handoff_id is None:
                    rejected += 1
                    continue
                receipt = self._interrupted_handoff_receipts.get((task.id, candidate_handoff_id))
                if type(receipt) is not TaskInterruptedHandoffReceipt or receipt.task != task:
                    rejected += 1
                    continue
                claimed = task.model_copy(
                    update={
                        "worker_id": worker_id,
                        "lease_expires_at": now + timedelta(seconds=lease_seconds),
                        "interrupted_handoff_id": handoff_id,
                        "updated_at": now,
                    }
                )
                self._store_task(claimed)
                self._interrupted_continuation_claims[handoff_id_sha256] = (
                    claimed.id,
                    worker_id,
                )
                return InterruptedTaskContinuationClaimPage(
                    task=claimed,
                    next_after=(task.created_at, task.id),
                    scanned_candidates=index + 1,
                    rejected_candidates=rejected,
                    filtered_candidates=filtered,
                    exhausted=index == len(page) - 1 and len(page) < scan_limit,
                )
            return InterruptedTaskContinuationClaimPage(
                next_after=(page[-1].created_at, page[-1].id) if page else None,
                scanned_candidates=len(page),
                rejected_candidates=rejected,
                filtered_candidates=filtered,
                exhausted=len(page) < scan_limit,
            )

    async def reconcile_task_cancellation(
        self,
        request: TaskCancellationReconciliationRequest,
    ) -> TaskCancellationReconciliationResult:
        request, request_sha256 = prepare_task_cancellation_reconciliation(request)
        receipt_key = (request.task_id, request.cancellation_idempotency_key)
        rejection_key = (request.task_id, request.reconciliation_idempotency_key)
        async with self._lock:
            now = self._ownership_clock()
            rejection = self._cancellation_reconciliation_rejections.get(rejection_key)
            if rejection is not None:
                raise _replay_task_cancellation_reconciliation_rejection(
                    request,
                    request_sha256=request_sha256,
                    record=rejection,
                )
            existing = self._terminalization_receipts.get(receipt_key)
            if existing is not None:
                return _replay_task_cancellation_reconciliation(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=existing,
                    current_task=self._tasks.get(request.task_id),
                )
            task = self._tasks.get(request.task_id)
            if task is None:
                raise _task_cancellation_reconciliation_conflict(
                    request,
                    "Task cancellation reconciliation task was not found.",
                )
            if request.task_id in self._latest_admission_id_by_task:
                raise WorkAttemptExecutionClaimLost(
                    "Admitted work attempts cannot use ordinary cancellation reconciliation."
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
                self._cancellation_reconciliation_rejections[rejection_key] = rejection
                raise _rejected_task_cancellation_reconciliation(rejection)
            result = _reconciled_task_cancellation(
                task,
                request,
                request_sha256=request_sha256,
                committed_at=now,
            )
            self._store_task(
                result.task,
                settled_execution=(task.id, task.worker_id, task.started_at)
                if task.worker_id is not None and task.started_at is not None
                else None,
            )
            self._terminalization_receipts[receipt_key] = result.terminalization_receipt
            return _copy_task_cancellation_reconciliation_result(result)

    async def settle_task_retry_attempt(
        self,
        request: TaskRetrySettlementRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_settlement(request)
        receipt_key = (request.task_id, request.idempotency_key)
        async with self._lock:
            existing = self._retry_settlements.get(receipt_key)
            if existing is not None:
                return _replay_task_retry_settlement(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=existing,
                    current_task=self._tasks.get(request.task_id),
                )
            task = self._require_task(request.task_id)
            now = self._ownership_clock()
            if request.lease_expires_at is None:
                raise TaskClaimLost("Task retry settlement requires its exact worker lease.")
            _ensure_exact_owned_active_task_lease(
                task,
                request.worker_id,
                request.lease_expires_at,
                now=now,
            )
            if task.retry_series is None:
                raise ValueError("Task does not belong to a retry series.")
            series_now = self._clock()
            settled, successor = _settled_task_retry_attempt(
                task,
                request,
                now=now,
                series_now=series_now,
            )
            if successor is not None and (
                successor.id in self._tasks
                or successor.id in self._task_graph_by_task
                or successor.id in self._task_group_retry_lineage
            ):
                raise TaskTerminalizationConflict(
                    "Task retry successor identity is already occupied."
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
            from cayu.tasks._memory_graphs import store_graph_task

            if not store_graph_task(
                self, settled, schedule_operation_id=None, retry_successor=successor
            ):
                writes = tuple(
                    self._prepare_task_write(item)
                    for item in (settled, successor)
                    if item is not None
                )
                for write in writes:
                    self._publish_prepared_task_write(write)
            self._retry_settlements[receipt_key] = receipt
            committed = receipt.model_copy(deep=True)
        if successor is not None:
            self._publish_task_admission_wakeup(successor, now=series_now)
        return committed

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
            task = self._require_task(task_id)
            lease_now = self._ownership_clock()
            _ensure_exact_owned_active_task_lease(
                task,
                worker_id,
                expected_lease,
                now=lease_now,
            )
            series_now = self._clock()
            if not _claimed_task_retry_attempt_elapsed(task, series_now=series_now):
                return None
            receipt = _elapsed_claimed_task_retry_settlement(
                task,
                committed_at=lease_now,
                token_count=token_count,
                estimated_cost=estimated_cost,
            )
            self._store_task(receipt.task)
            self._retry_settlements[(receipt.task_id, receipt.idempotency_key)] = receipt
            return receipt.model_copy(deep=True)

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
            task = self._require_task(task_id)
            _ensure_exact_owned_active_task_lease(
                task,
                worker_id,
                expected_lease,
                now=self._ownership_clock(),
            )
            return _claimed_task_retry_attempt_elapsed(task, series_now=self._clock())

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
            receipt = self._retry_settlements.get((task_id, idempotency_key))
            return None if receipt is None else receipt.model_copy(deep=True)

    async def reconcile_task_retry_cancellation(
        self,
        request: TaskRetryCancellationReconciliationRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_cancellation_reconciliation(request)
        receipt_key = (request.task_id, request.cancellation_idempotency_key)
        rejection_key = (request.task_id, request.reconciliation_idempotency_key)
        async with self._lock:
            now = self._ownership_clock()
            rejection = self._retry_reconciliation_rejections.get(rejection_key)
            if rejection is not None:
                raise _replay_task_retry_cancellation_reconciliation_rejection(
                    request,
                    request_sha256=request_sha256,
                    record=rejection,
                )
            existing = self._retry_settlements.get(receipt_key)
            if existing is not None:
                return _replay_task_retry_cancellation_reconciliation(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=existing,
                    current_task=self._tasks.get(request.task_id),
                )
            task = self._tasks.get(request.task_id)
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
                self._retry_reconciliation_rejections[rejection_key] = rejection
                raise _rejected_task_retry_cancellation_reconciliation(rejection)
            receipt = _reconciled_task_retry_cancellation(
                task,
                request,
                request_sha256=request_sha256,
                committed_at=now,
            )
            self._store_task(
                receipt.task,
                settled_execution=(task.id, task.worker_id, task.started_at)
                if task.worker_id is not None and task.started_at is not None
                else None,
            )
            self._retry_settlements[receipt_key] = receipt
            return receipt.model_copy(deep=True)

    @runtime_task_mutation
    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        async with self._lock:
            if self._require_task(task_id).schedule is not None:
                raise TaskScheduleConflict(
                    "Managed schedules require revision-fenced cancellation."
                )
            return self._finish_task(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
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
        async with self._lock:
            current = self._require_task(task_id)
            if current.worker_id != worker_id or current.lease_expires_at != expected_lease:
                raise TaskClaimLost(
                    "Claimed-task cancellation no longer owns the expected worker lease."
                )
            if _task_cancellation_requested(current) or _task_retry_cancellation_requested(current):
                return current.model_copy(deep=True)
            if current.started_at is not None:
                requested = _expired_dispatched_task_cancellation(
                    current,
                    updated_at=self._ownership_clock(),
                    error=copied_error,
                )
                self._store_task(requested)
                return requested.model_copy(deep=True)
            return self._finish_task(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
                worker_id=worker_id,
                expected_lease_expires_at=expected_lease,
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
        async with self._lock:
            now = self._ownership_clock()
            current = self._require_task(task_id)
            if current.worker_id != worker_id or current.lease_expires_at != expected_lease:
                raise TaskClaimLost(
                    "Claimed-task execution no longer owns the expected worker lease."
                )
            _ensure_owned_active_task_lease(current, worker_id, now=now)
            if (
                current.status is not TaskStatus.CLAIMED
                or current.session_id is not None
                or _task_cancellation_requested(current)
                or _task_retry_cancellation_requested(current)
            ):
                raise TaskTerminalizationConflict(
                    "Claimed task cannot begin ordinary worker execution."
                )
            if current.started_at is not None:
                return current.model_copy(deep=True)
            started = current.model_copy(update={"started_at": now, "updated_at": now})
            self._store_task(started)
            return self._require_task(task_id).model_copy(deep=True)

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
            task = self._require_task(task_id)
            admission_id = self._latest_admission_id_by_task.get(task_id)
            if admission_id is not None:
                raise WorkAttemptExecutionClaimLost(
                    "Admitted work attempts cannot use ordinary task resumption."
                )
            _ensure_can_resume_task(task)
            now = self._ownership_clock()
            updated = task.model_copy(
                update={
                    "status": TaskStatus.PENDING,
                    "status_reason": None,
                    "status_payload": None,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return self._require_task(task_id).model_copy(deep=True)

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
        lease_seconds = _validate_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        async with self._lock:
            availability_now = self._clock()
            now = self._ownership_clock()
            # Pending retry authority expires before schedule expiry/misfire,
            # matching the persistent stores' admission order. Once terminal,
            # the task cannot be settled again by scheduling maintenance.
            for task in tuple(self._tasks.values()):
                if _task_retry_attempt_elapsed(
                    task, series_now=availability_now
                ) and not self._task_has_unsettled_local_execution_attempt(task.id):
                    receipt = _expired_task_retry_settlement(
                        task,
                        committed_at=now,
                        series_now=availability_now,
                    )
                    self._store_task(receipt.task)
                    self._retry_settlements[(receipt.task_id, receipt.idempotency_key)] = receipt
            for waiting in tuple(self._tasks.values()):
                # A prior expiry in this batch may have skipped this member.
                waiting = self._tasks[waiting.id]
                if (
                    waiting.schedule is None
                    or waiting.schedule.admitted_at is not None
                    or waiting.status
                    not in {
                        TaskStatus.PENDING,
                        TaskStatus.WAITING_DEPENDENCIES,
                        TaskStatus.WAITING_GROUP,
                        TaskStatus.PAUSED,
                        TaskStatus.BLOCKED,
                        TaskStatus.NEEDS_ATTENTION,
                    }
                    or waiting.session_id is not None
                    or not _task_matches_claim_filter(waiting, query)
                    or self._task_has_unsettled_local_execution_attempt(waiting.id)
                ):
                    continue
                assert waiting.available_at is not None
                eligibility = task_schedule_eligibility(
                    available_at=waiting.available_at,
                    policy=waiting.schedule.policy,
                    as_of=availability_now,
                )
                if eligibility in {
                    TaskScheduleEligibility.EXPIRED,
                    TaskScheduleEligibility.SKIPPED,
                }:
                    expired, retry_settlement = _scheduled_task_nonexecution(
                        waiting, eligibility=eligibility, now=now
                    )
                    self._store_task(expired)
                    if retry_settlement is not None:
                        self._retry_settlements[(expired.id, retry_settlement.idempotency_key)] = (
                            retry_settlement
                        )
            candidates = [
                task
                for task in self._tasks.values()
                if task.status is TaskStatus.PENDING
                and task.session_id is None
                and (task.available_at is None or task.available_at <= availability_now)
                and not _task_retry_attempt_elapsed(task, series_now=availability_now)
                and (task.retry_series is None or retry_worker_id_is_bounded)
                and not self._task_has_unsettled_local_execution_attempt(task.id)
                and _task_matches_claim_filter(task, query)
            ]
            if not candidates:
                return None
            # Claiming is always FIFO by creation time, independent of the query's
            # display ordering, so the oldest pending task is dispatched first.
            task = _sort_tasks(candidates, TaskOrder.CREATED_AT_ASC)[0]
            if task.work_contract is not None:
                validate_work_completion_linked_id(worker_id, "worker_id")
            updated = task.model_copy(
                update={
                    "status": TaskStatus.CLAIMED,
                    "schedule": admitted_schedule(task, now=availability_now),
                    "worker_id": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

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
        extend_seconds = _validate_positive_int(extend_seconds, "extend_seconds")
        async with self._lock:
            now = self._ownership_clock()
            task = self._require_owned_leased_task(task_id, worker_id, now=now)
            if task.lease_expires_at != expected_lease:
                raise TaskClaimLost("Task heartbeat no longer owns the expected worker lease.")
            _ensure_task_handoff_authority(task, handoff_id)
            admission_id = self._latest_admission_id_by_task.get(task_id)
            if admission_id is not None:
                raise WorkAttemptExecutionClaimLost(
                    "Admitted work attempts require claim-fenced lease renewal."
                )
            updated = task.model_copy(
                update={
                    "lease_expires_at": now + timedelta(seconds=extend_seconds),
                    "updated_at": now,
                }
            )
            self._store_task(updated)
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
            now = self._ownership_clock()
            task = self._require_owned_leased_task(task_id, worker_id, now=now)
            if task.lease_expires_at != expected_lease:
                raise TaskClaimLost("Task release no longer owns the expected worker lease.")
            admission_id = self._latest_admission_id_by_task.get(task_id)
            if admission_id is not None:
                raise WorkAttemptExecutionClaimLost(
                    "Admitted work attempts release ownership through proposal publication."
                )
            if task.session_id is not None:
                raise ValueError(
                    f"Task {task.id} is already attached to session {task.session_id}."
                )
            if task.status is not TaskStatus.CLAIMED:
                raise ValueError(f"Task {task.id} is not claimed.")
            if _task_retry_cancellation_requested(task) or _task_cancellation_requested(task):
                raise TaskTerminalizationConflict(
                    "Task cancellation is still draining under its current owner."
                )
            updated = task.model_copy(
                update={
                    "status": TaskStatus.PENDING,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "started_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
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
            now = self._ownership_clock()
            task = self._require_owned_leased_task(task_id, worker_id, now=now)
            if task.lease_expires_at != expected_lease:
                raise TaskClaimLost(
                    "Attached-task release no longer owns the expected worker lease."
                )
            admission_id = self._latest_admission_id_by_task.get(task_id)
            if admission_id is not None:
                raise WorkAttemptExecutionClaimLost(
                    "Admitted work attempts release ownership through proposal publication."
                )
            if task.status is not TaskStatus.RUNNING:
                raise ValueError(f"Task {task.id} is not running.")
            if task.session_id is None:
                raise ValueError(f"Task {task.id} is not attached to a session.")
            if task.interrupted_handoff_id is not None:
                raise TaskInterruptedHandoffConflict(
                    "Recovery-owned attached tasks must publish an interrupted handoff."
                )
            if _task_cancellation_requested(task):
                raise TaskTerminalizationConflict(
                    "Task cancellation is still draining under its current owner."
                )
            updated = task.model_copy(
                update={
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        max_reclaims = _validate_positive_int(max_reclaims, "max_reclaims")
        if query.status is not None and query.status is not TaskStatus.CLAIMED:
            return []
        async with self._lock:
            now = self._ownership_clock()
            expired = [
                task
                for task in self._tasks.values()
                if task.status is TaskStatus.CLAIMED
                and task.session_id is None
                and not _task_retry_cancellation_requested(task)
                and not _task_cancellation_requested(task)
                and not self._task_has_unsettled_local_execution_attempt(task.id)
                and task.lease_expires_at is not None
                and task.lease_expires_at <= now
                and _task_matches_claim_filter(task, query)
            ]
            expired = _sort_tasks(expired, TaskOrder.UPDATED_AT_ASC)
            reclaimed: list[Task] = []
            for task in expired[:max_reclaims]:
                if task.started_at is not None:
                    requested = _expired_dispatched_task_cancellation(
                        task,
                        updated_at=now,
                    )
                    self._store_task(requested)
                    continue
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.PENDING,
                        "worker_id": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    }
                )
                self._store_task(updated)
                reclaimed.append(updated.model_copy(deep=True))
            return reclaimed

    def _task_has_unsettled_local_execution_attempt(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        retry_series_id = (
            None if task is None or task.retry_series is None else task.retry_series.series_id
        )
        return any(
            not record.retry_admissible
            and (
                record.authority.task_id == task_id
                or (
                    retry_series_id is not None
                    and record.authority.retry_series_id == retry_series_id
                )
            )
            for record in self._local_execution_attempts.values()
        )

    async def prepare_local_execution_attempt(
        self,
        authority: LocalExecutionAttemptAuthority,
    ) -> LocalExecutionAttemptRecord:
        authority = _copy_local_execution_attempt_authority(authority)
        async with self._lock:
            existing = self._local_execution_attempts.get(authority.attempt_id)
            lease_now = self._ownership_clock()
            task = (
                None
                if existing is not None
                else self._require_owned_leased_task(
                    authority.task_id,
                    authority.worker_id,
                    now=lease_now,
                )
            )
            lineage_key = local_execution_effect_scope(authority)
            prior_id = self._local_execution_attempt_by_lineage.get(lineage_key)
            evidence_now = self._clock()
            record = prepare_local_execution_attempt_record(
                authority=authority,
                task=task,
                existing=existing,
                prior=(None if prior_id is None else self._local_execution_attempts[prior_id]),
                evidence_now=evidence_now,
                lease_now=lease_now,
            )
            if existing is not None:
                return record
            self._local_execution_attempts[authority.attempt_id] = record
            self._local_execution_attempt_by_lineage[lineage_key] = authority.attempt_id
            return record.model_copy(deep=True)

    async def start_local_execution_attempt(
        self,
        start: LocalExecutionAttemptStart,
    ) -> LocalExecutionAttemptRecord:
        start = _copy_local_execution_attempt_start(start)
        async with self._lock:
            record = self._local_execution_attempts.get(start.attempt_id)
            if record is None:
                raise LocalExecutionAttemptConflict(
                    "Local execution start has no prepared attempt."
                )
            evidence_now = self._clock()
            lease_now = self._ownership_clock()
            if record.start is None:
                require_local_execution_task_authority(
                    self._require_task(record.authority.task_id),
                    record.authority,
                    now=lease_now,
                )
            updated = advance_local_execution_attempt_start(
                record,
                start,
                evidence_now=evidence_now,
                lease_now=lease_now,
            )
            self._local_execution_attempts[start.attempt_id] = updated
            return updated.model_copy(deep=True)

    async def settle_local_execution_attempt(
        self,
        settlement: LocalExecutionAttemptSettlement,
    ) -> LocalExecutionAttemptRecord:
        settlement = _copy_authenticated_local_execution_attempt_settlement(settlement)
        async with self._lock:
            record = self._local_execution_attempts.get(settlement.attempt_id)
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
            self._local_execution_attempts[settlement.attempt_id] = updated
            return updated.model_copy(deep=True)

    async def load_local_execution_attempt(
        self,
        attempt_id: str,
    ) -> LocalExecutionAttemptRecord | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            record = self._local_execution_attempts.get(attempt_id)
            return None if record is None else record.model_copy(deep=True)

    async def list_unsettled_local_execution_attempts(
        self,
        *,
        limit: int = 100,
        after: LocalExecutionAttemptListCursor | None = None,
    ) -> tuple[LocalExecutionAttemptRecord, ...]:
        limit = _validate_positive_int(limit, "limit")
        after = _copy_local_execution_attempt_list_cursor(after)
        async with self._lock:
            after_key = None if after is None else (after.created_at, after.attempt_id)
            records = sorted(
                (
                    record
                    for record in self._local_execution_attempts.values()
                    if not record.containment_settled
                    and (
                        after_key is None
                        or (
                            record.created_at,
                            record.authority.attempt_id,
                        )
                        > after_key
                    )
                ),
                key=lambda item: (
                    item.created_at,
                    item.authority.attempt_id,
                ),
            )
            return tuple(record.model_copy(deep=True) for record in records[:limit])

    async def claim_local_execution_attempt_recovery(
        self,
        claim: LocalExecutionAttemptRecoveryClaim,
    ) -> LocalExecutionAttemptRecord:
        claim = _copy_local_execution_attempt_recovery_claim(claim)
        async with self._lock:
            record = self._local_execution_attempts.get(claim.attempt_id)
            if record is None:
                raise LocalExecutionAttemptConflict(
                    "Local execution recovery authority conflicted."
                )
            task = self._require_task(record.authority.task_id)
            evidence_now = self._clock()
            lease_now = self._ownership_clock()
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
            self._local_execution_attempts[claim.attempt_id] = updated
            return updated.model_copy(deep=True)

    def _require_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        from cayu.tasks.access import require_mutation

        require_mutation(task)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    @staticmethod
    def _copy_work_attempt_admission(
        admission: WorkAttemptAdmission,
    ) -> WorkAttemptAdmission:
        return WorkAttemptAdmission.model_validate(
            admission.model_dump(mode="python", warnings=False)
        )

    def _require_work_attempt_admission(
        self,
        admission_id: str,
    ) -> WorkAttemptAdmission:
        admission = self._work_attempt_admissions.get(admission_id)
        if admission is None:
            raise KeyError(f"Work-attempt admission not found: {admission_id}")
        return admission

    @staticmethod
    def _ensure_live_work_attempt_claim(
        admission: WorkAttemptAdmission,
        *,
        now: datetime,
    ) -> None:
        if admission.claim.lease_expires_at <= now:
            raise WorkAttemptExecutionClaimLost("Work-attempt execution claim has expired.")

    def _work_attempt_continuation_context(
        self,
        task: Task,
        contract: WorkContract,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptContinuationContext | None:
        attempt_ids = self._attempt_ids_by_task.get(task.id, [])
        if not attempt_ids:
            return None
        prior_attempt_id = attempt_ids[-1]
        prior_admission_id = self._admission_id_by_attempt.get(prior_attempt_id)
        prior_admission = (
            None
            if prior_admission_id is None
            else self._work_attempt_admissions.get(prior_admission_id)
        )
        if (
            prior_admission is None
            or prior_admission.state is not WorkAttemptAdmissionState.RELEASED
            or prior_admission.task_id != task.id
            or prior_admission.session_id != task.session_id
        ):
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no exact released admission authority."
            )
        prior_proposal_id = self._proposal_id_by_attempt.get(prior_attempt_id)
        if prior_admission.run_semantics != request.run_semantics:
            raise WorkAttemptAdmissionConflict(
                "Continuation admission cannot change the source run settings."
            )
        if prior_proposal_id is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no durable completion proposal."
            )
        prior_decision_id = self._decision_id_by_proposal.get(prior_proposal_id)
        if prior_decision_id is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no durable completion decision."
            )
        decision = self._completion_decisions.get(prior_decision_id)
        if decision is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work-attempt decision index is incomplete."
            )
        receipt_key = self._decision_application_key_by_decision.get(prior_decision_id)
        if receipt_key is None or receipt_key not in self._decision_application_receipts:
            raise WorkAttemptAdmissionConflict(
                "The latest completion decision has not been applied durably."
            )
        receipt = self._decision_application_receipts[receipt_key]
        if (
            receipt.decision_id != decision.decision_id
            or receipt.task_id != task.id
            or receipt.task != task
        ):
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
            proposal_id=prior_proposal_id,
            decision=decision,
            application_idempotency_key=receipt_key[1],
            gap_fingerprint=decision.gap_fingerprint,
        )

    def _require_work_contract(self, reference: WorkContractRef) -> WorkContract:
        contract = self._work_contracts.get((reference.contract_id, reference.version))
        if contract is None:
            raise WorkContractConflict("Referenced work contract has not been published.")
        if contract.fingerprint != reference.fingerprint:
            raise WorkContractConflict(
                "Work-contract reference conflicts with the published fingerprint."
            )
        return contract

    def _ensure_task_contract_matches(
        self,
        task: Task,
        reference: WorkContractRef,
    ) -> WorkContract:
        contract = self._require_work_contract(reference)
        if task.work_contract is None:
            raise WorkCompletionConflict("Task is not bound to a work contract.")
        if task.work_contract != reference:
            raise WorkCompletionConflict(
                "Work operation conflicts with the task's frozen contract binding."
            )
        return contract

    def _require_work_attempt(self, attempt_id: str) -> WorkAttempt:
        attempt = self._work_attempts.get(attempt_id)
        if attempt is None:
            raise KeyError(f"Work attempt not found: {attempt_id}")
        return attempt

    def _require_completion_proposal(self, proposal_id: str) -> CompletionProposal:
        proposal = self._completion_proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Completion proposal not found: {proposal_id}")
        return proposal

    def _ensure_attempt_worker_matches(self, task: Task, worker_id: str | None) -> None:
        if task.worker_id != worker_id:
            raise TaskClaimLost("Work attempt does not carry the task's current worker authority.")
        if worker_id is not None:
            # Task ownership uses the store-owned clock. ``self._clock`` is the
            # independently injectable availability/verifier lifecycle clock.
            _ensure_active_task_lease(task, worker_id, now=self._ownership_clock())

    def _ensure_attempt_is_current(self, task: Task, attempt: WorkAttempt) -> None:
        self._ensure_attempt_state_is_current(task, attempt)
        self._ensure_attempt_worker_matches(task, attempt.worker_id)

    def _ensure_decision_attempt_is_current(self, task: Task, attempt: WorkAttempt) -> None:
        self._ensure_attempt_state_is_current(task, attempt)
        if task.worker_id not in {None, attempt.worker_id}:
            raise TaskClaimLost(
                "Completion decision conflicts with replacement task-worker authority."
            )

    def _ensure_attempt_state_is_current(self, task: Task, attempt: WorkAttempt) -> None:
        self._ensure_task_contract_matches(task, attempt.contract)
        attempt_ids = self._attempt_ids_by_task.get(task.id, [])
        if not attempt_ids or attempt_ids[-1] != attempt.attempt_id:
            raise WorkCompletionConflict(
                "Work operation does not reference the latest task attempt."
            )
        if task.status is not TaskStatus.RUNNING or task.session_id != attempt.session_id:
            raise WorkCompletionConflict("Work attempt no longer owns the live task session.")

    def _ensure_completion_proposal_is_current(self, proposal: CompletionProposal) -> None:
        attempt = self._require_work_attempt(proposal.attempt_id)
        if attempt.task_id != proposal.task_id or attempt.contract != proposal.contract:
            raise WorkCompletionConflict(
                "Completion proposal conflicts with its durable work attempt."
            )
        task = self._require_task(proposal.task_id)
        self._ensure_attempt_state_is_current(task, attempt)

    def _active_work_contract_task_for_session(self, session_id: str) -> Task | None:
        task_ids = self._contracted_task_ids_by_session.get(session_id)
        if not task_ids:
            return None
        task_id = next(iter(task_ids))
        task = self._tasks.get(task_id)
        if task is None or task.session_id != session_id or task.work_contract is None:
            raise TaskTopologyInconsistent(
                "The in-memory contracted-session authority index is inconsistent."
            )
        return task

    def _ensure_contract_session_accepts_attachment(
        self,
        contract: WorkContractRef | None,
        session_id: str | None,
        *,
        require_session: bool = False,
    ) -> None:
        if contract is None:
            return
        if session_id is None:
            if require_session:
                raise WorkCompletionConflict(
                    "Contracted tasks require a session binding before starting."
                )
            return
        validate_work_completion_linked_id(session_id, "session_id")
        if session_id in self._ordinary_execution_session_ids:
            raise WorkCompletionConflict(
                "Work-contract attachment conflicts with prior ordinary session execution."
            )

    def _task_parent_for_create(
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
        parent = self._tasks.get(parent_task_id)
        if parent is None:
            raise ValueError(f"Parent task not found: {parent_task_id}")
        return TaskInvocationSnapshot(
            id=parent.id,
            session_id=parent.session_id,
            session_instance_id=parent.session_instance_id,
            invocation=parent.invocation,
        )

    def _require_owned_leased_task(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> Task:
        task = self._require_task(task_id)
        _ensure_owned_active_task_lease(task, worker_id, now=now)
        return task

    def _finish_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
        expected_lease_expires_at: datetime | None = None,
        handoff_id: str | None = None,
        accepted_decision_id: str | None = None,
        now: datetime | None = None,
    ) -> Task:
        updated, retry_settlement = self._prepare_finished_task(
            task_id,
            status,
            result=result,
            error=error,
            worker_id=worker_id,
            expected_lease_expires_at=expected_lease_expires_at,
            handoff_id=handoff_id,
            accepted_decision_id=accepted_decision_id,
            now=self._ownership_clock() if now is None else now,
        )
        self._store_task(updated)
        if retry_settlement is not None:
            self._retry_settlements[(updated.id, retry_settlement.idempotency_key)] = (
                retry_settlement
            )
        return updated.model_copy(deep=True)

    def _prepare_finished_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None,
        expected_lease_expires_at: datetime | None,
        handoff_id: str | None = None,
        accepted_decision_id: str | None,
        now: datetime,
    ) -> tuple[Task, TaskRetrySettlementResult | None]:
        """Prepare state and optional retry evidence without publishing either."""
        task = self._require_task(task_id)
        if worker_id is not None:
            if expected_lease_expires_at is None:
                _ensure_owned_active_task_lease(task, worker_id, now=now)
            else:
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease_expires_at,
                    now=now,
                )
            _ensure_task_handoff_authority(task, handoff_id)
        admission_id = self._latest_admission_id_by_task.get(task_id)
        if admission_id is not None and accepted_decision_id is None:
            raise WorkAttemptExecutionClaimLost(
                "Admitted work attempts cannot use ordinary terminalization."
            )
        if (
            status is TaskStatus.COMPLETED
            and task.work_contract is not None
            and accepted_decision_id is None
        ):
            raise TaskCompletionDecisionRequired(
                "Contracted task completion requires an accepted durable verifier decision."
            )
        _ensure_can_transition(task, status)
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
                return cancellation_requested, None
            cancellation = _cancelled_task_retry_settlement(
                task,
                error=error,
                committed_at=now,
            )
            return cancellation.task, cancellation
        if (
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
            return cancellation_requested, None
        if _task_cancellation_requested(task) and not (
            status is TaskStatus.CANCELLED and worker_id is not None
        ):
            raise TaskTerminalizationConflict(
                "Task cancellation is still draining under its current owner."
            )
        updated = task.model_copy(
            update={
                "status": status,
                "status_reason": None,
                "status_payload": None,
                "result": deepcopy(result),
                "error": deepcopy(error),
                "worker_id": None,
                "lease_expires_at": None,
                "interrupted_handoff_id": None,
                "started_at": task.started_at or now,
                "completed_at": now,
                "updated_at": now,
                "retry_series": None,
            }
        )
        return updated, None

    def _matching_completion_gap_count(self, decision: CompletionDecision) -> int:
        matching_gap_count = 0
        for attempt_id in self._attempt_ids_by_task.get(decision.task_id, []):
            proposal_id = self._proposal_id_by_attempt.get(attempt_id)
            decision_id = (
                None if proposal_id is None else self._decision_id_by_proposal.get(proposal_id)
            )
            candidate = None if decision_id is None else self._completion_decisions.get(decision_id)
            if (
                candidate is not None
                and candidate.verdict is CompletionVerdict.REJECTED
                and candidate.gap_fingerprint == decision.gap_fingerprint
            ):
                matching_gap_count += 1
        return matching_gap_count

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
            task = self._require_task(task_id)
            admission_id = self._latest_admission_id_by_task.get(task_id)
            if admission_id is not None:
                raise WorkAttemptExecutionClaimLost(
                    "Admitted work attempts cannot use ordinary task holds."
                )
            _ensure_can_hold_task(task, status)
            now = self._ownership_clock()
            updated = task.model_copy(
                update={
                    "status": status,
                    "status_reason": reason,
                    "status_payload": deepcopy(payload),
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    def _task_topology_candidates(
        self,
        keys: Sequence[tuple[datetime, str]],
        *,
        cursor: str | None,
        scope_kind: Literal["session", "parent_task"],
        scope_id: str,
        limit: int,
    ) -> list[Task]:
        start_index = 0
        if cursor is not None:
            cursor_created_at, cursor_id = decode_task_topology_cursor(
                cursor,
                scope_kind=scope_kind,
                scope_id=scope_id,
            )
            start_index = bisect_right(keys, (cursor_created_at, cursor_id))
        candidates: list[Task] = []
        for _, task_id in keys[start_index : start_index + limit + 1]:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskTopologyInconsistent(
                    "The in-memory task topology index references a missing task."
                )
            if (scope_kind == "session" and task.session_id != scope_id) or (
                scope_kind == "parent_task" and task.parent_task_id != scope_id
            ):
                raise TaskTopologyInconsistent(
                    "The in-memory task topology index contains a contradictory link."
                )
            candidates.append(task)
        return candidates

    def _store_task(
        self,
        task: Task,
        *,
        schedule_operation_id: str | None = None,
        settled_execution: tuple[str, str, datetime] | None = None,
    ) -> None:
        from cayu.tasks._memory_graphs import store_graph_task

        if store_graph_task(
            self,
            task,
            schedule_operation_id=schedule_operation_id,
            settled_execution=settled_execution,
        ):
            return
        self._store_task_without_graph(task, schedule_operation_id=schedule_operation_id)

    def _store_task_without_graph(
        self, task: Task, *, schedule_operation_id: str | None = None
    ) -> None:
        self._publish_prepared_task_write(
            self._prepare_task_write(task, schedule_operation_id=schedule_operation_id)
        )

    def _prepare_task_write(
        self, task: Task, *, schedule_operation_id: str | None = None
    ) -> _PreparedMemoryTaskWrite:
        prior = self._tasks.get(task.id)
        for session_id in (task.session_id, None if prior is None else prior.session_id):
            if session_id is not None and session_id in self._session_closure_claims:
                raise ValueError("Task session is owned by closure.")
        # ``model_copy(update=...)`` intentionally skips Pydantic validation.
        # Revalidate every contracted lifecycle snapshot at the final in-memory
        # publication boundary so no transition can outgrow the bounded task
        # representation that decision receipts rely on.
        if task.work_contract is not None:
            task = copy_task(task)
        events = schedule_transition_events(
            prior,
            task,
            first_sequence=len(self._schedule_events.get(task.id, ())) + 1,
            operation_id=schedule_operation_id,
        )
        next_handoff_id = task.interrupted_handoff_id
        if next_handoff_id is not None:
            indexed_task_id = self._task_id_by_interrupted_handoff_id.get(next_handoff_id)
            if indexed_task_id is not None and indexed_task_id != task.id:
                raise TaskClaimLost("Interrupted-task handoff generation is already in use.")
        if prior is not None:
            for index, scope_id in (
                (self._task_keys_by_session, prior.session_id),
                (self._task_keys_by_parent, prior.parent_task_id),
            ):
                if scope_id is None:
                    continue
                keys = index.get(scope_id, ())
                key = (prior.created_at, prior.id)
                position = bisect_left(keys, key)
                if position >= len(keys) or keys[position] != key:
                    raise TaskTopologyInconsistent(
                        "The in-memory task topology index is incomplete."
                    )
            if (
                prior.session_id is not None
                and prior.work_contract is not None
                and not self._task_contract_binding_is_retired(prior.id)
                and prior.id not in self._contracted_task_ids_by_session.get(prior.session_id, {})
            ):
                raise TaskTopologyInconsistent(
                    "The in-memory contracted-session index is incomplete."
                )
        return _PreparedMemoryTaskWrite(task, prior, tuple(events))

    def _publish_prepared_task_write(self, prepared: _PreparedMemoryTaskWrite) -> None:
        """Publish prevalidated state while retaining the preparation lock."""
        task, prior, events = prepared
        prior_handoff_id = None if prior is None else prior.interrupted_handoff_id
        next_handoff_id = task.interrupted_handoff_id
        if prior_handoff_id != next_handoff_id:
            if prior_handoff_id is not None:
                self._task_id_by_interrupted_handoff_id.pop(prior_handoff_id, None)
            if next_handoff_id is not None:
                self._task_id_by_interrupted_handoff_id[next_handoff_id] = task.id
        if prior is not None and (
            prior.created_at,
            prior.session_id,
            prior.parent_task_id,
            prior.work_contract is None,
        ) == (
            task.created_at,
            task.session_id,
            task.parent_task_id,
            task.work_contract is None,
        ):
            # Lifecycle/status updates are the hot path. They do not change
            # either topology index, so avoid an O(n) list removal/reinsert.
            self._tasks[task.id] = task
            if events:
                self._schedule_events.setdefault(task.id, []).extend(events)
            return
        if prior is not None:
            self._remove_task_index_entry(self._task_keys_by_session, prior.session_id, prior)
            self._remove_task_index_entry(self._task_keys_by_parent, prior.parent_task_id, prior)
            self._remove_contracted_session_index_entry(prior)
        self._tasks[task.id] = task
        self._add_task_index_entry(self._task_keys_by_session, task.session_id, task)
        self._add_task_index_entry(self._task_keys_by_parent, task.parent_task_id, task)
        self._add_contracted_session_index_entry(task)
        if events:
            self._schedule_events.setdefault(task.id, []).extend(events)

    def _add_contracted_session_index_entry(self, task: Task) -> None:
        if task.session_id is None or task.work_contract is None:
            return
        if self._task_contract_binding_is_retired(task.id):
            return
        self._contracted_task_ids_by_session.setdefault(task.session_id, {}).setdefault(
            task.id,
            None,
        )

    def _remove_contracted_session_index_entry(self, task: Task) -> None:
        if task.session_id is None or task.work_contract is None:
            return
        if self._task_contract_binding_is_retired(task.id):
            return
        task_ids = self._contracted_task_ids_by_session.get(task.session_id)
        if task_ids is None or task.id not in task_ids:
            raise TaskTopologyInconsistent(
                "The in-memory contracted-session authority index is incomplete."
            )
        del task_ids[task.id]
        if not task_ids:
            del self._contracted_task_ids_by_session[task.session_id]

    def _task_contract_binding_is_retired(self, task_id: str) -> bool:
        admission_id = self._latest_admission_id_by_task.get(task_id)
        receipt = self._work_attempt_lifecycle_receipts.get(admission_id or "")
        return receipt is not None and receipt.retired_contract_binding

    @staticmethod
    def _add_task_index_entry(
        index: dict[str, list[tuple[datetime, str]]],
        scope_id: str | None,
        task: Task,
    ) -> None:
        if scope_id is None:
            return
        insort(index.setdefault(scope_id, []), (task.created_at, task.id))

    @staticmethod
    def _remove_task_index_entry(
        index: dict[str, list[tuple[datetime, str]]],
        scope_id: str | None,
        task: Task,
    ) -> None:
        if scope_id is None:
            return
        keys = index.get(scope_id)
        if keys is None:
            raise TaskTopologyInconsistent("The in-memory task topology index is incomplete.")
        key = (task.created_at, task.id)
        position = bisect_left(keys, key)
        if position >= len(keys) or keys[position] != key:
            raise TaskTopologyInconsistent("The in-memory task topology index is incomplete.")
        keys.pop(position)
        if not keys:
            del index[scope_id]


def _require_interrupted_task_handoff_authority(
    task: Task,
    request: TaskInterruptedHandoffRequest,
    *,
    now: datetime,
    recover_expired: bool,
) -> None:
    if (
        task.id != request.task_id
        or task.status is not TaskStatus.RUNNING
        or task.session_id != request.session_id
        or task.session_instance_id != request.session_instance_id
        or task.worker_id != request.worker_id
        or task.lease_expires_at != request.lease_expires_at
    ):
        raise TaskInterruptedHandoffConflict(
            "Interrupted-task handoff authority no longer matches the task."
        )
    if _task_cancellation_requested(task) or _task_retry_cancellation_requested(task):
        raise TaskInterruptedHandoffConflict(
            "Task cancellation is still draining under its current owner."
        )
    expired = request.lease_expires_at <= now
    if recover_expired is not expired:
        boundary = "expired" if recover_expired else "live"
        raise TaskInterruptedHandoffConflict(
            f"Interrupted-task handoff requires an exact {boundary} worker lease."
        )


def _task_cancellation_requested_task(
    task: Task,
    *,
    error: dict[str, Any] | None,
    updated_at: datetime,
) -> Task:
    """Fence an ordinary active task until its worker terminalizes cancellation."""

    if (
        task.retry_series is not None
        or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.worker_id is None
        or task.lease_expires_at is None
    ):
        raise TaskTerminalizationConflict("Task cannot drain ordinary cancellation.")
    _validate_task_retry_reconciliation_identity(task.id, "task_id")
    _validate_task_retry_reconciliation_identity(task.worker_id, "worker_id")
    if _task_cancellation_requested(task):
        payload = task.status_payload
        if type(payload) is not dict or set(payload) != {
            "terminalization_idempotency_key",
            "error",
            "event",
        }:
            raise TaskTerminalizationConflict(
                "Task cancellation request conflicts with active ownership."
            )
        key = payload["terminalization_idempotency_key"]
        event_payload = payload["event"]
        if type(key) is not str or type(event_payload) is not dict:
            raise TaskTerminalizationConflict(
                "Task cancellation request conflicts with active ownership."
            )
        event = TaskCancellationReconciliationEvent.model_validate(event_payload)
        if event != _task_cancellation_requested_event(
            task,
            cancellation_idempotency_key=key,
            occurred_at=event.occurred_at,
        ):
            raise TaskTerminalizationConflict(
                "Task cancellation request conflicts with its event identity."
            )
        return task.model_copy(deep=True)
    cancellation_error = (
        {"code": TaskStatus.CANCELLED.value}
        if error is None
        else copy_durable_json_object(error, "error")
    )
    identity = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-cancellation.v1",
            "task_id": task.id,
            "worker_id": task.worker_id,
        },
        "task_cancellation",
    )
    cancellation_idempotency_key = f"task-cancellation:v1:{sha256(identity).hexdigest()}"
    requested_at = normalize_utc_datetime(updated_at, "updated_at")
    requested_event = _task_cancellation_requested_event(
        task,
        cancellation_idempotency_key=cancellation_idempotency_key,
        occurred_at=requested_at,
    )
    return task.model_copy(
        update={
            "status_reason": _TASK_CANCELLATION_REQUESTED_REASON,
            "status_payload": {
                "terminalization_idempotency_key": cancellation_idempotency_key,
                "error": cancellation_error,
                "event": requested_event.model_dump(mode="json", warnings=False),
            },
            "updated_at": requested_at,
        },
        deep=True,
    )


def _expired_dispatched_task_cancellation(
    task: Task,
    *,
    updated_at: datetime,
    error: dict[str, Any] | None = None,
) -> Task:
    """Retain an expired dispatched claim until positive quiescence evidence exists."""

    if task.started_at is None:
        raise TaskTerminalizationConflict(
            "A task without durable dispatch evidence cannot enter drain reconciliation."
        )
    cancellation_error = (
        {"code": "task_worker_lease_expired_after_dispatch"}
        if error is None
        else copy_durable_json_object(error, "error")
    )
    if task.retry_series is not None:
        return _task_retry_cancellation_requested_task(
            task,
            error=cancellation_error,
            updated_at=updated_at,
        )
    return _task_cancellation_requested_task(
        task,
        error=cancellation_error,
        updated_at=updated_at,
    )


def _task_cancellation_terminalization_request(
    task: Task,
    *,
    worker_id: str,
) -> TaskTerminalizationRequest | None:
    """Build the exact worker terminalization for a requested cancellation."""

    if not _task_cancellation_requested(task):
        return None
    if task.worker_id != worker_id or task.lease_expires_at is None:
        raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id}.")
    payload = task.status_payload
    if type(payload) is not dict or set(payload) != {
        "terminalization_idempotency_key",
        "error",
        "event",
    }:
        raise TaskTerminalizationConflict("Task cancellation request payload is invalid.")
    key = payload["terminalization_idempotency_key"]
    error = payload["error"]
    event_payload = payload["event"]
    if type(key) is not str or type(error) is not dict or type(event_payload) is not dict:
        raise TaskTerminalizationConflict("Task cancellation request payload is invalid.")
    event = TaskCancellationReconciliationEvent.model_validate(event_payload)
    if event != _task_cancellation_requested_event(
        task,
        cancellation_idempotency_key=key,
        occurred_at=event.occurred_at,
    ):
        raise TaskTerminalizationConflict(
            "Task cancellation request conflicts with its event identity."
        )
    return TaskTerminalizationRequest(
        task_id=task.id,
        worker_id=worker_id,
        lease_expires_at=task.lease_expires_at,
        handoff_id=task.interrupted_handoff_id,
        kind=TaskTerminalKind.CANCELLED,
        error=error,
        idempotency_key=key,
    )


def _validate_ordinary_task_terminalization_against_cancellation(
    task: Task,
    request: TaskTerminalizationRequest,
) -> None:
    cancellation = _task_cancellation_terminalization_request(
        task,
        worker_id=request.worker_id,
    )
    if cancellation is None:
        if request.kind is TaskTerminalKind.CANCELLED:
            raise TaskTerminalizationConflict(
                "Worker cancellation requires a durable cancellation request."
            )
        return
    if cancellation != request:
        raise TaskTerminalizationConflict("Task cancellation request must win terminalization.")


def _validated_owner_lost_task_cancellation(
    task: Task,
    request: TaskCancellationReconciliationRequest,
    *,
    now: datetime,
) -> tuple[
    TaskTerminalizationRequest,
    TaskCancellationReconciliationEvent,
]:
    """Fence reconciliation to the exact expired ordinary-task owner."""

    return _validated_task_cancellation(
        task,
        request,
        now=now,
        require_owner_lost=True,
    )


def _validated_task_cancellation(
    task: Task,
    request: TaskCancellationReconciliationRequest,
    *,
    now: datetime,
    require_owner_lost: bool,
) -> tuple[
    TaskTerminalizationRequest,
    TaskCancellationReconciliationEvent,
]:
    """Fence reconciliation to the exact ordinary task and optional owner loss."""

    now = normalize_utc_datetime(now, "now")
    payload = task.status_payload
    conflict: str | None = None
    if (
        task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.status_reason != request.expected_status_reason
        or task.retry_series is not None
        or type(payload) is not dict
        or set(payload) != {"terminalization_idempotency_key", "error", "event"}
        or type(payload.get("terminalization_idempotency_key")) is not str
        or type(payload.get("error")) is not dict
        or type(payload.get("event")) is not dict
    ):
        conflict = "Task is not the expected cancellation-requested ordinary task."
    elif (
        task.id != request.task_id
        or task.worker_id != request.original_worker_id
        or task.interrupted_handoff_id != request.original_handoff_id
        or task.lease_expires_at != request.original_lease_expires_at
        or payload["terminalization_idempotency_key"] != request.cancellation_idempotency_key
    ):
        conflict = "Task cancellation reconciliation identity is stale."
    elif require_owner_lost and (task.lease_expires_at is None or task.lease_expires_at > now):
        conflict = "Task cancellation owner lease is still active."
    elif request.reconciliation_requested_at > now:
        conflict = "Task cancellation reconciliation request is from the future."

    for metadata_key, expected in (
        (
            "execution_profile_fingerprint",
            request.expected_execution_profile_fingerprint,
        ),
        ("effect_fingerprint", request.expected_effect_fingerprint),
    ):
        if conflict is None and metadata_key in task.metadata:
            stored = task.metadata[metadata_key]
            if type(stored) is not str or stored != expected:
                conflict = (
                    f"Task cancellation reconciliation conflicts with the stored {metadata_key}."
                )

    if conflict is not None:
        raise _task_cancellation_reconciliation_conflict(request, conflict)

    assert task.worker_id is not None
    assert task.lease_expires_at is not None
    assert type(payload) is dict
    terminalization = _task_cancellation_terminalization_request(
        task,
        worker_id=request.original_worker_id,
    )
    if (
        terminalization is None
        or terminalization.idempotency_key != request.cancellation_idempotency_key
    ):
        raise _task_cancellation_reconciliation_conflict(
            request,
            "Task cancellation terminalization identity is stale.",
        )
    requested_event = TaskCancellationReconciliationEvent.model_validate(payload["event"])
    if requested_event.occurred_at != request.cancellation_requested_at:
        raise _task_cancellation_reconciliation_conflict(
            request,
            "Task cancellation request event identity is stale.",
        )
    return terminalization, requested_event


def _reconciled_task_cancellation(
    task: Task,
    request: TaskCancellationReconciliationRequest,
    *,
    request_sha256: str,
    committed_at: datetime,
) -> TaskCancellationReconciliationResult:
    """Build a cancelled terminal receipt with bounded reconciliation evidence."""

    committed_at = normalize_utc_datetime(committed_at, "committed_at")
    terminalization, requested_event = _validated_owner_lost_task_cancellation(
        task,
        request,
        now=committed_at,
    )
    durable_actor = copy_resolution_actor(request.reconciled_by)
    if durable_actor is None:  # pragma: no cover - required request invariant
        raise AssertionError("Task cancellation reconciliation lost actor provenance.")
    durable_actor = ResolutionActor(
        subject=durable_actor.subject,
        tenant=durable_actor.tenant,
        source=durable_actor.source,
        claims={},
    )
    evidence = TaskCancellationReconciliationEvidence.model_validate(
        request.evidence.model_dump(mode="python")
    )
    reconciliation = TaskCancellationReconciliation(
        request_sha256=request_sha256,
        task_id=request.task_id,
        original_worker_id=request.original_worker_id,
        original_handoff_id=request.original_handoff_id,
        original_lease_expires_at=request.original_lease_expires_at,
        cancellation_requested_at=request.cancellation_requested_at,
        cancellation_idempotency_key=request.cancellation_idempotency_key,
        reconciliation_idempotency_key=request.reconciliation_idempotency_key,
        reconciliation_requested_at=request.reconciliation_requested_at,
        reconciled_by=durable_actor,
        evidence=evidence,
        events=(
            requested_event,
            _task_cancellation_reconciliation_event(
                request,
                event_type=TaskCancellationReconciliationEventType.STARTED,
                occurred_at=request.reconciliation_requested_at,
            ),
            _task_cancellation_reconciliation_event(
                request,
                event_type=TaskCancellationReconciliationEventType.RECONCILED,
                occurred_at=committed_at,
            ),
        ),
    )
    terminal_task = task.model_copy(
        update={
            "status": TaskStatus.CANCELLED,
            "status_reason": None,
            "status_payload": {
                "cancellation_reconciliation": reconciliation.model_dump(
                    mode="json",
                    warnings=False,
                )
            },
            "result": None,
            "error": copy_durable_json_object(terminalization.error, "error"),
            "worker_id": None,
            "lease_expires_at": None,
            "interrupted_handoff_id": None,
            "started_at": task.started_at or committed_at,
            "completed_at": committed_at,
            "updated_at": committed_at,
        },
        deep=True,
    )
    _, terminalization_sha256 = prepare_task_terminalization(terminalization)
    receipt = TaskTerminalizationReceipt(
        task_id=request.task_id,
        idempotency_key=request.cancellation_idempotency_key,
        worker_id=request.original_worker_id,
        kind=TaskTerminalKind.CANCELLED,
        request_sha256=terminalization_sha256,
        task=terminal_task,
        committed_at=committed_at,
    )
    return TaskCancellationReconciliationResult(
        request_sha256=request_sha256,
        task=terminal_task,
        terminalization_receipt=receipt,
        reconciliation=reconciliation,
        committed_at=committed_at,
    )


def _task_retry_cancellation_requested_task(
    task: Task,
    *,
    error: dict[str, Any] | None,
    updated_at: datetime,
) -> Task:
    """Fence an active attempt while its worker proves dispatched work quiescent."""

    series = task.retry_series
    if (
        series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.worker_id is None
        or task.lease_expires_at is None
    ):
        raise TaskTerminalizationConflict("Task retry attempt cannot drain cancellation.")
    if _task_retry_cancellation_requested(task):
        payload = task.status_payload
        if type(payload) is dict and set(payload) == {
            "settlement_idempotency_key",
            "error",
            "event",
        }:
            return task.model_copy(deep=True)
        # Cayu 0.3.0 persisted the same fenced cancellation intent before the
        # bounded cancellation event was added. Replaying that exact request is
        # the only supported upgrade path: preserve its original occurrence
        # time and identities, then let the public reconciliation API validate
        # and settle it normally.
        expected_key = _task_retry_runtime_idempotency_key(task, "cancellation")
        if (
            type(payload) is not dict
            or set(payload) != {"settlement_idempotency_key", "error"}
            or payload.get("settlement_idempotency_key") != expected_key
            or type(payload.get("error")) is not dict
        ):
            raise TaskTerminalizationConflict(
                "Task retry cancellation request conflicts with active ownership."
            )
        requested_event = _task_retry_cancellation_requested_event(
            task,
            occurred_at=task.updated_at,
        )
        return task.model_copy(
            update={
                "status_payload": {
                    "settlement_idempotency_key": expected_key,
                    "error": copy_durable_json_object(payload["error"], "error"),
                    "event": requested_event.model_dump(mode="json", warnings=False),
                }
            },
            deep=True,
        )
    cancellation_error = (
        {"code": TaskRetrySeriesDisposition.CANCELLED.value}
        if error is None
        else copy_durable_json_object(error, "error")
    )
    requested_event = _task_retry_cancellation_requested_event(
        task,
        occurred_at=updated_at,
    )
    return task.model_copy(
        update={
            "status_reason": _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
            "status_payload": {
                "settlement_idempotency_key": _task_retry_runtime_idempotency_key(
                    task,
                    "cancellation",
                ),
                "error": cancellation_error,
                "event": requested_event.model_dump(mode="json", warnings=False),
            },
            "updated_at": normalize_utc_datetime(updated_at, "updated_at"),
        },
        deep=True,
    )


def _validated_owner_lost_task_retry_cancellation(
    task: Task,
    request: TaskRetryCancellationReconciliationRequest,
    *,
    now: datetime,
) -> tuple[dict[str, Any], TaskRetryCancellationReconciliationEvent]:
    """Fence reconciliation to the exact expired owner and cancellation marker."""

    return _validated_task_retry_cancellation(
        task,
        request,
        now=now,
        require_owner_lost=True,
    )


def _validated_task_retry_cancellation(
    task: Task,
    request: TaskRetryCancellationReconciliationRequest,
    *,
    now: datetime,
    require_owner_lost: bool,
) -> tuple[dict[str, Any], TaskRetryCancellationReconciliationEvent]:
    """Fence reconciliation to the exact task and optionally an expired owner."""

    now = normalize_utc_datetime(now, "now")
    series = task.retry_series
    payload = task.status_payload
    conflict: str | None = None
    if (
        task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.status_reason != request.expected_status_reason
        or series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or type(payload) is not dict
        or set(payload) != {"settlement_idempotency_key", "error", "event"}
        or type(payload.get("settlement_idempotency_key")) is not str
        or type(payload.get("error")) is not dict
        or type(payload.get("event")) is not dict
    ):
        conflict = "Task is not the expected cancellation-requested retry attempt."
    elif (
        task.id != request.task_id
        or series.series_id != request.series_id
        or series.attempt != request.attempt
        or series.causal_budget_id != request.causal_budget_id
        or task.worker_id != request.original_worker_id
        or task.lease_expires_at != request.original_lease_expires_at
        or payload["settlement_idempotency_key"] != request.cancellation_idempotency_key
        or request.cancellation_idempotency_key
        != _task_retry_runtime_idempotency_key(task, "cancellation")
    ):
        conflict = "Task retry cancellation reconciliation identity is stale."
    elif require_owner_lost and (task.lease_expires_at is None or task.lease_expires_at > now):
        conflict = "Task retry cancellation owner lease is still active."
    elif request.reconciliation_requested_at > now:
        conflict = "Task retry cancellation reconciliation request is from the future."

    for metadata_key, expected in (
        (
            "execution_profile_fingerprint",
            request.expected_execution_profile_fingerprint,
        ),
        ("effect_fingerprint", request.expected_effect_fingerprint),
    ):
        if conflict is None and metadata_key in task.metadata:
            stored = task.metadata[metadata_key]
            if type(stored) is not str or stored != expected:
                conflict = (
                    "Task retry cancellation reconciliation conflicts with the "
                    f"stored {metadata_key}."
                )

    if conflict is not None:
        raise _task_retry_cancellation_reconciliation_conflict(
            request,
            conflict,
        )

    assert task.lease_expires_at is not None
    assert type(payload) is dict
    assert type(payload["error"]) is dict
    requested_event = TaskRetryCancellationReconciliationEvent.model_validate(payload["event"])
    expected_event = _task_retry_cancellation_requested_event(
        task,
        occurred_at=requested_event.occurred_at,
    )
    if (
        requested_event != expected_event
        or requested_event.occurred_at != request.cancellation_requested_at
    ):
        raise _task_retry_cancellation_reconciliation_conflict(
            request,
            "Task retry cancellation request event identity is stale.",
        )
    return (
        copy_durable_json_object(payload["error"], "error"),
        requested_event,
    )


def _reconciled_task_retry_cancellation(
    task: Task,
    request: TaskRetryCancellationReconciliationRequest,
    *,
    request_sha256: str,
    committed_at: datetime,
) -> TaskRetrySettlementResult:
    """Build an ordinary cancelled receipt with bounded reconciliation evidence."""

    cancellation_error, requested_event = _validated_owner_lost_task_retry_cancellation(
        task,
        request,
        now=committed_at,
    )
    base = _cancelled_task_retry_settlement(
        task,
        error=cancellation_error,
        committed_at=committed_at,
    )
    durable_actor = copy_resolution_actor(request.reconciled_by)
    if durable_actor is None:  # pragma: no cover - required request invariant
        raise AssertionError("Task retry reconciliation lost actor provenance.")
    durable_actor = ResolutionActor(
        subject=durable_actor.subject,
        tenant=durable_actor.tenant,
        source=durable_actor.source,
        claims={},
    )
    evidence = TaskRetryCancellationReconciliationEvidence.model_validate(
        request.evidence.model_dump(mode="python")
    )
    reconciliation = TaskRetryCancellationReconciliation(
        request_sha256=request_sha256,
        task_id=request.task_id,
        series_id=request.series_id,
        attempt=request.attempt,
        causal_budget_id=request.causal_budget_id,
        original_worker_id=request.original_worker_id,
        original_lease_expires_at=request.original_lease_expires_at,
        cancellation_requested_at=request.cancellation_requested_at,
        cancellation_idempotency_key=request.cancellation_idempotency_key,
        reconciliation_idempotency_key=request.reconciliation_idempotency_key,
        reconciliation_requested_at=request.reconciliation_requested_at,
        reconciled_by=durable_actor,
        evidence=evidence,
        events=(
            requested_event,
            _task_retry_cancellation_reconciliation_event(
                request,
                event_type=TaskRetryCancellationReconciliationEventType.STARTED,
                occurred_at=request.reconciliation_requested_at,
            ),
            _task_retry_cancellation_reconciliation_event(
                request,
                event_type=TaskRetryCancellationReconciliationEventType.RECONCILED,
                occurred_at=committed_at,
            ),
        ),
    )
    status_payload = copy_durable_json_object(base.task.status_payload, "status_payload")
    status_payload["cancellation_reconciliation"] = reconciliation.model_dump(
        mode="json",
        warnings=False,
    )
    settled = base.task.model_copy(
        update={"status_payload": status_payload},
        deep=True,
    )
    return TaskRetrySettlementResult(
        task_id=request.task_id,
        idempotency_key=request.cancellation_idempotency_key,
        request_sha256=request_sha256,
        task=settled,
        successor=None,
        reconciliation=reconciliation,
        events=_task_retry_events(settled, occurred_at=committed_at),
        committed_at=committed_at,
    )


async def terminalize_task_with_retry(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
    *,
    policy: TaskTerminalizationRetryPolicy | None = None,
) -> TaskTerminalizationRetryResult:
    """Terminalize once, reconciling only acknowledgement-ambiguous failures."""

    if not isinstance(task_store, TaskStore):
        raise TypeError("task_store must be a TaskStore instance.")
    if not task_store.supports_idempotent_terminalization:
        raise ValueError("task_store must support idempotent task terminalization and receipts.")
    request, request_sha256 = prepare_task_terminalization(request)
    if policy is None:
        policy = TaskTerminalizationRetryPolicy()
    elif type(policy) is not TaskTerminalizationRetryPolicy:
        raise TypeError("policy must be a TaskTerminalizationRetryPolicy instance.")
    else:
        policy = TaskTerminalizationRetryPolicy.model_validate(policy.model_dump(mode="python"))

    clock = asyncio.get_running_loop()
    started_at = clock.time()
    applied_backoff_seconds = 0.0
    delay = min(policy.initial_backoff_seconds, policy.max_backoff_seconds)
    last_error_category = "store_error"
    for attempt in range(1, policy.max_attempts + 1):
        attempt_request = TaskTerminalizationRequest.model_validate(
            request.model_dump(mode="python")
        )
        try:
            task = await asyncio.wait_for(
                task_store.terminalize_task(attempt_request),
                timeout=policy.attempt_timeout_seconds,
            )
            return TaskTerminalizationRetryResult(
                task=task,
                attempt_count=attempt,
                receipt_reconciled=False,
                elapsed_seconds=max(0.0, clock.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)

        try:
            receipt = await asyncio.wait_for(
                task_store.load_task_terminalization_receipt(
                    request.task_id,
                    request.idempotency_key,
                ),
                timeout=policy.attempt_timeout_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)
            receipt = None

        if receipt is not None:
            if type(receipt) is not TaskTerminalizationReceipt:
                raise TypeError(
                    "Task terminalization receipt loads must return "
                    "TaskTerminalizationReceipt instances."
                )
            if (
                receipt.task_id != request.task_id
                or receipt.idempotency_key != request.idempotency_key
                or receipt.worker_id != request.worker_id
                or receipt.kind is not request.kind
                or not _task_terminalization_request_matches_sha256(
                    request,
                    request_sha256=request_sha256,
                    candidate_sha256=receipt.request_sha256,
                )
            ):
                raise TaskTerminalizationConflict(
                    "Task terminalization receipt conflicts with the retry request."
                )
            try:
                current_task = await asyncio.wait_for(
                    task_store.load_task(request.task_id),
                    timeout=policy.attempt_timeout_seconds,
                )
            except Exception as exc:
                if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                    raise
                last_error_category = _task_terminalization_error_category(exc)
            else:
                if current_task is not None and type(current_task) is not Task:
                    raise TypeError("Task loads must return Task instances.")
                reconciled_task = _replay_task_terminalization_receipt(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=receipt,
                    current_task=current_task,
                )
                return TaskTerminalizationRetryResult(
                    task=reconciled_task,
                    attempt_count=attempt,
                    receipt_reconciled=True,
                    elapsed_seconds=max(0.0, clock.time() - started_at),
                    applied_backoff_seconds=applied_backoff_seconds,
                )

        if attempt == policy.max_attempts:
            raise TaskTerminalizationUncertain(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                attempt_count=attempt,
                error_category=last_error_category,
                elapsed_seconds=max(0.0, clock.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        if delay > 0:
            await asyncio.sleep(delay)
            applied_backoff_seconds += delay
        delay = min(delay * policy.backoff_multiplier, policy.max_backoff_seconds)

    raise AssertionError("Task terminalization retry loop exited without an outcome.")


async def settle_task_retry_attempt_with_retry(
    task_store: TaskStore,
    request: TaskRetrySettlementRequest,
    *,
    policy: TaskTerminalizationRetryPolicy | None = None,
) -> TaskRetrySettlementResult:
    """Settle once, reconciling only acknowledgement-ambiguous store failures."""

    if not isinstance(task_store, TaskStore):
        raise TypeError("task_store must be a TaskStore instance.")
    if not task_store.supports_task_retry_series:
        raise ValueError("task_store must support atomic task retry-series settlement.")
    request, request_sha256 = prepare_task_retry_settlement(request)
    if policy is None:
        policy = TaskTerminalizationRetryPolicy()
    elif type(policy) is not TaskTerminalizationRetryPolicy:
        raise TypeError("policy must be a TaskTerminalizationRetryPolicy instance.")
    else:
        policy = TaskTerminalizationRetryPolicy.model_validate(
            policy.model_dump(mode="python", warnings=False)
        )

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    applied_backoff_seconds = 0.0
    delay = min(policy.initial_backoff_seconds, policy.max_backoff_seconds)
    last_error_category = "store_error"
    for attempt in range(1, policy.max_attempts + 1):
        try:
            receipt = await asyncio.wait_for(
                task_store.settle_task_retry_attempt(
                    TaskRetrySettlementRequest.model_validate(
                        request.model_dump(mode="python", warnings=False)
                    )
                ),
                timeout=policy.attempt_timeout_seconds,
            )
            return _validate_task_retry_settlement_receipt_identity(
                receipt,
                request=request,
                request_sha256=request_sha256,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)

        try:
            receipt = await asyncio.wait_for(
                task_store.load_task_retry_settlement(
                    request.task_id,
                    request.idempotency_key,
                ),
                timeout=policy.attempt_timeout_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)
            receipt = None

        if receipt is not None:
            receipt = _validate_task_retry_settlement_receipt_identity(
                receipt,
                request=request,
                request_sha256=request_sha256,
            )
            try:
                current_task = await asyncio.wait_for(
                    task_store.load_task(request.task_id),
                    timeout=policy.attempt_timeout_seconds,
                )
            except Exception as exc:
                if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                    raise
                last_error_category = _task_terminalization_error_category(exc)
            else:
                return _replay_task_retry_settlement(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=receipt,
                    current_task=current_task,
                )

        if attempt == policy.max_attempts:
            raise TaskTerminalizationUncertain(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                attempt_count=attempt,
                error_category=last_error_category,
                elapsed_seconds=max(0.0, loop.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        if delay > 0:
            await asyncio.sleep(delay)
            applied_backoff_seconds += delay
        delay = min(delay * policy.backoff_multiplier, policy.max_backoff_seconds)

    raise AssertionError("Task retry settlement loop exited without an outcome.")


async def _terminalize_claimed_task(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
) -> Task:
    """Use receipt-safe terminalization when supported, with a legacy fallback."""

    if task_store.supports_idempotent_terminalization:
        return (await terminalize_task_with_retry(task_store, request)).task

    request, _request_sha256 = prepare_task_terminalization(request)
    if request.kind is TaskTerminalKind.CANCELLED:
        return await task_store.cancel_task(request.task_id, request.error)

    async def apply(lease_expires_at: datetime | None) -> Task:
        if request.kind is TaskTerminalKind.COMPLETED:
            if request.result is None:  # pragma: no cover - request model invariant
                raise AssertionError("Completed task terminalization requires a result.")
            return await task_store.complete_task(
                request.task_id,
                request.result,
                worker_id=request.worker_id,
                lease_expires_at=lease_expires_at,
                handoff_id=request.handoff_id,
            )
        if request.error is None:  # pragma: no cover - request model invariant
            raise AssertionError("Failed task terminalization requires an error.")
        return await task_store.fail_task(
            request.task_id,
            request.error,
            worker_id=request.worker_id,
            lease_expires_at=lease_expires_at,
            handoff_id=request.handoff_id,
        )

    if request.worker_id is None or request.lease_expires_at is not None:
        return await apply(request.lease_expires_at)

    current = await task_store.load_task(request.task_id)
    if (
        current is None
        or current.worker_id != request.worker_id
        or current.interrupted_handoff_id != request.handoff_id
        or current.lease_expires_at is None
    ):
        raise TaskClaimLost("Task terminalization cannot reconstruct its exact live worker lease.")
    return await apply(current.lease_expires_at)


async def _terminalize_claimed_task_or_detect_peer_winner(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
) -> bool:
    """Terminalize the claim, or conservatively identify a peer winner.

    ``True`` means the task is already terminal and this request's key has no
    receipt, so another terminalization won. A receipt under this request's key
    remains an explicit conflict because it may prove changed intent.
    """

    request, _request_sha256 = prepare_task_terminalization(request)
    try:
        await _terminalize_claimed_task(task_store, request)
    except TaskTerminalizationConflict:
        if not task_store.supports_idempotent_terminalization:
            raise
        receipt = await task_store.load_task_terminalization_receipt(
            request.task_id,
            request.idempotency_key,
        )
        if receipt is not None:
            raise
        task = await task_store.load_task(request.task_id)
        if task is not None:
            cancellation = _task_cancellation_terminalization_request(
                task,
                worker_id=request.worker_id,
            )
            if cancellation is not None:
                await _terminalize_claimed_task(task_store, cancellation)
                return True
        if task is not None and task.status in _TERMINAL_TASK_STATUSES:
            return True
        raise
    return False


def _task_terminalization_error_is_acknowledgement_ambiguous(exc: Exception) -> bool:
    if isinstance(
        exc,
        (TaskClaimLost, TaskTerminalizationConflict, TypeError, ValueError),
    ):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    for error_type in type(exc).__mro__:
        module = error_type.__module__
        name = error_type.__name__
        if module == "sqlite3" and name == "OperationalError":
            error_name = getattr(exc, "sqlite_errorname", None)
            return isinstance(error_name, str) and (
                error_name == "SQLITE_IOERR" or error_name.startswith("SQLITE_IOERR_")
            )
        if module.startswith("psycopg") and name == "OperationalError":
            sqlstate = getattr(exc, "sqlstate", None)
            return sqlstate is None or (isinstance(sqlstate, str) and sqlstate.startswith("08"))
    return False


def _task_terminalization_error_category(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection"
    return "database_operational"


def _task_lifecycle_now(task: Task) -> datetime:
    """Return a wall-clock lifecycle time that cannot move ``task`` backward."""

    timestamps = [datetime.now(UTC), task.created_at, task.updated_at]
    if task.started_at is not None:
        timestamps.append(task.started_at)
    if task.completed_at is not None:
        timestamps.append(task.completed_at)
    return max(timestamps)


def _ensure_can_transition(task: Task, next_status: TaskStatus) -> None:
    if (
        task.schedule is not None
        and task.schedule.admitted_at is None
        and next_status is TaskStatus.RUNNING
    ):
        raise TaskScheduleConflict("Managed schedules must be claimed before execution starts.")
    _ensure_task_status_can_transition(task.id, task.status, next_status)


def _ensure_retry_series_queue_attempt(
    retry_series: TaskRetrySeriesSnapshot | None,
) -> None:
    if retry_series is not None:
        raise ValueError(
            "Retry-series attempts are settled by task workers and cannot attach to sessions."
        )


def _ensure_task_status_can_transition(
    task_id: str,
    status: TaskStatus,
    next_status: TaskStatus,
) -> None:
    if status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.DEPENDENCY_SKIPPED,
    }:
        raise ValueError(f"Task {task_id} is already terminal: {status}")
    if next_status == TaskStatus.RUNNING and status != TaskStatus.PENDING:
        raise ValueError(f"Task {task_id} cannot transition to running from {status}")


def _ensure_can_hold_task(task: Task, next_status: TaskStatus) -> None:
    if next_status not in _HELD_TASK_STATUSES:
        raise ValueError(f"Task {task.id} cannot be held as {next_status}.")
    _ensure_not_terminal(task)
    if _task_retry_cancellation_requested(task) or _task_cancellation_requested(task):
        raise TaskTerminalizationConflict(
            "Task cancellation is still draining under its current owner."
        )
    if task.status is TaskStatus.RUNNING and task.session_id is not None:
        raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
    if task.status not in {
        TaskStatus.PENDING,
        TaskStatus.WAITING_DEPENDENCIES,
        TaskStatus.WAITING_GROUP,
        TaskStatus.CLAIMED,
        TaskStatus.RUNNING,
        *_HELD_TASK_STATUSES,
    }:
        raise ValueError(f"Task {task.id} cannot transition to {next_status} from {task.status}")


def _ensure_can_resume_task(task: Task) -> None:
    _ensure_not_terminal(task)
    if task.status not in _HELD_TASK_STATUSES:
        raise ValueError(f"Task {task.id} is not paused, blocked, or waiting for attention.")


def _ensure_not_terminal(task: Task) -> None:
    if task.status in _TERMINAL_TASK_STATUSES:
        raise ValueError(f"Task {task.id} is already terminal: {task.status}")


def _can_attach_claimed_task(
    task: Task,
    *,
    worker_id: str,
    now: datetime,
) -> bool:
    return _can_attach_claimed_task_state(
        status=task.status,
        session_id=task.session_id,
        worker_id=task.worker_id,
        lease_expires_at=task.lease_expires_at,
        expected_worker_id=worker_id,
        now=now,
    )


def _can_attach_claimed_task_state(
    *,
    status: TaskStatus,
    session_id: str | None,
    worker_id: str | None,
    lease_expires_at: datetime | None,
    expected_worker_id: str,
    now: datetime,
) -> bool:
    return (
        status is TaskStatus.CLAIMED
        and worker_id == expected_worker_id
        and session_id is None
        and lease_expires_at is not None
        and lease_expires_at > now
    )


def _ensure_active_task_lease(task: Task, worker_id: str, *, now: datetime) -> None:
    if task.lease_expires_at is None:
        raise TaskClaimLost(f"Task {task.id} has no active lease.")
    if task.lease_expires_at <= now:
        raise TaskClaimLost(f"Task {task.id} lease for worker {worker_id} has expired.")


def _ensure_owned_active_task_lease(
    task: Task,
    worker_id: str,
    *,
    now: datetime,
) -> None:
    """Require the supplied worker to own the task's current live lease."""

    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskClaimLost(f"Task {task.id} is not claimed or running.")
    if task.worker_id != worker_id:
        raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id}.")
    _ensure_active_task_lease(task, worker_id, now=now)


def _ensure_exact_owned_active_task_lease(
    task: Task,
    worker_id: str,
    lease_expires_at: datetime,
    *,
    now: datetime,
) -> None:
    """Require one exact live worker-lease generation."""

    expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
    _ensure_owned_active_task_lease(task, worker_id, now=now)
    if task.lease_expires_at != expected_lease:
        raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id} lease generation.")


def _ensure_task_terminalization_lease_authority(
    task: Task,
    request: TaskTerminalizationRequest,
    *,
    now: datetime,
) -> None:
    """Fence reclaimable claims exactly while preserving attached-task handoffs."""

    _ensure_owned_active_task_lease(task, request.worker_id, now=now)
    if request.lease_expires_at is not None:
        _ensure_exact_owned_active_task_lease(
            task,
            request.worker_id,
            request.lease_expires_at,
            now=now,
        )
        return
    if task.status is TaskStatus.CLAIMED and task.session_id is None:
        raise TaskClaimLost(
            "Unattached task terminalization requires the exact worker lease generation."
        )


def _ensure_recovered_attached_task_session(
    task: Task,
    *,
    session_id: str,
    session_instance_id: str,
) -> None:
    """Require a task to belong to one exact durable session incarnation."""

    if task.session_id != session_id or task.session_instance_id != session_instance_id:
        raise TaskClaimLost(
            "Attached-task recovery no longer owns the expected session incarnation."
        )


def _ensure_recovered_attached_task_failure_authority(
    task: Task,
    request: TaskTerminalizationRequest,
    *,
    session_id: str,
    session_instance_id: str,
    now: datetime,
) -> None:
    """Fence recovery failure to one exact expired attached-task generation."""

    now = normalize_utc_datetime(now, "now")
    _ensure_recovered_attached_task_session(
        task,
        session_id=session_id,
        session_instance_id=session_instance_id,
    )
    if request.kind is not TaskTerminalKind.FAILED:
        raise ValueError("Attached-task recovery can only publish task failure.")
    if task.retry_series is not None:
        raise ValueError(
            "Retry-series tasks require settle_task_retry_attempt for completion or failure."
        )
    if task.status is not TaskStatus.RUNNING:
        raise TaskClaimLost("Attached-task recovery requires a running task.")
    if (
        task.worker_id != request.worker_id
        or task.lease_expires_at is None
        or request.lease_expires_at is None
        or task.lease_expires_at != request.lease_expires_at
    ):
        raise TaskClaimLost(
            "Attached-task recovery no longer owns the expected worker lease generation."
        )
    _ensure_task_handoff_authority(task, request.handoff_id)
    if task.lease_expires_at > now:
        raise TaskClaimLost("Attached-task recovery owner lease is still active.")
    _validate_ordinary_task_terminalization_against_cancellation(task, request)


def _ensure_task_handoff_authority(task: Task, handoff_id: str | None) -> None:
    """Fence a task mutation to the exact continuation generation.

    A worker identifier is deliberately insufficient: an operator may reuse one
    stable worker name after a lease handoff. Both ``None`` and non-null values
    therefore compare exactly against the stored generation.
    """

    if task.interrupted_handoff_id != handoff_id:
        raise TaskClaimLost(
            f"Worker {task.worker_id} does not own task {task.id} handoff generation."
        )


def _require_active_attached_task_worker(
    task: Task,
    *,
    worker_id: str,
    session_id: str,
    session_instance_id: str,
    now: datetime,
) -> Task:
    """Validate and detach one store-authoritative resume owner snapshot."""

    _ensure_owned_active_task_lease(task, worker_id, now=now)
    if (
        task.status is not TaskStatus.RUNNING
        or task.session_id != session_id
        or task.session_instance_id != session_instance_id
    ):
        raise TaskClaimLost(f"Worker {worker_id} does not own the requested attached task session.")
    return task.model_copy(deep=True)


def _require_direct_attached_task_resume(
    task: Task,
    *,
    session_id: str,
    session_instance_id: str,
) -> Task:
    """Validate a workerless direct attachment with no handoff generation."""

    if (
        task.status is not TaskStatus.RUNNING
        or task.session_id != session_id
        or task.session_instance_id != session_instance_id
        or task.worker_id is not None
        or task.lease_expires_at is not None
        or task.interrupted_handoff_id is not None
    ):
        raise TaskClaimLost(
            "Ordinary resume does not own the requested direct attached task session."
        )
    return task.model_copy(deep=True)


def _raise_task_claim_attach_error(
    task: Task,
    worker_id: str,
    *,
    now: datetime,
) -> None:
    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskClaimLost(f"Task {task.id} is not claimed by worker {worker_id}.")
    _ensure_owned_active_task_lease(task, worker_id, now=now)
    if task.status is TaskStatus.RUNNING:
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        raise ValueError(f"Task {task.id} is already running.")
    if task.session_id is not None:
        raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
    raise RuntimeError(f"Task {task.id} active claim could not be attached.")


def _copy_optional_status_reason(value: str | None) -> str | None:
    if value is None:
        return None
    return require_nonblank(value, "reason")


def _copy_optional_status_payload(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return copy_durable_json_object(value, "payload")
