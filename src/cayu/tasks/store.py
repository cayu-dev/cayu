"""Backend-independent task store contract and worker polling coordination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from datetime import datetime
from decimal import Decimal
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from cayu.tasks._group_maintenance import TaskGroupMaintenance
    from cayu.tasks.graphs import (
        TaskGraphCreate,
        TaskGraphCreationReceipt,
        TaskGraphEvent,
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
from cayu._clock import normalize_utc_datetime
from cayu.runtime._durable_worker_loop import (
    DurableWorkerDemandPolicy,
    DurableWorkerPoller,
    DurableWorkerPollerGroup,
)
from cayu.runtime._task_admission_wakeup import TaskAdmissionWakeup, TaskAdmissionWakeupBroker
from cayu.runtime.local_execution_attempts import (
    LocalExecutionAttemptAuthority,
    LocalExecutionAttemptListCursor,
    LocalExecutionAttemptRecord,
    LocalExecutionAttemptRecoveryClaim,
    LocalExecutionAttemptSettlement,
    LocalExecutionAttemptStart,
)
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.runtime.work_attempt_lifecycle import (
    WorkAttemptLifecycleSettlement,
    WorkAttemptPreparationHold,
)
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.tasks._lifecycle import _require_direct_attached_task_resume
from cayu.tasks.admission import (
    AdmittedCompletionProposalRequest,
    WorkAttemptAdmission,
    WorkAttemptAdmissionActivate,
    WorkAttemptAdmissionPrepare,
    WorkAttemptExecutionClaim,
    WorkAttemptExecutionClaimRequest,
    WorkAttemptExecutionEntryRequest,
    WorkAttemptExecutionEntryResult,
    WorkAttemptExecutionStopRequest,
    WorkAttemptRecoveryActivate,
)
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationRequest,
    TaskCancellationReconciliationResult,
    TaskRetryCancellationReconciliationRequest,
)
from cayu.tasks.completion_evaluations import (
    CompletionEvaluationRun,
    CompletionEvaluationRunRequest,
    CompletionEvaluationSettlementRequest,
)
from cayu.tasks.completion_verifier_dispatches import (
    CompletionVerifierDispatch,
    CompletionVerifierDispatchRequest,
    CompletionVerifierDispatchSettlementRequest,
)
from cayu.tasks.completion_verifier_profiles import (
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierProfileRecord,
)
from cayu.tasks.contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionVerificationClaim,
    CompletionVerificationClaimRequest,
    WorkAttempt,
    WorkAttemptCreate,
    WorkContract,
    WorkContractRef,
)
from cayu.tasks.creation import TaskCreate, TaskInvocationSnapshot
from cayu.tasks.handoff import (
    _TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    InterruptedTaskContinuationClaimPage,
    TaskInterruptedHandoffReceipt,
    TaskInterruptedHandoffRequest,
)
from cayu.tasks.queries import (
    TaskAggregateFilter,
    TaskOperationalSnapshot,
    TaskQuery,
    _ensure_claim_query_supported,
    _task_matches_claim_filter,
    copy_task_query,
)
from cayu.tasks.records import (
    Task,
    TaskSessionClosureClaim,
    TaskStatus,
)
from cayu.tasks.retry import TaskRetrySettlementRequest, TaskRetrySettlementResult
from cayu.tasks.scheduling import (
    TaskRescheduleRequest,
    TaskScheduleCancelRequest,
    TaskScheduleEvent,
    TaskScheduleReceipt,
    TaskScheduleWakeup,
)
from cayu.tasks.terminalization import TaskTerminalizationReceipt, TaskTerminalizationRequest
from cayu.tasks.topology import (
    TaskTopologyQuery,
    TaskTopologyStoreResult,
)
from cayu.tasks.work_receipts import (
    CompletionDecisionApplicationReceipt,
    WorkAttemptLifecycleReceipt,
    WorkAttemptPreparationHoldReceipt,
)

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
