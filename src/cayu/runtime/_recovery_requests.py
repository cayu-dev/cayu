"""Typed requests shared by execution and continuation recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
    from cayu.runtime._producer_completion_replay import _ProducerCompletionReplay

from dataclasses import dataclass
from typing import Any

from cayu.approvals.tools import (
    PendingToolApproval,
    ToolApprovalDecision,
)
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
)
from cayu.budgets.base import (
    BudgetLimit,
)
from cayu.budgets.pricing import SessionCostTotals
from cayu.budgets.run_limits import RunLimits
from cayu.budgets.usage import SessionUsageSummary
from cayu.collaboration.access import CollaborationAccessContext
from cayu.context.structured_output import (
    StructuredOutputSpec,
)
from cayu.context.thinking import ThinkingConfig
from cayu.deadlines import (
    ExecutionDeadline,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.execution_units import (
    ModelStepIdentity,
)
from cayu.messages import Message
from cayu.observability.hooks import RuntimeHookPhase
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._run_limits import (
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    ActiveSessionRun,
)
from cayu.runtime.stop_policy import StopDecision
from cayu.runtime.tool_completion import ToolCompletionPolicy, ToolCompletionResult
from cayu.sessions.base import (
    InteractionTransitionSpec,
)
from cayu.sessions.records import (
    Session,
)
from cayu.tasks.records import Task
from cayu.tools.exposure import (
    ResolvedToolExposureAuthority,
)


@dataclass(frozen=True)
class RecoverySessionRunRequest:
    session: Session
    invocation_context: InvocationContext
    messages: list[Message]
    messages_to_append: list[Message]
    max_steps: int
    limits: RunLimits
    budget_limits: tuple[BudgetLimit, ...]
    retry_policy: RetryPolicy
    structured_output: StructuredOutputSpec | None
    thinking: ThinkingConfig | None
    request_metadata: dict[str, Any]
    task_id: str | None
    task_worker_id: str | None
    task_handoff_id: str | None
    start_event_type: EventType | None
    start_event_payload: dict[str, Any]
    start_task_on_enter: bool
    release_run_fence_on_exit: bool
    tool_completion_replay: ToolCompletionResult | None = None
    tool_completion: ToolCompletionPolicy | None = None
    run_limit_accounting: RunLimitAccountingContext | None = None
    initial_model_step_identity: ModelStepIdentity | None = None
    initial_model_step_number: int | None = None
    completed_tool_round_model_step: int | None = None
    initial_model_step_tool_exposure: ResolvedToolExposureAuthority | None = None
    previous_tool_exposure_profile_id: str | None = None
    preserve_failure_until_initial_provider_dispatch: bool = False
    participant_context: CollaborationAccessContext | None = None
    producer_replay: _ProducerCompletionReplay | None = None
    execution_to_wait: _ExternalExecutionToWait | None = None

    def __post_init__(self) -> None:
        if type(self.invocation_context) is not InvocationContext:
            raise TypeError("Recovery requires an authenticated InvocationContext.")
        self.invocation_context._validate()
        self.invocation_context.with_admitted_session(self.session)


@dataclass(frozen=True)
class RecoveryTerminalEventRequest:
    event: Event
    phase: RuntimeHookPhase
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    execution_profile: ExecutionProfileIdentity | None = None
    invocation_context: InvocationContext | None = None
    run_runtime_hooks: bool = True
    terminal_event_already_durable: bool = False
    yield_durable_terminal_event: bool = True


@dataclass(frozen=True)
class ProviderOperationFailureRequest:
    resolution_event: Event
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    execution_profile: ExecutionProfileIdentity
    task_id: str | None = None
    task_worker_id: str | None = None
    task_handoff_id: str | None = None
    legacy_resolution_without_profile: bool = False
    invocation_context: InvocationContext | None = None


@dataclass(frozen=True)
class RecoveryLimitStopRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    decision: StopDecision
    usage_summary: SessionUsageSummary
    cost_summary: SessionCostTotals | None
    messages: list[Message]
    tool_calls: list[runtime_records.ToolCallRequest]
    completed_tool_outcomes: list[runtime_records.ToolCallOutcome]
    pending_approval_to_clear: PendingToolApproval | None
    deferred_messages: list[Message]
    requested_approval_decision: ToolApprovalDecision | None
    approval_resolution_request_digest: str | None
    execution_profile: ExecutionProfileIdentity | None = None
    invocation_context: InvocationContext | None = None


@dataclass(frozen=True)
class RecoveryTaskEventRequest:
    event_type: EventType
    task: Task
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None


@dataclass(frozen=True)
class RecoveryInterruptionRequest:
    """Interrupt an epoch, optionally preserving an authenticated replacement interaction."""

    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    execution_profile: ExecutionProfileIdentity | None = None
    invocation_context: InvocationContext | None = None
    run_terminal_hooks: bool = True
    preserve_interaction_id: str | None = None
    recovery_claim_id: str | None = None


@dataclass(frozen=True)
class RecoveryAbandonedTurnRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    run_started_at: float | None
    usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None
    execution_profile: ExecutionProfileIdentity | None = None
    invocation_context: InvocationContext | None = None


@dataclass(frozen=True)
class RecoveryAbandonedSessionRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    run_started_at: float | None = None
    turn_usage_tracker: SessionUsageTracker | None = None
    active_run: ActiveSessionRun[SessionUsageTracker] | None = None
    interaction_transition_failures: tuple[dict[str, Any], ...] = ()
    interaction_transition: InteractionTransitionSpec | None = None
    interaction_transition_recovery_claim_id: str | None = None
    provider_cancellation_failures: tuple[dict[str, Any], ...] = ()
    native_admission_deadline: ExecutionDeadline | None = None
    execution_profile: ExecutionProfileIdentity | None = None
    invocation_context: InvocationContext | None = None
    run_terminal_hooks: bool = True
    # Some admission boundaries have already settled their predecessor and
    # therefore cannot rely on the outer cancellation handler to retry. Keep a
    # durable repair marker if terminal-event publication fails.
    retain_terminal_publication_repair: bool = False
