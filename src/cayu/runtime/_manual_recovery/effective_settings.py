"""Invocation settings a manual continuation restores from its pending record.

Approval, user-input and tool-round continuations inherit the paused run's
recorded step limit, run and budget limits, retry policy, structured output and
thinking whenever the resolver supplies no explicit value. Profile admission
decides whether an explicit value preserves the frozen invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cayu.approvals.tools import (
    PendingToolApproval,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.approvals.user_input import (
    PendingUserInput,
    UserInputRecoveryRequest,
    UserInputResponse,
    pending_user_input_from_checkpoint,
)
from cayu.budgets.base import BudgetLimit, copy_request_budget_limits
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.context.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.context.thinking import ThinkingConfig
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime._incomplete_session_recovery import _require_recovery_max_steps
from cayu.runtime._recovery_admission import _RecoveryInvocationSemantics
from cayu.runtime.tool_effects import ToolEffectReconciliationRequest
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.records import Session
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults.redaction import SecretRedactor

EffectiveRetryPolicy = Callable[[RetryPolicy | None], RetryPolicy]


def _effective_tool_round_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_round: pending_rounds.PendingToolRound,
) -> StructuredOutputSpec | None:
    if type(pending_round) is not pending_rounds.PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    if structured_output is None:
        return copy_structured_output_spec(pending_round.structured_output)
    if pending_round.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if structured_output.model_dump(mode="json") != pending_round.structured_output.model_dump(
        mode="json"
    ):
        raise ValueError("structured_output does not match the crashed run contract.")
    return copy_structured_output_spec(pending_round.structured_output)


def _effective_tool_round_invocation_semantics(
    *,
    request: ToolRoundRecoveryRequest | ToolEffectReconciliationRequest,
    pending_round: pending_rounds.PendingToolRound,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(request) is ToolEffectReconciliationRequest:
        recorded_max_steps = _require_recovery_max_steps(pending_round)
        if (
            request.max_steps is not None
            and recorded_max_steps is not None
            and request.max_steps > recorded_max_steps
        ):
            raise ValueError("Receipt continuation cannot increase the recorded step limit.")
        return _RecoveryInvocationSemantics(
            max_steps=recorded_max_steps if request.max_steps is None else request.max_steps,
            limits=copy_run_limits(pending_round.limits or RunLimits()),
            budget_limits=copy_request_budget_limits(pending_round.budget_limits or ()),
            retry_policy=effective_retry_policy(pending_round.retry_policy),
            structured_output=copy_structured_output_spec(structured_output),
            thinking=pending_round.thinking,
        )
    if type(request) is not ToolRoundRecoveryRequest:
        raise TypeError("Tool-round recovery requires a ToolRoundRecoveryRequest.")
    if type(pending_round) is not pending_rounds.PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    recorded_max_steps = _require_recovery_max_steps(pending_round)
    return _RecoveryInvocationSemantics(
        max_steps=request.max_steps if request.max_steps is not None else recorded_max_steps,
        limits=copy_run_limits(
            request.limits if request.limits is not None else pending_round.limits or RunLimits()
        ),
        budget_limits=copy_request_budget_limits(
            request.budget_limits
            if request.budget_limits is not None
            else pending_round.budget_limits or ()
        ),
        retry_policy=effective_retry_policy(
            request.retry_policy if request.retry_policy is not None else pending_round.retry_policy
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=request.thinking if request.thinking is not None else pending_round.thinking,
    )


def _effective_user_input_max_steps(
    *,
    max_steps: int | None,
    pending: PendingUserInput,
) -> int | None:
    # Restore recorded invocation settings; profile admission validates overrides.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    recorded = _require_recovery_max_steps(pending)
    return recorded if max_steps is None else max_steps


def _effective_user_input_run_limits(
    *,
    limits: RunLimits | None,
    pending: PendingUserInput,
) -> RunLimits:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if limits is not None:
        return copy_run_limits(limits)
    if pending.limits is not None:
        return copy_run_limits(pending.limits)
    return RunLimits()


def _effective_user_input_budget_limits(
    *,
    budget_limits: tuple[BudgetLimit, ...] | None,
    pending: PendingUserInput,
) -> tuple[BudgetLimit, ...]:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if budget_limits is not None:
        return copy_request_budget_limits(budget_limits)
    if pending.budget_limits is not None:
        return copy_request_budget_limits(pending.budget_limits)
    return ()


def _effective_user_input_retry_policy(
    *,
    retry_policy: RetryPolicy | None,
    pending: PendingUserInput,
) -> RetryPolicy | None:
    # RetryPolicy is frozen, so the persisted reference is safe to reuse.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if retry_policy is not None:
        return retry_policy
    return pending.retry_policy


def _effective_user_input_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending: PendingUserInput,
) -> StructuredOutputSpec | None:
    # Mirror _effective_approval_structured_output: inherit the paused run's spec when
    # the resolver supplies none; profile admission owns the final equality decision.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if structured_output is None:
        return copy_structured_output_spec(pending.structured_output)
    if pending.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(structured_output, pending.structured_output):
        raise ValueError("structured_output does not match the paused run contract.")
    return copy_structured_output_spec(pending.structured_output)


def _effective_user_input_invocation_semantics(
    *,
    response: UserInputResponse | UserInputRecoveryRequest,
    pending: PendingUserInput,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(response) not in (UserInputResponse, UserInputRecoveryRequest):
        raise TypeError("User-input continuation requires a validated response.")
    return _RecoveryInvocationSemantics(
        max_steps=_effective_user_input_max_steps(max_steps=response.max_steps, pending=pending),
        limits=_effective_user_input_run_limits(limits=response.limits, pending=pending),
        budget_limits=_effective_user_input_budget_limits(
            budget_limits=response.budget_limits,
            pending=pending,
        ),
        retry_policy=effective_retry_policy(
            _effective_user_input_retry_policy(
                retry_policy=response.retry_policy,
                pending=pending,
            )
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=response.thinking if response.thinking is not None else pending.thinking,
    )


def _effective_approval_thinking(
    *,
    thinking: ThinkingConfig | None,
    pending_approval: PendingToolApproval,
) -> ThinkingConfig | None:
    # Restore the original run's thinking config on an approval continuation. Profile
    # admission decides whether an explicit value preserves the frozen invocation.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if thinking is not None:
        return thinking
    return pending_approval.thinking


def _effective_approval_max_steps(
    *,
    max_steps: int | None,
    pending_approval: PendingToolApproval,
) -> int | None:
    # Restore recorded invocation settings; profile admission validates overrides.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    recorded = _require_recovery_max_steps(pending_approval)
    return recorded if max_steps is None else max_steps


def _effective_approval_run_limits(
    *,
    limits: RunLimits | None,
    pending_approval: PendingToolApproval,
) -> RunLimits:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if limits is not None:
        return copy_run_limits(limits)
    if pending_approval.limits is not None:
        return copy_run_limits(pending_approval.limits)
    return RunLimits()


def _effective_approval_budget_limits(
    *,
    budget_limits: tuple[BudgetLimit, ...] | None,
    pending_approval: PendingToolApproval,
) -> tuple[BudgetLimit, ...]:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if budget_limits is not None:
        return copy_request_budget_limits(budget_limits)
    if pending_approval.budget_limits is not None:
        return copy_request_budget_limits(pending_approval.budget_limits)
    return ()


def _effective_approval_retry_policy(
    *,
    retry_policy: RetryPolicy | None,
    pending_approval: PendingToolApproval,
) -> RetryPolicy | None:
    # RetryPolicy is frozen, so the persisted reference is safe to reuse.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if retry_policy is not None:
        return retry_policy
    return pending_approval.retry_policy


def _effective_approval_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_approval: PendingToolApproval,
) -> StructuredOutputSpec | None:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if structured_output is None:
        return copy_structured_output_spec(pending_approval.structured_output)
    if pending_approval.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(
        structured_output,
        pending_approval.structured_output,
    ):
        raise ValueError("Tool approval structured_output does not match the pending run contract.")
    return copy_structured_output_spec(pending_approval.structured_output)


def _effective_approval_invocation_semantics(
    *,
    request: ToolApprovalRequest | ToolApprovalRecoveryRequest | ToolEffectReconciliationRequest,
    pending_approval: PendingToolApproval,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(request) is ToolEffectReconciliationRequest:
        return _RecoveryInvocationSemantics(
            max_steps=_effective_approval_max_steps(
                max_steps=request.max_steps, pending_approval=pending_approval
            ),
            limits=_effective_approval_run_limits(limits=None, pending_approval=pending_approval),
            budget_limits=_effective_approval_budget_limits(
                budget_limits=None, pending_approval=pending_approval
            ),
            retry_policy=effective_retry_policy(
                _effective_approval_retry_policy(
                    retry_policy=None, pending_approval=pending_approval
                )
            ),
            structured_output=copy_structured_output_spec(structured_output),
            thinking=_effective_approval_thinking(thinking=None, pending_approval=pending_approval),
        )
    if type(request) not in (ToolApprovalRequest, ToolApprovalRecoveryRequest):
        raise TypeError("Tool-approval continuation requires a validated request.")
    assert isinstance(request, (ToolApprovalRequest, ToolApprovalRecoveryRequest))
    return _RecoveryInvocationSemantics(
        max_steps=_effective_approval_max_steps(
            max_steps=request.max_steps,
            pending_approval=pending_approval,
        ),
        limits=_effective_approval_run_limits(
            limits=request.limits,
            pending_approval=pending_approval,
        ),
        budget_limits=_effective_approval_budget_limits(
            budget_limits=request.budget_limits,
            pending_approval=pending_approval,
        ),
        retry_policy=effective_retry_policy(
            _effective_approval_retry_policy(
                retry_policy=request.retry_policy,
                pending_approval=pending_approval,
            )
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=_effective_approval_thinking(
            thinking=request.thinking,
            pending_approval=pending_approval,
        ),
    )


def _structured_output_specs_equal(
    left: StructuredOutputSpec,
    right: StructuredOutputSpec,
) -> bool:
    if type(left) is not StructuredOutputSpec or type(right) is not StructuredOutputSpec:
        raise TypeError("Structured output comparison requires StructuredOutputSpec values.")
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def recorded_pending_invocation_semantics(
    checkpoint: dict[str, Any] | None,
    *,
    session: Session,
    redactor: SecretRedactor,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics | None:
    """Return the semantics a continuation without overrides would restore, read-only.

    This uses the same recorded fields and defaults as approval, user-input and
    tool-round continuation when the resolver supplies no explicit values. It
    returns ``None`` when no such pending record exists.
    """

    pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
        checkpoint, redactor=redactor
    )
    if pending_approval is not None:
        return _RecoveryInvocationSemantics(
            max_steps=_effective_approval_max_steps(
                max_steps=None, pending_approval=pending_approval
            ),
            limits=_effective_approval_run_limits(limits=None, pending_approval=pending_approval),
            budget_limits=_effective_approval_budget_limits(
                budget_limits=None, pending_approval=pending_approval
            ),
            retry_policy=effective_retry_policy(
                _effective_approval_retry_policy(
                    retry_policy=None, pending_approval=pending_approval
                )
            ),
            structured_output=_effective_approval_structured_output(
                structured_output=None, pending_approval=pending_approval
            ),
            thinking=_effective_approval_thinking(thinking=None, pending_approval=pending_approval),
        )
    pending_input = pending_user_input_from_checkpoint(
        checkpoint, redactor=redactor, runtime_session=session
    )
    if pending_input is not None:
        return _RecoveryInvocationSemantics(
            max_steps=_effective_user_input_max_steps(max_steps=None, pending=pending_input),
            limits=_effective_user_input_run_limits(limits=None, pending=pending_input),
            budget_limits=_effective_user_input_budget_limits(
                budget_limits=None, pending=pending_input
            ),
            retry_policy=effective_retry_policy(
                _effective_user_input_retry_policy(retry_policy=None, pending=pending_input)
            ),
            structured_output=_effective_user_input_structured_output(
                structured_output=None, pending=pending_input
            ),
            thinking=pending_input.thinking,
        )
    pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
        checkpoint, redactor=redactor, runtime_session=session
    )
    if pending_round is not None:
        return _RecoveryInvocationSemantics(
            max_steps=_require_recovery_max_steps(pending_round),
            limits=copy_run_limits(pending_round.limits or RunLimits()),
            budget_limits=copy_request_budget_limits(pending_round.budget_limits or ()),
            retry_policy=effective_retry_policy(pending_round.retry_policy),
            structured_output=_effective_tool_round_structured_output(
                structured_output=None, pending_round=pending_round
            ),
            thinking=pending_round.thinking,
        )
    return None
