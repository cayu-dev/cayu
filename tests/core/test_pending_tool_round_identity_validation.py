from __future__ import annotations

import pytest
from pydantic import ValidationError

from cayu.core.events import Event, EventType
from cayu.core.tools import ToolResult
from cayu.runtime import PendingToolApproval, ToolPolicyEvidence
from cayu.runtime import _approval_support as approval_support
from cayu.runtime._assistant_tool_round_publication import StagedToolCallTerminal
from cayu.runtime._tool_round_recovery import PendingToolRound
from cayu.runtime.approvals import PendingToolCallApproval
from cayu.runtime.tool_exposure import (
    ResolvedToolExposureAuthority,
    unexposed_tool_result,
)
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult
from cayu.runtime.user_input import PendingUserInput


def _identity() -> dict[str, str]:
    return {
        "model_step_id": f"mstep_{'1' * 32}",
        "model_attempt_id": f"matt_{'2' * 32}",
        "tool_round_id": f"tround_{'3' * 32}",
    }


def _call(
    tool_call_id: str = "call_1",
    *,
    tool_name: str = "deploy",
) -> PendingToolCallApproval:
    return PendingToolCallApproval(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments={"target": "production"},
    )


def _exposure_authority(
    *,
    profile_id: str,
    tool_names: tuple[str, ...],
    fingerprint: str = "a" * 64,
) -> ResolvedToolExposureAuthority:
    return ResolvedToolExposureAuthority(
        profile_id=profile_id,
        tool_names=tool_names,
        registered_count=2,
        ceiling_count=2,
        fingerprint=fingerprint,
    )


def _unexposed_call() -> PendingToolCallApproval:
    return PendingToolCallApproval(
        tool_call_id="call_hidden",
        tool_name="hidden",
        arguments={"secret": "private"},
        policy_evidence=ToolPolicyEvidence.UNEXPOSED,
    )


def _authoritative_call(
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: dict | None = None,
    decision: ToolPolicyDecision = ToolPolicyDecision.ALLOW,
) -> PendingToolCallApproval:
    return PendingToolCallApproval(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments={} if arguments is None else arguments,
        policy_evidence=ToolPolicyEvidence.AUTHORITATIVE,
        policy_decision=decision.value,
    )


def _unexposed_terminal(
    exposure: ResolvedToolExposureAuthority,
) -> StagedToolCallTerminal:
    return StagedToolCallTerminal(
        tool_call_id="call_hidden",
        event=Event(
            type=EventType.TOOL_CALL_BLOCKED,
            session_id="session_1",
            tool_name="hidden",
            payload={
                **_identity(),
                "tool_call_id": "call_hidden",
                "blocked_by": "tool_exposure",
                "reason": "not_exposed_in_request",
                "profile_id": exposure.profile_id,
                "exposure_fingerprint": exposure.fingerprint,
                "arguments_state": "unavailable",
                "result": unexposed_tool_result().model_dump(mode="json"),
            },
        ),
        hooks_state="completed",
    )


def _ordinary_round_with_unexposed_stage() -> PendingToolRound:
    exposure = _exposure_authority(profile_id="tool-free", tool_names=())
    return PendingToolRound(
        **_identity(),
        agent_name="assistant",
        tool_exposure=exposure,
        tool_calls=[_unexposed_call()],
        policy_state="planned",
        policy_context_version=1,
        staged_terminals=[_unexposed_terminal(exposure)],
    )


def _approval_round_with_unexposed_stage() -> PendingToolRound:
    exposure = _exposure_authority(profile_id="deploy-only", tool_names=("deploy",))
    return PendingToolRound(
        **_identity(),
        agent_name="assistant",
        tool_exposure=exposure,
        tool_calls=[
            _unexposed_call(),
            _authoritative_call(
                tool_call_id="call_approval",
                tool_name="deploy",
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            ),
        ],
        policy_state="planned",
        policy_context_version=1,
        staged_terminals=[_unexposed_terminal(exposure)],
    )


def _user_input_with_unexposed_stage() -> PendingUserInput:
    exposure = _exposure_authority(profile_id="input-only", tool_names=("ask_user",))
    arguments = {"question": "Continue?"}
    return PendingUserInput(
        input_id="input_1",
        **_identity(),
        tool_call_id="call_input",
        tool_name="ask_user",
        question="Continue?",
        arguments=arguments,
        agent_name="assistant",
        tool_exposure=exposure,
        tool_calls=[
            _unexposed_call(),
            _authoritative_call(
                tool_call_id="call_input",
                tool_name="ask_user",
                arguments=arguments,
            ),
        ],
        staged_terminals=[_unexposed_terminal(exposure)],
    )


@pytest.mark.parametrize(
    "build",
    (
        lambda calls: PendingToolApproval(
            approval_id="approval_1",
            **_identity(),
            tool_call_id="call_1",
            tool_name="deploy",
            arguments={"target": "production"},
            agent_name="assistant",
            publish_arguments=True,
            tool_calls=calls,
        ),
        lambda calls: PendingUserInput(
            input_id="input_1",
            **_identity(),
            tool_call_id="call_1",
            tool_name="deploy",
            question="Continue?",
            arguments={"target": "production"},
            agent_name="assistant",
            tool_calls=calls,
        ),
        lambda calls: PendingToolRound(
            **_identity(),
            agent_name="assistant",
            tool_calls=calls,
        ),
    ),
)
def test_pending_tool_round_state_rejects_duplicate_call_identity(build) -> None:
    with pytest.raises(ValidationError, match="duplicate tool-call identities"):
        build([_call(), _call()])


@pytest.mark.parametrize(
    "build",
    (
        lambda: PendingToolApproval(
            approval_id="approval_1",
            **_identity(),
            tool_call_id="call_missing",
            tool_name="deploy",
            arguments={"target": "production"},
            agent_name="assistant",
            publish_arguments=True,
            tool_calls=[_call()],
        ),
        lambda: PendingUserInput(
            input_id="input_1",
            **_identity(),
            tool_call_id="call_missing",
            tool_name="deploy",
            question="Continue?",
            arguments={"target": "production"},
            agent_name="assistant",
            tool_calls=[_call()],
        ),
    ),
)
def test_pending_pause_state_requires_its_gating_call_in_the_round(build) -> None:
    with pytest.raises(ValidationError, match="exactly one call"):
        build()


@pytest.mark.parametrize(
    "build",
    (
        lambda: PendingToolApproval(
            approval_id="approval_1",
            **_identity(),
            tool_call_id="call_1",
            tool_name="rollback",
            arguments={"target": "production"},
            agent_name="assistant",
            publish_arguments=True,
            tool_calls=[_call()],
        ),
        lambda: PendingUserInput(
            input_id="input_1",
            **_identity(),
            tool_call_id="call_1",
            tool_name="rollback",
            question="Continue?",
            arguments={"target": "production"},
            agent_name="assistant",
            tool_calls=[_call()],
        ),
    ),
)
def test_pending_pause_state_rejects_conflicting_gating_call_details(build) -> None:
    with pytest.raises(ValidationError, match="details do not match"):
        build()


@pytest.mark.parametrize(
    "build",
    (
        _ordinary_round_with_unexposed_stage,
        _approval_round_with_unexposed_stage,
        _user_input_with_unexposed_stage,
    ),
    ids=("ordinary-recovery", "approval-resolution", "user-input-resolution"),
)
@pytest.mark.parametrize(
    ("mutate_stage", "error"),
    (
        (
            lambda stage: stage["event"].update(
                {
                    "type": EventType.TOOL_CALL_COMPLETED.value,
                    "payload": {
                        **stage["event"]["payload"],
                        "blocked_by": None,
                        "result": ToolResult(content="forged success").model_dump(mode="json"),
                    },
                }
            ),
            "conflicts with its frozen exposure authority",
        ),
        (
            lambda stage: stage["event"]["payload"].update({"exposure_fingerprint": "b" * 64}),
            "conflicts with its frozen exposure authority",
        ),
        (
            lambda stage: stage["event"]["payload"]["result"].update(
                {"content": "forged recovery instruction"}
            ),
            "conflicts with its frozen exposure authority",
        ),
        (
            lambda stage: stage.update({"hooks_state": "pending"}),
            "cannot retain executable hook work",
        ),
    ),
    ids=(
        "forged-success",
        "foreign-fingerprint",
        "forged-error-content",
        "pending-hooks",
    ),
)
def test_pending_paths_reject_mutated_unexposed_staged_terminals(
    build,
    mutate_stage,
    error: str,
) -> None:
    pending = build()
    payload = pending.model_dump(mode="json")
    [stage] = payload["staged_terminals"]
    mutate_stage(stage)

    with pytest.raises(ValidationError, match=error):
        type(pending).model_validate(payload)


@pytest.mark.parametrize(
    "build",
    (
        _ordinary_round_with_unexposed_stage,
        _approval_round_with_unexposed_stage,
        _user_input_with_unexposed_stage,
    ),
    ids=("ordinary-recovery", "approval-resolution", "user-input-resolution"),
)
def test_pending_paths_reject_exposure_attribution_on_non_unexposed_calls(build) -> None:
    pending = build()
    payload = pending.model_dump(mode="json")
    hidden_call = next(
        call for call in payload["tool_calls"] if call["tool_call_id"] == "call_hidden"
    )
    hidden_call["policy_evidence"] = ToolPolicyEvidence.UNREGISTERED.value
    hidden_call["policy_decision"] = None

    with pytest.raises(
        ValidationError,
        match="requires unexposed tool-call evidence",
    ):
        type(pending).model_validate(payload)


def test_pending_approval_scope_requires_matching_paired_round_evidence() -> None:
    pending_round = PendingToolRound(
        **_identity(),
        agent_name="assistant",
        tool_calls=[_call()],
    )
    approval = PendingToolApproval(
        approval_id="approval_1",
        **_identity(),
        tool_call_id="call_1",
        tool_name="deploy",
        arguments={"target": "production"},
        agent_name="assistant",
        publish_arguments=True,
        secret_resolution_scope="static",
        tool_calls=[_call()],
    )

    assert not approval_support.pending_approval_scope_matches_round(
        approval,
        pending_round,
    )
    assert approval_support.pending_approval_scope_matches_round(
        approval.model_copy(update={"secret_resolution_scope": "unknown"}),
        pending_round,
    )


def test_approval_closure_policy_output_requires_static_scope() -> None:
    approval = PendingToolApproval(
        approval_id="approval_1",
        **_identity(),
        tool_call_id="call_1",
        tool_name="deploy",
        arguments={"target": "production"},
        agent_name="assistant",
        publish_arguments=True,
        secret_resolution_scope="dynamic",
        tool_calls=[_call()],
    )
    denial = ToolPolicyResult(
        decision=ToolPolicyDecision.DENY,
        reason="private policy reason",
        metadata={"private": "policy metadata"},
    )

    assert approval_support.public_policy_denial_result(
        secret_resolution_scope=approval.secret_resolution_scope,
        policy_result=denial,
    ) == ToolPolicyResult(decision=ToolPolicyDecision.DENY)
    assert (
        approval_support.public_policy_denial_result(
            secret_resolution_scope="static",
            policy_result=denial,
        )
        == denial
    )
