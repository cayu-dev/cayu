"""Saved foreground-child waits and continuation evidence shared with session stores.

These records do not grant execution authority. Runtime and native publication
owners authenticate the exact parent, child and action before consuming them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, copy_durable_metadata
from cayu.approvals.user_input import user_input_lifecycle_authority_from_checkpoint
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
    has_run_limit_accounting_authority,
    resume_run_limit_accounting_context,
)
from cayu.runtime.tool_effects import _bounded_text
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._pending_tool_round import PendingToolRound
from cayu.sessions._tool_effect_intent import ToolEffectIntent
from cayu.sessions.base import ResumeRequest, SessionRunFenced
from cayu.sessions.records import MAX_SESSION_ID_BYTES, Session

FOREGROUND_CHILD_WAIT_KEY = "foreground_child_wait"
FOREGROUND_CHILD_TERMINAL_KEY = "foreground_child_terminal"
FOREGROUND_PARENT_CONTINUATION_KEY = "foreground_parent_continuation"
FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY = "foreground_child_post_action_continuation"


class ForegroundChildResumeRequest(ResumeRequest):
    """Private continuation input; public resume still requires new user messages.

    Only the exact-wait internal entrance accepts this type. It does not carry
    new input or grants, and the enclosing invocation admission checks the
    persisted wait under its checkpoint and incarnation fence.
    """

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        if value:
            raise ValueError("A foreground child continuation cannot introduce new messages.")
        return []


class ForegroundChildWait(BaseModel):
    """One exact parent execution waiting for one exact child action.

    The complete effect intent retains the parent incarnation, interaction,
    effective arguments and execution profile. The child identity is independent
    of its current run epoch: normal action resolution advances that epoch but
    must not replace the original child incarnation or interaction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    parent_effect: ToolEffectIntent
    child_session_id: StrictStr
    child_session_instance_id: StrictStr
    child_interaction_id: StrictStr
    child_spawn_fingerprint: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    child_action_kind: Literal["tool_approval", "user_input", "delegated_action"]
    child_action_id: StrictStr
    child_action_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    revision: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Foreground child wait has an unsupported schema version.")
        return value

    @field_validator(
        "child_session_id",
        "child_session_instance_id",
        "child_interaction_id",
        "child_action_id",
    )
    @classmethod
    def bounded_identity(cls, value: str, info) -> str:
        return _bounded_text(
            value,
            info.field_name,
            maximum=MAX_SESSION_ID_BYTES if info.field_name == "child_session_id" else 256,
            identifier=True,
        )

    def delegated_action_reference(self) -> dict[str, str]:
        """Public discovery only; never copy the child's question or arguments."""

        return {
            "child_session_id": self.child_session_id,
            "action_kind": self.child_action_kind,
            "action_id": self.child_action_id,
            "status": "waiting_on_child_action",
        }


class ForegroundChildTerminal(BaseModel):
    """Exact child outcome selected for one suspended parent execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    wait: ForegroundChildWait
    child_released_run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    event_id: StrictStr = Field(min_length=1, max_length=256)
    event_type: Literal["session.completed", "session.failed", "session.interrupted"]
    event_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("event_id")
    @classmethod
    def bounded_event_id(cls, value: str) -> str:
        return _bounded_text(value, "event_id", maximum=256, identifier=True)

    @model_validator(mode="after")
    def terminal_follows_pause(self) -> ForegroundChildTerminal:
        if self.child_released_run_epoch <= self.wait.child_action_run_epoch:
            raise ValueError("Child terminal selection must follow its action pause epoch.")
        return self


class ForegroundParentContinuation(BaseModel):
    """Result attached; the original parent interaction still owns continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    terminal: ForegroundChildTerminal
    publication_id: StrictStr = Field(min_length=1, max_length=256)
    request: ForegroundChildResumeRequest
    completed_model_step: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    run_limit_accounting: RunLimitAccountingContext | None = None
    task_id: StrictStr | None = None

    @model_validator(mode="after")
    def exact_parent_scope(self) -> ForegroundParentContinuation:
        intent = self.terminal.wait.parent_effect
        publications = {f"tool-round:{intent.tool_round_id}"}
        if intent.approval_id is not None:
            publications.add(f"approval-close:{intent.approval_id}")
        if intent.pause_id is not None:
            publications.add(f"user-input-close:{intent.pause_id}")
        if (
            self.publication_id not in publications
            or self.request.session_id != intent.session_id
            or self.request.target is not None
            or self.request.failover is not None
            or self.request.profile_adoption is not None
            or self.request.tool_grants
            or self.request.tool_capability_ceiling is not None
            or self.request.loop_policies
            or self.completed_model_step > self.request.max_steps
            or (
                has_run_limit_accounting_authority(self.request.limits, self.request.budget_limits)
                and self.run_limit_accounting is None
            )
            or (
                self.run_limit_accounting is not None
                and self.run_limit_accounting.baseline.session_id != intent.session_id
            )
        ):
            raise ValueError("Attached foreground continuation conflicts with its original scope.")
        return self


class ForegroundChildPostActionContinuation(BaseModel):
    """Durable claim that a closed child action still needs its next model turn.

    This is evidence only: recovery must authenticate the parent effect, child
    incarnation, and close publication receipt before consuming the claim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    wait: ForegroundChildWait
    action_kind: Literal["tool_approval", "user_input"]
    action_id: StrictStr = Field(min_length=1, max_length=256)
    close_publication_id: StrictStr = Field(min_length=1, max_length=256)
    completed_model_step: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    continuation_revision: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    pending_tool_round: dict[str, Any]
    request_metadata: dict[str, Any]

    @field_validator("request_metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: Any) -> dict[str, Any]:
        return copy_durable_metadata(value, "post_action.request_metadata")

    @model_validator(mode="after")
    def validate_scope(self) -> ForegroundChildPostActionContinuation:
        prefix = "approval-close" if self.action_kind == "tool_approval" else "user-input-close"
        if self.close_publication_id != f"{prefix}:{self.action_id}":
            raise ValueError("Post-action continuation has an invalid close publication.")
        return self


def post_action_continuation_round_from_checkpoint(
    checkpoint: dict[str, Any],
) -> PendingToolRound | None:
    """Retain execution configuration, not merely pending-action display evidence."""
    from cayu.sessions.pending_actions import pending_action_evidence_round_from_checkpoint

    pending_round = pending_action_evidence_round_from_checkpoint(checkpoint)
    pending_input, input_intent = user_input_lifecycle_authority_from_checkpoint(checkpoint)
    if pending_round is not None and pending_input is not None:
        pending_round = pending_round.model_copy(
            update={
                field: getattr(pending_input, field)
                for field in (
                    "max_steps",
                    "limits",
                    "run_limit_accounting",
                    "budget_limits",
                    "retry_policy",
                    "thinking",
                    "source_run_epoch",
                )
            },
            deep=True,
        )
    approval = pending_approval_reader.pending_approval_from_checkpoint(checkpoint)
    if approval is not None:
        context = approval.run_limit_accounting
        intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(checkpoint)
    else:
        context = None if pending_input is None else pending_input.run_limit_accounting
        intent = input_intent
    if pending_round is not None and context is not None:
        pending_round = pending_round.model_copy(
            update={
                "run_limit_accounting": resume_run_limit_accounting_context(
                    context, resolved_at=None if intent is None else intent.pause_resolved_at
                )
            },
            deep=True,
        )
    return pending_round


def post_action_continuation_for_close(
    checkpoint: dict[str, Any] | None,
    *,
    session: Session,
    wait_checkpoint: dict[str, Any] | None = None,
    close_publication_id: str,
    completed_model_step: int,
    request_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind the current child action separately from parent discovery evidence.

    The retained wait supplies original spawn linkage and its observed revision;
    discovery can still name an earlier action. Only the child-owned checkpoint
    supplies the action being closed. This never refreshes or revives the parent.
    """
    if checkpoint is None:
        return None
    wait_source = checkpoint if wait_checkpoint is None else wait_checkpoint
    if FOREGROUND_CHILD_WAIT_KEY not in wait_source:
        return None
    wait = ForegroundChildWait.model_validate(wait_source[FOREGROUND_CHILD_WAIT_KEY])
    if wait_checkpoint is None and wait.parent_effect.session_id == session.id:
        # This session is closing its own resolved gate after a delegated child,
        # not closing an action on behalf of an ancestor waiting for this session.
        return None
    pending_round = post_action_continuation_round_from_checkpoint(checkpoint)
    if pending_round is None:
        return None
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if (
        wait.child_session_id != session.id
        or wait.child_session_instance_id != session.instance_id
        or wait.parent_effect.session_id != session.parent_session_id
        or profile is None
        or profile.session_id != session.id
        or profile.interaction_id != wait.child_interaction_id
        or profile.run_epoch != session.run_epoch
    ):
        raise RuntimeError("Action-close continuation conflicts with its child invocation.")
    approval = pending_approval_reader.pending_approval_from_checkpoint(checkpoint)
    pending_input, _ = user_input_lifecycle_authority_from_checkpoint(
        checkpoint, current_run_epoch=session.run_epoch
    )
    if (approval is None) == (pending_input is None):
        raise RuntimeError("Action-close continuation requires one current child action.")
    if approval is not None:
        action_kind = "tool_approval"
        action_id = approval.approval_id
    else:
        assert pending_input is not None
        action_kind = "user_input"
        action_id = pending_input.input_id
    marker = ForegroundChildPostActionContinuation(
        wait=wait,
        action_kind=action_kind,
        action_id=action_id,
        close_publication_id=close_publication_id,
        completed_model_step=completed_model_step,
        continuation_revision=wait.revision,
        pending_tool_round=pending_round.model_dump(mode="json"),
        request_metadata=request_metadata,
    )
    return marker.model_dump(mode="json")


def post_action_continuation_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> ForegroundChildPostActionContinuation | None:
    """Parse a persisted post-close claim without granting execution authority."""
    if checkpoint is None or FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY not in checkpoint:
        return None
    return ForegroundChildPostActionContinuation.model_validate(
        checkpoint[FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY]
    )


def foreground_child_state_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> tuple[ForegroundChildWait | None, ForegroundChildTerminal | None]:
    """Reconstruct one coherent wait/selection topology, rejecting orphan claims."""
    if checkpoint is None:
        return None, None
    wait = (
        ForegroundChildWait.model_validate(checkpoint[FOREGROUND_CHILD_WAIT_KEY])
        if FOREGROUND_CHILD_WAIT_KEY in checkpoint
        else None
    )
    terminal = (
        ForegroundChildTerminal.model_validate(checkpoint[FOREGROUND_CHILD_TERMINAL_KEY])
        if FOREGROUND_CHILD_TERMINAL_KEY in checkpoint
        else None
    )
    if terminal is not None and (wait is None or terminal.wait != wait):
        raise RuntimeError("Foreground terminal selection has no exact retained wait.")
    return wait, terminal


def gate_close_continuation(checkpoint, *, pending, publication_id, metadata):
    wait, selected = foreground_child_state_from_checkpoint(checkpoint)
    if wait is None:
        return None
    if selected is None or pending.tool_round_id != wait.parent_effect.tool_round_id:
        raise SessionRunFenced("Gate closure lacks its selected child outcome.")
    pending = post_action_continuation_round_from_checkpoint(checkpoint)
    if pending is None:
        raise SessionRunFenced("Gate closure lost its original round.")
    if pending.max_steps is None or pending.limits is None or pending.budget_limits is None:
        raise SessionRunFenced("Gate closure lacks original execution semantics.")
    pending_input, _ = user_input_lifecycle_authority_from_checkpoint(checkpoint)
    completed_model_step = pending.model_step if pending_input is None else pending_input.model_step
    if type(completed_model_step) is not int:
        raise SessionRunFenced("Gate closure lacks its consumed model step.")
    return ForegroundParentContinuation(
        terminal=selected,
        publication_id=publication_id,
        completed_model_step=completed_model_step,
        run_limit_accounting=pending.run_limit_accounting,
        task_id=pending.task_id,
        request=ForegroundChildResumeRequest(
            session_id=wait.parent_effect.session_id,
            messages=[],
            metadata=metadata,
            max_steps=pending.max_steps,
            limits=pending.limits,
            budget_limits=pending.budget_limits,
            retry_policy=pending.retry_policy,
            structured_output=pending.structured_output,
            thinking=pending.thinking,
        ),
    ).model_dump(mode="json")
