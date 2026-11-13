"""Tool calls proposed by application code instead of a model.

Workflow code that already knows the exact call it wants a person to approve
does not need a model turn. ``tool_call_proposal_request`` builds an ordinary
``RunRequest`` for the agent whose model step is replaced by a runtime-owned,
zero-priced control step that raises exactly the proposed call. Everything
after that is the ordinary session loop: the agent's tool policy (including
``GuardedToolPolicy`` guards) decides, an approval pauses the session in the
ordinary approval checkpoint where the Developer dashboard, ``cayu session``
and ``CayuApp.resolve_tool_approval`` find it, approval runs the tool, a denial
is recorded as ``tool.call.approval_denied``, and recovery after a restart is
the same as for a model-raised call. No model provider is called.

``CayuApp.propose_tool_call`` runs that request and summarizes the outcome. Run
the request yourself instead to attach it to a durable task (pass ``task_id``,
``task_worker_id`` and ``task_lease_expires_at``) or to stream its events.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cayu._validation import copy_durable_json_object
from cayu.events import Event, EventType
from cayu.messages import TextPart
from cayu.runtime._tool_call_proposals import (
    TOOL_CALL_PROPOSAL_MODEL,
    TOOL_CALL_PROPOSAL_PROVIDER_NAME,
    proposal_message,
)
from cayu.session_metadata import MAX_SESSION_TITLE_LENGTH
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions._tool_call_proposal_record import tool_call_proposal_record
from cayu.sessions.requests import RunRequest, bind_runtime_tool_call_proposal
from cayu.tools.base import ToolResult
from cayu.tools.policy import ToolPolicyDecision
from cayu.vaults.redaction import SecretRedactor

# One step raises the call; the second records that it settled.
_PROPOSAL_MAX_STEPS = 2
_PROPOSAL_OWNED_FIELDS = frozenset(
    {
        "agent_name",
        "messages",
        "target",
        "failover",
        "max_steps",
        "structured_output",
        "tool_completion",
        "loop_policies",
    }
)


class ToolCallProposalStatus(StrEnum):
    """Where a proposed tool call stands."""

    PENDING_APPROVAL = "pending_approval"
    """The call waits in the ordinary approval checkpoint."""
    COMPLETED = "completed"
    """The tool ran and returned a result."""
    FAILED = "failed"
    """The tool ran, or was meant to, and failed."""
    BLOCKED = "blocked"
    """A tool policy, guard or hook refused the call; no approval was requested."""
    DENIED = "denied"
    """A person denied the approval."""
    EXPIRED = "expired"
    """The approval expired before anyone approved it."""
    UNSETTLED = "unsettled"
    """The session stopped before the call settled, for example on an interruption."""


_TERMINAL_STATUS: dict[str, ToolCallProposalStatus] = {
    EventType.TOOL_CALL_COMPLETED: ToolCallProposalStatus.COMPLETED,
    EventType.TOOL_CALL_FAILED: ToolCallProposalStatus.FAILED,
    EventType.TOOL_CALL_BLOCKED: ToolCallProposalStatus.BLOCKED,
    EventType.TOOL_CALL_APPROVAL_DENIED: ToolCallProposalStatus.DENIED,
    EventType.TOOL_CALL_APPROVAL_EXPIRED: ToolCallProposalStatus.EXPIRED,
}


class ToolCallProposalOutcome(BaseModel):
    """The state of one proposed call, read from its session's public events.

    ``approval_id``, ``tool_round_id`` and ``tool_call_id`` are the identifiers
    ``ToolApprovalRequest`` takes. ``result`` is set when the tool ran.
    ``reason`` carries a block or denial reason when the event recorded one,
    or the error the session failed with before the call settled.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    status: ToolCallProposalStatus
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_round_id: str | None = None
    approval_id: str | None = None
    result: ToolResult | None = None
    reason: str | None = None

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> ToolCallProposalOutcome:
        """Summarize a proposal session from its events, in order.

        Pass the events ``app.run`` yielded for the proposal, or later the
        session's persisted events, to learn whether it is still pending.
        """

        session_id: str | None = None
        status = ToolCallProposalStatus.UNSETTLED
        fields: dict[str, Any] = {}
        session_error: str | None = None
        for event in events:
            if type(event) is not Event:
                raise TypeError("events must contain Event instances.")
            session_id = session_id or event.session_id
            payload = event.payload
            terminal = _TERMINAL_STATUS.get(event.type)
            if event.type == EventType.SESSION_FAILED:
                error = payload.get("error")
                session_error = error if type(error) is str else None
                if status is ToolCallProposalStatus.PENDING_APPROVAL:
                    # A failed (for example closed) session no longer waits
                    # for anyone, so nobody can answer its approval.
                    status = ToolCallProposalStatus.UNSETTLED
                continue
            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
                status = ToolCallProposalStatus.PENDING_APPROVAL
            elif terminal is not None:
                # Expiry closes the round with a denial marked expired.
                expired = event.type == EventType.TOOL_CALL_APPROVAL_DENIED and (
                    payload.get("expired") is True
                )
                status = ToolCallProposalStatus.EXPIRED if expired else terminal
                result = payload.get("result")
                if type(result) is dict:
                    fields["result"] = ToolResult.model_validate(result)
                reason = payload.get("reason") or payload.get("block_reason")
                if type(reason) is str:
                    fields["reason"] = reason
            elif event.type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_APPROVED}:
                # No longer awaiting a person, and not settled yet.
                status = ToolCallProposalStatus.UNSETTLED
            else:
                continue
            # A later call event supersedes an earlier session failure.
            session_error = None
            if event.tool_name is not None:
                fields["tool_name"] = event.tool_name
            for name in ("tool_call_id", "tool_round_id", "approval_id"):
                value = payload.get(name)
                if type(value) is str:
                    fields[name] = value
        if session_id is None:
            raise ValueError("A proposal outcome needs at least one session event.")
        if status is ToolCallProposalStatus.UNSETTLED and session_error is not None:
            # The call stays unsettled: a session that failed after the tool
            # started may have had its effect. The error says why it stopped.
            fields["reason"] = session_error
        return cls(session_id=session_id, status=status, **fields)


class ToolCallPreview(BaseModel):
    """What proposing one call would do now, decided without recording anything.

    ``decision`` is the tool policy's, or DENY when the call is stopped before the
    policy runs; ``blocked_by`` then says where (``unregistered``,
    ``tool_exposure`` or ``invalid_arguments``), and is ``tool_policy`` when the
    policy denied it. ``metadata`` is the policy result's metadata, such as the
    setting reading a ``SettingToolPolicy`` decided from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    agent_name: str
    tool_name: str
    decision: ToolPolicyDecision
    blocked_by: (
        Literal["unregistered", "tool_exposure", "invalid_arguments", "tool_policy"] | None
    ) = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    proposal_only: bool = False
    effect: str | None = None


def _string_leaves(value: Any) -> Iterable[str]:
    if type(value) is str:
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _string_leaves(key)
            yield from _string_leaves(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        # Any sequence, including Cayu's immutable JSON arrays.
        for item in value:
            yield from _string_leaves(item)


def _require_secret_free_proposal(
    redactor: SecretRedactor,
    *,
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    tool_call_id: str | None,
) -> None:
    """Refuse a proposal that would record a registered secret in the transcript.

    Every string is checked as written, before JSON encoding: an escaped quote,
    backslash or newline would otherwise hide a secret from the redactor while
    ``json.loads`` still recovers it from the stored proposal. The arguments are
    first normalized by the same durable JSON copy the proposal message uses, so
    every container Cayu accepts as JSON (including its immutable arrays and
    mappings) is walked.
    """

    if arguments is not None and not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping.")
    normalized = copy_durable_json_object(dict(arguments or {}), "arguments")
    for text in _string_leaves([tool_name, tool_call_id, normalized]):
        if redactor.redact_text(text) != text:
            raise ValueError(
                "Tool call proposal contains a registered secret; bind the credential "
                "to the tool instead of passing it as an argument."
            )


def _require_unredacted_proposal_message(redactor: SecretRedactor, request: RunRequest) -> None:
    """Refuse a proposal whose stored message redaction would still change.

    Each string passed the check above, but the stored message escapes every
    quoted redaction marker, and a short registered secret can match inside
    that escape. The provider would then refuse the proposal, or not find it.
    """

    for message in request.messages:
        for part in message.content:
            if isinstance(part, TextPart) and redactor.redact_text(part.text) != part.text:
                raise ValueError(
                    "Tool call proposal would be altered by secret redaction when it is "
                    "recorded; a registered secret matches part of its stored form."
                )


def tool_call_proposal_request(
    agent_name: str,
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    tool_call_id: str | None = None,
    **request_fields: Any,
) -> RunRequest:
    """Build a ``RunRequest`` that raises one exact tool call for ``agent_name``.

    The call uses the agent's registered tools and tool policy. Other
    ``RunRequest`` fields, such as ``session_id``, ``title``, ``labels``,
    ``metadata``, ``budget_limits`` or the task attachment fields, pass through.
    The proposal owns ``messages``, ``target``, ``failover``, ``max_steps``,
    ``structured_output``, ``tool_completion`` and ``loop_policies``.

    The arguments are recorded in the session transcript as the proposal, so
    keep secrets out of them; bind credentials to the tool instead.
    """

    owned = sorted(_PROPOSAL_OWNED_FIELDS.intersection(request_fields))
    if owned:
        raise ValueError(f"A tool call proposal owns these RunRequest fields: {', '.join(owned)}.")
    if arguments is not None and not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping.")
    call_id = tool_call_id or f"proposal-{uuid4().hex}"
    copied_arguments = copy_durable_json_object(dict(arguments or {}), "arguments")
    message = proposal_message(
        tool_name=tool_name, arguments=copied_arguments, tool_call_id=call_id
    )
    request_fields.setdefault("title", f"Proposed {tool_name}"[:MAX_SESSION_TITLE_LENGTH])
    request = RunRequest(
        agent_name=agent_name,
        messages=[message],
        target=ModelTarget(
            provider_name=TOOL_CALL_PROPOSAL_PROVIDER_NAME, model=TOOL_CALL_PROPOSAL_MODEL
        ),
        max_steps=_PROPOSAL_MAX_STEPS,
        **request_fields,
    )
    # The runtime-owned record is what admits a proposal-only tool: the
    # transcript text alone never does, because any caller that can start a
    # run can write a proposal envelope.
    bind_runtime_tool_call_proposal(
        request,
        tool_call_proposal_record(
            tool_name=tool_name, tool_call_id=call_id, arguments=copied_arguments
        ),
    )
    return request


__all__ = [
    "ToolCallPreview",
    "ToolCallProposalOutcome",
    "ToolCallProposalStatus",
    "tool_call_proposal_request",
]
