"""The agent's inbox: what is waiting on a person, and how a person settles it.

Approvals, questions and recovery items are the agent's own pending actions;
knowledge proposals are its pending knowledge entries. The inbox lists them as
one kind of record and settles approvals, questions and knowledge proposals
through the Runtime's exact-action APIs, so a decision made from Cayu Cloud,
the agent's own app or the Developer dashboard shows as settled in the other
two. The person who decides is always the verified caller.

``InboxAccess`` is the deployment's rule for who may see and settle an item. It
is one decision point: the server applies it on every route that lists or
settles an approval, question, recovery item or knowledge proposal, not only on
the inbox routes. Without one, every caller the server's authentication admits
may, as with the rest of the control plane.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank, require_durable_nonblank
from cayu.approvals.actors import ResolutionActorSource
from cayu.approvals.review import HumanReviewReference
from cayu.approvals.revisions import ApprovalRevision
from cayu.approvals.tools import copy_approved_tool_arguments
from cayu.deadlines import ExecutionDeadlineReport
from cayu.execution_profiles import (
    EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS,
    EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS,
)
from cayu.server.auth import AuthContext

InboxItemKind = Literal["approval", "question", "recovery", "knowledge"]
InboxDecision = Literal["approve", "deny", "answer", "keep", "drop"]
InboxSettledDecision = Literal["approve", "deny", "expire", "answer", "recover", "keep", "drop"]
InboxItemStatus = Literal["pending", "settled", "superseded", "unavailable"]
# ``profile_changed``: the application changed while the approval was pending and
# the decision did not adopt the current execution profile; settle it again with
# ``profile_adoption``.
InboxConflictCode = Literal[
    "already_settled",
    "superseded",
    "expired",
    "unavailable",
    "not_settleable",
    "conflict",
    "profile_changed",
]
InboxIssueCode = Literal[
    "source_too_large",
    "source_too_complex",
    "source_invalid",
    "knowledge_truncated",
    "inbox_truncated",
]

INBOX_REASON_MAX_CHARS = 4096
INBOX_ANSWER_MAX_CHARS = 16_384

#: The decisions each kind of item accepts. Recovery items are listed so the inbox
#: is complete, but a person settles them in the Developer dashboard, where the
#: recovery evidence can be supplied.
INBOX_DECISIONS: dict[str, tuple[InboxDecision, ...]] = {
    "approval": ("approve", "deny"),
    "question": ("answer",),
    "recovery": (),
    "knowledge": ("keep", "drop"),
}


class _InboxModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class InboxKnowledgeEntry(_InboxModel):
    """The exact knowledge revision a knowledge proposal asks a person to keep or drop."""

    entry_id: str
    revision: int = Field(ge=1)
    kind: str


class InboxItem(_InboxModel):
    """One thing waiting on a person.

    Text fields are untrusted, already redacted for known secrets, and must be
    rendered as escaped text. ``id`` names this exact pause or proposal
    revision: a new question or a revised proposal is a new item.
    """

    id: str
    kind: InboxItemKind
    attention_id: str | None = Field(
        default=None,
        description=(
            "Human-attention identity of an approval, question or recovery item, as "
            "pending-action reads, supervision events and the attention handoff name it."
        ),
    )
    session_id: str | None = None
    agent_name: str | None = None
    created_at: datetime
    title: str
    detail: str | None = None
    tool_name: str | None = None
    arguments: None = Field(
        default=None, description="Proposed arguments remain quarantined before execution."
    )
    arguments_unavailable_reason: Literal["quarantined", "no_review_policy"] | None = Field(
        default=None,
        description=(
            "Why an approval or recovery item shows no arguments. no_review_policy: the "
            "app configured no human review policy, so it shows no arguments for "
            "approvals; configure one such as ToolArgumentReview. quarantined: the "
            "inbox never lists them, and the app's review policy decides what protected "
            "human review shows."
        ),
    )
    question: str | None = None
    options: list[str] = Field(default_factory=list)
    knowledge: InboxKnowledgeEntry | None = None
    decisions: list[InboxDecision] = Field(
        description="What this item accepts; empty when it is settled elsewhere."
    )
    execution_deadline: ExecutionDeadlineReport | None = Field(
        default=None,
        description=(
            "The paused session's execution deadline, for approvals and questions; null "
            "when it has none. Settling after it passes is refused unless waiting_counts "
            "is false."
        ),
    )


class InboxDecider(_InboxModel):
    """Who settled an item, as the Runtime recorded it."""

    subject: str
    tenant: str | None = None
    source: ResolutionActorSource | None = None


class InboxSettlement(_InboxModel):
    """How an item was settled.

    ``decided_by`` is ``None`` when the Runtime has no record of the person: a
    knowledge proposal dropped or kept outside the inbox, or a request settled
    under open local access.
    """

    decision: InboxSettledDecision
    reason: str | None = None
    decided_by: InboxDecider | None = None
    decided_at: datetime | None = None


class InboxItemState(_InboxModel):
    """The current state of one item: the item while pending, its settlement after."""

    id: str
    kind: InboxItemKind
    status: InboxItemStatus
    session_id: str | None = None
    agent_name: str | None = None
    item: InboxItem | None = None
    settlement: InboxSettlement | None = None

    @model_validator(mode="after")
    def validate_status_detail(self) -> InboxItemState:
        if (self.status == "pending") != (self.item is not None):
            raise ValueError("Only a pending inbox item carries the item.")
        if (self.status == "settled") != (self.settlement is not None):
            raise ValueError("Only a settled inbox item carries a settlement.")
        return self


class InboxIssue(_InboxModel):
    """Something the inbox could not list; review it in the Developer dashboard.

    A caller who sees the whole store gets the pending-action issues of sessions
    too large or invalid to list. A caller limited by tenant or access rule gets
    only ``inbox_truncated``, when other people's items fill its scan.
    """

    code: InboxIssueCode
    detail: str
    session_id: str | None = None
    agent_name: str | None = None


class InboxPage(_InboxModel):
    """One page of the inbox. Filtering can leave a page with fewer than ``limit`` items."""

    items: list[InboxItem]
    next_cursor: str | None
    has_more: bool
    issues: list[InboxIssue] = Field(default_factory=list)


class InboxCounts(_InboxModel):
    """Exact live-inbox totals, unavailable when access or source evidence is incomplete."""

    counts: dict[InboxItemKind, int] | None = None
    total_count: int | None = Field(default=None, ge=0)
    complete: bool = False
    unavailable_reason: Literal["access_limited", "source_incomplete"] | None = None

    @classmethod
    def unavailable(cls, reason: Literal["access_limited", "source_incomplete"]) -> InboxCounts:
        return cls(unavailable_reason=reason)


class InboxProfileAdoption(_InboxModel):
    """Settle an approval after the application's execution profile changed.

    A pending approval keeps the profile its run started with. After an upgrade
    that changed more than the Cayu version, the decision settles only when the
    application's ``ExecutionProfilePolicy`` adopts the current profile for this
    explicit request. The person settling is the requester. Without
    authentication, ``requested_by`` names them, and the request is otherwise
    recorded as ``unauthenticated-local-development``. Use a new
    ``idempotency_key`` for each adoption.
    """

    idempotency_key: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS)
    reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    requested_by: str | None = Field(
        default=None, max_length=EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS
    )

    @field_validator("idempotency_key", "requested_by")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        return None if value is None else require_durable_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return require_durable_nonblank(value, "reason")


class InboxSettleRequest(_InboxModel):
    """A person's decision on one inbox item.

    ``approve`` and ``deny`` settle an approval. Every decision accepts an optional
    durable ``reason``.
    ``approve`` can carry the changed ``arguments`` the tool runs with, or an
    ``approval_revision`` naming an edit the application stored itself.
    ``answer`` settles a question. ``keep`` and ``drop`` settle a knowledge
    proposal. An approval can carry ``profile_adoption`` to settle after an
    upgrade changed the application's execution profile. ``review_reference`` is
    required when the application configures a human-review policy; read it from
    ``GET /api/sessions/{id}/human-review``.
    Task-linked approvals and questions require ordinary worker/handoff authority,
    or explicit ``task_continuation="claim"`` to acquire a new workerless handoff.
    Controlled claims never borrow an active worker's authority.
    """

    decision: InboxDecision
    reason: str | None = Field(default=None, max_length=INBOX_REASON_MAX_CHARS)
    arguments: dict[str, Any] | None = None
    approval_revision: ApprovalRevision | None = None
    answer: str | None = Field(default=None, max_length=INBOX_ANSWER_MAX_CHARS)
    review_reference: HumanReviewReference | None = None
    task_worker_id: str | None = None
    task_handoff_id: str | None = None
    task_continuation: Literal["claim"] | None = None
    profile_adoption: InboxProfileAdoption | None = None

    @field_validator("task_worker_id", "task_handoff_id")
    @classmethod
    def validate_task_authority(cls, value: str | None, info) -> str | None:
        return None if value is None else require_durable_clean_nonblank(value, info.field_name)

    @field_validator("reason", "answer")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        return None if value is None else require_durable_nonblank(value, info.field_name)

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: Any) -> dict[str, Any] | None:
        return copy_approved_tool_arguments(value)

    @model_validator(mode="after")
    def validate_decision_fields(self) -> InboxSettleRequest:
        if self.task_continuation is not None and (
            self.task_worker_id is not None
            or self.task_handoff_id is not None
            or self.decision in {"keep", "drop"}
        ):
            raise ValueError("Controlled continuation is only for decisions without worker IDs.")
        if self.task_handoff_id is not None and self.task_worker_id is None:
            raise ValueError("task_handoff_id requires task_worker_id.")
        if self.task_worker_id is not None and self.decision in {"keep", "drop"}:
            raise ValueError("Task continuation authority applies to approvals and questions.")
        if (self.arguments is not None or self.approval_revision is not None) and (
            self.decision != "approve"
        ):
            raise ValueError("arguments and approval_revision require decision=approve.")
        if self.profile_adoption is not None and self.decision not in {"approve", "deny"}:
            raise ValueError("profile_adoption applies to approvals.")
        if (self.answer is not None) != (self.decision == "answer"):
            raise ValueError("answer is required for decision=answer and allowed only there.")
        if self.review_reference is not None and self.decision in {"keep", "drop"}:
            raise ValueError("review_reference applies to approvals and questions.")
        if self.review_reference is not None and self.arguments is not None:
            # The review binds the proposal it showed; changed arguments would run
            # content the review never covered.
            raise ValueError("A human-reviewed approval cannot change the arguments.")
        return self


class InboxSettleResponse(_InboxModel):
    """The settled item. ``replayed`` is true when this exact decision was already recorded."""

    replayed: bool
    item: InboxItemState


class InboxConflict(_InboxModel):
    """Why a settle was refused, with the item's current state when it is known."""

    code: InboxConflictCode
    message: str
    item: InboxItemState | None = None


class InboxConflictResponse(_InboxModel):
    """The 409 body of a refused settle."""

    detail: InboxConflict


class InboxItemScope(_InboxModel):
    """What an inbox access rule decides on: which item, from which agent and session.

    ``origin_tenant`` is the tenant the item's session was started under, when it
    recorded one, so a rule can keep each tenant to its own sessions. A
    knowledge proposal belongs to no session and carries ``knowledge`` instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: InboxItemKind
    agent_name: str | None = None
    session_id: str | None = None
    attention_id: str | None = None
    origin_tenant: str | None = None
    knowledge: InboxKnowledgeEntry | None = None


#: Decides whether an authenticated caller may see and settle one item. Return
#: ``True`` to allow; anything else hides the item from the caller on every route.
InboxAccess = Callable[[AuthContext, InboxItemScope], Awaitable[bool]]


__all__ = [
    "INBOX_DECISIONS",
    "InboxAccess",
    "InboxConflict",
    "InboxConflictCode",
    "InboxConflictResponse",
    "InboxDecider",
    "InboxDecision",
    "InboxIssue",
    "InboxItem",
    "InboxItemKind",
    "InboxItemScope",
    "InboxItemState",
    "InboxKnowledgeEntry",
    "InboxPage",
    "InboxProfileAdoption",
    "InboxSettleRequest",
    "InboxSettleResponse",
    "InboxSettledDecision",
    "InboxSettlement",
]
