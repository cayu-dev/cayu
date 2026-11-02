"""Trusted eval-time reviewers for agents whose tool calls wait for a person's approval.

An agent can end its turns with a tool that needs a person's approval. A corpus
trial of such an agent stops at the approval and never reaches the answer the
corpus checks. An eval reviewer, declared on the trusted ``CorpusTarget`` and
never in a corpus, plays that person inside each trial: it approves each pending
call as the agent proposed it, as a named subject, at most a bounded number of
times. Its decisions are ordinary runtime facts (``tool_approval_requested`` and
``tool_approved``), and its policy is part of the target's identity.

The reviewer acts only inside an eval trial: on the trial's own root session,
while the eval runner runs that trial, and only on a session the runtime marked
as an eval trial when it created it. Nothing outside the eval runner calls it,
so it cannot approve anything in a served session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu.approvals.actors import (
    RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX,
    ResolutionActor,
    ResolutionActorSource,
)
from cayu.approvals.review import HumanReviewContext, HumanReviewDenied
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.evals.corpus import _bounded_durable_text, _content_revision, _sha256_revision
from cayu.sessions._run_identity import in_eval_trial_scope, is_eval_trial
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import PendingActionKind, SessionStatus

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.events import Event
    from cayu.runtime.loop_policies import LoopPolicy
    from cayu.sessions.pending_action_contracts import PendingActionRecord

# The most approvals one trial's reviewer may give; a trial's deadline bounds it too.
EVAL_REVIEWER_MAX_APPROVALS = 64
# Bounds of the human-review context the reviewer presents (``HumanReviewContext``).
_SUBJECT_MAX_CHARS = 512
_REVIEW_PURPOSE_MAX_CHARS = 128
_REVIEW_TENANT_MAX_CHARS = 512
APPROVE_AS_PROPOSED = "approve_as_proposed"
# Scenario approval checkpoints each take a fresh operator decision instead.
SCENARIO_WITH_EVAL_REVIEWER = (
    "A scenario decides each approval at its own checkpoint, so it cannot run on a target "
    "that declares an eval reviewer."
)


def _reviewer_subject(value: str, field_name: str) -> str:
    value = _bounded_durable_text(
        value, field_name, max_chars=_SUBJECT_MAX_CHARS, nonblank=True, clean=True
    )
    if value.startswith(RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX):
        raise ValueError(
            f"`{field_name}` cannot use the reserved "
            f"{RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX!r} prefix."
        )
    return value


def _optional_text(value: str | None, field_name: str, max_chars: int) -> str | None:
    if value is None:
        return None
    return _bounded_durable_text(value, field_name, max_chars=max_chars, nonblank=True, clean=True)


class _ApproveAsProposedPolicy(BaseModel):
    """The approve-as-proposed policy's settings, shared by its declaration and identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    policy: Literal["approve_as_proposed"] = APPROVE_AS_PROPOSED
    subject: StrictStr
    max_approvals: StrictInt = Field(ge=1, le=EVAL_REVIEWER_MAX_APPROVALS)
    review_purpose: StrictStr | None = Field(default=None, exclude_if=lambda value: value is None)
    review_tenant: StrictStr | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str, info) -> str:
        return _reviewer_subject(value, info.field_name)

    @field_validator("review_purpose")
    @classmethod
    def validate_review_purpose(cls, value: str | None, info) -> str | None:
        return _optional_text(value, info.field_name, _REVIEW_PURPOSE_MAX_CHARS)

    @field_validator("review_tenant")
    @classmethod
    def validate_review_tenant(cls, value: str | None, info) -> str | None:
        return _optional_text(value, info.field_name, _REVIEW_TENANT_MAX_CHARS)


class EvalReviewerIdentityV1(_ApproveAsProposedPolicy):
    """Portable identity of the eval reviewer a corpus target declares.

    A target's reviewer is part of its execution identity: results under
    different reviewers are not compared as equals.
    """

    schema_version: Literal[1] = 1
    revision: StrictStr
    # An identity always names its policy.
    policy: Literal["approve_as_proposed"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_revision(self) -> EvalReviewerIdentityV1:
        if self.revision != _content_revision(self.model_dump(mode="json"), "eval reviewer"):
            raise ValueError("Eval reviewer identity revision does not match its content.")
        return self


class ApproveAsProposedReviewer(_ApproveAsProposedPolicy):
    """Trusted eval-time reviewer that approves each pending tool call unchanged.

    Declare it on a ``CorpusTarget`` as ``eval_reviewer``; a corpus can never
    supply one. In each trial it approves the trial's pending tool approval as
    the agent proposed it, recording ``subject`` as the resolver, at most
    ``max_approvals`` times. A trial still waiting for an approval after that
    is reported as stopped awaiting approval.

    Under a human review policy (``CayuApp(human_review_policy=...)``) every
    decision needs the review the policy shows. The reviewer then inspects it
    as the recipient ``subject``, with ``review_purpose`` (by default the
    policy's ``dashboard_purpose``) and ``review_tenant``, and the policy must
    authorize that context to inspect and decide. It approves only a permitted
    review, and never changes the proposed arguments.
    """

    def identity(self) -> EvalReviewerIdentityV1:
        """The reviewer's portable identity, as a target's identity publishes it."""

        document = {"schema_version": 1, **self.model_dump(mode="json")}
        return EvalReviewerIdentityV1.model_validate(
            {**document, "revision": _content_revision(document, "eval reviewer")}
        )


def eval_reviewer_text(identity: EvalReviewerIdentityV1) -> str:
    """How a report names the eval reviewer a run's target declared."""

    approvals = "approval" if identity.max_approvals == 1 else "approvals"
    return (
        f"Eval reviewer: {identity.policy} as {identity.subject}, at most "
        f"{identity.max_approvals} {approvals} per trial (revision {identity.revision})."
    )


def copy_eval_reviewer(value: object) -> ApproveAsProposedReviewer | None:
    """Validate a target's ``eval_reviewer`` and return a detached copy."""

    if value is None:
        return None
    if type(value) is not ApproveAsProposedReviewer:
        raise TypeError("eval_reviewer must be an exact ApproveAsProposedReviewer or None.")
    return ApproveAsProposedReviewer.model_validate(value.model_dump(mode="python"))


class EvalReviewerScopeError(PermissionError):
    """An eval reviewer was asked to act outside the eval trial it belongs to."""


class _TrialReviewer:
    """One trial's reviewer: approves the trial session's pending tool calls.

    ``review`` yields the trial's own run, then each approval's continuation, as
    one stream. ``stop_reason`` says why it left an approval pending, if it did.
    ``loop_policies`` are the trial request's: a session's request loop policies
    are not checkpointed, so every continuation passes them again.
    """

    def __init__(
        self,
        app: CayuApp,
        reviewer: ApproveAsProposedReviewer,
        session_id: str,
        *,
        loop_policies: tuple[LoopPolicy, ...] = (),
    ) -> None:
        self._app = app
        self._reviewer = reviewer
        self._session_id = session_id
        self._loop_policies = loop_policies
        self._actor = ResolutionActor(subject=reviewer.subject, source=ResolutionActorSource.SYSTEM)
        self.approvals = 0
        self.stop_reason: str | None = None

    async def review(self, stream: AsyncIterator[Event]) -> AsyncIterator[Event]:
        async for event in stream:
            yield event
        while (pending := await self._pending_approval()) is not None:
            if self.approvals >= self._reviewer.max_approvals:
                self.stop_reason = (
                    f"the eval reviewer had approved its limit of {self.approvals} "
                    f"{'call' if self.approvals == 1 else 'calls'}"
                )
                return
            request = await self._approval(pending)
            if request is None:
                return
            self.approvals += 1
            async for event in self._app.resolve_tool_approval(request):
                yield event

    async def _pending_approval(self) -> PendingActionRecord | None:
        """The trial session's first pending tool approval, once it stopped on one."""

        session = await self._app.session_store.load(self._session_id)
        if session is None or not is_eval_trial(session) or not in_eval_trial_scope():
            raise EvalReviewerScopeError(
                "An eval reviewer acts only on its own eval trial's session, inside the trial."
            )
        if session.status is not SessionStatus.INTERRUPTED:
            return None
        result = await self._app.session_store.query_pending_actions(
            PendingActionQuery(
                session_id=self._session_id,
                statuses=frozenset({SessionStatus.INTERRUPTED}),
                kind=PendingActionKind.TOOL_APPROVAL,
                limit=1,
            )
        )
        return result.actions[0] if result.actions else None

    async def _approval(self, pending: PendingActionRecord) -> ToolApprovalRequest | None:
        """Approve ``pending`` as proposed, through the review a human review policy shows."""

        if pending.approval_id is None or pending.round_id is None or pending.tool_call_id is None:
            self.stop_reason = "the pending approval had no runtime linkage to resolve"
            return None
        if not self._app.human_review_configured:
            return ToolApprovalRequest(
                session_id=self._session_id,
                approval_id=pending.approval_id,
                tool_round_id=pending.round_id,
                tool_call_id=pending.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                resolved_by=self._actor,
                loop_policies=self._loop_policies,
            )
        purpose = self._reviewer.review_purpose or self._app.human_review_purpose
        if purpose is None:
            self.stop_reason = "the eval reviewer has no human-review purpose to review it with"
            return None
        context = HumanReviewContext(
            recipient=self._reviewer.subject,
            tenant=self._reviewer.review_tenant,
            purpose=purpose,
        )
        try:
            view = await self._app.inspect_human_review(self._session_id, context=context)
        except HumanReviewDenied:
            self.stop_reason = (
                "the application's human review policy did not let the eval reviewer inspect it"
            )
            return None
        if (
            view.kind != "tool_approval"
            or view.status != "permitted"
            or view.reference is None
            or view.interaction_id is None
            or view.tool_round_id is None
            or view.tool_call_id is None
        ):
            self.stop_reason = (
                "the application's human review policy did not show the eval reviewer the "
                f"call (review {view.status})"
            )
            return None
        try:
            await self._app.require_human_review_resolution_authority(
                self._session_id, view.reference
            )
        except HumanReviewDenied:
            self.stop_reason = (
                "the application's human review policy did not let the eval reviewer decide it"
            )
            return None
        return ToolApprovalRequest(
            session_id=self._session_id,
            approval_id=view.interaction_id,
            tool_round_id=view.tool_round_id,
            tool_call_id=view.tool_call_id,
            decision=ToolApprovalDecision.APPROVE,
            review_reference=view.reference,
            resolved_by=self._actor,
            loop_policies=self._loop_policies,
        )


__all__ = [
    "EVAL_REVIEWER_MAX_APPROVALS",
    "ApproveAsProposedReviewer",
    "EvalReviewerIdentityV1",
    "EvalReviewerScopeError",
]
