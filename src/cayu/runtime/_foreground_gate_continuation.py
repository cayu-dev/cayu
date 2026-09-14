"""Store-owned continuation of an accepted human resolution across delegation.

The stored request is a redacted execution projection, not a second public
resolution. Its original digests remain bound to the existing resolution intent.
Only the private child-terminal entrance may use those digests with this projection.
"""

from dataclasses import dataclass
from typing import Any, Literal

from cayu._validation import canonical_durable_json_bytes
from cayu.approvals.tools import ToolApprovalRequest
from cayu.approvals.user_input import (
    UserInputRecoveryRequest,
    UserInputResponse,
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.runtime import _approval_support as approval_support
from cayu.runtime._foreground_child_wait import (
    FOREGROUND_CHILD_TERMINAL_KEY,
    ForegroundChildResumeRequest,
    ForegroundChildTerminal,
    ForegroundChildWait,
    ForegroundParentContinuation,
    foreground_child_state_from_checkpoint,
    post_action_continuation_round_from_checkpoint,
)
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.loop_policies import LoopPolicy
from cayu.sessions.base import (
    Session,
    SessionOperationPublication,
    SessionRunFenced,
    SessionStore,
    _invocation_lifecycle_authority_read_scope,
)


def _key(kind: str, action_id: str, resolution_digest: str) -> str:
    return f"foreground-gate:{kind}:{action_id}:{resolution_digest}"


class ForegroundGatePolicyOwner:
    """Own live request policies across ordinary and accepted-gate child waits.

    A digest authenticates behavior but cannot reconstruct executable objects.
    Missing process-owned behavior must therefore refuse continuation, including
    after restart. The normal profile validator still checks the live objects.
    """

    def __init__(self) -> None:
        self._policies: dict[tuple[str, ...], tuple[LoopPolicy, ...]] = {}
        self._wait_policies: dict[tuple[str, str, str], tuple[LoopPolicy, ...]] = {}

    @staticmethod
    def _wait_identity(session: Session, wait: ForegroundChildWait) -> tuple[str, str, str]:
        return (
            session.instance_id,
            wait.parent_effect.interaction_id,
            wait.parent_effect.execution_profile_fingerprint,
        )

    def retain_wait(
        self, session: Session, wait: ForegroundChildWait, policies: tuple[LoopPolicy, ...]
    ) -> None:
        """Retain executable inputs before releasing the paused invocation owner."""
        if policies:
            self._wait_policies[self._wait_identity(session, wait)] = policies

    def for_wait(self, session: Session, wait: ForegroundChildWait) -> tuple[LoopPolicy, ...]:
        # Missing objects (including after restart) still face the authoritative
        # invocation-profile check; a durable fingerprint cannot recreate them.
        return self._wait_policies.get(self._wait_identity(session, wait), ())

    @staticmethod
    def _identity(record: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            record[field]
            for field in (
                "session_instance_id",
                "interaction_id",
                "profile_fingerprint",
                "kind",
                "action_id",
                "resolution_request_digest",
            )
        )

    def retain(self, record: dict[str, Any], policies: tuple[LoopPolicy, ...]) -> None:
        if policies:
            self._policies[self._identity(record)] = policies

    def resolve(self, record: dict[str, Any]) -> tuple[LoopPolicy, ...]:
        count = record.get("loop_policy_count")
        if type(count) is not int or count < 0:
            raise SessionRunFenced("Retained gate lacks exact loop-policy authority.")
        policies = self._policies.get(self._identity(record), ())
        if len(policies) != count:
            raise SessionRunFenced("Accepted gate requires its original live loop policies.")
        return policies

    def release(self, *, session: Session, kind: str, action_id: str) -> None:
        for identity in tuple(self._policies):
            if identity[0] == session.instance_id and identity[3:5] == (kind, action_id):
                del self._policies[identity]

    def release_interaction(self, *, session: Session, interaction_id: str) -> None:
        for identity in tuple(self._wait_policies):
            if identity[:2] == (session.instance_id, interaction_id):
                del self._wait_policies[identity]
        for identity in tuple(self._policies):
            if identity[:2] == (session.instance_id, interaction_id):
                del self._policies[identity]

    async def for_attached(
        self, store: SessionStore, *, parent: Session, continuation: ForegroundParentContinuation
    ) -> tuple[LoopPolicy, ...]:
        if continuation.publication_id.split(":", 1)[0] not in {
            "approval-close",
            "user-input-close",
        }:
            return self.for_wait(parent, continuation.terminal.wait)
        return self.resolve(
            await self.attached_record(store, parent=parent, continuation=continuation)
        )

    async def attached_record(
        self, store: SessionStore, *, parent: Session, continuation: ForegroundParentContinuation
    ) -> dict[str, Any]:
        prefix, action_id = continuation.publication_id.split(":", 1)
        if prefix not in {"approval-close", "user-input-close"}:
            raise SessionRunFenced("Policy restoration requires an accepted gate closure.")
        kind = "approval" if prefix == "approval-close" else "input"
        receipt = await store.load_runtime_publication_receipt(
            parent.id, continuation.publication_id
        )
        digest = None if receipt is None else receipt.intent.get("resolution_request_digest")
        if type(digest) is not str:
            raise SessionRunFenced("Attached gate lacks its exact resolution identity.")
        record = await store.load_session_operation(parent.id, _key(kind, action_id, digest))
        effect = continuation.terminal.wait.parent_effect
        if (
            record is None
            or receipt is None
            or record.get("session_instance_id") != parent.instance_id
            or record.get("interaction_id") != effect.interaction_id
            or record.get("profile_fingerprint") != effect.execution_profile_fingerprint
            or record.get("kind") != kind
            or record.get("action_id") != action_id
            or record.get("resolution_request_digest")
            != receipt.intent.get("resolution_request_digest")
        ):
            raise SessionRunFenced("Attached gate lost its accepted policy authority.")
        return record


def validate_gate_restoration_request(record, request, *, redactor) -> None:
    """Compare public retry material before allowing live policy restoration."""
    if isinstance(request, UserInputRecoveryRequest):
        request = gate_input_response_from_recovery(request)
    if (
        type(record.get("loop_policy_count")) is not int
        or record["loop_policy_count"] != len(request.loop_policies)
        or canonical_durable_json_bytes(record["request"], "gate_request")
        != canonical_durable_json_bytes(_gate_request_projection(request, redactor), "gate_request")
    ):
        raise SessionRunFenced("Policy restoration conflicts with the accepted resolution.")


def _gate_request_projection(request, redactor):
    projection = request.model_dump(mode="json", exclude={"review_reference"})
    for field in ("metadata", "reason", "answer", "structured", "artifacts"):
        if field in projection:
            projection[field] = redactor.redact_json_values(projection[field])
    return projection


def gate_input_response_from_recovery(request: UserInputRecoveryRequest) -> UserInputResponse:
    """Project an authenticated recovery's accepted answer, not its tool outcome."""
    return UserInputResponse(
        review_reference=request.answer_review_reference or request.review_reference,
        session_id=request.session_id,
        task_worker_id=request.task_worker_id,
        task_handoff_id=request.task_handoff_id,
        input_id=request.input_id,
        answer=request.answer,
        structured=request.structured,
        artifacts=request.artifacts,
        metadata=request.metadata,
        resolved_by=request.resolved_by,
        max_steps=request.max_steps,
        limits=request.limits,
        budget_limits=request.budget_limits,
        retry_policy=request.retry_policy,
        structured_output=request.structured_output,
        thinking=request.thinking,
        loop_policies=request.loop_policies,
    )


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


def _intent(checkpoint, kind):
    if kind == "approval":
        intent = approval_support.approval_resolution_intent_from_checkpoint(checkpoint)
        return intent, None
    _, intent = user_input_lifecycle_authority_from_checkpoint(checkpoint)
    return intent, None if intent is None else intent.answer_request_digest


async def retain_gate_request(
    store: SessionStore,
    *,
    session: Session,
    request: ToolApprovalRequest | UserInputResponse,
    redactor,
    policy_owner: ForegroundGatePolicyOwner,
) -> None:
    """Retain accepted semantics before a child can be dispatched or paused."""
    kind = "approval" if isinstance(request, ToolApprovalRequest) else "input"
    action_id = (
        request.approval_id if isinstance(request, ToolApprovalRequest) else request.input_id
    )
    checkpoint = await store.load_checkpoint(session.id)
    intent, answer_digest = _intent(checkpoint, kind)
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if intent is None or intent.resolution_request_digest is None or profile is None:
        raise SessionRunFenced("Delegated resolution lacks its accepted request authority.")
    key = _key(kind, action_id, intent.resolution_request_digest)
    projection = _gate_request_projection(request, redactor)
    record = {
        "session_instance_id": session.instance_id,
        "interaction_id": profile.interaction_id,
        "profile_fingerprint": profile.profile.fingerprint,
        "kind": kind,
        "action_id": action_id,
        "resolution_request_digest": intent.resolution_request_digest,
        "answer_request_digest": answer_digest,
        "resolution_stage": intent.resolution_stage if kind == "input" else None,
        "loop_policy_count": len(request.loop_policies),
        "request": projection,
    }

    def publish(current, source, existing):
        current_intent, current_answer = _intent(source, kind)
        current_profile = active_invocation_execution_profile_from_checkpoint(source)
        if (
            current.instance_id != session.instance_id
            or current_intent is None
            or current_intent.resolution_request_digest != intent.resolution_request_digest
            or current_answer != answer_digest
            or current_profile is None
            or current_profile.interaction_id != profile.interaction_id
            or current_profile.profile != profile.profile
        ):
            raise SessionRunFenced("Accepted resolution changed before retention.")
        if existing is not None and existing != record:
            raise SessionRunFenced("Accepted delegated resolution has conflicting semantics.")
        return SessionOperationPublication(checkpoint=source, operation_records={key: record})

    # Retain before the publication await: acknowledgement loss must not leave a
    # committed request without its live execution owner.
    policy_owner.retain(record, request.loop_policies)
    with _invocation_lifecycle_authority_read_scope():
        await store.publish_session_operation(
            session.id,
            idempotency_key=key,
            operation_transform=publish,
            events=[],
            expected_statuses={session.status},
            expected_run_epoch=session.run_epoch,
        )


@dataclass(frozen=True)
class GateReplay:
    """Private, store-resolved projection; never accepted by a public API."""

    terminal: ForegroundChildTerminal
    record: dict[str, Any]

    @property
    def input_resolution_stage(self) -> Literal["answer", "manual-recovery"]:
        stage = self.record.get("resolution_stage")
        if stage not in {"answer", "manual-recovery"}:
            raise SessionRunFenced("Delegated input lacks its authenticated resolution stage.")
        return stage

    @property
    def resolution_digest(self) -> str:
        return self.record["resolution_request_digest"]

    @property
    def answer_digest(self) -> str:
        return self.record["answer_request_digest"]

    def select(self, session: Session, checkpoint):
        wait, selected = foreground_child_state_from_checkpoint(checkpoint)
        profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        intent, answer_digest = _intent(checkpoint, self.record["kind"])
        if (
            wait != self.terminal.wait
            or selected not in (None, self.terminal)
            or session.instance_id != self.record["session_instance_id"]
            or profile is None
            or profile.interaction_id != self.record["interaction_id"]
            or profile.profile.fingerprint != self.record["profile_fingerprint"]
            or intent is None
            or intent.resolution_request_digest != self.resolution_digest
            or answer_digest != self.record["answer_request_digest"]
            or self.record.get("resolution_stage")
            != (intent.resolution_stage if self.record["kind"] == "input" else None)
        ):
            raise SessionRunFenced("Delegated gate continuation lost its exact source.")
        return {
            **checkpoint,
            FOREGROUND_CHILD_TERMINAL_KEY: self.terminal.model_dump(mode="json"),
        }


async def load_gate_request(
    store: SessionStore, *, parent: Session, wait: ForegroundChildWait
) -> dict[str, Any] | None:
    checkpoint = await store.load_checkpoint(parent.id)
    approval = approval_support.pending_approval_from_checkpoint(checkpoint)
    pending_input, _ = user_input_lifecycle_authority_from_checkpoint(checkpoint)
    if approval is None and pending_input is None:
        return None
    if approval is not None and pending_input is not None:
        raise SessionRunFenced("Delegated continuation has conflicting gates.")
    kind = "approval" if approval is not None else "input"
    if approval is not None:
        action_id = approval.approval_id
    else:
        assert pending_input is not None
        action_id = pending_input.input_id
    effect = wait.parent_effect
    if (
        effect.session_id != parent.id
        or effect.session_instance_id != parent.instance_id
        or (effect.approval_id if approval is not None else effect.pause_id) != action_id
    ):
        raise SessionRunFenced("Delegated terminal belongs to a different accepted gate.")
    intent, answer_digest = _intent(checkpoint, kind)
    if intent is None or intent.resolution_request_digest is None:
        raise SessionRunFenced("Delegated resolution lost its accepted request identity.")
    record = await store.load_session_operation(
        parent.id, _key(kind, action_id, intent.resolution_request_digest)
    )
    if record is None or record.get("kind") != kind or record.get("action_id") != action_id:
        raise SessionRunFenced("Delegated resolution has no retained accepted request.")
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if (
        profile is None
        or intent is None
        or record.get("session_instance_id") != parent.instance_id
        or record.get("interaction_id") != effect.interaction_id
        or profile.interaction_id != effect.interaction_id
        or record.get("profile_fingerprint") != profile.profile.fingerprint
        or record.get("profile_fingerprint") != effect.execution_profile_fingerprint
        or record.get("resolution_request_digest") != intent.resolution_request_digest
        or record.get("answer_request_digest") != answer_digest
        or record.get("resolution_stage") != (intent.resolution_stage if kind == "input" else None)
    ):
        raise SessionRunFenced("Retained resolution lost its original authority.")
    request_type = ToolApprovalRequest if kind == "approval" else UserInputResponse
    request = request_type.model_validate(record["request"])
    if request.session_id != parent.id:
        raise SessionRunFenced("Retained resolution belongs to another session.")
    # Round-trip validation must not silently normalize decision-bearing fields.
    if canonical_durable_json_bytes(request.model_dump(mode="json"), "gate_request") != (
        canonical_durable_json_bytes(record["request"], "gate_request")
    ):
        raise SessionRunFenced("Retained resolution projection is noncanonical.")
    return record


async def load_gate_replay(
    store: SessionStore, *, parent: Session, terminal: ForegroundChildTerminal
) -> GateReplay | None:
    record = await load_gate_request(store, parent=parent, wait=terminal.wait)
    if record is None:
        return None
    replay = GateReplay(terminal, record)
    replay.select(parent, await store.load_checkpoint(parent.id))
    return replay
