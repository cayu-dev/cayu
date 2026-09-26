"""One exact root-execution tuple for admission and read-only reconciliation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING

from cayu._validation import canonical_durable_json_bytes

if TYPE_CHECKING:
    from cayu.collaboration._permits import PermitReceipt
    from cayu.messages import Message
    from cayu.sessions.base import Session
    from cayu.sessions.context_views import (
        ParticipantSessionBinding,
        ParticipantSessionCreationReceipt,
        ParticipantSessionExecutionRequest,
    )


def require_execution_creation(
    execution: ParticipantSessionExecutionRequest,
    receipt: ParticipantSessionCreationReceipt,
) -> None:
    """Compare against native creation, including its store-retained provenance.

    The caller must load the receipt from its registered SessionStore. Recipient
    metadata is not accepted from the execution request or replaced on replay.
    """
    from cayu.sessions.context_views import ParticipantSessionCreationRequest

    binding = receipt.binding
    if (
        execution.request.session_id != binding.session_id
        or execution.session_instance_id != binding.session_instance_id
    ):
        raise ValueError("Participant execution conflicts with its creation identity.")
    request = execution.request
    if receipt.requested_session_id is None:
        request = request.model_copy(update={"session_id": None}, deep=True)
    expected = ParticipantSessionCreationRequest(
        request=request,
        creation_key=binding.creation_key,
        metadata_json=receipt.recipient_metadata_json,
    )
    if expected.request_commitment != binding.request_commitment:
        raise ValueError("Participant execution conflicts with its creation request.")


def require_initial_execution_input(
    session: Session,
    receipt: ParticipantSessionCreationReceipt | None,
    current: list[Message],
    expected: list[Message],
) -> None:
    """Allow native first-activation adoption of exactly the retained inert input.

    Called under the store mutation lock alongside deferred-input validation.
    It does not authorize arbitrary nonempty transcript replacement.

    Ordinary participant creation can leave the transcript empty and defer the
    request's first input to runtime admission. Recipient creation can instead
    retain that input immediately. The receipt commits the prepared input in
    either case. Only ordinary creation may defer materialization; a recipient
    transcript must still match even when it has become empty.
    """
    from cayu.sessions.context_views import json_commitment

    deferred = receipt is not None and receipt.recipient_metadata_json is None and not current
    committed_input = expected if deferred else current
    if (
        receipt is None
        or (session.status, session.run_epoch) not in {("pending", 0), ("running", 1)}
        or receipt.binding.session_id != session.id
        or receipt.binding.session_instance_id != session.instance_id
        or (not deferred and current != expected)
        or receipt.initial_input_commitment
        != json_commitment(
            canonical_durable_json_bytes(
                [message.model_dump(mode="json") for message in committed_input], "initial_input"
            ).decode(),
            "initial_input",
        )
    ):
        raise RuntimeError("Initial transcript changed before finalization.")


@dataclass(frozen=True, slots=True)
class ParticipantExecutionIdentity:
    operation_key: str
    admission_commitment: str
    _material: bytes = field(repr=False)

    def permit_commitment(self, receipt: PermitReceipt) -> str:
        material = json.loads(self._material)
        material["permit"] = receipt.model_dump(mode="json")
        return sha256(
            canonical_durable_json_bytes(material, "participant_execution_admission")
        ).hexdigest()


def participant_execution_identity(
    execution: ParticipantSessionExecutionRequest,
    binding: ParticipantSessionBinding,
    *,
    execution_profile_fingerprint: str,
    wait_commitment: str | None = None,
) -> ParticipantExecutionIdentity:
    if (
        execution.request.session_id != binding.session_id
        or execution.session_instance_id != binding.session_instance_id
    ):
        raise ValueError("Participant execution identity conflicts with its immutable binding.")
    material = {
        "request": execution.request.model_dump(mode="json"),
        "binding": binding.model_dump(mode="json"),
        "session_instance_id": binding.session_instance_id,
        "execution_key": execution.execution_key,
        "execution_profile_fingerprint": execution_profile_fingerprint,
    }
    encoded = canonical_durable_json_bytes(material, "participant_execution_admission")
    if wait_commitment is not None:
        material["execution_wait"] = wait_commitment
    return ParticipantExecutionIdentity(
        operation_key="participant-execution:"
        + sha256(
            f"{binding.session_id}:{binding.session_instance_id}:{execution.execution_key}".encode()
        ).hexdigest(),
        admission_commitment=sha256(
            canonical_durable_json_bytes(material, "participant_execution_admission")
        ).hexdigest(),
        _material=encoded,
    )
