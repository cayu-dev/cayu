"""One exact root-execution tuple for admission and read-only reconciliation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING

from cayu._validation import canonical_durable_json_bytes

if TYPE_CHECKING:
    from cayu.collaboration._permits import PermitReceipt
    from cayu.sessions.context_views import (
        ParticipantSessionBinding,
        ParticipantSessionExecutionRequest,
    )


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
