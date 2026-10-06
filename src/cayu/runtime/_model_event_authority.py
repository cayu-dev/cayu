"""Model execution identity attestation shared by runtime publication owners."""

from __future__ import annotations

from cayu.events import Event, event_with_runtime_payload_authority
from cayu.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
)


def _event_with_model_identity_authority(
    event: Event,
    identity: ModelStepIdentity | ModelAttemptIdentity,
) -> Event:
    """Attest model execution linkage supplied by a typed runtime identity."""

    if type(identity) is ModelAttemptIdentity:
        payload = copy_model_attempt_identity(identity).payload()
    elif type(identity) is ModelStepIdentity:
        payload = copy_model_step_identity(identity).payload()
    else:
        raise TypeError("Model event identity has an unsupported type.")
    fields = [
        field_name
        for field_name, value in payload.items()
        if event.payload.get(field_name) == value
    ]
    return event_with_runtime_payload_authority(event, *fields) if fields else event
