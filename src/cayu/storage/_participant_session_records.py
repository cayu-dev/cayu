"""Shared reconstruction and exact replay checks for native session bindings."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from cayu.sessions.base import Session, SessionIdentity
from cayu.sessions.context_views import ParticipantSessionBinding, ParticipantSessionCreationReceipt
from cayu.storage._participant_bindings_schema import PARTICIPANT_BINDING_COLUMNS


def reconstruct(
    row: Mapping[str, Any], session: Session | None
) -> ParticipantSessionCreationReceipt:
    row = row_mapping(row)

    def document(value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    binding = ParticipantSessionBinding.model_validate(document(row["binding_json"]))
    receipt = ParticipantSessionCreationReceipt.model_validate(document(row["receipt_json"]))
    fields = {
        "creation_key": binding.creation_key,
        "session_id": binding.session_id,
        "session_instance_id": binding.session_instance_id,
        "application_scope": binding.application_scope,
        "participant_owner_id": binding.participant.owner.owner_id,
        "participant_owner_incarnation": binding.participant.owner.incarnation,
        "participant_id": binding.participant.participant_id,
        "participant_incarnation": binding.participant.incarnation,
        "lifecycle_revision": binding.lifecycle_revision,
        "configuration_revision": binding.configuration_revision,
        "admission_generation": binding.admission_generation,
        "creator_commitment": binding.creator_commitment,
        "authorization_commitment": binding.authorization_commitment,
        "initial_input_commitment": binding.initial_input_commitment,
        "execution_profile_commitment": binding.execution_profile_commitment,
        "request_commitment": binding.request_commitment,
    }
    if any(type(row[key]) is not type(value) or row[key] != value for key, value in fields.items()):
        raise RuntimeError("Participant binding indexes conflict with its content.")
    if receipt.binding != binding:
        raise RuntimeError("Participant receipt conflicts with its binding.")
    if session is None or (session.id, session.instance_id) != (
        binding.session_id,
        binding.session_instance_id,
    ):
        raise RuntimeError("Participant binding conflicts with its session incarnation.")
    return receipt


def row_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    return dict(zip(PARTICIPANT_BINDING_COLUMNS, row, strict=True))


def validate_replay(
    result: tuple[Session, ParticipantSessionCreationReceipt],
    identity: SessionIdentity,
    binding_factory: Callable[
        [Session], tuple[ParticipantSessionBinding, ParticipantSessionCreationReceipt]
    ],
) -> tuple[Session, ParticipantSessionCreationReceipt]:
    session, receipt = result
    expected_binding, expected_receipt = binding_factory(session.model_copy(deep=True))
    if expected_binding != receipt.binding or expected_receipt != receipt:
        raise ValueError("Participant creation key conflicts with its receipt.")
    if (session.provider_name, session.model) != (identity.provider_name, identity.model):
        raise ValueError("Participant creation key conflicts with the execution target.")
    return session, receipt
