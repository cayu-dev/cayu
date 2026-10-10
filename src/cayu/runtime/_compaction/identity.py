"""Bind compactor completion evidence to runtime-issued dispatch identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cayu._validation import (
    copy_durable_json_object,
)
from cayu.context.base import (
    _COMPACTION_ATTEMPT_ID_KEY,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
)


@dataclass
class _CompactionExecutionIdentityLedger:
    """Bind internal compaction evidence to pre-dispatch runtime identities."""

    model_step_identity: ModelStepIdentity
    active_model_attempt_identity: ModelAttemptIdentity | None = None
    model_attempts_by_compaction_id: dict[str, ModelAttemptIdentity] = field(default_factory=dict)
    compaction_ids_by_model_attempt_id: dict[str, str] = field(default_factory=dict)
    issued_model_attempts_by_id: dict[str, ModelAttemptIdentity] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_step_identity = copy_model_step_identity(self.model_step_identity)

    def begin_dispatch(self, identity: ModelAttemptIdentity) -> ModelAttemptIdentity:
        if self.active_model_attempt_identity is not None:
            raise RuntimeError("Compaction provider dispatches cannot overlap.")
        identity = copy_model_attempt_identity(identity)
        self.active_model_attempt_identity = identity
        self.issued_model_attempts_by_id[identity.model_attempt_id] = identity
        return copy_model_attempt_identity(identity)

    def end_dispatch(self, identity: ModelAttemptIdentity) -> None:
        identity = copy_model_attempt_identity(identity)
        if self.active_model_attempt_identity != identity:
            raise RuntimeError("Compaction provider dispatch identity was not active.")
        self.active_model_attempt_identity = None

    def identify_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_identity: ModelAttemptIdentity | None = None,
    ) -> dict[str, Any]:
        """Bind an already copied payload to its exact issued provider dispatch."""

        expected = (
            None if expected_identity is None else copy_model_attempt_identity(expected_identity)
        )
        compaction_id = payload.get(_COMPACTION_ATTEMPT_ID_KEY)
        if type(compaction_id) is not str:
            raise RuntimeError("Compaction completion evidence lost its attempt identity.")
        identity = self.model_attempts_by_compaction_id.get(compaction_id)
        candidate = expected or self.active_model_attempt_identity
        payload_identity: ModelAttemptIdentity | None = None
        if "model_step_id" in payload or "model_attempt_id" in payload:
            try:
                payload_identity = ModelAttemptIdentity.model_validate(
                    {
                        "model_step_id": payload.get("model_step_id"),
                        "model_attempt_id": payload.get("model_attempt_id"),
                    }
                )
            except (TypeError, ValueError):
                raise ValueError(
                    "Compaction completion carries an invalid model attempt identity."
                ) from None
            issued_identity = self.issued_model_attempts_by_id.get(
                payload_identity.model_attempt_id
            )
            if issued_identity != payload_identity:
                raise ValueError(
                    "Compaction completion carries a model attempt identity that "
                    "was not issued for this logical step."
                )
        if candidate is not None and payload_identity is not None and candidate != payload_identity:
            raise ValueError("Compaction completion identity conflicts with its provider dispatch.")
        candidate = candidate or payload_identity
        if identity is None:
            if candidate is None:
                raise RuntimeError(
                    "Compaction completion was observed outside its provider dispatch."
                )
            existing_compaction_id = self.compaction_ids_by_model_attempt_id.get(
                candidate.model_attempt_id
            )
            if existing_compaction_id is not None and existing_compaction_id != compaction_id:
                raise ValueError(
                    "Compaction provider dispatch produced conflicting completion identities."
                )
            identity = copy_model_attempt_identity(candidate)
            self.model_attempts_by_compaction_id[compaction_id] = identity
        elif candidate is not None and identity != candidate:
            raise ValueError("Compaction completion identity conflicts with its provider dispatch.")
        existing_compaction_id = self.compaction_ids_by_model_attempt_id.setdefault(
            identity.model_attempt_id,
            compaction_id,
        )
        if existing_compaction_id != compaction_id:
            raise ValueError(
                "Compaction provider dispatch produced conflicting completion identities."
            )
        payload.update(identity.payload())
        return payload

    def identify_payloads(
        self,
        payloads: list[dict[str, Any]],
        *,
        expected_identity: ModelAttemptIdentity | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.identify_payload(
                copy_durable_json_object(payload, "compaction_model_completed_payload"),
                expected_identity=expected_identity,
            )
            for payload in payloads
        ]
