"""Exact future-session decisions shared by creation and receiving owners.

Delivery withdrawal is not creation exclusion. Peer owners may refer to this
target, but retain their own per-obligation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from pydantic import StrictBool, model_validator

from cayu.collaboration._contracts import ContractValue, Identifier, OwnerRef
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration._preparation import prepare_contract
from cayu.vaults.redaction import SecretRedactor

_SESSION_CREATION_AUTHORITY = object()


class SessionCreationConflict(ValueError):
    """The same future-target identity was used with different authority."""


class SessionCreationExcluded(PermissionError):
    """An exact future target has been durably excluded."""


class SessionCreationTarget(ContractValue):
    schema_version: Literal[1] = 1
    permit: PermitCommand
    receiving_owner: OwnerRef
    creation_key: Identifier
    requested_session_id: Identifier | None
    request_commitment: Identifier
    material_commitment: Identifier
    execution_identity_commitment: Identifier

    @model_validator(mode="after")
    def validate_receiver(self) -> SessionCreationTarget:
        registration = self.permit.intent.request
        if (
            registration.target.owner != self.receiving_owner
            or registration.target_state != "future"
            or registration.target.kind != "recipient_creation"
            or registration.effect_scope != "recipient_session_creation"
            or registration.expected_configuration_revision is None
            or self.receiving_owner.application_scope != self.permit.source.application_scope
        ):
            raise ValueError("Session creation target authority conflicts.")
        # Reserve room for the terminal identity and ExactMatch envelope before
        # admitting responsibility to this receiving store.
        if len(self.model_dump_json().encode("utf-8")) > 48 * 1024:
            raise ValueError("Session creation authority exceeds its durable bound.")
        return self

    @property
    def key(self) -> str:
        # The operation, not its mutable intent or requested public ID, owns
        # conflict detection. The complete target is compared on every access.
        return sha256(
            self.permit.intent.request.source_operation.model_dump_json().encode()
        ).hexdigest()


class SessionCreationDecision(ContractValue):
    target: SessionCreationTarget
    state: Literal["pending", "created", "excluded"]
    responsibility_registered: StrictBool = False
    settlement_acknowledged: StrictBool = False
    session_id: Identifier | None = None
    session_instance_id: Identifier | None = None
    creation_receipt_commitment: Identifier | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> SessionCreationDecision:
        if self.state == "pending" and self.settlement_acknowledged:
            raise ValueError("Pending creation cannot acknowledge settlement.")
        if self.state == "created" and not self.responsibility_registered:
            raise ValueError("Created decisions require admitted responsibility.")
        if (
            (self.state == "created")
            != (self.session_id is not None and self.session_instance_id is not None)
            or ((self.session_id is None) != (self.session_instance_id is None))
            or ((self.state == "created") != (self.creation_receipt_commitment is not None))
        ):
            raise ValueError("Session creation decision identity conflicts.")
        return self


@dataclass(frozen=True, slots=True)
class SessionCreationPage:
    decisions: tuple[SessionCreationDecision, ...]
    next_cursor: str | None


def owner_key(owner: OwnerRef) -> str:
    owner = prepare_contract(OwnerRef, owner, redactor=SecretRedactor())
    return sha256(owner.model_dump_json().encode()).hexdigest()


def discovery_bounds(cursor: str | None, limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError("Creation discovery limit must be between one and 64.")
    if cursor is not None and (
        type(cursor) is not str
        or len(cursor) != 64
        or any(character not in "0123456789abcdef" for character in cursor)
    ):
        raise ValueError("Invalid creation discovery cursor.")


def discovery_page(decisions: list[SessionCreationDecision], limit: int) -> SessionCreationPage:
    return SessionCreationPage(
        decisions=tuple(decisions[:limit]),
        next_cursor=decisions[limit - 1].target.key if len(decisions) > limit else None,
    )


def snapshot_target(value: SessionCreationTarget) -> SessionCreationTarget:
    if type(value) is not SessionCreationTarget:
        raise TypeError("Session creation requires an exact target.")
    return prepare_contract(SessionCreationTarget, value, redactor=SecretRedactor())


def require_authority(authority: object) -> None:
    if authority is not _SESSION_CREATION_AUTHORITY:
        raise PermissionError("Session creation decisions require a trusted receiving owner.")


def decide(
    target: SessionCreationTarget,
    current: SessionCreationDecision | None,
    *,
    exclude: bool = False,
    register: bool = False,
    acknowledge: bool = False,
) -> SessionCreationDecision:
    if acknowledge:
        if current is None or current.target != target:
            raise SessionCreationConflict("Settlement target conflicts with durable evidence.")
        if current.state == "pending":
            raise SessionCreationConflict("Pending creation cannot settle.")
        return SessionCreationDecision.model_validate(
            {**current.model_dump(), "settlement_acknowledged": True}
        )
    if current is not None:
        if current.target != target:
            raise SessionCreationConflict("Session creation authority conflicts.")
        if current.state != "pending" or not (exclude or register):
            return current
    if register and current is None:
        raise PermissionError("Creation responsibility requires an exact prepared target.")
    return SessionCreationDecision(
        target=target,
        state="excluded" if exclude else "pending",
        responsibility_registered=(
            register or (current.responsibility_registered if current is not None else False)
        ),
    )


def require_pending(
    target: SessionCreationTarget,
    current: SessionCreationDecision | None,
    *,
    request_commitment: str | None,
    requested_session_id: str | None,
) -> None:
    if current is None:
        raise PermissionError("Session creation target has not been registered.")
    decide(target, current)
    if current.state == "excluded":
        raise SessionCreationExcluded("Session creation target is excluded.")
    if current.state != "pending":
        raise SessionCreationConflict("Session creation target is already bound.")
    if not current.responsibility_registered:
        raise PermissionError("Session creation responsibility has not been registered.")
    if target.request_commitment != request_commitment or (
        target.requested_session_id is not None
        and target.requested_session_id != requested_session_id
    ):
        raise SessionCreationConflict("Session creation request conflicts with its target.")


def validate_binding(
    target: SessionCreationTarget, binding: Any, *, requested_session_id: str | None
) -> None:
    registration = target.permit.intent.request
    if (
        binding.participant != registration.participant
        or target.requested_session_id != requested_session_id
        or binding.creation_key != target.creation_key
        or binding.request_commitment != target.request_commitment
        or binding.initial_input_commitment != target.material_commitment
        or binding.execution_profile_commitment != target.execution_identity_commitment
        or binding.lifecycle_revision != registration.expected_lifecycle_revision
        or binding.configuration_revision != registration.expected_configuration_revision
        or binding.admission_generation != registration.admission_generation
    ):
        raise SessionCreationConflict("Session creation binding conflicts with admitted authority.")


def created(target: SessionCreationTarget, session: Any, receipt: Any) -> SessionCreationDecision:
    return SessionCreationDecision(
        target=target,
        state="created",
        responsibility_registered=True,
        session_id=session.id,
        session_instance_id=session.instance_id,
        creation_receipt_commitment=receipt.receipt_commitment,
    )
