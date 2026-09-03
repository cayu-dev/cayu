"""Typed authority for one operation embedded in a feature-owned record.

This module deliberately owns no journal or feature lifecycle.  A consuming
store calls :func:`transition_durable_operation_ownership` while it holds the
atomic write boundary for its own record and supplies the time sampled by that
same transaction.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import MAX_DURABLE_JSON_INTEGER, require_durable_clean_nonblank

DURABLE_OPERATION_OWNERSHIP_SCHEMA_VERSION = 1
DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS = 256
DURABLE_OPERATION_OWNERSHIP_MAX_LEASE_SECONDS = 86_400

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)


def _ownership_id(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS:
        raise ValueError(
            f"{field_name} cannot exceed {DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS} characters."
        )
    return value


class DurableOperationOwnershipState(StrEnum):
    ACTIVE = "active"
    RELEASED = "released"
    SETTLED = "settled"


class DurableOperationOwnership(BaseModel):
    """Bounded fencing authority embedded in one feature-owned record."""

    model_config = _MODEL_CONFIG

    record_type: Literal["cayu.durable-operation-ownership"] = "cayu.durable-operation-ownership"
    schema_version: Literal[1] = DURABLE_OPERATION_OWNERSHIP_SCHEMA_VERSION
    operation_id: StrictStr = Field(max_length=DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS)
    claim_id: StrictStr = Field(max_length=DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS)
    owner_id: StrictStr = Field(max_length=DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS)
    generation: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    state: DurableOperationOwnershipState = DurableOperationOwnershipState.ACTIVE
    acquired_at: datetime
    renewed_at: datetime
    lease_expires_at: datetime | None
    released_at: datetime | None = None
    settled_at: datetime | None = None

    @field_validator("schema_version", "generation", mode="before")
    @classmethod
    def validate_exact_integers(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator("operation_id", "claim_id", "owner_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _ownership_id(value, info.field_name)

    @field_validator(
        "acquired_at",
        "renewed_at",
        "lease_expires_at",
        "released_at",
        "settled_at",
    )
    @classmethod
    def validate_timestamps(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.renewed_at < self.acquired_at:
            raise ValueError("renewed_at cannot precede acquired_at.")
        if self.state is DurableOperationOwnershipState.ACTIVE:
            if (
                self.lease_expires_at is None
                or self.lease_expires_at <= self.renewed_at
                or self.released_at is not None
                or self.settled_at is not None
            ):
                raise ValueError("Active ownership requires one live-shaped lease.")
            return self
        if self.lease_expires_at is not None:
            raise ValueError("Inactive ownership cannot retain a lease expiry.")
        if self.state is DurableOperationOwnershipState.RELEASED:
            if self.released_at is None or self.settled_at is not None:
                raise ValueError("Released ownership requires exact release evidence.")
            if self.released_at < self.renewed_at:
                raise ValueError("released_at cannot precede renewed_at.")
            return self
        if self.settled_at is None or self.released_at is not None:
            raise ValueError("Settled ownership requires exact settlement evidence.")
        if self.settled_at < self.renewed_at:
            raise ValueError("settled_at cannot precede renewed_at.")
        return self


class DurableOperationOwnershipAction(StrEnum):
    CLAIM = "claim"
    RENEW = "renew"
    RELEASE = "release"
    SETTLE = "settle"


class DurableOperationOwnershipTransition(BaseModel):
    """One requested mutation or exact acknowledgement reconciliation."""

    model_config = _MODEL_CONFIG

    operation_id: StrictStr = Field(max_length=DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS)
    claim_id: StrictStr = Field(max_length=DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS)
    owner_id: StrictStr = Field(max_length=DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS)
    action: DurableOperationOwnershipAction
    generation: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    lease_seconds: StrictInt | None = Field(
        default=None,
        ge=1,
        le=DURABLE_OPERATION_OWNERSHIP_MAX_LEASE_SECONDS,
    )

    @field_validator("generation", "lease_seconds", mode="before")
    @classmethod
    def validate_optional_exact_integers(cls, value: object, info) -> object:
        if value is not None and type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator("operation_id", "claim_id", "owner_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _ownership_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_action_shape(self) -> Self:
        if self.action is DurableOperationOwnershipAction.CLAIM:
            if self.generation is not None or self.lease_seconds is None:
                raise ValueError("Claim requires a lease and no caller-selected generation.")
        elif self.action is DurableOperationOwnershipAction.RENEW:
            if self.generation is None or self.lease_seconds is None:
                raise ValueError("Renew requires the exact generation and a lease.")
        elif self.generation is None or self.lease_seconds is not None:
            raise ValueError("Release and settle require the exact generation and no lease.")
        return self


class DurableOperationOwnershipDisposition(StrEnum):
    ACQUIRED = "acquired"
    RENEWED = "renewed"
    EQUIVALENT_LIVE_OWNER = "equivalent_live_owner"
    EXPIRED_TAKEN_OVER = "expired_taken_over"
    RELEASED = "released"
    SETTLED = "settled"
    FENCED = "fenced"
    OPERATION_ADVANCED = "operation_advanced"
    IDENTITY_CONFLICT = "identity_conflict"
    INDETERMINATE = "indeterminate"


class DurableOperationOwnershipResult(BaseModel):
    """Typed result returned by a consuming store's atomic transition."""

    model_config = _MODEL_CONFIG

    disposition: DurableOperationOwnershipDisposition
    observed_at: datetime
    ownership: DurableOperationOwnership | None = None

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "observed_at")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.disposition is DurableOperationOwnershipDisposition.INDETERMINATE:
            if self.ownership is not None:
                raise ValueError("Indeterminate ownership cannot assert durable evidence.")
            return self
        if self.disposition in {
            DurableOperationOwnershipDisposition.OPERATION_ADVANCED,
            DurableOperationOwnershipDisposition.FENCED,
        }:
            return self
        if self.ownership is None:
            raise ValueError("A determinate ownership result requires durable evidence.")
        if self.disposition in {
            DurableOperationOwnershipDisposition.ACQUIRED,
            DurableOperationOwnershipDisposition.RENEWED,
            DurableOperationOwnershipDisposition.EQUIVALENT_LIVE_OWNER,
            DurableOperationOwnershipDisposition.EXPIRED_TAKEN_OVER,
        }:
            if (
                self.ownership.state is not DurableOperationOwnershipState.ACTIVE
                or self.ownership.lease_expires_at is None
                or self.ownership.lease_expires_at <= self.observed_at
            ):
                raise ValueError("A live-owner result requires an unexpired active lease.")
        elif (
            self.disposition is DurableOperationOwnershipDisposition.RELEASED
            and self.ownership.state is not DurableOperationOwnershipState.RELEASED
        ):
            raise ValueError("A release result requires released ownership evidence.")
        elif (
            self.disposition is DurableOperationOwnershipDisposition.SETTLED
            and self.ownership.state is not DurableOperationOwnershipState.SETTLED
        ):
            raise ValueError("A settlement result requires settled ownership evidence.")
        return self

    @property
    def owns_exact_claim(self) -> bool:
        return self.disposition in {
            DurableOperationOwnershipDisposition.ACQUIRED,
            DurableOperationOwnershipDisposition.RENEWED,
            DurableOperationOwnershipDisposition.EQUIVALENT_LIVE_OWNER,
            DurableOperationOwnershipDisposition.EXPIRED_TAKEN_OVER,
        }


def _result(
    disposition: DurableOperationOwnershipDisposition,
    *,
    observed_at: datetime,
    ownership: DurableOperationOwnership | None,
) -> DurableOperationOwnershipResult:
    return DurableOperationOwnershipResult(
        disposition=disposition,
        observed_at=observed_at,
        ownership=ownership,
    )


def _active_ownership(
    request: DurableOperationOwnershipTransition,
    *,
    generation: int,
    store_now: datetime,
) -> DurableOperationOwnership:
    assert request.lease_seconds is not None
    try:
        lease_expires_at = store_now + timedelta(seconds=request.lease_seconds)
    except OverflowError as error:
        raise ValueError("The requested ownership lease exceeds the store timeline.") from error
    return DurableOperationOwnership(
        operation_id=request.operation_id,
        claim_id=request.claim_id,
        owner_id=request.owner_id,
        generation=generation,
        acquired_at=store_now,
        renewed_at=store_now,
        lease_expires_at=lease_expires_at,
    )


def transition_durable_operation_ownership(
    current: DurableOperationOwnership | None,
    request: DurableOperationOwnershipTransition,
    *,
    store_now: datetime,
    operation_active: bool,
) -> DurableOperationOwnershipResult:
    """Decide one ownership transition at a feature store's write boundary.

    ``store_now`` must be sampled by the store transaction that persists the
    returned ownership.  A caller clock is never an accepted argument.  The
    feature supplies ``operation_active`` from its own phase/status machine.
    """

    if current is not None and type(current) is not DurableOperationOwnership:
        raise TypeError("current must be an exact DurableOperationOwnership or None.")
    if type(request) is not DurableOperationOwnershipTransition:
        raise TypeError("request must be an exact DurableOperationOwnershipTransition.")
    store_now = normalize_utc_datetime(store_now, "store_now")
    if type(operation_active) is not bool:
        raise TypeError("operation_active must be a bool.")
    if current is not None and current.operation_id != request.operation_id:
        return _result(
            DurableOperationOwnershipDisposition.IDENTITY_CONFLICT,
            observed_at=store_now,
            ownership=current,
        )
    exact_identity = bool(
        current is not None
        and current.claim_id == request.claim_id
        and current.owner_id == request.owner_id
        and current.generation == request.generation
    )
    if current is not None and exact_identity:
        terminal_replay: DurableOperationOwnershipDisposition | None = None
        if (
            request.action is DurableOperationOwnershipAction.RELEASE
            and current.state is DurableOperationOwnershipState.RELEASED
        ):
            terminal_replay = DurableOperationOwnershipDisposition.RELEASED
        elif (
            request.action is DurableOperationOwnershipAction.SETTLE
            and current.state is DurableOperationOwnershipState.SETTLED
        ):
            terminal_replay = DurableOperationOwnershipDisposition.SETTLED
        if terminal_replay is not None:
            return _result(
                terminal_replay,
                observed_at=store_now,
                ownership=current,
            )
    if not operation_active:
        return _result(
            DurableOperationOwnershipDisposition.OPERATION_ADVANCED,
            observed_at=store_now,
            ownership=current,
        )

    if request.action is DurableOperationOwnershipAction.CLAIM:
        if current is None or current.state is DurableOperationOwnershipState.RELEASED:
            if current is not None and current.claim_id == request.claim_id:
                return _result(
                    DurableOperationOwnershipDisposition.FENCED,
                    observed_at=store_now,
                    ownership=current,
                )
            generation = 1 if current is None else current.generation + 1
            if generation > MAX_DURABLE_JSON_INTEGER:
                return _result(
                    DurableOperationOwnershipDisposition.IDENTITY_CONFLICT,
                    observed_at=store_now,
                    ownership=current,
                )
            ownership = _active_ownership(
                request,
                generation=generation,
                store_now=store_now,
            )
            return _result(
                DurableOperationOwnershipDisposition.ACQUIRED,
                observed_at=store_now,
                ownership=ownership,
            )
        if current.state is DurableOperationOwnershipState.SETTLED:
            return _result(
                DurableOperationOwnershipDisposition.OPERATION_ADVANCED,
                observed_at=store_now,
                ownership=current,
            )
        equivalent = current.claim_id == request.claim_id and current.owner_id == request.owner_id
        if current.claim_id == request.claim_id and not equivalent:
            return _result(
                DurableOperationOwnershipDisposition.IDENTITY_CONFLICT,
                observed_at=store_now,
                ownership=current,
            )
        if current.lease_expires_at is not None and current.lease_expires_at > store_now:
            return _result(
                (
                    DurableOperationOwnershipDisposition.EQUIVALENT_LIVE_OWNER
                    if equivalent
                    else DurableOperationOwnershipDisposition.FENCED
                ),
                observed_at=store_now,
                ownership=current,
            )
        if current.generation == MAX_DURABLE_JSON_INTEGER:
            return _result(
                DurableOperationOwnershipDisposition.IDENTITY_CONFLICT,
                observed_at=store_now,
                ownership=current,
            )
        ownership = _active_ownership(
            request,
            generation=current.generation + 1,
            store_now=store_now,
        )
        return _result(
            DurableOperationOwnershipDisposition.EXPIRED_TAKEN_OVER,
            observed_at=store_now,
            ownership=ownership,
        )

    if current is None:
        return _result(
            DurableOperationOwnershipDisposition.FENCED,
            observed_at=store_now,
            ownership=None,
        )
    if current.state is DurableOperationOwnershipState.SETTLED:
        return _result(
            DurableOperationOwnershipDisposition.OPERATION_ADVANCED,
            observed_at=store_now,
            ownership=current,
        )
    exact_claim = (
        current.state is DurableOperationOwnershipState.ACTIVE
        and current.claim_id == request.claim_id
        and current.owner_id == request.owner_id
        and current.generation == request.generation
    )
    if not exact_claim:
        return _result(
            DurableOperationOwnershipDisposition.FENCED,
            observed_at=store_now,
            ownership=current,
        )
    if current.lease_expires_at is None or current.lease_expires_at <= store_now:
        return _result(
            DurableOperationOwnershipDisposition.FENCED,
            observed_at=store_now,
            ownership=current,
        )

    if request.action is DurableOperationOwnershipAction.RENEW:
        assert request.lease_seconds is not None
        try:
            requested_expiry = store_now + timedelta(seconds=request.lease_seconds)
        except OverflowError as error:
            raise ValueError("The requested ownership lease exceeds the store timeline.") from error
        ownership = DurableOperationOwnership.model_validate(
            {
                **current.model_dump(mode="python"),
                "renewed_at": max(current.renewed_at, store_now),
                "lease_expires_at": max(current.lease_expires_at, requested_expiry),
            }
        )
        return _result(
            DurableOperationOwnershipDisposition.RENEWED,
            observed_at=store_now,
            ownership=ownership,
        )

    terminal_field = (
        "released_at" if request.action is DurableOperationOwnershipAction.RELEASE else "settled_at"
    )
    ownership = DurableOperationOwnership.model_validate(
        {
            **current.model_dump(mode="python"),
            "state": (
                DurableOperationOwnershipState.RELEASED
                if request.action is DurableOperationOwnershipAction.RELEASE
                else DurableOperationOwnershipState.SETTLED
            ),
            "lease_expires_at": None,
            terminal_field: max(current.renewed_at, store_now),
        }
    )
    return _result(
        (
            DurableOperationOwnershipDisposition.RELEASED
            if request.action is DurableOperationOwnershipAction.RELEASE
            else DurableOperationOwnershipDisposition.SETTLED
        ),
        observed_at=store_now,
        ownership=ownership,
    )


__all__ = [
    "DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS",
    "DURABLE_OPERATION_OWNERSHIP_MAX_LEASE_SECONDS",
    "DURABLE_OPERATION_OWNERSHIP_SCHEMA_VERSION",
    "DurableOperationOwnership",
    "DurableOperationOwnershipAction",
    "DurableOperationOwnershipDisposition",
    "DurableOperationOwnershipResult",
    "DurableOperationOwnershipState",
    "DurableOperationOwnershipTransition",
    "transition_durable_operation_ownership",
]
