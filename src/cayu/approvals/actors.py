"""Shared operator identity and redacted audit payloads."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_durable_json_value, require_durable_clean_nonblank

RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX = "cayu:"


EXPIRY_RESOLUTION_ACTOR_SUBJECT = "cayu:approval-expiry"


class ResolutionActorSource(StrEnum):
    """How a ``ResolutionActor``'s identity claim was established.

    ``HTTP_AUTH`` is produced only by the server layer from a verified
    ``AuthContext``; ``REQUEST`` marks a caller-asserted identity (SDK or
    open-access HTTP body); ``SYSTEM`` marks runtime-generated actors such as
    deterministic approval expiry. Direct SDK callers are a trusted boundary
    and may construct system actors; HTTP bodies cannot — the server re-stamps
    open-access bodies to ``REQUEST`` and rejects them entirely under auth.
    """

    HTTP_AUTH = "http_auth"
    REQUEST = "request"
    SYSTEM = "system"


class ResolutionActor(BaseModel):
    """Typed actor identity for trusted runtime operator actions.

    Stamped into approval, user-input, recovery, and interruption event payloads
    so the audit trail answers who performed an operator action without
    consulting app-side state. ``reason`` and ``metadata`` on requests remain
    caller-claimed free-form data; this model is the provenance field.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    subject: str
    tenant: str | None = None
    source: ResolutionActorSource | None = None
    claims: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subject", "tenant")
    @classmethod
    def validate_nonblank_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("claims", mode="before")
    @classmethod
    def copy_claims(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_value(value, "claims")

    @model_validator(mode="after")
    def validate_reserved_subject(self) -> ResolutionActor:
        if (
            self.subject.startswith(RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX)
            and self.source != ResolutionActorSource.SYSTEM
        ):
            raise ValueError(
                "ResolutionActor subjects prefixed "
                f"{RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX!r} are reserved for system actors."
            )
        return self


def copy_resolution_actor(actor: ResolutionActor | None) -> ResolutionActor | None:
    if actor is None:
        return None
    if type(actor) is not ResolutionActor:
        raise TypeError("Resolution actors must be ResolutionActor instances.")
    return ResolutionActor(
        subject=actor.subject,
        tenant=actor.tenant,
        source=actor.source,
        claims=copy_durable_json_value(actor.claims, "claims"),
    )


def expiry_resolution_actor() -> ResolutionActor:
    """The system actor stamped on deterministic approval-expiry resolutions."""

    return ResolutionActor(
        subject=EXPIRY_RESOLUTION_ACTOR_SUBJECT,
        source=ResolutionActorSource.SYSTEM,
    )


def resolution_actor_payload(actor: ResolutionActor | None) -> dict[str, Any] | None:
    """JSON-safe event payload form of an actor (``None`` stays ``None``).

    ``claims`` are deliberately excluded: they carry deployment authorization
    state (scopes/roles) for in-process use on the request, and nothing
    redacts durable event payloads. The audit trail's who/how is
    ``subject``/``tenant``/``source``.
    """

    if actor is None:
        return None
    return actor.model_dump(mode="json", exclude={"claims"})
