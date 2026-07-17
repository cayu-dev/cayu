from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import (
    copy_json_value,
    freeze_json_value,
    require_clean_nonblank,
    require_durable_json_text,
    thaw_json_value,
)


class PricingContext(BaseModel):
    """One exact set of commercial dimensions that may price a dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimensions: Mapping[str, str]

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_dimensions_input(cls, value: Any) -> dict[str, str]:
        copied = copy_json_value(value, "dimensions")
        if type(copied) is not dict:
            raise ValueError("dimensions must be an object.")
        if not copied:
            raise ValueError("dimensions must not be empty.")
        result: dict[str, str] = {}
        for key, item in copied.items():
            clean_key = require_clean_nonblank(key, "dimension name")
            if type(item) is not str:
                raise ValueError(f"Pricing dimension {clean_key!r} must be a string.")
            result[clean_key] = require_clean_nonblank(item, f"dimensions.{clean_key}")
        return result

    @field_validator("dimensions")
    @classmethod
    def freeze_dimensions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return freeze_json_value(dict(value))

    @field_serializer("dimensions")
    def serialize_dimensions(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(thaw_json_value(value))

    def storage_key(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.dimensions.items()))


class BillingIdentity(BaseModel):
    """Provider-neutral commercial identity and possible pricing contexts.

    Providers own the meaning of their evidence. Core preserves it durably and
    enforces that completion cannot rewrite request evidence or widen the set of
    pricing outcomes established before dispatch.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: str
    resource_id: str
    request_evidence: Mapping[str, Any] = Field(default_factory=dict)
    completion_evidence: Mapping[str, Any] = Field(default_factory=dict)
    pricing_contexts: tuple[PricingContext, ...] = ()

    @field_validator("provider_name", "resource_id")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("request_evidence", "completion_evidence", mode="before")
    @classmethod
    def validate_evidence_input(cls, value: Any, info) -> dict[str, Any]:
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not dict:
            raise ValueError(f"{info.field_name} must be an object.")
        require_durable_json_text(copied, info.field_name)
        return copied

    @field_validator("request_evidence", "completion_evidence")
    @classmethod
    def freeze_evidence(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_json_value(dict(value))

    @field_serializer("request_evidence", "completion_evidence")
    def serialize_evidence(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(thaw_json_value(value))

    @model_validator(mode="after")
    def validate_unique_pricing_contexts(self) -> BillingIdentity:
        keys = [context.storage_key() for context in self.pricing_contexts]
        if len(keys) != len(set(keys)):
            raise ValueError("pricing_contexts must be distinct.")
        return self


class UnresolvedBillingIdentity(BaseModel):
    """The provider's request hook has not resolved commercial identity yet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["unresolved"] = "unresolved"


class ResolvedBillingIdentity(BaseModel):
    """The provider hook ran; the dispatch may or may not have an identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["resolved"] = "resolved"
    identity: BillingIdentity | None = None

    @field_validator("identity")
    @classmethod
    def validate_identity(cls, value: BillingIdentity | None) -> BillingIdentity | None:
        return copy_billing_identity(value)


BillingIdentityState = Annotated[
    UnresolvedBillingIdentity | ResolvedBillingIdentity,
    Field(discriminator="status"),
]

UNRESOLVED_BILLING_IDENTITY = UnresolvedBillingIdentity()


def copy_billing_identity(identity: BillingIdentity | None) -> BillingIdentity | None:
    """Validate and detach one provider-supplied billing identity."""

    if identity is None:
        return None
    if not isinstance(identity, BillingIdentity):
        raise TypeError("Provider billing identity must be a BillingIdentity or None.")
    return BillingIdentity.model_validate(identity.model_dump(mode="json"))


def resolved_billing_identity(identity: BillingIdentity | None) -> ResolvedBillingIdentity:
    return ResolvedBillingIdentity(identity=identity)


def billing_identity_value(state: BillingIdentityState) -> BillingIdentity | None:
    return state.identity if isinstance(state, ResolvedBillingIdentity) else None


def completed_billing_identity(
    requested: BillingIdentity | None,
    completed: BillingIdentity | None,
) -> BillingIdentity | None:
    """Validate provider completion evidence against the request identity."""

    if requested is None:
        if completed is not None:
            raise ValueError("Completion billing identity has no request identity.")
        return None
    requested = copy_billing_identity(requested)
    assert requested is not None
    if completed is None:
        raise ValueError("Completion billing identity discarded the request identity.")
    completed = copy_billing_identity(completed)
    assert completed is not None
    if (
        requested.provider_name != completed.provider_name
        or requested.resource_id != completed.resource_id
        or requested.request_evidence != completed.request_evidence
    ):
        raise ValueError("Completion billing identity conflicts with request identity.")
    requested_contexts = {context.storage_key() for context in requested.pricing_contexts}
    completed_contexts = {context.storage_key() for context in completed.pricing_contexts}
    if requested_contexts and not completed_contexts:
        raise ValueError("Completion billing identity discarded request pricing contexts.")
    if not completed_contexts.issubset(requested_contexts):
        raise ValueError("Completion billing identity widened request pricing contexts.")
    return completed
