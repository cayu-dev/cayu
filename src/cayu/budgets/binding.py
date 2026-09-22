"""Trusted common-root budget bindings.

Bindings are runtime-owned authority records. They are deliberately separate
from :class:`BudgetLimit`: a limit describes accounting semantics, while a
binding proves which authenticated root is allowed to use those semantics.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_clean_nonblank,
)
from cayu.budgets.base import BudgetLimit


class BudgetBindingError(ValueError):
    """Raised when a common-root binding is malformed or conflicts."""


class BudgetBinding(BaseModel):
    """Immutable authority for one common-root budget domain.

    Construction does not authenticate a caller. Applications must obtain
    instances from a registered trusted receiver and persist the resulting
    digest with every reservation and settlement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    protocol_version: StrictStr = "cayu:budget-binding:v1"
    binding_id: StrictStr
    application_scope: StrictStr
    initiator: StrictStr
    sponsor: StrictStr
    purpose: StrictStr
    root_budget_id: StrictStr
    ancestor_budget_ids: tuple[StrictStr, ...] = ()
    limits: tuple[BudgetLimit, ...] = Field(min_length=1)
    ledger_owner: StrictStr
    receiver_id: StrictStr
    receiver_generation: StrictInt = Field(ge=0)
    # One allowance unit is one atomically admitted model or auxiliary
    # dispatch. The ledger consumes it exactly once per operation identity.
    allowance: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    retention_policy: StrictStr
    settlement_policy: StrictStr
    provider_name: StrictStr | None = None
    model: StrictStr | None = None
    environment_name: StrictStr | None = None

    @field_validator(
        "protocol_version",
        "binding_id",
        "application_scope",
        "initiator",
        "sponsor",
        "purpose",
        "root_budget_id",
        "ledger_owner",
        "receiver_id",
        "retention_policy",
        "settlement_policy",
        "provider_name",
        "model",
        "environment_name",
    )
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("ancestor_budget_ids")
    @classmethod
    def validate_ancestors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(type(item) is not str or not item.strip() for item in value):
            raise BudgetBindingError("ancestor_budget_ids must contain non-empty strings.")
        if len(set(value)) != len(value):
            raise BudgetBindingError("ancestor_budget_ids must be unique.")
        return tuple(value)

    @field_validator("limits", mode="before")
    @classmethod
    def copy_limits(cls, value: object) -> tuple[BudgetLimit, ...]:
        if not isinstance(value, (tuple, list)) or not value:
            raise BudgetBindingError("A common-root binding requires at least one budget limit.")
        copied: list[BudgetLimit] = []
        for item in value:
            if isinstance(item, BudgetLimit):
                copied.append(BudgetLimit.model_validate(item.model_dump(mode="python")))
            elif isinstance(item, dict):
                # Durable reconstruction re-enters through plain JSON. This
                # remains data validation, not authentication; only a trusted
                # receiver may authorize the reconstructed binding.
                copied.append(BudgetLimit.model_validate(item))
            else:
                raise TypeError("Budget bindings require BudgetLimit instances or mappings.")
        return tuple(copied)

    @model_validator(mode="after")
    def require_root_ceiling(self) -> BudgetBinding:
        required_keys = (self.root_budget_id, *self.ancestor_budget_ids)
        if self.root_budget_id in self.ancestor_budget_ids:
            raise BudgetBindingError("root_budget_id cannot also be an ancestor budget id.")
        causal_limits: dict[str, BudgetLimit] = {}
        for limit in self.limits:
            if limit.scope != "causal" or limit.key is None:
                continue
            if limit.key in causal_limits:
                raise BudgetBindingError(
                    "A common-root binding cannot contain duplicate causal limit keys."
                )
            causal_limits[limit.key] = limit
        if any(key not in causal_limits for key in required_keys):
            raise BudgetBindingError(
                "A common-root binding must include a causal limit keyed by root_budget_id."
            )
        # A binding ceiling is part of strict admission, not descriptive
        # metadata. A ceiling without a reservation cannot participate in the
        # atomic ledger transaction and would otherwise be silently omitted
        # by the runtime.
        if any(causal_limits[key].reservation is None for key in required_keys):
            raise BudgetBindingError(
                "Every root and ancestor ceiling must define a reservation for strict admission."
            )
        return self

    @property
    def authority_digest(self) -> str:
        """Return the stable digest of every decision-bearing authority field."""

        return sha256(
            canonical_durable_json_bytes(self.model_dump(mode="json"), "budget_binding")
        ).hexdigest()

    def exact_match(self, other: BudgetBinding) -> bool:
        """Compare complete authority, not only the binding identifier."""

        return isinstance(other, BudgetBinding) and self == other


def copy_budget_binding(binding: BudgetBinding) -> BudgetBinding:
    """Defensively reconstruct a trusted binding at an ownership boundary."""

    if type(binding) is not BudgetBinding:
        raise TypeError("binding must be a BudgetBinding.")
    # ``model_dump`` intentionally serializes nested limits to mappings, while
    # the binding validator requires already-authenticated BudgetLimit values.
    # A deep model copy preserves those validated nested objects without
    # weakening the type boundary.
    return binding.model_copy(deep=True)


@runtime_checkable
class BudgetBindingReceiver(Protocol):
    """Trusted application/runtime receiver for common-root bindings."""

    async def resolve_budget_binding(self, *, request: Any) -> BudgetBinding:
        """Resolve authenticated authority; caller-shaped values are not enough.

        Receivers may expose a ``register`` method with the same signature;
        runtime admission prefers that registration boundary when present.
        """
