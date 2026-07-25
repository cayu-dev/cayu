from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import EXECUTION_UNIT_ID_MAX_CHARS, require_execution_unit_id

RUNTIME_OWNED_EXECUTION_IDENTITY_FIELDS = frozenset(
    {
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "budget_limit_id",
        "reservation_id",
    }
)


def strip_runtime_owned_execution_identity(payload: dict[str, Any]) -> None:
    """Remove identity authority that an extension or provider cannot supply."""

    if type(payload) is not dict:
        raise TypeError("Runtime identity payload must be an object.")
    for field_name in RUNTIME_OWNED_EXECUTION_IDENTITY_FIELDS:
        payload.pop(field_name, None)


class ModelStepIdentity(BaseModel):
    """Runtime-owned identity of one logical model-loop decision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model_step_id: str = Field(max_length=EXECUTION_UNIT_ID_MAX_CHARS)

    @field_validator("model_step_id")
    @classmethod
    def validate_model_step_id(cls, value: str) -> str:
        return require_execution_unit_id(value, "model_step_id")

    def payload(self) -> dict[str, str]:
        return {"model_step_id": self.model_step_id}

    def new_attempt(self) -> ModelAttemptIdentity:
        return ModelAttemptIdentity(
            model_step_id=self.model_step_id,
            model_attempt_id=f"matt_{uuid4().hex}",
        )


class ModelAttemptIdentity(ModelStepIdentity):
    """Runtime-owned identity of one actual provider-dispatch attempt."""

    model_attempt_id: str = Field(max_length=EXECUTION_UNIT_ID_MAX_CHARS)

    @field_validator("model_attempt_id")
    @classmethod
    def validate_model_attempt_id(cls, value: str) -> str:
        return require_execution_unit_id(value, "model_attempt_id")

    def payload(self) -> dict[str, str]:
        return {
            "model_step_id": self.model_step_id,
            "model_attempt_id": self.model_attempt_id,
        }

    def new_tool_round(self) -> ToolRoundIdentity:
        return ToolRoundIdentity(
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
            tool_round_id=f"tround_{uuid4().hex}",
        )


class ToolRoundIdentity(ModelAttemptIdentity):
    """Runtime-owned identity of one tool group requested by a model attempt."""

    tool_round_id: str = Field(max_length=EXECUTION_UNIT_ID_MAX_CHARS)

    @field_validator("tool_round_id")
    @classmethod
    def validate_tool_round_id(cls, value: str) -> str:
        return require_execution_unit_id(value, "tool_round_id")

    def payload(self) -> dict[str, str]:
        return {
            "model_step_id": self.model_step_id,
            "model_attempt_id": self.model_attempt_id,
            "tool_round_id": self.tool_round_id,
        }

    def matches_payload(self, payload: Mapping[str, object]) -> bool:
        """Return whether a payload carries this complete, unmodified identity."""

        return all(payload.get(key) == value for key, value in self.payload().items())


class BudgetLimitIdentity(BaseModel):
    """Opaque runtime identity of one effective configured budget limit."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    budget_limit_id: str = Field(max_length=EXECUTION_UNIT_ID_MAX_CHARS)

    @field_validator("budget_limit_id")
    @classmethod
    def validate_budget_limit_id(cls, value: str) -> str:
        return require_execution_unit_id(value, "budget_limit_id")

    def payload(self) -> dict[str, str]:
        return {"budget_limit_id": self.budget_limit_id}


def new_model_step_identity() -> ModelStepIdentity:
    """Allocate a globally unique logical model-step identity."""

    return ModelStepIdentity(model_step_id=f"mstep_{uuid4().hex}")


def copy_model_step_identity(identity: ModelStepIdentity) -> ModelStepIdentity:
    if type(identity) is not ModelStepIdentity:
        raise TypeError("Model step identity must be a ModelStepIdentity.")
    return ModelStepIdentity(model_step_id=identity.model_step_id)


def copy_model_attempt_identity(identity: ModelAttemptIdentity) -> ModelAttemptIdentity:
    if type(identity) is not ModelAttemptIdentity:
        raise TypeError("Model attempt identity must be a ModelAttemptIdentity.")
    return ModelAttemptIdentity(
        model_step_id=identity.model_step_id,
        model_attempt_id=identity.model_attempt_id,
    )


def copy_tool_round_identity(identity: ToolRoundIdentity) -> ToolRoundIdentity:
    if type(identity) is not ToolRoundIdentity:
        raise TypeError("Tool round identity must be a ToolRoundIdentity.")
    return ToolRoundIdentity(
        model_step_id=identity.model_step_id,
        model_attempt_id=identity.model_attempt_id,
        tool_round_id=identity.tool_round_id,
    )
