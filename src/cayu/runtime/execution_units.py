from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import EXECUTION_UNIT_ID_MAX_CHARS, require_execution_unit_id


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
