"""Application-owned bounds for optional invocation-scoped inference.

These declarations carry no runtime authority, provider clients, or credentials.
Only the runtime may bind them to an executing tool's inference capability.
"""

from __future__ import annotations

import re
from math import isfinite
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import MAX_DURABLE_JSON_INTEGER

if TYPE_CHECKING:
    from cayu.providers.base import ModelRequest
    from cayu.providers.response import ModelResponse

MAX_INFERENCE_BYTES = 8 * 1024 * 1024
MAX_INFERENCE_PURPOSES = 32
_PURPOSE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")


class InferenceInvoker(Protocol):
    """Optional invocation-scoped capability supplied only by the runtime.

    This interface exposes neither provider clients nor identity/credential
    selection. A declaration or structurally matching caller object grants no
    runtime authority.
    """

    async def invoke(
        self, request: ModelRequest, *, purpose: str, limits: InferenceLimits
    ) -> ModelResponse: ...


class InferenceLimits(BaseModel):
    """Declared token envelope and local collection/deadline bounds.

    Token bounds describe a conservative request envelope, not measured usage.
    Accounting must retain the provider's actual usage, including any overrun.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_input_tokens: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    max_output_tokens: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    timeout_seconds: float = Field(gt=0, le=3600)
    max_request_bytes: StrictInt = Field(default=1024 * 1024, gt=0, le=MAX_INFERENCE_BYTES)
    max_response_bytes: StrictInt = Field(default=1024 * 1024, gt=0, le=MAX_INFERENCE_BYTES)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def validate_timeout(cls, value: object) -> float:
        if type(value) is not int and type(value) is not float:
            raise ValueError("Inference timeout must be a finite number.")
        if not 0 < value <= 3600 or not isfinite(value):
            raise ValueError("Inference timeout must be positive and at most 3600 seconds.")
        return float(value)

    def bounded_by(self, ceiling: InferenceLimits) -> InferenceLimits:
        """Validate a caller's narrower envelope without silently clipping it."""

        requested = copy_inference_limits(self)
        allowed = copy_inference_limits(ceiling)
        if any(
            getattr(requested, name) > getattr(allowed, name)
            for name in InferenceLimits.model_fields
        ):
            raise ValueError("Inference request exceeds the application-owned limits.")
        return requested


def copy_inference_limits(value: InferenceLimits) -> InferenceLimits:
    """Revalidate scalar fields without invoking potentially unsafe serialization."""

    if type(value) is not InferenceLimits:
        raise TypeError("Inference limits must be an exact InferenceLimits instance.")
    return InferenceLimits.model_validate(
        {name: getattr(value, name) for name in InferenceLimits.model_fields}
    )


def validate_inference_purpose(value: object) -> str:
    if type(value) is not str or not 1 <= len(value) <= 128 or _PURPOSE.fullmatch(value) is None:
        raise ValueError("Inference purpose must be a bounded lowercase dotted identifier.")
    return value


class AuxiliaryInferencePolicy(BaseModel):
    """Explicit application opt-in; omission grants no inference capability."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    limits: InferenceLimits
    purposes: tuple[str, ...] = Field(min_length=1, max_length=MAX_INFERENCE_PURPOSES)

    @field_validator("limits", mode="before")
    @classmethod
    def validate_limits(cls, value: object) -> InferenceLimits:
        if type(value) is InferenceLimits:
            return copy_inference_limits(value)
        if type(value) is dict:
            return InferenceLimits.model_validate(value)
        raise ValueError("Inference policy requires validated limits.")

    @field_validator("purposes", mode="before")
    @classmethod
    def validate_purposes(cls, value: object) -> tuple[str, ...]:
        if type(value) is not tuple and type(value) is not list:
            raise ValueError("Inference policy purposes must be a list or tuple.")
        if not 1 <= len(value) <= MAX_INFERENCE_PURPOSES:
            raise ValueError("Inference policy requires between 1 and 32 purposes.")
        purposes = tuple(validate_inference_purpose(item) for item in value)
        if len(set(purposes)) != len(purposes):
            raise ValueError("Inference policy purposes must be unique.")
        return tuple(sorted(purposes))


def copy_auxiliary_inference_policy(value: AuxiliaryInferencePolicy) -> AuxiliaryInferencePolicy:
    if type(value) is not AuxiliaryInferencePolicy:
        raise TypeError("Inference policy must be an exact AuxiliaryInferencePolicy instance.")
    return AuxiliaryInferencePolicy(limits=value.limits, purposes=value.purposes)


def validate_optional_auxiliary_inference_policy(value: object) -> AuxiliaryInferencePolicy | None:
    """Validate declarations and their serialized form without granting authority."""

    if value is None:
        return None
    if type(value) is AuxiliaryInferencePolicy:
        return copy_auxiliary_inference_policy(value)
    if type(value) is dict:
        return AuxiliaryInferencePolicy.model_validate(value)
    raise ValueError("Auxiliary inference requires an AuxiliaryInferencePolicy or None.")
