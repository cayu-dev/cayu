"""Provider-neutral record of the generation controls a model request carries.

Adapters report the controls they send after Cayu's own mapping and defaults
(``ModelProvider.request_controls``), and may report the values a provider
echoes back in its completion payload under ``ECHOED_REQUEST_CONTROLS_KEY``.
Neither record contains prompts, messages, tool definitions or raw options.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    require_durable_clean_nonblank,
)

REQUEST_CONTROLS_PAYLOAD_KEY = "request_controls"
"""``model.started`` payload key holding the controls the adapter sent."""

ECHOED_REQUEST_CONTROLS_KEY = "echoed_request_controls"
"""Completion payload key holding the controls the provider echoed back."""

REQUEST_CONTROLS_MISMATCH_KEY = "request_controls_mismatch"
"""``model.completed`` payload key naming echoed controls that differ from those sent."""

REQUEST_CONTROL_FIELDS: tuple[str, ...] = (
    "reasoning_effort",
    "thinking_budget_tokens",
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
)

_MAX_REASONING_EFFORT_CHARS = 64
_INTEGER_FIELDS = frozenset({"thinking_budget_tokens", "top_k", "max_output_tokens"})
_NUMBER_FIELDS = frozenset({"temperature", "top_p"})


class ModelRequestControls(BaseModel):
    """Effective generation controls one model request sends to its provider.

    ``None`` means the adapter does not send that control, so the provider's own
    default applies. Effort values are the provider's native vocabulary, recorded
    unchanged. ``max_output_tokens`` is the cap the adapter sends, whichever wire
    field carries it (``max_output_tokens``, ``max_tokens``, ``max_completion_tokens``
    or Bedrock ``maxTokens``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def validate_reasoning_effort(cls, value: object) -> str | None:
        if value is None:
            return None
        if not _valid_control_value("reasoning_effort", value):
            raise ValueError("reasoning_effort must be a short printable nonblank string.")
        return cast("str", value)

    @field_validator("thinking_budget_tokens", "top_k", "max_output_tokens", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info) -> int | None:
        if value is None:
            return None
        if not _valid_control_value(info.field_name, value):
            raise ValueError(f"{info.field_name} must be a nonnegative durable integer.")
        return cast("int", value)

    @field_validator("temperature", "top_p", mode="before")
    @classmethod
    def validate_number(cls, value: object, info) -> float | None:
        if value is None:
            return None
        if not _valid_control_value(info.field_name, value):
            raise ValueError(f"{info.field_name} must be a finite number.")
        return float(cast("int | float", value))

    def payload(self) -> dict[str, Any]:
        """The controls actually sent, as durable JSON; unsent controls are omitted."""

        return {
            field: value
            for field in REQUEST_CONTROL_FIELDS
            if (value := getattr(self, field)) is not None
        }


def request_controls_from_values(**values: object) -> ModelRequestControls:
    """Build controls from wire values, dropping any value of an unusable shape.

    Adapters forward raw provider options unchanged, so a caller can send, say, a
    string temperature. Such a value cannot be recorded as a typed control and is
    left out instead of failing the request.
    """

    unknown = set(values) - set(REQUEST_CONTROL_FIELDS)
    if unknown:
        raise TypeError(f"Unknown request control: {sorted(unknown)[0]}")
    return ModelRequestControls.model_validate(
        {
            field: value
            for field, value in values.items()
            if value is not None and _valid_control_value(field, value)
        }
    )


def copy_request_controls_payload(value: object) -> dict[str, Any] | None:
    """Validate a recorded ``request_controls`` payload, or return None."""

    if type(value) is not dict:
        return None
    try:
        return ModelRequestControls.model_validate(value).payload()
    except (TypeError, ValueError):
        return None


def copy_echoed_request_controls(value: object) -> dict[str, Any] | None:
    """Validate echoed controls: known fields only, each a valid value or null.

    A key that is present with ``None`` means the provider echoed the control as
    null, which differs from not echoing it at all. Returns None when nothing
    usable remains.
    """

    if not isinstance(value, Mapping):
        return None
    echoed = cast("Mapping[str, Any]", value)
    copied: dict[str, Any] = {}
    for field in REQUEST_CONTROL_FIELDS:
        if field not in echoed:
            continue
        item = echoed[field]
        if item is None:
            copied[field] = None
        elif _valid_control_value(field, item):
            copied[field] = _normalized_control_value(field, item)
    return copied or None


def request_controls_mismatch(
    sent: Mapping[str, Any] | None,
    echoed: Mapping[str, Any] | None,
) -> list[str]:
    """Name the controls Cayu sent whose echoed value differs, in field order.

    Only controls that were sent and also echoed are compared. An echoed value for
    a control Cayu did not send is the provider's default, not a mismatch. An
    echoed null for a control that was sent (a dropped output cap, say) is one.
    """

    if not sent or not echoed:
        return []
    mismatched: list[str] = []
    for field in REQUEST_CONTROL_FIELDS:
        if field not in echoed or sent.get(field) is None:
            continue
        if not _same_control_value(sent[field], echoed[field]):
            mismatched.append(field)
    return mismatched


def apply_echoed_request_controls(
    payload: dict[str, Any],
    *,
    request_controls: Mapping[str, Any] | None,
) -> None:
    """Validate provider-echoed controls on a completion payload and flag mismatches.

    The echoed record is provider evidence: an unusable one is dropped. The
    mismatch flag is runtime-owned, so a provider-supplied one is always removed.
    It is set only when the sent controls are known; without them the reader
    compares against the call's ``model.started`` record instead.
    """

    payload.pop(REQUEST_CONTROLS_MISMATCH_KEY, None)
    echoed = copy_echoed_request_controls(payload.pop(ECHOED_REQUEST_CONTROLS_KEY, None))
    if echoed is None:
        return
    payload[ECHOED_REQUEST_CONTROLS_KEY] = echoed
    mismatched = request_controls_mismatch(
        copy_request_controls_payload(None if request_controls is None else dict(request_controls)),
        echoed,
    )
    if mismatched:
        payload[REQUEST_CONTROLS_MISMATCH_KEY] = mismatched


def openai_responses_echoed_request_controls(response: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read the controls an OpenAI Responses object echoes back.

    Responses report ``reasoning.effort``, ``max_output_tokens``, ``temperature``
    and ``top_p`` on the response. Only fields present on the response are kept,
    so a compatible endpoint that omits them reports nothing rather than nulls.
    """

    echoed: dict[str, Any] = {}
    if "reasoning" in response:
        reasoning = response.get("reasoning")
        if reasoning is None:
            echoed["reasoning_effort"] = None
        elif isinstance(reasoning, Mapping) and "effort" in reasoning:
            echoed["reasoning_effort"] = reasoning.get("effort")
    for field in ("max_output_tokens", "temperature", "top_p"):
        if field in response:
            echoed[field] = response.get(field)
    return copy_echoed_request_controls(echoed)


def _valid_control_value(field: str, value: object) -> bool:
    if field == "reasoning_effort":
        if (
            type(value) is not str
            or not value.strip()
            or len(value) > _MAX_REASONING_EFFORT_CHARS
            # Echoed values come from the provider and are printed by the CLI, so a
            # control character (a terminal escape, say) is never a usable effort.
            or not value.isprintable()
        ):
            return False
        try:
            require_durable_clean_nonblank(value, field)
        except (TypeError, ValueError):
            return False
        return True
    if field in _INTEGER_FIELDS:
        return type(value) is int and 0 <= value <= MAX_DURABLE_JSON_INTEGER
    if field in _NUMBER_FIELDS:
        if type(value) is int:
            # An int too large for a float would raise in float().
            if abs(value) > MAX_DURABLE_JSON_INTEGER:
                return False
        elif type(value) is not float:
            return False
        number = float(value)
        # The record is stored as a float, which must cross every durable JSON
        # boundary: an integral float outside int64 (1e20, or 2**63 - 1 after
        # rounding) is rejected there and would otherwise fail the model call.
        return math.isfinite(number) and _is_durable_json(number, field)
    return False


def _is_durable_json(value: object, field: str) -> bool:
    try:
        copy_durable_json_value(value, field)
    except (TypeError, ValueError):
        return False
    return True


def _normalized_control_value(field: str, value: object) -> Any:
    if field in _NUMBER_FIELDS:
        return float(cast("int | float", value))
    return value


def _same_control_value(sent: object, echoed: object) -> bool:
    if echoed is None:
        return False
    if type(sent) in {int, float} and type(echoed) in {int, float}:
        return math.isclose(
            float(cast("int | float", sent)),
            float(cast("int | float", echoed)),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    return sent == echoed


__all__ = [
    "ECHOED_REQUEST_CONTROLS_KEY",
    "REQUEST_CONTROLS_MISMATCH_KEY",
    "REQUEST_CONTROLS_PAYLOAD_KEY",
    "REQUEST_CONTROL_FIELDS",
    "ModelRequestControls",
    "apply_echoed_request_controls",
    "copy_echoed_request_controls",
    "copy_request_controls_payload",
    "openai_responses_echoed_request_controls",
    "request_controls_from_values",
    "request_controls_mismatch",
]
