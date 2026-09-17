"""Finite-vocabulary, untrusted rejection evidence; never a retry input.

No response text, arbitrary parameter, numeric constraint, or request identifier
crosses this boundary. Explanations are authored here, not redacted upstream prose.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypeVar, cast

import httpx

_PARAMETERS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "messages",
        "input",
        "model",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "stop",
        "seed",
        "reasoning_effort",
        "previous_response_id",
        "metadata",
    }
)
_REASONS = {
    "unsupported_parameter": "Remove this parameter; the endpoint or model does not support it.",
    "missing_required_parameter": "Supply this required parameter.",
    "invalid_value": "Correct this parameter's value for the endpoint or model.",
    "positive_integer_required": "Set this parameter to a positive integer.",
}
_UNAVAILABLE = frozenset(
    {
        "absent_body",
        "body_unavailable",
        "body_too_large",
        "non_json_body",
        "malformed_body",
        "absent_details",
        "unrecognized_details",
        "unsafe_parameter",
        "credential_overlap",
    }
)
PREFIX = "provider_rejection_"
REJECTION_KEYS = tuple(
    PREFIX + key
    for key in (
        "reason",
        "parameter",
        "explanation",
        "unavailable_reason",
        "request_id_state",
    )
)


def rejection_fields(value: object, *, credential_values: tuple[str, ...] = ()) -> dict[str, str]:
    """Revalidate even caller-mutated evidence, returning fresh constant-only fields."""
    if type(value) is not dict or not any(key in value for key in REJECTION_KEYS):
        return {}
    value = cast("dict[str, Any]", value)
    reason = value.get(PREFIX + "reason")
    parameter = value.get(PREFIX + "parameter")
    unavailable = value.get(PREFIX + "unavailable_reason")
    state = value.get(PREFIX + "request_id_state")
    fields = {
        PREFIX + "request_id_state": state
        if type(state) is str and state in {"absent", "omitted_untrusted", "unavailable"}
        else "unavailable"
    }
    if (
        type(reason) is str
        and reason in _REASONS
        and type(parameter) is str
        and parameter in _PARAMETERS
    ):
        fields.update(
            {
                PREFIX + "reason": reason,
                PREFIX + "parameter": parameter,
                PREFIX + "explanation": _REASONS[reason],
            }
        )
    elif type(unavailable) is str and unavailable in _UNAVAILABLE:
        fields[PREFIX + "unavailable_reason"] = unavailable
    else:
        fields[PREFIX + "unavailable_reason"] = "unrecognized_details"
    if any(secret and secret in text for secret in credential_values for text in fields.values()):
        return {PREFIX + "unavailable_reason": "credential_overlap"}
    return fields


def unavailable(reason: str) -> dict[str, str]:
    return rejection_fields({PREFIX + "unavailable_reason": reason})


def project_rejection_response(response: httpx.Response) -> dict[str, str]:
    try:
        body = response.content
        if not body:
            return unavailable("absent_body")
        if len(body) > 64 * 1024:
            return unavailable("body_too_large")
        if "application/json" not in response.headers.get("content-type", ""):
            return unavailable("non_json_body")
        decoded = json.loads(body)
    except httpx.ResponseNotRead:
        return unavailable("body_unavailable")
    except (ValueError, RecursionError):
        return unavailable("malformed_body")
    if type(decoded) is not dict:
        return unavailable("malformed_body")
    error = decoded.get("error", decoded)
    if type(error) is not dict:
        return unavailable("malformed_body")
    fields = project_rejection_error(error)
    fields[PREFIX + "request_id_state"] = (
        "omitted_untrusted"
        if any(name in response.headers for name in ("x-request-id", "request-id"))
        or "request_id" in decoded
        or "request_id" in error
        else "absent"
    )
    return fields


def project_rejection_error(error: Mapping[str, Any]) -> dict[str, str]:
    """Recognize flat proxy/OpenAI codes and exact provider message templates."""
    code, parameter, message = error.get("code"), error.get("param"), error.get("message")
    if type(code) is str and code in _REASONS:
        if type(parameter) is str and parameter in _PARAMETERS:
            return rejection_fields({PREFIX + "reason": code, PREFIX + "parameter": parameter})
        return unavailable("unsafe_parameter")
    # Full equality prevents partial matches from accepting echoed values or
    # instructions. Unknown nested bodies and arbitrary constraints stay private.
    if type(message) is str and len(message) <= 512:
        for param in sorted(_PARAMETERS):
            for text, reason in (
                (f"Unsupported parameter: '{param}'.", "unsupported_parameter"),
                (f"Missing required parameter: '{param}'.", "missing_required_parameter"),
                (f"{param}: Input should be a valid integer", "invalid_value"),
                (f"{param}: must be a positive integer", "positive_integer_required"),
            ):
                if message == text:
                    return rejection_fields(
                        {PREFIX + "reason": reason, PREFIX + "parameter": param}
                    )
    return unavailable(
        "absent_details"
        if all(v is None for v in (code, parameter, message))
        else "unrecognized_details"
    )


_Error = TypeVar("_Error", bound=Exception)


def retain_rejection_diagnostic(error: _Error, diagnostic: object) -> _Error:
    """Carry revalidated evidence when an adapter rebuilds its exception type."""
    cast("Any", error).rejection_diagnostic = rejection_fields(diagnostic)
    return error
