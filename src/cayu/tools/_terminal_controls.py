"""Validation and redaction of durable tool-terminal result controls."""

from __future__ import annotations

from typing import Any

from cayu._validation import DurableValueError, safe_durable_value_error_details
from cayu.failure_evidence import FailureEvidence
from cayu.tools.base import ToolEffect
from cayu.vaults.redaction import SecretRedactor

_TERMINAL_OUTCOMES = frozenset(
    {
        "invalid_tool_output",
        "tool_execution_error",
        "tool_execution_timeout",
    }
)


_TOOL_EFFECT_VALUES = frozenset(effect.value for effect in ToolEffect)


def _redacted_failure_evidence(
    evidence: FailureEvidence, *, redactor: SecretRedactor | None
) -> dict[str, Any]:
    """Keep typed diagnostic structure without exempting variable strings.

    Omit secret-bearing type names and identities rather than inventing their
    replacements. Deadline labels use a fixed schema-valid redaction marker;
    a secret-bearing expiry cannot remain a valid timestamp, so omit that
    deadline and downgrade its classification to unknown.
    """
    payload = evidence.model_dump(mode="json")
    if redactor is None or not redactor.has_values:
        return payload

    def contains_secret(value: str) -> bool:
        return redactor.redact_text(value) != value

    for snapshot in (payload, *payload.get("branch_failures", [])):
        names = snapshot["exception_types"]
        snapshot["exception_types"] = [name for name in names if not contains_secret(name)]
        if len(snapshot["exception_types"]) != len(names):
            snapshot["truncated"] = True
        for key in ("session_id", "terminal_event_id"):
            if snapshot[key] is not None and contains_secret(snapshot[key]):
                snapshot[key] = None
        deadline = snapshot["deadline"]
        if deadline is not None:
            if deadline["expires_at"] is not None and contains_secret(deadline["expires_at"]):
                snapshot["deadline"] = None
                snapshot["deadline_phase"] = None
                snapshot["classification"] = "unknown"
                snapshot["truncated"] = True
            else:
                for key in ("source", "scope"):
                    if contains_secret(deadline[key]):
                        deadline[key] = "redacted"
    return payload


def runtime_terminal_controls(
    payload: dict[str, Any], *, redactor: SecretRedactor | None = None
) -> dict[str, Any]:
    """Validate runtime controls before exempting them from secret redaction."""

    if "terminal_outcome" not in payload:
        return {}
    terminal_outcome = payload.get("terminal_outcome")
    tool_effect = payload.get("tool_effect")
    outcome_unknown = payload.get("outcome_unknown")
    manual_reconciliation_required = payload.get("manual_reconciliation_required")
    if type(terminal_outcome) is not str or terminal_outcome not in _TERMINAL_OUTCOMES:
        raise ValueError("Invalid runtime terminal_outcome control.")
    if type(tool_effect) is not str or tool_effect not in _TOOL_EFFECT_VALUES:
        raise ValueError("Invalid runtime tool_effect control.")
    if type(outcome_unknown) is not bool:
        raise TypeError("Runtime outcome_unknown control must be a boolean.")
    if type(manual_reconciliation_required) is not bool:
        raise TypeError("Runtime manual_reconciliation_required control must be a boolean.")
    if outcome_unknown != (tool_effect != ToolEffect.NONE.value):
        raise ValueError("Runtime outcome_unknown control conflicts with tool_effect.")
    if manual_reconciliation_required != (tool_effect == ToolEffect.EXTERNAL.value):
        raise ValueError(
            "Runtime manual_reconciliation_required control conflicts with tool_effect."
        )
    controls: dict[str, Any] = {
        "terminal_outcome": terminal_outcome,
        "tool_effect": tool_effect,
        "outcome_unknown": outcome_unknown,
        "manual_reconciliation_required": manual_reconciliation_required,
    }
    if "failure_evidence" in payload:
        evidence = FailureEvidence.model_validate(payload["failure_evidence"])
        controls["failure_evidence"] = _redacted_failure_evidence(evidence, redactor=redactor)
    code_present = "durable_value_error_code" in payload
    path_present = "durable_value_error_path" in payload
    if code_present is not path_present:
        raise ValueError("Runtime durable-value error controls must be paired.")
    if code_present:
        code = payload["durable_value_error_code"]
        path = payload["durable_value_error_path"]
        if type(code) is not str or type(path) is not str:
            raise TypeError("Runtime durable-value error controls must be strings.")
        try:
            trusted_error = DurableValueError(code, "tool_result", path=path)
        except Exception as exc:
            raise ValueError("Invalid runtime durable-value error controls.") from exc
        safe_code, safe_path = safe_durable_value_error_details(trusted_error)
        if (safe_code, safe_path) != (code, path):
            raise ValueError("Invalid runtime durable-value error controls.")
        controls["durable_value_error_code"] = safe_code
        controls["durable_value_error_path"] = safe_path
    return controls
