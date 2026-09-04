from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any

from cayu._validation import (
    DurableValueError,
    FrozenJsonDict,
    FrozenJsonList,
    compact_json_utf8_size,
    copy_durable_json_value,
    json_utf8_size_within_limit,
    require_durable_text,
    safe_durable_value_error_details,
)
from cayu.core.events import Event, EventType, event_payload_authority_is_runtime_generated
from cayu.core.tools import ToolEffect, ToolResult
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _shared_artifact_results as shared_artifact_results
from cayu.runtime import _web_access_results as web_access_results
from cayu.runtime._diagnostics import (
    MAX_DIAGNOSTIC_UTF8_BYTES,
    ExceptionDiagnostic,
    bound_diagnostic_text,
    exception_diagnostic,
)
from cayu.runtime.tool_result_projection import (
    _TOOL_RESULT_PROJECTION_AUTHORITY_FIELD,
    TOOL_RESULT_ARTIFACT_TYPE,
    redact_tool_result_projection_content,
)
from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime.tool_result_projection import ToolResultProjection

_MAX_DIAGNOSTIC_UTF8_BYTES = MAX_DIAGNOSTIC_UTF8_BYTES
_MAX_PORTABLE_EVIDENCE_UTF8_BYTES = 12 * 1024
_MAX_PORTABLE_EVIDENCE_DEPTH = 16
_MAX_PORTABLE_EVIDENCE_NODES = 256
_MAX_PRIORITY_KEY_SCAN = 4 * 1024
_TRUNCATION_MARKER = "\u2026[truncated]"
_OMITTED = object()
_NO_EVIDENCE = object()
_PRIORITY_EVIDENCE_KEYS = (
    "receipt_id",
    "receiptId",
    "receipt",
    "confirmation_id",
    "confirmationId",
    "transaction_id",
    "transactionId",
    "transaction",
    "idempotency_key",
    "idempotencyKey",
    "request_id",
    "requestId",
    "operation_id",
    "operationId",
    "tool_effect",
    "effect",
    "outcome",
    "status",
)
_TERMINAL_OUTCOMES = frozenset(
    {
        "invalid_tool_output",
        "tool_execution_error",
        "tool_execution_timeout",
    }
)
_RUNTIME_TERMINAL_CONTROL_FIELDS = frozenset(
    {
        "terminal_outcome",
        "tool_effect",
        "outcome_unknown",
        "manual_reconciliation_required",
        "durable_value_error_code",
        "durable_value_error_path",
        "isolated_tool_failure_code",
        "isolated_tool_cleanup_failure_code",
        "tool_execution_boundary",
        "tool_timeout_strength",
    }
)
_RUNTIME_TOOL_EVENT_LINKAGE_FIELDS = frozenset(
    {
        "approval_id",
        "idempotency_key",
        "input_id",
        "model_attempt_id",
        "model_step_id",
        "tool_call_id",
        "tool_name",
        "tool_round_id",
    }
)
_RUNTIME_TOOL_EVENT_FIXED_FIELDS = frozenset(
    {
        "workspace_mutation_capture_detail_code",
        "workspace_mutation_capture_status",
    }
)
_EXECUTION_PROFILE_FINGERPRINT_FIELD = "execution_profile_fingerprint"
_TOOL_EFFECT_VALUES = frozenset(effect.value for effect in ToolEffect)


@dataclass(frozen=True, slots=True)
class _PortableEvidence:
    value: Any
    included: bool
    incomplete: bool


@dataclass(slots=True)
class _SalvageState:
    remaining_nodes: int = _MAX_PORTABLE_EVIDENCE_NODES
    incomplete: bool = False


def normalize_tool_result(result: ToolResult) -> ToolResult:
    if result.is_error and not result.content.strip():
        return result.model_copy(update={"content": "Tool returned an error without details."})
    return result


def validate_tool_result(result: ToolResult) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Tool results must be ToolResult instances.")
    return ToolResult(
        content=result.content,
        structured=copy_durable_json_value(result.structured, "structured"),
        artifacts=copy_durable_json_value(result.artifacts, "artifacts"),
        is_error=result.is_error,
    )


def strip_untrusted_runtime_tool_result_control_authority(
    structured: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Remove complete runtime-control projections from application output.

    Runtime-owned controls travel beside ``ToolResult`` in a private execution
    envelope.  A tool result with the same public dictionary shape therefore
    remains application data and cannot acquire that private authority.
    Invalid lookalikes are retained as ordinary data for normal redaction.
    """

    if structured is None:
        return None
    copied = copy_durable_json_value(structured, "structured")
    if type(copied) is not dict:
        raise AssertionError("Tool-result structure copied as a non-object.")
    try:
        controls = runtime_terminal_controls(copied)
        controls.update(runtime_tool_execution_boundary_controls(copied))
    except (TypeError, ValueError):
        return copied
    for key in controls:
        copied.pop(key, None)
    return copied


def restore_runtime_tool_result_control_authority(
    structured: Mapping[str, Any] | None,
    authority: Mapping[str, Any],
    *,
    include_terminal_controls: bool = True,
) -> dict[str, Any] | None:
    """Strip caller-shaped controls and restore only runtime-owned authority."""

    if type(include_terminal_controls) is not bool:
        raise TypeError("include_terminal_controls must be a bool.")
    sanitized = strip_untrusted_runtime_tool_result_control_authority(structured)
    copied_authority = copy_durable_json_value(authority, "tool_result_control_authority")
    if type(copied_authority) is not dict:
        raise AssertionError("Tool-result control authority copied as a non-object.")
    controls = runtime_terminal_controls(copied_authority) if include_terminal_controls else {}
    controls.update(runtime_tool_execution_boundary_controls(copied_authority))
    if sanitized is not None and controls:
        # A malformed or partial caller-shaped tuple is ordinary data at the
        # untrusted boundary, but it cannot remain beside runtime authority.
        # Remove the complete reserved family before restoring the exact
        # runtime-owned tuple so one conflicting sibling cannot poison its
        # validation or secret-safe projection downstream.
        for key in _RUNTIME_TERMINAL_CONTROL_FIELDS:
            sanitized.pop(key, None)
    if not controls:
        return sanitized
    restored = {} if sanitized is None else sanitized
    restored.update(controls)
    return restored


def redact_tool_result(result: ToolResult, redactor: SecretRedactor) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Tool results must be ToolResult instances.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if not redactor.has_values:
        return result
    return ToolResult(
        content=redactor.redact_text(result.content),
        structured=redactor.redact_json(result.structured),
        artifacts=redactor.redact_json(result.artifacts),
        is_error=result.is_error,
    )


def strip_runtime_tool_result_projection_authority(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove runtime-only ownership evidence from application-authored artifacts."""

    copied = copy_durable_json_value(artifacts, "artifacts")
    if type(copied) is not list:
        raise AssertionError("Tool-result artifacts copied as a non-list.")
    for artifact in copied:
        if type(artifact) is dict and artifact.get("type") == TOOL_RESULT_ARTIFACT_TYPE:
            artifact.pop(_TOOL_RESULT_PROJECTION_AUTHORITY_FIELD, None)
    return copied


def redact_tool_result_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
    include_terminal_controls: bool = True,
) -> tuple[Event, ToolResult]:
    """Redact a terminal event payload and keep its result field synchronized."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if type(include_terminal_controls) is not bool:
        raise TypeError("include_terminal_controls must be a bool.")
    event_controls = runtime_terminal_controls(event.payload)
    boundary_controls = runtime_tool_execution_boundary_controls(event.payload)
    event_controls.update(boundary_controls)
    result_controls = dict(event_controls) if include_terminal_controls else dict(boundary_controls)
    # Result bodies can originate from tools, hooks, or operator-authored
    # recovery. Their public dictionary shape is never execution authority.
    # Strip every complete lookalike before redaction, including when the
    # current registry is empty, then restore only controls owned by the event.
    result_to_redact = result.model_copy(
        update={
            "structured": strip_untrusted_runtime_tool_result_control_authority(result.structured)
        }
    )
    redacted_result = (
        _redact_terminal_result(result_to_redact, redactor)
        if event_controls
        else redact_tool_result(result_to_redact, redactor)
    )
    if result_to_redact.structured is None and redacted_result.structured == {}:
        redacted_result = redacted_result.model_copy(update={"structured": None})
    linkage_fields = _runtime_tool_event_linkage_fields(event.payload)
    profile_attribution = _runtime_execution_profile_attribution(event)
    exposure_attribution = _runtime_tool_exposure_attribution(event)
    payload_to_redact = {
        key: value
        for key, value in event.payload.items()
        if key != "result"
        and key not in linkage_fields
        and key not in profile_attribution
        and key not in exposure_attribution
        and not (event_controls and key in _RUNTIME_TERMINAL_CONTROL_FIELDS)
    }
    payload = redactor.redact_json(payload_to_redact)
    if type(payload) is not dict:
        raise AssertionError("Event payload redaction returned non-object payload.")
    payload.update(linkage_fields)
    payload.update(profile_attribution)
    payload.update(exposure_attribution)
    if result_controls:
        structured = dict(redacted_result.structured or {})
        structured.update(result_controls)
        redacted_result = ToolResult(
            content=redacted_result.content,
            structured=structured,
            artifacts=redacted_result.artifacts,
            is_error=redacted_result.is_error,
        )
    if event_controls:
        payload.update(event_controls)
    payload["result"] = redacted_result.model_dump()
    restored_event, restored_result = web_access_results.restore_attested_tool_result(
        event.model_copy(update={"payload": payload}),
        original=result,
        redacted=redacted_result,
    )
    return shared_artifact_results.restore_attested_tool_result(
        restored_event,
        original=result,
        redacted=restored_result,
    )


def _runtime_execution_profile_attribution(event: Event) -> dict[str, str]:
    """Preserve only the exact profile digest attested by the runtime producer."""

    value = event.payload.get(_EXECUTION_PROFILE_FINGERPRINT_FIELD)
    if type(value) is not str or not event_payload_authority_is_runtime_generated(
        event,
        field_name=_EXECUTION_PROFILE_FINGERPRINT_FIELD,
        value=value,
    ):
        return {}
    return {_EXECUTION_PROFILE_FINGERPRINT_FIELD: value}


def _runtime_tool_exposure_attribution(event: Event) -> dict[str, str]:
    """Preserve one exact runtime-attested unexposed-call classification."""

    if (
        event.type is not EventType.TOOL_CALL_BLOCKED
        or event.payload.get("blocked_by") != "tool_exposure"
        or event.payload.get("reason") != "not_exposed_in_request"
    ):
        return {}
    profile_id = event.payload.get("profile_id")
    fingerprint = event.payload.get("exposure_fingerprint")
    if (
        type(profile_id) is not str
        or not profile_id.strip()
        or len(profile_id) > 256
        or type(fingerprint) is not str
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or not event_payload_authority_is_runtime_generated(
            event,
            field_name="profile_id",
            value=profile_id,
        )
        or not event_payload_authority_is_runtime_generated(
            event,
            field_name="exposure_fingerprint",
            value=fingerprint,
        )
    ):
        return {}
    return {
        "blocked_by": "tool_exposure",
        "reason": "not_exposed_in_request",
        "profile_id": profile_id,
        "exposure_fingerprint": fingerprint,
    }


def _runtime_tool_event_linkage_fields(payload: dict[str, Any]) -> dict[str, str]:
    """Validate runtime-owned identities before exempting them from redaction."""

    linkage: dict[str, str] = {}
    for key in _RUNTIME_TOOL_EVENT_LINKAGE_FIELDS | _RUNTIME_TOOL_EVENT_FIXED_FIELDS:
        if key not in payload:
            continue
        value = require_durable_text(payload[key], key)
        if not value.strip():
            raise ValueError(f"Runtime tool event `{key}` cannot be blank.")
        linkage[key] = value
    return linkage


def runtime_terminal_controls(payload: dict[str, Any]) -> dict[str, Any]:
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


def runtime_tool_execution_boundary_controls(payload: dict[str, Any]) -> dict[str, str]:
    """Validate execution-boundary controls from a runtime-owned event payload."""

    fields = {
        key: payload[key]
        for key in (
            "isolated_tool_failure_code",
            "isolated_tool_cleanup_failure_code",
            "tool_execution_boundary",
            "tool_timeout_strength",
        )
        if key in payload
    }
    if not fields:
        return {}
    if fields.get("tool_execution_boundary") == "in_process":
        if fields != {
            "tool_execution_boundary": "in_process",
            "tool_timeout_strength": "cooperative_in_process",
        }:
            raise ValueError("Runtime in-process tool controls are inconsistent.")
        return {key: str(value) for key, value in fields.items()}
    if (
        fields.get("tool_execution_boundary") != "posix_process"
        or fields.get("tool_timeout_strength") != "hard_process_deadline"
        or "isolated_tool_failure_code" not in fields
    ):
        raise ValueError("Runtime isolated-tool controls are incomplete.")
    for key in ("isolated_tool_failure_code", "isolated_tool_cleanup_failure_code"):
        if key not in fields:
            continue
        value = fields[key]
        if (
            type(value) is not str
            or not value
            or len(value) > 128
            or value[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)
        ):
            raise ValueError(f"Invalid runtime {key} control.")
    return {key: str(value) for key, value in fields.items()}


def runtime_tool_event_boundary_controls(
    payload: dict[str, Any],
    *,
    include_terminal_controls: bool = True,
    redactor: SecretRedactor | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Return validated controls and strict projection references for publication."""

    if type(payload) is not dict:
        raise TypeError("Runtime tool event payload must be a dict.")
    if type(include_terminal_controls) is not bool:
        raise TypeError("include_terminal_controls must be a bool.")
    if redactor is not None and not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor or None.")
    controls: dict[str, Any] = _runtime_tool_event_linkage_fields(payload)
    projection_references: dict[int, dict[str, Any]] = {}
    if include_terminal_controls:
        controls.update(runtime_terminal_controls(payload))
        controls.update(runtime_tool_execution_boundary_controls(payload))
    evidence = _runtime_tool_result_projection(payload)
    if evidence is not None:
        projection, projection_references = evidence
        controls["tool_result_projection"] = _projection_record_for_event_boundary(
            projection,
            redactor=redactor,
        )
    return (
        copy_durable_json_value(controls, "runtime_tool_event_controls"),
        {
            index: copy_durable_json_value(
                reference,
                "runtime_tool_event_projection_reference",
            )
            for index, reference in projection_references.items()
        },
    )


def _projection_record_for_event_boundary(
    projection: ToolResultProjection,
    *,
    redactor: SecretRedactor | None,
) -> dict[str, Any]:
    """Omit secret-bearing optional extension metadata before control restoration."""

    record = projection.record.model_dump(mode="json", exclude_none=True)
    if redactor is None:
        return record
    settlement = record.get("artifact_write_settlement")
    if type(settlement) is not dict:
        return record
    settlement = dict(settlement)
    for field in ("backend_locator", "backend_version"):
        value = settlement.get(field)
        if type(value) is str and redactor.redact_text(value) != value:
            settlement.pop(field, None)
    record["artifact_write_settlement"] = settlement
    return record


def project_runtime_tool_result_for_boundary(
    *,
    original_content: object,
    redacted_artifacts: object,
    references: dict[int, dict[str, Any]],
    redactor: SecretRedactor,
) -> tuple[str, list[object]]:
    """Re-redact projected text and restore exactly one strict artifact reference."""

    if type(original_content) is not str or type(redacted_artifacts) is not list:
        raise AssertionError("Validated projection lost its content or artifacts.")
    if len(references) != 1:
        raise AssertionError("Validated projection must own exactly one artifact reference.")
    artifact_index, reference = next(iter(references.items()))
    if artifact_index >= len(redacted_artifacts):
        raise AssertionError("Validated projection artifact position changed during redaction.")
    artifact_id = reference.get("artifact_id")
    if type(artifact_id) is not str:
        raise AssertionError("Validated projection reference lost its artifact identity.")
    readback_max_bytes = reference.get("readback_max_bytes")
    if type(readback_max_bytes) is not int:
        raise AssertionError("Validated projection reference lost its readback bound.")
    content = redact_tool_result_projection_content(
        original_content,
        artifact_id=artifact_id,
        readback_max_bytes=readback_max_bytes,
        redact_text=redactor.redact_text,
    )
    artifacts = list(redacted_artifacts)
    artifacts[artifact_index] = reference
    return content, artifacts


def _runtime_tool_result_projection(
    payload: dict[str, Any],
) -> tuple[ToolResultProjection, dict[int, dict[str, Any]]] | None:
    record_payload = payload.get("tool_result_projection")
    if record_payload is None:
        return None
    result_payload = payload.get("result")
    if type(record_payload) is not dict or type(result_payload) is not dict:
        raise TypeError("Projected tool-result events require object result and record fields.")

    from cayu.runtime.tool_result_projection import (
        ToolResultProjection,
        ToolResultProjectionRecord,
        ToolResultProjectionStatus,
        _validate_externalized_projection_reference,
    )

    try:
        projection = ToolResultProjection(
            result=ToolResult(**copy_durable_json_value(result_payload, "projected_tool_result")),
            record=ToolResultProjectionRecord.model_validate(
                copy_durable_json_value(record_payload, "tool_result_projection")
            ),
        )
    except (TypeError, ValueError):
        return None
    references: dict[int, dict[str, Any]] = {}
    if projection.record.status is ToolResultProjectionStatus.EXTERNALIZED:
        try:
            reference = _validate_externalized_projection_reference(projection)
        except (TypeError, ValueError):
            return None
        references = {
            index: reference
            for index, artifact in enumerate(projection.result.artifacts)
            if artifact == reference
        }
    return projection, references


def _tool_result_without_terminal_controls(
    result: ToolResult,
    *,
    terminal_controls: dict[str, Any],
) -> ToolResult:
    if not terminal_controls or result.structured is None:
        return result
    structured = {
        key: value
        for key, value in result.structured.items()
        if key not in _RUNTIME_TERMINAL_CONTROL_FIELDS
    }
    return ToolResult(
        content=result.content,
        structured=structured,
        artifacts=result.artifacts,
        is_error=result.is_error,
    )


def redact_tool_call_outcomes(
    outcomes: list[runtime_records.ToolCallOutcome],
    redactor: SecretRedactor,
) -> list[runtime_records.ToolCallOutcome]:
    """Redact tool outcomes while preserving their call identity and order."""
    if not redactor.has_values:
        return outcomes
    return [
        runtime_records.ToolCallOutcome(
            call=outcome.call,
            result=redact_tool_result(outcome.result, redactor),
        )
        for outcome in outcomes
    ]


def redact_runtime_owned_tool_call_outcomes(
    outcomes: list[runtime_records.ToolCallOutcome],
    redactor: SecretRedactor,
) -> list[runtime_records.ToolCallOutcome]:
    """Redact trusted runtime outcomes without rewriting validated controls.

    This entrance is deliberately separate from ``redact_tool_call_outcomes``:
    a value merely shaped like a runtime control is not authority to bypass
    workload-secret redaction.
    """

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    redacted_outcomes: list[runtime_records.ToolCallOutcome] = []
    for outcome in outcomes:
        structured = dict(outcome.result.structured or {})
        controls: dict[str, Any] = runtime_terminal_controls(structured)
        controls.update(runtime_tool_execution_boundary_controls(structured))
        if not controls:
            redacted_result = redact_tool_result(outcome.result, redactor)
        else:
            redacted_result = _redact_terminal_result(
                _tool_result_without_terminal_controls(
                    outcome.result,
                    terminal_controls=controls,
                ),
                redactor,
            )
            redacted_structured = dict(redacted_result.structured or {})
            redacted_structured.update(controls)
            redacted_result = ToolResult(
                content=redacted_result.content,
                structured=redacted_structured,
                artifacts=redacted_result.artifacts,
                is_error=redacted_result.is_error,
            )
        redacted_outcomes.append(
            runtime_records.ToolCallOutcome(
                call=outcome.call,
                result=redacted_result,
            )
        )
    return redacted_outcomes


def tool_result_from_payload(payload: dict[str, Any]) -> ToolResult:
    return normalize_tool_result(validate_tool_result(ToolResult(**deepcopy(payload))))


def exception_message(exc: Exception) -> str:
    """Backward-compatible safe message used by ordinary tool failures."""

    return exception_diagnostic(exc, empty_message="tool execution failed").message


def terminal_failure_result(
    *,
    terminal_outcome: str,
    effect: ToolEffect,
    message: str,
    redactor: SecretRedactor,
    raw_evidence: Any = _NO_EVIDENCE,
    diagnostic: ExceptionDiagnostic | None = None,
) -> tuple[ToolResult, dict[str, Any]]:
    """Build the runtime-owned terminal result for a post-dispatch failure.

    The control fields are deliberately separate from the tool-supplied value.
    Valid siblings from a malformed return value are retained as bounded evidence;
    invalid leaves and custom container implementations are never invoked.
    """

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    terminal_outcome = require_durable_text(terminal_outcome, "terminal_outcome")
    message = redactor.redact_text_bounded(
        require_durable_text(message, "message"),
        max_bytes=_MAX_DIAGNOSTIC_UTF8_BYTES,
    )
    outcome_unknown = effect is not ToolEffect.NONE
    manual_reconciliation_required = effect is ToolEffect.EXTERNAL
    controls: dict[str, Any] = {
        "terminal_outcome": terminal_outcome,
        "tool_effect": effect.value,
        "outcome_unknown": outcome_unknown,
        "manual_reconciliation_required": manual_reconciliation_required,
    }
    structured = dict(controls)
    if diagnostic is not None and diagnostic.durable_value_error_code is not None:
        controls["durable_value_error_code"] = diagnostic.durable_value_error_code
        controls["durable_value_error_path"] = diagnostic.durable_value_error_path
        structured["durable_value_error_code"] = diagnostic.durable_value_error_code
        structured["durable_value_error_path"] = diagnostic.durable_value_error_path
    if raw_evidence is not _NO_EVIDENCE:
        evidence = portable_result_evidence(raw_evidence, redactor=redactor)
        if evidence.included:
            structured["portable_result_evidence"] = evidence.value
        structured["portable_result_evidence_incomplete"] = evidence.incomplete
    result = ToolResult(content=message, structured=structured, is_error=True)
    return result, copy_durable_json_value(controls, "terminal_tool_controls")


def raw_tool_result_evidence(value: Any) -> Any:
    """Expose exact ToolResult fields without invoking forged model serializers."""

    if type(value) is not ToolResult:
        return {"returned_value": value}
    try:
        fields = object.__getattribute__(value, "__dict__")
    except BaseException:
        return {}
    if type(fields) is not dict:
        return {}
    recognized = ("structured", "content", "artifacts", "is_error")
    evidence: dict[str, Any] = {}
    for index, (name, field_value) in enumerate(fields.items()):
        if index >= _MAX_PRIORITY_KEY_SCAN or len(evidence) == len(recognized):
            break
        if type(name) is str and name in recognized:
            evidence[name] = field_value
    return evidence


def portable_result_evidence(
    value: Any,
    *,
    redactor: SecretRedactor | None = None,
) -> _PortableEvidence:
    """Project portable siblings from an untrusted result under fixed limits."""

    resolved_redactor = redactor or SecretRedactor()
    state = _SalvageState()
    try:
        salvaged = _salvage_portable_value(
            value,
            max_bytes=_MAX_PORTABLE_EVIDENCE_UTF8_BYTES,
            depth=0,
            active_container_ids=set(),
            state=state,
            redactor=resolved_redactor,
        )
    except BaseException:
        return _PortableEvidence(value=None, included=False, incomplete=True)
    if salvaged is _OMITTED:
        return _PortableEvidence(value=None, included=False, incomplete=True)
    if not json_utf8_size_within_limit(
        salvaged,
        _MAX_PORTABLE_EVIDENCE_UTF8_BYTES,
        ensure_ascii=False,
    ):
        # The recursive accounting is exact for ordinary JSON values. Keep this
        # defensive fail-closed guard at the final durable boundary.
        return _PortableEvidence(value=None, included=False, incomplete=True)
    return _PortableEvidence(value=salvaged, included=True, incomplete=state.incomplete)


def _salvage_portable_value(
    value: Any,
    *,
    max_bytes: int,
    depth: int,
    active_container_ids: set[int],
    state: _SalvageState,
    redactor: SecretRedactor,
) -> Any:
    if state.remaining_nodes <= 0 or max_bytes <= 0:
        state.incomplete = True
        return _OMITTED
    state.remaining_nodes -= 1

    value_type = type(value)
    if value is None or value_type in {bool, int, float}:
        try:
            copied = copy_durable_json_value(value, "tool_result_evidence")
        except Exception:
            state.incomplete = True
            return _OMITTED
        if compact_json_utf8_size(copied) > max_bytes:
            state.incomplete = True
            return _OMITTED
        return copied
    if value_type is str:
        try:
            require_durable_text(value, "tool_result_evidence")
        except Exception:
            state.incomplete = True
            return _OMITTED
        value = redactor.redact_text(value)
        if json_utf8_size_within_limit(value, max_bytes, ensure_ascii=False):
            return value
        state.incomplete = True
        bounded = _truncate_json_string(value, max_bytes)
        return bounded if bounded is not None else _OMITTED

    if value_type not in {dict, list, FrozenJsonDict, FrozenJsonList}:
        state.incomplete = True
        return _OMITTED
    if depth >= _MAX_PORTABLE_EVIDENCE_DEPTH:
        state.incomplete = True
        return _OMITTED
    container_id = id(value)
    if container_id in active_container_ids:
        state.incomplete = True
        return _OMITTED
    active_container_ids.add(container_id)
    try:
        if value_type in {dict, FrozenJsonDict}:
            return _salvage_portable_object(
                value,
                max_bytes=max_bytes,
                depth=depth,
                active_container_ids=active_container_ids,
                state=state,
                redactor=redactor,
            )
        return _salvage_portable_array(
            value,
            max_bytes=max_bytes,
            depth=depth,
            active_container_ids=active_container_ids,
            state=state,
            redactor=redactor,
        )
    finally:
        active_container_ids.remove(container_id)


def _salvage_portable_object(
    value: dict[Any, Any] | FrozenJsonDict,
    *,
    max_bytes: int,
    depth: int,
    active_container_ids: set[int],
    state: _SalvageState,
    redactor: SecretRedactor,
) -> Any:
    if max_bytes < 2:
        state.incomplete = True
        return _OMITTED
    result: dict[str, Any] = {}
    used_bytes = 2
    for key, child_value in _prioritized_object_items(value, state=state):
        if state.remaining_nodes <= 0:
            state.incomplete = True
            break
        if type(key) is not str:
            state.incomplete = True
            continue
        try:
            require_durable_text(key, "tool_result_evidence key")
        except Exception:
            state.incomplete = True
            continue
        key = redactor.redact_text(key)
        if key in result:
            state.incomplete = True
            continue
        remaining_for_key = max_bytes - used_bytes - 1 - (1 if result else 0)
        if remaining_for_key <= 0:
            state.incomplete = True
            break
        if not json_utf8_size_within_limit(key, remaining_for_key, ensure_ascii=False):
            state.incomplete = True
            continue
        key_bytes = compact_json_utf8_size(key)
        entry_overhead = key_bytes + 1 + (1 if result else 0)
        child_budget = max_bytes - used_bytes - entry_overhead
        if child_budget <= 0:
            state.incomplete = True
            break
        child = _salvage_portable_value(
            child_value,
            max_bytes=child_budget,
            depth=depth + 1,
            active_container_ids=active_container_ids,
            state=state,
            redactor=redactor,
        )
        if child is _OMITTED:
            continue
        child_bytes = compact_json_utf8_size(child)
        result[key] = child
        used_bytes += entry_overhead + child_bytes
    return result


def _prioritized_object_items(
    value: dict[Any, Any] | FrozenJsonDict,
    *,
    state: _SalvageState,
) -> Iterator[tuple[Any, Any]]:
    """Yield reconciliation identifiers first, then ordinary insertion order."""

    if len(value) > _MAX_PRIORITY_KEY_SCAN:
        state.incomplete = True
    # Never use mapping lookup here: even an exact dict can invoke a hostile
    # str-subclass key's equality method on a hash collision. Scan a bounded
    # prefix once and retain only exact-string reconciliation fields.
    scanned = list(islice(value.items(), _MAX_PRIORITY_KEY_SCAN))
    found = {
        key: child_value
        for key, child_value in scanned
        if type(key) is str and key in _PRIORITY_EVIDENCE_KEYS
    }
    for key in _PRIORITY_EVIDENCE_KEYS:
        if key in found:
            yield key, found[key]
    for key, child_value in scanned:
        if type(key) is str and key in found:
            continue
        yield key, child_value


def _salvage_portable_array(
    value: list[Any] | FrozenJsonList,
    *,
    max_bytes: int,
    depth: int,
    active_container_ids: set[int],
    state: _SalvageState,
    redactor: SecretRedactor,
) -> Any:
    if max_bytes < 2:
        state.incomplete = True
        return _OMITTED
    result: list[Any] = []
    used_bytes = 2
    for child_value in value:
        if state.remaining_nodes <= 0:
            state.incomplete = True
            break
        entry_overhead = 1 if result else 0
        child_budget = max_bytes - used_bytes - entry_overhead
        if child_budget <= 0:
            state.incomplete = True
            break
        child = _salvage_portable_value(
            child_value,
            max_bytes=child_budget,
            depth=depth + 1,
            active_container_ids=active_container_ids,
            state=state,
            redactor=redactor,
        )
        if child is _OMITTED:
            continue
        child_bytes = compact_json_utf8_size(child)
        result.append(child)
        used_bytes += entry_overhead + child_bytes
    return result


def _truncate_json_string(value: str, max_bytes: int) -> str | None:
    marker_size = compact_json_utf8_size(_TRUNCATION_MARKER) - 2
    available = max_bytes - 2 - marker_size
    if available < 0:
        return None
    characters: list[str] = []
    used_bytes = 0
    for character in value:
        character_bytes = _json_string_character_size(character)
        if used_bytes + character_bytes > available:
            break
        characters.append(character)
        used_bytes += character_bytes
    return "".join(characters) + _TRUNCATION_MARKER


def _json_string_character_size(character: str) -> int:
    codepoint = ord(character)
    if character in {'"', "\\"} or character in "\b\f\n\r\t":
        return 2
    if codepoint < 0x20:
        return 6
    return len(character.encode("utf-8"))


def _redact_terminal_result(result: ToolResult, redactor: SecretRedactor) -> ToolResult:
    """Redact terminal evidence while preserving runtime-owned wrapper keys and bounds."""

    structured = dict(result.structured or {})
    raw_evidence = structured.pop("portable_result_evidence", _NO_EVIDENCE)
    evidence_was_incomplete = structured.pop("portable_result_evidence_incomplete", False)
    redacted_structured = redactor.redact_json(structured)
    if type(redacted_structured) is not dict:
        raise AssertionError("Terminal tool result redaction returned non-object structured data.")
    if raw_evidence is not _NO_EVIDENCE:
        evidence = portable_result_evidence(raw_evidence, redactor=redactor)
        if evidence.included:
            redacted_structured["portable_result_evidence"] = evidence.value
        redacted_structured["portable_result_evidence_incomplete"] = (
            bool(evidence_was_incomplete) or evidence.incomplete
        )
    elif evidence_was_incomplete:
        redacted_structured["portable_result_evidence_incomplete"] = True
    return ToolResult(
        content=_bound_diagnostic_text(redactor.redact_text(result.content)),
        structured=redacted_structured,
        artifacts=redactor.redact_json(result.artifacts),
        is_error=result.is_error,
    )


def _bound_diagnostic_text(value: str) -> str:
    return bound_diagnostic_text(value)
