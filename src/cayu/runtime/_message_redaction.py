from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from cayu.artifacts.attachments import FILE_ATTACHMENT_TYPE
from cayu.messages import Message
from cayu.runtime import _tool_results as tool_results
from cayu.runtime._tool_result_adapters import prepare_tool_result_message
from cayu.runtime._tool_result_projection import (
    ValidatedToolResultMessage,
    read_runtime_tool_result_message,
    require_secret_free_tool_result_artifact_keys,
)
from cayu.tools import _terminal_controls as tool_terminal_controls
from cayu.tools import _web_access_result_schema as web_access_result_schema
from cayu.tools import _web_access_results as web_access_results
from cayu.vaults import SecretRedactor

_MESSAGE_ROOT_STRUCTURE_KEYS = frozenset({"content", "role"})
_MESSAGE_PART_STRUCTURE_KEYS = {
    "structured_output": frozenset({"type", "name", "output"}),
    "text": frozenset({"text", "type"}),
    "tool_call": frozenset(
        {
            "arguments",
            "arguments_state",
            "model_attempt_id",
            "model_step_id",
            "tool_call_id",
            "tool_name",
            "tool_round_id",
            "type",
        }
    ),
    "tool_result": frozenset(
        {
            "artifacts",
            "content",
            "is_error",
            "model_attempt_id",
            "model_step_id",
            "structured",
            "tool_call_id",
            "tool_name",
            "tool_round_id",
            "type",
        }
    ),
    "provider_state": frozenset({"provider", "state", "type"}),
    "thinking": frozenset({"provider_state", "text", "type"}),
    "file": frozenset({"attachment", "type"}),
    "hosted_tool_call": frozenset(
        {
            "action",
            "call_id",
            "hosted_tool",
            "model",
            "model_attempt_id",
            "model_step_id",
            "provider_name",
            "status",
            "type",
        }
    ),
    "citation": frozenset(
        {
            "citation_type",
            "end_index",
            "model_attempt_id",
            "model_step_id",
            "provenance",
            "start_index",
            "title",
            "type",
            "url",
        }
    ),
    "peer_content": frozenset(
        {
            "append_key_json",
            "executable",
            "occurrence_id",
            "operation_key",
            "projection_id",
            "provenance_sha256",
            "sender_participant_id",
            "sender_participant_incarnation",
            "sender_session_id",
            "sender_session_instance_id",
            "text",
            "type",
        }
    ),
}
_MESSAGE_PART_UNTRUSTED_CONTAINERS = {
    "structured_output": frozenset({"output"}),
    "tool_call": frozenset({"arguments"}),
    "tool_result": frozenset({"artifacts", "structured"}),
    "provider_state": frozenset({"state"}),
    "thinking": frozenset({"provider_state"}),
    "hosted_tool_call": frozenset({"action"}),
    "citation": frozenset({"provenance"}),
}
_FILE_ATTACHMENT_STRUCTURE_KEYS = frozenset(
    {
        "artifact_id",
        "content_type",
        "filename",
        "kind",
        "metadata",
        "size_bytes",
        "type",
    }
)
_TERMINAL_RESULT_STRUCTURE_KEYS = frozenset(
    {
        "durable_value_error_code",
        "durable_value_error_path",
        "durable_value_error_limit",
        "durable_value_error_observed_lower_bound",
        "isolated_tool_cleanup_failure_code",
        "isolated_tool_failure_code",
        "manual_reconciliation_required",
        "outcome_unknown",
        "portable_result_evidence",
        "portable_result_evidence_incomplete",
        "terminal_outcome",
        "tool_effect",
        "tool_execution_boundary",
        "tool_timeout_strength",
    }
)
_TOOL_POLICY_DENIAL_RESULT_STRUCTURE_KEYS = frozenset({"decision", "metadata", "reason"})
_COMMAND_POLICY_DENIAL_RESULT_STRUCTURE_KEYS = frozenset({"decision", "error", "reason"})
_COMMAND_POLICY_DENIAL_IDENTITIES = frozenset(
    {
        ("command_denied", "deny"),
        ("command_approval_required", "require_command_approval"),
    }
)
_APPROVAL_DENIAL_RESULT_STRUCTURE_KEYS = frozenset(
    {
        "approval_id",
        "approval_required",
        "decision",
        "denied_by_approval",
        "denied_tool_call_id",
        "denied_tool_name",
        "metadata",
        "reason",
        "skipped_due_to_approval_denial",
        "tool_call_id",
        "tool_name",
    }
)
_MESSAGE_AUTHORITY_STRING_FIELDS = frozenset(
    {
        "model_attempt_id",
        "model_step_id",
        "call_id",
        "hosted_tool",
        "model",
        "provider",
        "provider_name",
        "tool_call_id",
        "tool_name",
        "tool_round_id",
    }
)
_MESSAGE_RUNTIME_EXECUTION_AUTHORITY_FIELDS = frozenset(
    {
        "model_attempt_id",
        "model_step_id",
        "tool_round_id",
    }
)
_MESSAGE_PRESERVED_STRING_FIELDS = _MESSAGE_AUTHORITY_STRING_FIELDS | {
    "type",
    "arguments_state",
}


@dataclass(slots=True)
class _MessageAdmission:
    """Positive per-part facts established by one ingress or runtime entry point."""

    projections: dict[int, ValidatedToolResultMessage] = field(default_factory=dict)
    controls: dict[int, dict[str, object]] = field(default_factory=dict)
    execution_authority_fields: frozenset[str] = frozenset()


def _message_payload(message: Message, redactor: SecretRedactor) -> dict[str, Any]:
    if type(message) is not Message:
        raise TypeError("Messages must be Message instances.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    payload = message.model_dump(mode="json")
    content = payload.get("content")
    if type(content) is not list:
        raise AssertionError("Message serialization returned non-list content.")
    if any(type(part) is not dict for part in content):
        raise AssertionError("Message content serialized as a non-object.")
    return payload


def redact_untrusted_message_for_boundary(
    message: Message,
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> Message:
    """Redact ingress after stripping runtime projection and control lookalikes."""

    payload = _message_payload(message, redactor)
    _strip_untrusted_runtime_tool_result_projection_authority(payload)
    admission = _MessageAdmission()
    for index, part in enumerate(payload["content"]):
        structured = part.get("structured")
        if type(structured) is dict:
            admission.controls[index] = _recognized_untrusted_terminal_controls(structured)
    return _redact_message_for_boundary(
        payload, admission=admission, redactor=redactor, field_name=field_name
    )


def redact_runtime_message_for_boundary(
    message: Message,
    *,
    redactor: SecretRedactor,
    field_name: str,
    reject_secret_bearing_runtime_projection_authority: bool = False,
) -> Message:
    """Redact runtime transcript state using positively admitted per-part facts."""

    payload = _message_payload(message, redactor)
    admission = _MessageAdmission(
        execution_authority_fields=_MESSAGE_RUNTIME_EXECUTION_AUTHORITY_FIELDS
    )
    for index, part in enumerate(payload["content"]):
        projection = read_runtime_tool_result_message(part)
        if projection is not None:
            admission.projections[index] = projection
        structured = part.get("structured")
        if type(structured) is dict:
            admission.controls[index] = _recognized_runtime_terminal_controls(structured)
    return _redact_message_for_boundary(
        payload,
        admission=admission,
        redactor=redactor,
        field_name=field_name,
        reject_secret_bearing_runtime_projection_authority=(
            reject_secret_bearing_runtime_projection_authority
        ),
    )


def _redact_message_for_boundary(
    payload: dict[str, Any],
    *,
    admission: _MessageAdmission,
    redactor: SecretRedactor,
    field_name: str,
    reject_secret_bearing_runtime_projection_authority: bool = False,
) -> Message:
    """Redact one admitted message without rewriting executable authorities."""

    content = payload["content"]
    _require_secret_free_message_keys(
        payload,
        content=content,
        redactor=redactor,
        field_name=field_name,
        admission=admission,
    )
    redacted_content: list[object] = []
    for index, part in enumerate(content):
        if type(part) is not dict:
            raise AssertionError("Message content serialized as a non-object.")
        part = cast("dict[str, object]", part)
        projection = admission.projections.get(index)
        if reject_secret_bearing_runtime_projection_authority and projection is not None:
            projection.require_secret_free_authority(
                redactor, f"{field_name}.content[{index}].artifacts"
            )
        # Protocol linkage and routing fields are executable authority. They
        # must fail closed instead of being rewritten into a different value.
        for authority_field in _MESSAGE_AUTHORITY_STRING_FIELDS:
            value = part.get(authority_field)
            if authority_field in admission.execution_authority_fields:
                continue
            if type(value) is str and redactor.redact_text(value) != value:
                raise ValueError(
                    f"{field_name}.content[{index}].{authority_field} contains a "
                    "workload secret and cannot be used as execution authority."
                )
        _require_secret_free_file_attachment_authority(
            part,
            redactor=redactor,
            field_name=f"{field_name}.content[{index}]",
        )
        redacted_part = redactor.redact_json_values(
            part,
            preserve_string_fields=_MESSAGE_PRESERVED_STRING_FIELDS,
        )
        if type(redacted_part) is not dict:
            raise AssertionError("Message part redaction returned a non-object.")
        prepare_tool_result_message(redacted_part, projection=projection, redactor=redactor)
        structured = part.get("structured")
        if type(structured) is dict:
            terminal_controls = admission.controls.get(index, {})
            redacted_structured = redacted_part.get("structured")
            if terminal_controls and type(redacted_structured) is dict:
                redacted_structured.update(terminal_controls)
            web_access_results.restore_finite_message_controls(
                structured,
                redacted_structured,
            )
        redacted_content.append(redacted_part)
    return Message.model_validate(
        {
            "role": payload["role"],
            "content": redacted_content,
        }
    )


def _require_secret_free_file_attachment_authority(
    part: dict[str, object],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> None:
    """Reject file handles that redaction would rewrite into different authority."""

    part_type = part.get("type")
    attachments: list[tuple[str, object]] = []
    if part_type == "file":
        attachments.append(("attachment", part.get("attachment")))
    elif part_type == "tool_result":
        artifacts = part.get("artifacts")
        if type(artifacts) is not list:
            raise AssertionError("Tool result artifacts serialized as a non-list.")
        for index, untyped_artifact in enumerate(artifacts):
            if type(untyped_artifact) is not dict:
                continue
            artifact = cast("dict[str, object]", untyped_artifact)
            if artifact.get("type") == FILE_ATTACHMENT_TYPE:
                attachments.append((f"artifacts[{index}]", artifact))

    for path, untyped_attachment in attachments:
        if type(untyped_attachment) is not dict:
            raise AssertionError("File attachment serialized as a non-object.")
        attachment = cast("dict[str, object]", untyped_attachment)
        artifact_id = attachment.get("artifact_id")
        if type(artifact_id) is str and redactor.redact_text(artifact_id) != artifact_id:
            raise ValueError(
                f"{field_name}.{path}.artifact_id contains a workload secret and cannot "
                "be used as execution authority."
            )


def _require_secret_free_message_keys(
    payload: dict[str, object],
    *,
    content: list[object],
    redactor: SecretRedactor,
    field_name: str,
    admission: _MessageAdmission,
) -> None:
    """Validate message keys against the schema that owns each typed container."""

    redactor.require_no_secret_keys(
        {
            "role": payload.get("role"),
            "content": None,
        },
        field_name=field_name,
        preserve_keys=_MESSAGE_ROOT_STRUCTURE_KEYS,
        match_short_substrings=True,
    )
    for index, untyped_part in enumerate(content):
        if type(untyped_part) is not dict:
            raise AssertionError("Message content serialized as a non-object.")
        part = cast("dict[str, object]", untyped_part)
        part_type = part.get("type")
        if type(part_type) is not str or part_type not in _MESSAGE_PART_STRUCTURE_KEYS:
            raise AssertionError("Message content serialized with an unknown part type.")
        part_to_validate = part
        attachment: object | None = None
        artifacts: object | None = None
        structured: object | None = None
        if part_type == "file":
            part_to_validate = dict(part)
            attachment = part_to_validate.get("attachment")
            part_to_validate["attachment"] = None
        elif part_type == "tool_result":
            part_to_validate = dict(part)
            artifacts = part_to_validate.get("artifacts")
            structured = part_to_validate.get("structured")
            part_to_validate["artifacts"] = None
            part_to_validate["structured"] = None
        redactor.require_no_secret_keys(
            part_to_validate,
            field_name=f"{field_name}.content[{index}]",
            preserve_keys=_MESSAGE_PART_STRUCTURE_KEYS[part_type],
            untrusted_container_keys=_MESSAGE_PART_UNTRUSTED_CONTAINERS.get(
                part_type,
                (),
            ),
            match_short_substrings=True,
        )
        if part_type == "file":
            if type(attachment) is not dict:
                raise AssertionError("File message part serialized with a non-object attachment.")
            redactor.require_no_secret_keys(
                attachment,
                field_name=f"{field_name}.content[{index}].attachment",
                preserve_keys=_FILE_ATTACHMENT_STRUCTURE_KEYS,
                untrusted_container_keys={"metadata"},
                match_short_substrings=True,
            )
        elif part_type == "tool_result":
            require_secret_free_tool_result_artifact_keys(
                artifacts,
                source=admission.projections.get(index),
                redactor=redactor,
                field_name=f"{field_name}.content[{index}].artifacts",
            )
            if type(structured) is dict:
                _require_secret_free_tool_result_structure(
                    part,
                    structured=cast("dict[str, object]", structured),
                    redactor=redactor,
                    field_name=f"{field_name}.content[{index}].structured",
                    terminal_controls=admission.controls.get(index, {}),
                )


def _strip_untrusted_runtime_tool_result_projection_authority(
    payload: dict[str, object],
) -> None:
    content = payload.get("content")
    if type(content) is not list:
        return
    for part in content:
        if type(part) is not dict:
            continue
        part = cast("dict[str, object]", part)
        if part.get("type") != "tool_result":
            continue
        structured = part.get("structured")
        if type(structured) is dict:
            # Shape alone cannot turn application output into runtime control
            # authority.  Remove only a complete, positively recognized
            # runtime-control projection; invalid or similarly named ordinary
            # application fields remain untrusted data and are redacted below.
            part["structured"] = tool_results.strip_untrusted_runtime_tool_result_control_authority(
                cast("dict[str, Any]", structured)
            )
        artifacts = part.get("artifacts")
        if type(artifacts) is not list:
            continue
        part["artifacts"] = tool_results.strip_runtime_tool_result_projection_authority(
            cast("list[dict[str, Any]]", artifacts)
        )


def _require_secret_free_tool_result_structure(
    part: dict[str, object],
    *,
    structured: dict[str, object],
    redactor: SecretRedactor,
    field_name: str,
    terminal_controls: dict[str, object],
) -> None:
    """Exempt only positively identified runtime-owned tool-result schemas."""

    if web_access_results.finite_message_controls(structured):
        redactor.require_no_secret_keys(
            structured,
            field_name=field_name,
            preserve_keys=web_access_result_schema.WEB_ACCESS_MESSAGE_STRUCTURE_KEYS,
            match_short_substrings=True,
        )
        return

    if terminal_controls:
        runtime_fields = {
            key: value
            for key, value in structured.items()
            if key in _TERMINAL_RESULT_STRUCTURE_KEYS
        }
        untrusted_fields = {
            key: value
            for key, value in structured.items()
            if key not in _TERMINAL_RESULT_STRUCTURE_KEYS
        }
        redactor.require_no_secret_keys(
            runtime_fields,
            field_name=field_name,
            preserve_keys=_TERMINAL_RESULT_STRUCTURE_KEYS,
            untrusted_container_keys={"portable_result_evidence"},
            match_short_substrings=True,
        )
        redactor.require_no_secret_keys(
            untrusted_fields,
            field_name=field_name,
            match_short_substrings=True,
        )
        return

    is_tool_policy_denial = (
        part.get("is_error") is True
        and set(structured) == _TOOL_POLICY_DENIAL_RESULT_STRUCTURE_KEYS
        and structured.get("decision") == "deny"
        and type(structured.get("reason")) is str
        and structured.get("reason") == part.get("content")
        and type(structured.get("metadata")) is dict
    )
    is_command_policy_denial = (
        part.get("is_error") is True
        and set(structured)
        in (
            _COMMAND_POLICY_DENIAL_RESULT_STRUCTURE_KEYS,
            _COMMAND_POLICY_DENIAL_RESULT_STRUCTURE_KEYS | {"recovery_instruction"},
        )
        and (structured.get("error"), structured.get("decision"))
        in _COMMAND_POLICY_DENIAL_IDENTITIES
        and (structured.get("reason") is None or type(structured.get("reason")) is str)
    )
    approval_required = structured.get("approval_required")
    is_approval_denial = (
        part.get("is_error") is True
        and set(structured) == _APPROVAL_DENIAL_RESULT_STRUCTURE_KEYS
        and structured.get("decision") == "deny"
        and type(approval_required) is bool
        and structured.get("denied_by_approval") is approval_required
        and structured.get("skipped_due_to_approval_denial") is (not approval_required)
        and all(
            type(value) is str and bool(value.strip())
            for value in (
                structured.get(key)
                for key in (
                    "approval_id",
                    "denied_tool_call_id",
                    "denied_tool_name",
                    "tool_call_id",
                    "tool_name",
                )
            )
        )
        and type(structured.get("reason")) is str
        and type(structured.get("metadata")) is dict
    )
    if is_tool_policy_denial or is_command_policy_denial or is_approval_denial:
        structure_keys = (
            _TOOL_POLICY_DENIAL_RESULT_STRUCTURE_KEYS
            if is_tool_policy_denial
            else (
                _COMMAND_POLICY_DENIAL_RESULT_STRUCTURE_KEYS
                if is_command_policy_denial
                else _APPROVAL_DENIAL_RESULT_STRUCTURE_KEYS
            )
        )
        redactor.require_no_secret_keys(
            {key: None if key == "metadata" else structured[key] for key in structure_keys},
            field_name=field_name,
            preserve_keys=structure_keys,
            match_short_substrings=True,
        )
        if is_tool_policy_denial or is_approval_denial:
            redactor.require_no_secret_keys(
                structured["metadata"],
                field_name=f"{field_name}.metadata",
                match_short_substrings=True,
            )
        return

    redactor.require_no_secret_keys(
        structured,
        field_name=field_name,
        match_short_substrings=True,
    )


def _recognized_runtime_terminal_controls(structured: dict[str, object]) -> dict[str, object]:
    """Return validated terminal and execution controls from admitted runtime state."""

    try:
        controls: dict[str, object] = tool_terminal_controls.runtime_terminal_controls(structured)
        controls.update(tool_results.runtime_tool_execution_boundary_controls(structured))
        return controls
    except (TypeError, ValueError):
        return {}


def _recognized_untrusted_terminal_controls(structured: dict[str, object]) -> dict[str, object]:
    """Retain the ingress rule: boundary-only shapes cannot claim runtime authority."""

    try:
        controls: dict[str, object] = tool_terminal_controls.runtime_terminal_controls(structured)
        if controls:
            controls.update(tool_results.runtime_tool_execution_boundary_controls(structured))
        return controls
    except (TypeError, ValueError):
        return {}
