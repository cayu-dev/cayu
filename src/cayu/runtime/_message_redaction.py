from __future__ import annotations

from typing import Any, cast

from cayu.artifacts.attachments import FILE_ATTACHMENT_TYPE
from cayu.core.messages import Message
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.tool_result_projection import (
    _BUILTIN_TOOL_RESULT_ARTIFACT_REFERENCE_FIELDS,
    TOOL_RESULT_ARTIFACT_TYPE,
    _validate_builtin_tool_result_artifact_reference,
)
from cayu.vaults import SecretRedactor

_MESSAGE_ROOT_STRUCTURE_KEYS = frozenset({"content", "role"})
_MESSAGE_PART_STRUCTURE_KEYS = {
    "text": frozenset({"text", "type"}),
    "tool_call": frozenset(
        {
            "arguments",
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
}
_MESSAGE_PART_UNTRUSTED_CONTAINERS = {
    "tool_call": frozenset({"arguments"}),
    "tool_result": frozenset({"artifacts", "structured"}),
    "provider_state": frozenset({"state"}),
    "thinking": frozenset({"provider_state"}),
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
        "manual_reconciliation_required",
        "outcome_unknown",
        "portable_result_evidence",
        "portable_result_evidence_incomplete",
        "terminal_outcome",
        "tool_effect",
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
        "provider",
        "tool_call_id",
        "tool_name",
        "tool_round_id",
    }
)
_MESSAGE_PRESERVED_STRING_FIELDS = _MESSAGE_AUTHORITY_STRING_FIELDS | {
    "type",
}


def redact_untrusted_message_for_boundary(
    message: Message,
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> Message:
    """Redact untrusted ingress without accepting runtime projection authority."""

    return _redact_message_for_boundary(
        message,
        redactor=redactor,
        field_name=field_name,
        trust_runtime_tool_result_projection=False,
        reject_secret_bearing_runtime_projection_authority=False,
    )


def redact_runtime_message_for_boundary(
    message: Message,
    *,
    redactor: SecretRedactor,
    field_name: str,
    reject_secret_bearing_runtime_projection_authority: bool = False,
) -> Message:
    """Redact runtime-owned transcript state after validating projection authority."""

    return _redact_message_for_boundary(
        message,
        redactor=redactor,
        field_name=field_name,
        trust_runtime_tool_result_projection=True,
        reject_secret_bearing_runtime_projection_authority=(
            reject_secret_bearing_runtime_projection_authority
        ),
    )


def _redact_message_for_boundary(
    message: Message,
    *,
    redactor: SecretRedactor,
    field_name: str,
    trust_runtime_tool_result_projection: bool,
    reject_secret_bearing_runtime_projection_authority: bool,
) -> Message:
    """Redact one message without rewriting executable protocol authorities."""

    if type(message) is not Message:
        raise TypeError("Messages must be Message instances.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    payload = message.model_dump(mode="json")
    if not trust_runtime_tool_result_projection:
        _strip_untrusted_runtime_tool_result_projection_authority(payload)
    content = payload.get("content")
    if type(content) is not list:
        raise AssertionError("Message serialization returned non-list content.")
    _require_secret_free_message_keys(
        payload,
        content=content,
        redactor=redactor,
        field_name=field_name,
        trust_runtime_tool_result_projection=trust_runtime_tool_result_projection,
    )
    redacted_content: list[object] = []
    for index, part in enumerate(content):
        if type(part) is not dict:
            raise AssertionError("Message content serialized as a non-object.")
        part = cast("dict[str, object]", part)
        projection_references = _runtime_tool_result_artifact_references(
            part,
            trust_runtime_tool_result_projection=trust_runtime_tool_result_projection,
        )
        if reject_secret_bearing_runtime_projection_authority:
            _require_secret_free_runtime_projection_authority(
                projection_references,
                redactor=redactor,
                field_name=f"{field_name}.content[{index}].artifacts",
            )
        # Protocol linkage and routing fields are executable authority. They
        # must fail closed instead of being rewritten into a different value.
        for authority_field in _MESSAGE_AUTHORITY_STRING_FIELDS:
            value = part.get(authority_field)
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
        if projection_references:
            projected_content, projected_artifacts = (
                tool_results.project_runtime_tool_result_for_boundary(
                    original_content=part.get("content"),
                    redacted_artifacts=redacted_part.get("artifacts"),
                    references=projection_references,
                    redactor=redactor,
                )
            )
            redacted_part["content"] = projected_content
            redacted_part["artifacts"] = projected_artifacts
        structured = part.get("structured")
        if type(structured) is dict:
            terminal_controls = _recognized_runtime_terminal_controls(
                cast("dict[str, object]", structured)
            )
            redacted_structured = redacted_part.get("structured")
            if terminal_controls and type(redacted_structured) is dict:
                redacted_structured.update(terminal_controls)
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
    trust_runtime_tool_result_projection: bool,
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
            _require_secret_free_tool_result_artifacts(
                artifacts,
                redactor=redactor,
                field_name=f"{field_name}.content[{index}].artifacts",
                trust_runtime_tool_result_projection=(trust_runtime_tool_result_projection),
            )
            if type(structured) is dict:
                _require_secret_free_tool_result_structure(
                    part,
                    structured=cast("dict[str, object]", structured),
                    redactor=redactor,
                    field_name=f"{field_name}.content[{index}].structured",
                )


def _runtime_tool_result_artifact_references(
    part: dict[str, object],
    *,
    trust_runtime_tool_result_projection: bool,
) -> dict[int, dict[str, object]]:
    if not trust_runtime_tool_result_projection or part.get("type") != "tool_result":
        return {}
    artifacts = part.get("artifacts")
    if type(artifacts) is not list:
        return {}
    references: dict[int, dict[str, object]] = {}
    for index, artifact in enumerate(artifacts):
        if type(artifact) is not dict:
            continue
        artifact = cast("dict[str, Any]", artifact)
        if artifact.get("type") != TOOL_RESULT_ARTIFACT_TYPE:
            continue
        try:
            reference = _validate_builtin_tool_result_artifact_reference(artifact)
        except (TypeError, ValueError):
            continue
        references[index] = cast("dict[str, object]", reference)
    return references


def _require_secret_free_runtime_projection_authority(
    references: dict[int, dict[str, object]],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> None:
    """Reject unverified stored references that secret redaction would rewrite."""

    for index, reference in references.items():
        if redactor.redact_json_values(reference) != reference:
            raise ValueError(
                f"{field_name}[{index}] contains a workload secret and cannot be "
                "used as runtime projection authority."
            )


def _require_secret_free_tool_result_artifacts(
    artifacts: object,
    *,
    redactor: SecretRedactor,
    field_name: str,
    trust_runtime_tool_result_projection: bool,
) -> None:
    if type(artifacts) is not list:
        raise AssertionError("Tool result artifacts serialized as a non-list.")
    references = _runtime_tool_result_artifact_references(
        {
            "type": "tool_result",
            "artifacts": artifacts,
        },
        trust_runtime_tool_result_projection=trust_runtime_tool_result_projection,
    )
    for index, artifact in enumerate(artifacts):
        if type(artifact) is not dict:
            raise AssertionError("Tool result artifact serialized as a non-object.")
        reference = references.get(index)
        redactor.require_no_secret_keys(
            artifact if reference is None else reference,
            field_name=f"{field_name}[{index}]",
            preserve_keys=(
                () if reference is None else _BUILTIN_TOOL_RESULT_ARTIFACT_REFERENCE_FIELDS
            ),
            match_short_substrings=True,
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
) -> None:
    """Exempt only positively identified runtime-owned tool-result schemas."""

    terminal_controls = _recognized_runtime_terminal_controls(structured)
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
        and set(structured) == _COMMAND_POLICY_DENIAL_RESULT_STRUCTURE_KEYS
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


def _recognized_runtime_terminal_controls(
    structured: dict[str, object],
) -> dict[str, object]:
    """Return only positively validated runtime-owned terminal controls."""

    try:
        return tool_results.runtime_terminal_controls(structured)
    except (TypeError, ValueError):
        return {}
