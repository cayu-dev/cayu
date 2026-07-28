from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cayu._exception_state import exception_state, set_exception_state
from cayu._validation import (
    copy_durable_json_object,
    extract_durable_value_error,
    require_durable_text,
    safe_durable_value_error_details,
)
from cayu.vaults import SecretRedactor

MAX_DIAGNOSTIC_UTF8_BYTES = 4 * 1024
MAX_DIAGNOSTIC_TYPE_UTF8_BYTES = 128
_TRUNCATION_MARKER = "\u2026[truncated]"
_RUNTIME_EXCEPTION_PAYLOAD_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _RuntimeExceptionPayload:
    """Immutable runtime-owned payload attached to a propagated exception."""

    attribute_name: str
    encoded_payload: str
    token: object


def _attach_runtime_exception_payload(
    error: BaseException,
    *,
    attribute_name: str,
    payload: dict[str, Any],
) -> None:
    """Attach validated evidence without invoking exception subclass accessors."""

    copied = copy_durable_json_object(payload, "runtime exception payload")
    handoff = _RuntimeExceptionPayload(
        attribute_name=attribute_name,
        encoded_payload=json.dumps(
            copied,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        token=_RUNTIME_EXCEPTION_PAYLOAD_TOKEN,
    )
    if not set_exception_state(error, attribute_name, handoff):
        raise RuntimeError("Could not attach runtime exception payload.")


def _runtime_exception_payload(
    error: BaseException,
    *,
    attribute_name: str,
) -> dict[str, Any] | None:
    """Return only evidence attached through the runtime-owned handoff."""

    handoff = _runtime_exception_payload_handoff(
        error,
        attribute_name=attribute_name,
    )
    if handoff is None:
        return None
    try:
        decoded = json.loads(handoff.encoded_payload)
        return copy_durable_json_object(decoded, "runtime exception payload")
    except BaseException:
        return None


def _copy_runtime_exception_payload(
    source: BaseException,
    target: BaseException,
    *,
    attribute_name: str,
) -> None:
    """Propagate an authenticated immutable handoff to an aggregate error."""

    handoff = _runtime_exception_payload_handoff(
        source,
        attribute_name=attribute_name,
    )
    if handoff is not None and not set_exception_state(target, attribute_name, handoff):
        raise RuntimeError("Could not copy runtime exception payload.")


def _runtime_exception_payload_handoff(
    error: BaseException,
    *,
    attribute_name: str,
) -> _RuntimeExceptionPayload | None:
    handoff = exception_state(error, attribute_name)
    if (
        type(handoff) is not _RuntimeExceptionPayload
        or handoff.token is not _RUNTIME_EXCEPTION_PAYLOAD_TOKEN
        or handoff.attribute_name != attribute_name
    ):
        return None
    return handoff


@dataclass(frozen=True, slots=True)
class ExceptionDiagnostic:
    """Bounded, portable exception evidence for durable records."""

    message: str
    error_type: str
    durable_value_error_code: str | None = None
    durable_value_error_path: str | None = None

    def payload_fields(
        self,
        *,
        message_key: str = "error",
        error_type_key: str = "error_type",
        detail_prefix: str = "",
    ) -> dict[str, Any]:
        """Return portable fields under caller-selected record keys."""

        fields: dict[str, Any] = {
            message_key: self.message,
            error_type_key: self.error_type,
        }
        if self.durable_value_error_code is not None:
            fields[f"{detail_prefix}durable_value_error_code"] = self.durable_value_error_code
            fields[f"{detail_prefix}durable_value_error_path"] = self.durable_value_error_path
        return fields


def exception_diagnostic(
    exc: BaseException,
    *,
    empty_message: str = "operation failed",
    nonportable_message: str = "Operation failed with a non-portable diagnostic.",
    preserve_empty_message: bool = False,
    redactor: SecretRedactor | None = None,
) -> ExceptionDiagnostic:
    """Snapshot an exception without carrying rejected text into durability.

    Exception implementations and their text are caller-controlled. Attribute
    access and rendering are therefore guarded, while durable-value failures
    retain only their runtime-owned code and path.
    """

    resolved_redactor = redactor or SecretRedactor()
    error_type = _safe_exception_type_name(
        exc,
        redactor=resolved_redactor,
    )
    durable_error = extract_durable_value_error(exc)
    if durable_error is not None:
        code, path = safe_durable_value_error_details(durable_error)
        return ExceptionDiagnostic(
            message=resolved_redactor.redact_text_bounded(
                nonportable_message,
                max_bytes=MAX_DIAGNOSTIC_UTF8_BYTES,
            ),
            error_type=error_type,
            durable_value_error_code=code,
            durable_value_error_path=path,
        )
    rendering_failed = False
    try:
        rendered = str(exc).strip()
    except BaseException:
        rendering_failed = True
        rendered = ""
    if rendered:
        try:
            require_durable_text(rendered, "error")
        except BaseException as validation_error:
            durable_error = extract_durable_value_error(validation_error)
            code = path = None
            if durable_error is not None:
                code, path = safe_durable_value_error_details(durable_error)
            return ExceptionDiagnostic(
                message=resolved_redactor.redact_text_bounded(
                    nonportable_message,
                    max_bytes=MAX_DIAGNOSTIC_UTF8_BYTES,
                ),
                error_type=error_type,
                durable_value_error_code=code,
                durable_value_error_path=path,
            )
        message = rendered
    elif preserve_empty_message and not rendering_failed:
        message = ""
    else:
        message = f"{error_type}: {empty_message}"
    return ExceptionDiagnostic(
        message=resolved_redactor.redact_text_bounded(
            message,
            max_bytes=MAX_DIAGNOSTIC_UTF8_BYTES,
        ),
        error_type=error_type,
    )


def task_failure_payload_from_diagnostic(
    diagnostic: ExceptionDiagnostic,
    *,
    session_id: str,
    additional_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one trusted failure snapshot onto the task error contract."""

    if type(diagnostic) is not ExceptionDiagnostic:
        raise TypeError("diagnostic must be an ExceptionDiagnostic.")
    payload = {
        **diagnostic.payload_fields(
            message_key="message",
            error_type_key="type",
        ),
        "session_id": session_id,
    }
    if additional_fields is not None:
        payload.update(additional_fields)
    return payload


def task_update_error_payload(
    error: BaseException,
    *,
    redactor: SecretRedactor | None = None,
) -> dict[str, Any]:
    """Return prefixed portable evidence for a secondary task-store failure."""

    return exception_diagnostic(
        error,
        empty_message="task update failed",
        nonportable_message="Task update failed with a non-portable diagnostic.",
        redactor=redactor,
    ).payload_fields(
        message_key="task_update_error",
        error_type_key="task_update_error_type",
        detail_prefix="task_update_",
    )


def bound_diagnostic_text(value: str) -> str:
    """Bound already-portable diagnostic text without splitting UTF-8."""

    used_bytes = 0
    characters: list[str] = []
    for character in value:
        character_bytes = len(character.encode("utf-8"))
        if used_bytes + character_bytes > MAX_DIAGNOSTIC_UTF8_BYTES:
            marker_bytes = len(_TRUNCATION_MARKER.encode("utf-8"))
            while characters and used_bytes + marker_bytes > MAX_DIAGNOSTIC_UTF8_BYTES:
                removed = characters.pop()
                used_bytes -= len(removed.encode("utf-8"))
            return "".join(characters) + _TRUNCATION_MARKER
        characters.append(character)
        used_bytes += character_bytes
    return value


def _safe_exception_type_name(
    exc: BaseException,
    *,
    redactor: SecretRedactor,
) -> str:
    try:
        name = type.__getattribute__(type(exc), "__name__")
    except BaseException:
        name = "Exception"
    if type(name) is not str:
        name = "Exception"
    try:
        require_durable_text(name, "error_type")
    except BaseException:
        name = "Exception"
    return _bound_utf8_text(
        redactor.redact_text(name),
        MAX_DIAGNOSTIC_TYPE_UTF8_BYTES,
    )


def _bound_utf8_text(value: str, max_bytes: int) -> str:
    used_bytes = 0
    characters: list[str] = []
    for character in value:
        character_bytes = len(character.encode("utf-8"))
        if used_bytes + character_bytes > max_bytes:
            marker_bytes = len(_TRUNCATION_MARKER.encode("utf-8"))
            while characters and used_bytes + marker_bytes > max_bytes:
                removed = characters.pop()
                used_bytes -= len(removed.encode("utf-8"))
            return "".join(characters) + _TRUNCATION_MARKER
        characters.append(character)
        used_bytes += character_bytes
    return value
