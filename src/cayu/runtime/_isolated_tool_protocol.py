"""Bounded versioned protocol for one process-isolated tool invocation."""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import (
    canonical_bounded_durable_json_bytes,
    copy_bounded_durable_json_value,
    inspect_bounded_durable_json,
    require_durable_clean_nonblank,
)
from cayu.core.isolated_tools import (
    ISOLATED_TOOL_PROTOCOL_NAME,
    ISOLATED_TOOL_PROTOCOL_VERSION,
    MAX_ISOLATED_TOOL_JSON_NODES,
    MAX_ISOLATED_TOOL_MESSAGE_BYTES,
    ProcessIsolatedToolContext,
    ProcessIsolatedToolFactoryRef,
    ProcessIsolatedToolLimits,
)
from cayu.core.tools import ToolResult

_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_TERMINAL_FRAME_MAGIC = b"CIT1"
ISOLATED_TOOL_TERMINAL_FRAME_HEADER_BYTES = len(_TERMINAL_FRAME_MAGIC) + 4


class IsolatedToolChildErrorCode(StrEnum):
    """Fixed child-produced failure classifications safe for parent diagnostics."""

    FACTORY_IMPORT_FAILED = "factory_import_failed"
    FACTORY_IDENTITY_MISMATCH = "factory_identity_mismatch"
    FACTORY_INVALID = "factory_invalid"
    FACTORY_CONSTRUCTION_FAILED = "factory_construction_failed"
    CHILD_EXCEPTION = "child_exception"
    INVALID_RESULT = "invalid_result"
    INTERNAL_PROTOCOL_FAILURE = "internal_protocol_failure"
    ENVIRONMENT_INVALID = "environment_invalid"


class IsolatedToolProtocolError(ValueError):
    """Detached protocol failure that never retains peer-controlled diagnostics."""

    def __init__(self, code: str) -> None:
        self.code = require_durable_clean_nonblank(code, "code")
        super().__init__(f"Isolated tool protocol failed: {self.code}.")


class _InvocationAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    parent_task_id: str | None = None
    parent_run_epoch: StrictInt = Field(ge=0)
    model_step_id: str
    model_attempt_id: str
    tool_round_id: str
    tool_call_id: str
    tool_name: str
    idempotency_key: str
    effective_arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_profile_fingerprint: str
    environment_allocation_fingerprint: str | None = None

    @field_validator(
        "parent_task_id",
        "environment_allocation_fingerprint",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "tool_name",
        "idempotency_key",
        "execution_profile_fingerprint",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class IsolatedToolInvocationEnvelope(BaseModel):
    """Exact parent-to-child invocation document."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    protocol: Literal["cayu.isolated-tool"] = ISOLATED_TOOL_PROTOCOL_NAME
    version: Literal[1] = ISOLATED_TOOL_PROTOCOL_VERSION
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    session_id: str
    authority: _InvocationAuthority
    factory: ProcessIsolatedToolFactoryRef
    limits: ProcessIsolatedToolLimits
    factory_config: dict[str, Any]
    arguments: dict[str, Any]
    context: ProcessIsolatedToolContext
    environment: dict[str, str]

    @field_validator("protocol", mode="before")
    @classmethod
    def validate_protocol(cls, value: object) -> object:
        if type(value) is not str or value != ISOLATED_TOOL_PROTOCOL_NAME:
            raise ValueError("Invalid isolated tool protocol.")
        return value

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int or value != ISOLATED_TOOL_PROTOCOL_VERSION:
            raise ValueError("Invalid isolated tool protocol version.")
        return value

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "session_id")

    @field_validator("factory_config", "arguments", mode="before")
    @classmethod
    def copy_objects(cls, value: object, info) -> dict[str, Any]:
        copied = copy_bounded_durable_json_value(
            value,
            info.field_name,
            max_bytes=MAX_ISOLATED_TOOL_MESSAGE_BYTES,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )
        if type(copied) is not dict:
            raise TypeError(f"{info.field_name} must be a JSON object.")
        return copied

    @field_validator("environment", mode="before")
    @classmethod
    def copy_environment(cls, value: object) -> dict[str, str]:
        if type(value) is not dict or any(
            type(key) is not str or type(item) is not str for key, item in value.items()
        ):
            raise TypeError("environment must be a string map.")
        return dict(cast("dict[str, str]", value))

    def identity_material(self) -> dict[str, Any]:
        document = self.model_dump(mode="json")
        document.pop("request_sha256", None)
        return document


class IsolatedToolTerminalEnvelope(BaseModel):
    """Exact child-to-parent terminal document."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    protocol: Literal["cayu.isolated-tool"] = ISOLATED_TOOL_PROTOCOL_NAME
    version: Literal[1] = ISOLATED_TOOL_PROTOCOL_VERSION
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: Literal["ok", "error"]
    result: dict[str, Any] | None = None
    error_code: IsolatedToolChildErrorCode | None = None

    @field_validator("protocol", mode="before")
    @classmethod
    def validate_protocol(cls, value: object) -> object:
        if type(value) is not str or value != ISOLATED_TOOL_PROTOCOL_NAME:
            raise ValueError("Invalid isolated tool protocol.")
        return value

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int or value != ISOLATED_TOOL_PROTOCOL_VERSION:
            raise ValueError("Invalid isolated tool protocol version.")
        return value

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> dict[str, Any] | None:
        if value is None:
            return None
        copied = copy_bounded_durable_json_value(
            value,
            "result",
            max_bytes=MAX_ISOLATED_TOOL_MESSAGE_BYTES,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )
        if type(copied) is not dict:
            raise TypeError("result must be a JSON object.")
        return copied

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> IsolatedToolTerminalEnvelope:
        if self.status == "ok" and (self.result is None or self.error_code is not None):
            raise ValueError("Successful isolated tool output requires only result.")
        if self.status == "error" and (self.result is not None or self.error_code is None):
            raise ValueError("Failed isolated tool output requires only error_code.")
        return self


def build_isolated_tool_request(
    *,
    session_id: str,
    authority: dict[str, Any],
    factory: ProcessIsolatedToolFactoryRef,
    limits: ProcessIsolatedToolLimits,
    factory_config: dict[str, Any],
    arguments: dict[str, Any],
    context: ProcessIsolatedToolContext,
    environment: dict[str, str],
) -> tuple[IsolatedToolInvocationEnvelope, bytes]:
    """Build one content-bound request and its canonical bounded bytes."""

    identity_material = {
        "protocol": ISOLATED_TOOL_PROTOCOL_NAME,
        "version": ISOLATED_TOOL_PROTOCOL_VERSION,
        "session_id": session_id,
        "authority": authority,
        "factory": factory.model_dump(mode="json"),
        "limits": limits.model_dump(mode="json"),
        "factory_config": factory_config,
        "arguments": arguments,
        "context": context.model_dump(mode="json"),
        "environment": environment,
    }
    identity_bytes = canonical_bounded_durable_json_bytes(
        identity_material,
        "isolated_tool_request_identity",
        max_bytes=limits.max_request_bytes,
        max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
    )
    request_sha256 = f"sha256:{sha256(identity_bytes).hexdigest()}"
    document = {"request_sha256": request_sha256, **identity_material}
    encoded = canonical_bounded_durable_json_bytes(
        document,
        "isolated_tool_request",
        max_bytes=limits.max_request_bytes,
        max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
    )
    try:
        envelope = IsolatedToolInvocationEnvelope.model_validate(document)
    except Exception:
        raise IsolatedToolProtocolError("request_construction_failed") from None
    return envelope, encoded


def decode_isolated_tool_request(data: bytes) -> IsolatedToolInvocationEnvelope:
    """Decode one child request without propagating raw parser diagnostics."""

    if type(data) is not bytes or len(data) > MAX_ISOLATED_TOOL_MESSAGE_BYTES:
        raise IsolatedToolProtocolError("request_size_invalid")
    try:
        return _decode_isolated_tool_request_unchecked(data)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        data = b""
    failure = IsolatedToolProtocolError("request_invalid")
    failure.__context__ = None
    raise failure from None


def _decode_isolated_tool_request_unchecked(data: bytes) -> IsolatedToolInvocationEnvelope:
    document = json.loads(data.decode("utf-8"))
    if type(document) is not dict:
        raise ValueError
    inspect_bounded_durable_json(
        document,
        "isolated_tool_request",
        max_bytes=MAX_ISOLATED_TOOL_MESSAGE_BYTES,
        max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
    )
    envelope = IsolatedToolInvocationEnvelope.model_validate(document)
    identity_bytes = canonical_bounded_durable_json_bytes(
        envelope.identity_material(),
        "isolated_tool_request_identity",
        max_bytes=envelope.limits.max_request_bytes,
        max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
    )
    expected = f"sha256:{sha256(identity_bytes).hexdigest()}"
    if envelope.request_sha256 != expected or len(data) > envelope.limits.max_request_bytes:
        raise ValueError
    canonical = canonical_bounded_durable_json_bytes(
        envelope.model_dump(mode="json"),
        "isolated_tool_request",
        max_bytes=envelope.limits.max_request_bytes,
        max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
    )
    if data != canonical:
        raise ValueError
    return envelope


def encode_isolated_tool_success(
    *,
    request_sha256: str,
    result: ToolResult,
    max_bytes: int,
) -> bytes:
    if type(result) is not ToolResult:
        raise IsolatedToolProtocolError("result_type_invalid")
    try:
        validated = ToolResult(
            content=result.content,
            structured=result.structured,
            artifacts=result.artifacts,
            is_error=result.is_error,
        )
        document = IsolatedToolTerminalEnvelope(
            request_sha256=request_sha256,
            status="ok",
            result=validated.model_dump(mode="json", warnings=False),
        ).model_dump(mode="json")
        return canonical_bounded_durable_json_bytes(
            document,
            "isolated_tool_response",
            max_bytes=max_bytes,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )
    except Exception:
        raise IsolatedToolProtocolError("result_invalid") from None


def encode_isolated_tool_error(
    *,
    request_sha256: str,
    error_code: IsolatedToolChildErrorCode,
    max_bytes: int,
) -> bytes:
    try:
        document = IsolatedToolTerminalEnvelope(
            request_sha256=request_sha256,
            status="error",
            error_code=error_code,
        ).model_dump(mode="json")
        return canonical_bounded_durable_json_bytes(
            document,
            "isolated_tool_response",
            max_bytes=max_bytes,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )
    except Exception:
        raise IsolatedToolProtocolError("error_response_invalid") from None


def encode_isolated_tool_terminal_frame(data: bytes, *, max_bytes: int) -> bytes:
    """Wrap one canonical terminal envelope in a fixed bounded transport frame."""

    if type(data) is not bytes or not data or len(data) > max_bytes:
        raise IsolatedToolProtocolError("response_size_invalid")
    return _TERMINAL_FRAME_MAGIC + len(data).to_bytes(4, "big") + data


def isolated_tool_terminal_frame_payload_length(
    header: bytes,
    *,
    max_bytes: int,
) -> int:
    """Validate one fixed frame header and return its declared payload length."""

    if (
        type(header) is not bytes
        or len(header) != ISOLATED_TOOL_TERMINAL_FRAME_HEADER_BYTES
        or header[: len(_TERMINAL_FRAME_MAGIC)] != _TERMINAL_FRAME_MAGIC
    ):
        raise IsolatedToolProtocolError("response_invalid")
    payload_bytes = int.from_bytes(header[len(_TERMINAL_FRAME_MAGIC) :], "big")
    if payload_bytes <= 0:
        raise IsolatedToolProtocolError("response_invalid")
    if payload_bytes > max_bytes:
        raise IsolatedToolProtocolError("response_too_large")
    return payload_bytes


def decode_isolated_tool_response(
    data: bytes,
    *,
    expected_request_sha256: str,
    max_bytes: int,
) -> ToolResult | IsolatedToolChildErrorCode:
    """Decode one exact terminal frame and detach all peer-controlled failures."""

    if type(data) is not bytes or not data or len(data) > max_bytes:
        raise IsolatedToolProtocolError("response_size_invalid")
    try:
        return _decode_isolated_tool_response_unchecked(
            data,
            expected_request_sha256=expected_request_sha256,
            max_bytes=max_bytes,
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        data = b""
    failure = IsolatedToolProtocolError("response_invalid")
    failure.__context__ = None
    raise failure from None


def _decode_isolated_tool_response_unchecked(
    data: bytes,
    *,
    expected_request_sha256: str,
    max_bytes: int,
) -> ToolResult | IsolatedToolChildErrorCode:
    document = json.loads(data.decode("utf-8"))
    if type(document) is not dict:
        raise ValueError
    inspect_bounded_durable_json(
        document,
        "isolated_tool_response",
        max_bytes=max_bytes,
        max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
    )
    envelope = IsolatedToolTerminalEnvelope.model_validate(document)
    if envelope.request_sha256 != expected_request_sha256:
        raise ValueError
    canonical = canonical_bounded_durable_json_bytes(
        envelope.model_dump(mode="json"),
        "isolated_tool_response",
        max_bytes=max_bytes,
        max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
    )
    if data != canonical:
        raise ValueError
    if envelope.status == "error":
        if envelope.error_code is None:  # pragma: no cover - model invariant
            raise AssertionError
        return envelope.error_code
    if envelope.result is None:  # pragma: no cover - model invariant
        raise AssertionError
    return ToolResult.model_validate(envelope.result)


__all__ = [
    "ISOLATED_TOOL_TERMINAL_FRAME_HEADER_BYTES",
    "IsolatedToolChildErrorCode",
    "IsolatedToolInvocationEnvelope",
    "IsolatedToolProtocolError",
    "build_isolated_tool_request",
    "decode_isolated_tool_request",
    "decode_isolated_tool_response",
    "encode_isolated_tool_error",
    "encode_isolated_tool_success",
    "encode_isolated_tool_terminal_frame",
    "isolated_tool_terminal_frame_payload_length",
]
