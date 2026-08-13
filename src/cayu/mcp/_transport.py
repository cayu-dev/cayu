"""Shared resource bounds and failures for MCP transports."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._exception_groups import exception_cause, rebuild_exception_group, set_exception_cause
from cayu._validation import FrozenJsonDict, FrozenJsonList
from cayu.mcp._exception_handoffs import copy_mcp_failure_handoffs
from cayu.mcp._jsonrpc import McpProtocolError
from cayu.vaults import SecretRedactor

DEFAULT_MCP_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
DEFAULT_MCP_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_MCP_JSON_NESTING = 256
_MCP_JSON_NO_VALUE = object()
_OwnedTaskResultT = TypeVar("_OwnedTaskResultT")


class McpTransportLimits(BaseModel):
    """Allocation and timing bounds for one MCP transport exchange."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    max_message_bytes: int = DEFAULT_MCP_MAX_MESSAGE_BYTES
    max_response_bytes: int = DEFAULT_MCP_MAX_RESPONSE_BYTES
    idle_timeout_s: float = 30.0
    total_call_timeout_s: float = 30.0

    @field_validator("max_message_bytes", "max_response_bytes", mode="before")
    @classmethod
    def validate_byte_limit(cls, value: object, info) -> int:
        if type(value) is not int:
            raise TypeError(f"{info.field_name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{info.field_name} must be greater than zero.")
        return value

    @field_validator("idle_timeout_s", "total_call_timeout_s", mode="before")
    @classmethod
    def validate_timeout(cls, value: object, info) -> float:
        if type(value) not in {float, int}:
            raise TypeError(f"{info.field_name} must be a number.")
        try:
            numeric = float(cast("float | int", value))
        except OverflowError:
            raise ValueError(f"{info.field_name} must be finite and greater than zero.") from None
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{info.field_name} must be finite and greater than zero.")
        return numeric

    @model_validator(mode="after")
    def validate_response_limit(self) -> McpTransportLimits:
        if self.max_response_bytes < self.max_message_bytes:
            raise ValueError("max_response_bytes must be at least max_message_bytes.")
        return self


class McpIdleTimeoutError(TimeoutError):
    """Raised when an MCP transport stops making inbound progress."""


class McpCallDeadlineExceededError(TimeoutError):
    """Raised when an MCP exchange exceeds its absolute deadline."""


class McpMessageTooLargeError(McpProtocolError):
    """Raised when one JSON-RPC message or SSE event exceeds its byte limit."""


class McpResponseTooLargeError(McpProtocolError):
    """Raised when an aggregate HTTP response exceeds its byte limit."""


class McpPeerClosedError(McpProtocolError):
    """Raised when the peer closes before completing the expected response."""


@dataclass(frozen=True, slots=True)
class _McpOwnedTaskFatalSignal:
    """Keep a scalar fatal signal from escaping through asyncio task machinery."""

    signal: BaseException


async def _capture_mcp_owned_task_fatal_signal(
    awaitable: Awaitable[_OwnedTaskResultT],
) -> _OwnedTaskResultT | _McpOwnedTaskFatalSignal:
    """Return task-unsafe process-control signals as inert child outcomes."""

    try:
        return await awaitable
    except (KeyboardInterrupt, SystemExit) as signal:
        # CPython Task stores and then immediately re-raises these two signals.
        # Detach their extension-controlled traceback before the parent resumes;
        # the parent transport boundary remains responsible for safe diagnostics.
        return _McpOwnedTaskFatalSignal(BaseException.with_traceback(signal, None))


def _unwrap_mcp_owned_task_result(
    result: _OwnedTaskResultT | _McpOwnedTaskFatalSignal,
) -> _OwnedTaskResultT:
    """Re-deliver a captured process-control signal in the owning parent task."""

    if not isinstance(result, _McpOwnedTaskFatalSignal):
        return result
    # A retained settlement may need to observe the same exact child outcome
    # after the foreground owner already classified it. Clear the prior parent
    # traceback before each delivery while preserving the signal's scalar type.
    raise BaseException.with_traceback(result.signal, None) from None


@dataclass(frozen=True, slots=True)
class _McpJsonPreflightResult:
    exceeds_limit: bool
    nesting_too_deep: bool


@dataclass(slots=True)
class _McpJsonContainerFrame:
    value_id: int
    items: Iterator[Any]
    is_object: bool
    item_count: int = 0


def mcp_jsonrpc_request_preflight(
    request_id: int,
    method: str,
    params: dict[str, Any],
    *,
    max_bytes: int,
) -> _McpJsonPreflightResult:
    """Preflight one request without allocating its defensive JSON copy.

    Unsupported JSON values remain the payload validator's responsibility. The
    preflight classifies only a positively established byte overflow or nesting
    beyond the transport's bounded defensive-copy envelope.
    """

    payload_view = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    try:
        return _McpJsonUtf8SizeCounter(max_bytes).inspect(payload_view)
    finally:
        # Drop the shallow view so a typed preparation error cannot retain caller
        # data through this helper's locals.
        payload_view.clear()


def mcp_json_value_nesting_too_deep(value: Any) -> bool:
    """Return whether an MCP value exceeds the bounded JSON nesting envelope."""

    return _McpJsonUtf8SizeCounter(None).inspect(value).nesting_too_deep


class _McpJsonUtf8SizeCounter:
    """Iterative compact-JSON counter with the copier's value contract."""

    def __init__(self, limit: int | None) -> None:
        self._remaining = limit
        self._active_container_ids: set[int] = set()
        self._exceeded_limit = False
        self._nesting_too_deep = False
        self._invalid_value = False

    def _consume(self, count: int) -> bool:
        if self._remaining is None:
            return True
        self._remaining -= count
        if self._remaining < 0:
            self._exceeded_limit = True
            return False
        return True

    def _string(self, value: str) -> bool:
        if not self._consume(2):
            return False
        for character in value:
            codepoint = ord(character)
            if character in {'"', "\\"} or character in "\b\f\n\r\t":
                size = 2
            elif codepoint < 0x20 or 0x7F <= codepoint < 0x10000:
                size = 6
            elif codepoint >= 0x10000:
                size = 12
            else:
                size = 1
            if not self._consume(size):
                return False
        return True

    def _scalar(self, value: Any) -> bool:
        if self._remaining is None and (value is None or type(value) in {str, bool, int, float}):
            # A depth-only walk must not render a sibling scalar. In particular,
            # interpreter bigint conversion limits cannot hide a later deep branch.
            return True
        if value is None:
            return self._consume(4)
        if value is True:
            return self._consume(4)
        if value is False:
            return self._consume(5)
        if type(value) is str:
            return self._string(value)
        if type(value) in {int, float}:
            if type(value) is float and not math.isfinite(value):
                return False
            try:
                rendered = str(value)
            except (OverflowError, ValueError):
                self._exceeded_limit = True
                return False
            return self._consume(len(rendered.encode("utf-8")))
        self._invalid_value = True
        return False

    def _next_item(self, frame: _McpJsonContainerFrame) -> tuple[bool, Any]:
        try:
            entry = next(frame.items)
        except StopIteration:
            return False, None
        if frame.item_count and not self._consume(1):
            return False, None
        frame.item_count += 1
        if not frame.is_object:
            return True, entry
        key, item = entry
        if type(key) is not str:
            self._invalid_value = True
            return False, None
        if self._remaining is not None and (not self._string(key) or not self._consume(1)):
            return False, None
        return True, item

    def inspect(self, value: Any) -> _McpJsonPreflightResult:
        frames: list[_McpJsonContainerFrame] = []
        current = value
        value = None
        try:
            while True:
                if type(current) in {dict, FrozenJsonDict, list, FrozenJsonList}:
                    value_id = id(current)
                    if value_id in self._active_container_ids:
                        self._invalid_value = True
                        break
                    if not self._consume(2):
                        break
                    if len(frames) >= _MAX_MCP_JSON_NESTING:
                        self._nesting_too_deep = True
                        break
                    is_object = type(current) in {dict, FrozenJsonDict}
                    self._active_container_ids.add(value_id)
                    container_items = (
                        iter(cast("dict[str, Any] | FrozenJsonDict", current).items())
                        if is_object
                        else iter(cast("list[Any] | FrozenJsonList", current))
                    )
                    frames.append(
                        _McpJsonContainerFrame(
                            value_id=value_id,
                            items=container_items,
                            is_object=is_object,
                        )
                    )
                elif not self._scalar(current):
                    break

                current = _MCP_JSON_NO_VALUE
                while frames:
                    frame = frames[-1]
                    has_item, item = self._next_item(frame)
                    if self._exceeded_limit or self._invalid_value:
                        break
                    if has_item:
                        current = item
                        item = None
                        break
                    self._active_container_ids.remove(frame.value_id)
                    frames.pop()
                if self._exceeded_limit or self._invalid_value:
                    break
                if current is _MCP_JSON_NO_VALUE:
                    break
        finally:
            current = None
            frames.clear()
            self._active_container_ids.clear()
        return _McpJsonPreflightResult(
            exceeds_limit=self._exceeded_limit,
            nesting_too_deep=self._nesting_too_deep,
        )


_BASE_EXCEPTION_ARGS_DESCRIPTOR = BaseException.__dict__["args"]
_MAX_MCP_EXCEPTION_DIAGNOSTIC_ARGS = 16
_MAX_MCP_EXCEPTION_STRING_SOURCE_FACTOR = 4
_OVERSIZED_MCP_EXCEPTION_DIAGNOSTIC = "<oversized diagnostic>"
_OMITTED_MCP_EXCEPTION_DIAGNOSTICS = "<additional diagnostics omitted>"


def credential_safe_mcp_transport_failure(
    error: BaseException,
    *,
    redactor: SecretRedactor,
    context: str,
    max_message_bytes: int = 4096,
    preserve_cause: bool = False,
) -> BaseException:
    """Detach and redact an exception crossing an MCP transport boundary.

    Transport and cleanup implementations are extension-controlled. Never retain
    their raw exception object, traceback, cause, or context in a public MCP
    failure. Runtime-owned MCP transport types and explicitly preserved, detached
    causal diagnostics remain distinguishable after redaction.
    """

    safe_failure: BaseException
    if isinstance(error, BaseExceptionGroup):
        safe_failure = rebuild_exception_group(
            error,
            group_message=f"{context} reported multiple failures.",
            leaf_mapper=lambda leaf: credential_safe_mcp_transport_failure(
                leaf,
                redactor=redactor,
                context=context,
                max_message_bytes=max_message_bytes,
                preserve_cause=preserve_cause,
            ),
            invalid_leaf_factory=lambda: McpProtocolError(
                f"{context} reported an invalid failure group."
            ),
        )
    else:
        preserves_failure_type = isinstance(
            error,
            (
                McpCallDeadlineExceededError,
                McpIdleTimeoutError,
                McpResponseTooLargeError,
                McpMessageTooLargeError,
                McpPeerClosedError,
                McpProtocolError,
                TimeoutError,
            ),
        )
        safe_type = (
            ""
            if preserves_failure_type
            else _credential_safe_exception_type_name(error, redactor=redactor)
        )
        type_suffix = f": {safe_type}" if safe_type else ""
        context_budget = max_message_bytes - len(type_suffix.encode("utf-8"))
        safe_context = redactor.redact_text_bounded(
            context,
            max_bytes=max(1, context_budget),
        )
        message_prefix = f"{safe_context}{type_suffix}" if context_budget > 0 else safe_context
        message_budget = max_message_bytes - len(message_prefix.encode("utf-8")) - 2
        safe_message = _credential_safe_exception_message(
            error,
            redactor=redactor,
            max_bytes=max(1, message_budget),
        )
        detail = safe_message or ("failed" if not preserves_failure_type else "")
        message = (
            f"{message_prefix}: {detail}"
            if detail and len(detail.encode("utf-8")) <= message_budget
            else message_prefix
        )
        if isinstance(error, McpCallDeadlineExceededError):
            safe_failure = McpCallDeadlineExceededError(message)
        elif isinstance(error, McpIdleTimeoutError):
            safe_failure = McpIdleTimeoutError(message)
        elif isinstance(error, McpResponseTooLargeError):
            safe_failure = McpResponseTooLargeError(message)
        elif isinstance(error, McpMessageTooLargeError):
            safe_failure = McpMessageTooLargeError(message)
        elif isinstance(error, McpPeerClosedError):
            safe_failure = McpPeerClosedError(message)
        elif isinstance(error, McpProtocolError):
            safe_failure = McpProtocolError(message)
        elif isinstance(error, TimeoutError):
            safe_failure = TimeoutError(message)
        else:
            # A historical child signal is diagnostic evidence, not current
            # cancellation or process control reported by this boundary.
            safe_failure = McpProtocolError(message)

    _copy_credential_safe_mcp_failure_evidence(
        error,
        safe_failure,
        redactor=redactor,
        context=context,
        max_message_bytes=max_message_bytes,
        preserve_cause=preserve_cause,
    )
    return safe_failure


def credential_safe_mcp_fatal_signal(
    error: BaseException,
    *,
    redactor: SecretRedactor,
    context: str,
    max_message_bytes: int = 4096,
) -> BaseException:
    """Detach one current process-control signal while preserving its type."""

    args = _credential_safe_fatal_signal_args(
        error,
        redactor=redactor,
        max_bytes=max_message_bytes,
    )
    if isinstance(error, KeyboardInterrupt):
        safe_failure: BaseException = KeyboardInterrupt(*args)
    elif isinstance(error, SystemExit):
        safe_failure = SystemExit(*args)
    elif isinstance(error, GeneratorExit):
        safe_failure = GeneratorExit(*args)
    else:
        safe_failure = BaseException(*args)
    _copy_credential_safe_mcp_failure_evidence(
        error,
        safe_failure,
        redactor=redactor,
        context=context,
        max_message_bytes=max_message_bytes,
        preserve_cause=True,
    )
    return safe_failure


def _credential_safe_fatal_signal_args(
    error: BaseException,
    *,
    redactor: SecretRedactor,
    max_bytes: int,
) -> tuple[Any, ...]:
    try:
        raw_args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return ()
    if type(raw_args) is not tuple:
        return ()
    safe_args: list[Any] = []
    remaining = max_bytes
    source_complete = len(raw_args) <= _MAX_MCP_EXCEPTION_DIAGNOSTIC_ARGS
    for value in raw_args[:_MAX_MCP_EXCEPTION_DIAGNOSTIC_ARGS]:
        safe_value: Any
        if value is None or type(value) in {bool, int, float}:
            rendered = _credential_safe_exception_argument(
                value,
                redactor=redactor,
                max_bytes=max(1, remaining),
            )
            safe_value = _credential_safe_mcp_exception_argument_value(
                value,
                redactor=redactor,
                max_bytes=max(1, remaining),
            )
        elif type(value) is str:
            source_char_limit = max(1, remaining) * _MAX_MCP_EXCEPTION_STRING_SOURCE_FACTOR
            value_complete = len(value) <= source_char_limit
            source = value if value_complete else value[:source_char_limit]
            projected, projection_truncated = redactor.redact_utf8_head(
                source.encode("utf-8", "replace"),
                max_bytes=max(1, remaining),
                source_complete=value_complete,
            )
            projection_complete = value_complete and not projection_truncated
            if not projection_complete:
                source_complete = False
            rendered = (
                projected
                if projection_complete
                else projected or _OVERSIZED_MCP_EXCEPTION_DIAGNOSTIC
            )
            safe_value = rendered
        else:
            rendered = "<non-text diagnostic>"
            safe_value = rendered
        if len(rendered.encode("utf-8", "replace")) > remaining:
            rendered = _OVERSIZED_MCP_EXCEPTION_DIAGNOSTIC
            safe_value = rendered
        rendered_size = len(rendered.encode("utf-8", "replace"))
        if rendered_size > remaining:
            break
        safe_args.append(safe_value)
        remaining -= rendered_size
    if len(raw_args) > len(safe_args) or not source_complete:
        omitted_size = len(_OMITTED_MCP_EXCEPTION_DIAGNOSTICS.encode("utf-8"))
        if omitted_size <= remaining:
            safe_args.append(_OMITTED_MCP_EXCEPTION_DIAGNOSTICS)
    return tuple(safe_args)


def _copy_credential_safe_mcp_failure_evidence(
    source: BaseException,
    target: BaseException,
    *,
    redactor: SecretRedactor,
    context: str,
    max_message_bytes: int = 4096,
    preserve_cause: bool,
) -> None:
    copy_mcp_failure_handoffs(source, target)
    if not preserve_cause:
        return
    cause = exception_cause(source)
    if cause is None or cause is source:
        return
    safe_cause = credential_safe_mcp_transport_failure(
        cause,
        redactor=redactor,
        context=f"{context} diagnostic",
        max_message_bytes=max_message_bytes,
    )
    set_exception_cause(target, safe_cause)


def _credential_safe_exception_message(
    error: BaseException,
    *,
    redactor: SecretRedactor,
    max_bytes: int,
) -> str:
    try:
        args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return ""
    if type(args) is not tuple:
        return ""
    parts: list[str] = []
    retained_chars = 0
    for value in args[:_MAX_MCP_EXCEPTION_DIAGNOSTIC_ARGS]:
        rendered = _credential_safe_exception_argument(
            value,
            redactor=redactor,
            max_bytes=max_bytes,
        )
        added_chars = len(rendered) + (2 if parts else 0)
        if retained_chars + added_chars > max_bytes:
            break
        parts.append(rendered)
        retained_chars += added_chars
    if len(args) > len(parts):
        added_chars = len(_OMITTED_MCP_EXCEPTION_DIAGNOSTICS) + (2 if parts else 0)
        if retained_chars + added_chars <= max_bytes:
            parts.append(_OMITTED_MCP_EXCEPTION_DIAGNOSTICS)
    if not parts:
        return ""
    return redactor.redact_text_bounded(
        ", ".join(parts),
        max_bytes=max_bytes,
    )


def _credential_safe_exception_argument(
    value: object,
    *,
    redactor: SecretRedactor,
    max_bytes: int,
) -> str:
    if type(value) is str:
        if not value:
            return ""
        source_char_limit = max_bytes * _MAX_MCP_EXCEPTION_STRING_SOURCE_FACTOR
        source_complete = len(value) <= source_char_limit
        source = value if source_complete else value[:source_char_limit]
        projected, _ = redactor.redact_utf8_head(
            source.encode("utf-8", "replace"),
            max_bytes=max_bytes,
            source_complete=source_complete,
        )
        return projected or _OVERSIZED_MCP_EXCEPTION_DIAGNOSTIC
    if value is None or type(value) in {bool, float}:
        rendered = str(value)
    elif type(value) is int:
        # CPython may reject decimal conversion of a comparatively small bigint.
        # Avoid conversion when its bit length already proves the diagnostic cannot
        # fit, and retain the fallback for interpreter-configured lower thresholds.
        if int.bit_length(value) > max_bytes * 3:
            return _OVERSIZED_MCP_EXCEPTION_DIAGNOSTIC
        try:
            rendered = str(value)
        except (OverflowError, ValueError):
            return _OVERSIZED_MCP_EXCEPTION_DIAGNOSTIC
    else:
        return "<non-text diagnostic>"
    if len(rendered.encode("utf-8", "replace")) > max_bytes:
        return _OVERSIZED_MCP_EXCEPTION_DIAGNOSTIC
    return redactor.redact_text_bounded(rendered, max_bytes=max_bytes)


def _credential_safe_mcp_exception_argument_value(
    value: object,
    *,
    redactor: SecretRedactor,
    max_bytes: int,
    non_text_diagnostic: str = "<non-text diagnostic>",
) -> Any:
    """Detach one public exception argument while retaining safe primitive types."""

    if type(value) not in {str, bool, int, float} and value is not None:
        rendered = redactor.redact_text_bounded(non_text_diagnostic, max_bytes=max_bytes)
    else:
        rendered = _credential_safe_exception_argument(
            value,
            redactor=redactor,
            max_bytes=max_bytes,
        )
    if value is None or type(value) in {bool, int, float}:
        if rendered == _OVERSIZED_MCP_EXCEPTION_DIAGNOSTIC:
            return rendered
        try:
            original = str(value)
        except (OverflowError, ValueError):
            return rendered
        if rendered == original:
            return value
    return rendered


def _credential_safe_exception_type_name(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> str:
    try:
        name = type.__getattribute__(type(error), "__name__")
    except BaseException:
        name = "Exception"
    if type(name) is not str:
        name = "Exception"
    return redactor.redact_text_bounded(name, max_bytes=256)


def copy_mcp_transport_limits(limits: McpTransportLimits) -> McpTransportLimits:
    """Revalidate a public limits object before retaining it."""

    if type(limits) is not McpTransportLimits:
        raise TypeError("transport_limits must be an McpTransportLimits.")
    return McpTransportLimits.model_validate(limits.model_dump(mode="python", warnings=False))


def resolve_mcp_transport_limits(
    limits: McpTransportLimits | None,
    *,
    legacy_timeout_s: float | None,
    default_timeout_s: float,
    legacy_field_name: str,
) -> McpTransportLimits:
    """Resolve an explicit limits object or a compatible legacy timeout."""

    if limits is not None:
        if legacy_timeout_s is not None:
            raise ValueError(f"{legacy_field_name} cannot be combined with transport_limits.")
        return copy_mcp_transport_limits(limits)
    timeout_s = default_timeout_s
    if legacy_timeout_s is not None:
        if type(legacy_timeout_s) not in {float, int}:
            raise TypeError(f"{legacy_field_name} must be a number.")
        try:
            timeout_s = float(legacy_timeout_s)
        except OverflowError:
            raise ValueError(f"{legacy_field_name} must be finite and greater than zero.") from None
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(f"{legacy_field_name} must be finite and greater than zero.")
    return McpTransportLimits(
        idle_timeout_s=timeout_s,
        total_call_timeout_s=timeout_s,
    )
