"""Closed, bounded evidence owned by the local stream-close boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from cayu._exception_state import exception_state
from cayu._validation import EXECUTION_UNIT_ID_MAX_CHARS, require_execution_unit_id
from cayu.providers.base import ModelProviderError

# Exact classes only: extension class names and formatting hooks are untrusted.
_EXCEPTION_TYPES = (
    (RuntimeError, "RuntimeError"),
    (ValueError, "ValueError"),
    (OSError, "OSError"),
    (TimeoutError, "TimeoutError"),
    (ConnectionError, "ConnectionError"),
    (asyncio.CancelledError, "CancelledError"),
    (ModelProviderError, "ModelProviderError"),
    (httpx.ReadError, "ReadError"),
    (httpx.WriteError, "WriteError"),
    (httpx.CloseError, "CloseError"),
    (httpx.ConnectError, "ConnectError"),
    (httpx.ReadTimeout, "ReadTimeout"),
    (httpx.WriteTimeout, "WriteTimeout"),
    (httpx.ConnectTimeout, "ConnectTimeout"),
    (httpx.PoolTimeout, "PoolTimeout"),
    (httpx.TimeoutException, "TimeoutException"),
    (httpx.RemoteProtocolError, "RemoteProtocolError"),
)
_ENUM_FIELDS = {
    "cleanup_action": {"stream_close", "stream_close_lookup", "unknown"},
    "cleanup_reason": {
        "close_exception",
        "cleanup_timeout",
        "cleanup_cancelled",
        "cleanup_pending",
        "unknown_exception",
    },
    "cleanup_exception_type": {"unknown", *(name for _, name in _EXCEPTION_TYPES)},
    "stream_close_state": {"not_confirmed", "pending"},
    "remote_cancellation_state": {"unknown"},
    "remote_settlement_state": {"unknown"},
}
_REQUIRED_FIELDS = {"cleanup_diagnostic_version", "cancellation_requested", *_ENUM_FIELDS}
_IDENTITY_FIELDS = {"model_step_id", "model_attempt_id"}
_OPTIONAL_FIELDS = {
    *_IDENTITY_FIELDS,
    "cleanup_status_code",
    "cleanup_provider",
    "cleanup_error_code",
    "cleanup_exception_message",
    "cleanup_cause_type",
    "cleanup_cause_message",
    "cleanup_local_stack",
}
MAX_CLEANUP_DIAGNOSTIC_FIELDS = len(_REQUIRED_FIELDS | _OPTIONAL_FIELDS)
_PROVIDER_CODES = {
    "openai": {"rate_limit_exceeded", "server_error", "insufficient_quota"},
    "anthropic": {"rate_limit_error", "overloaded_error", "api_error"},
}


# Exception text is an untrusted payload. Only exact, content-free interpreter
# messages cross this boundary; arbitrary transport text is never exported.
_SAFE_MESSAGES = frozenset(
    {
        "aclose(): asynchronous generator is already running",
        "anext(): asynchronous generator is already running",
        "Cannot call send() once a close message has been sent.",
        "Event loop is closed",
    }
)
_STACK_FILES = frozenset(
    {
        "_credential_boundary.py",
        "_http.py",
        "_stream_lifecycle.py",
        "deadlines.py",
        "openai.py",
        "openai_subscription.py",
        "anthropic.py",
    }
)
_PROVIDER_ROOT = str(Path(__file__).parent)


def _safe_message(failure: BaseException) -> str:
    args = BaseException.__dict__["args"].__get__(failure, BaseException)
    if (
        type(args) is tuple
        and len(args) == 1
        and type(args[0]) is str
        and len(args[0]) <= 128
        and args[0] in _SAFE_MESSAGES
    ):
        return args[0]
    return "redacted"


def _local_exception_evidence(failure: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {"cleanup_exception_message": _safe_message(failure)}
    cause = BaseException.__dict__["__cause__"].__get__(failure, BaseException)
    if cause is None:
        cause = BaseException.__dict__["__context__"].__get__(failure, BaseException)
    if isinstance(cause, BaseException):
        result["cleanup_cause_type"] = next(
            (name for cls, name in _EXCEPTION_TYPES if type(cause) is cls), "unknown"
        )
        result["cleanup_cause_message"] = _safe_message(cause)
    tb = BaseException.__dict__["__traceback__"].__get__(failure, BaseException)
    frames = []
    for _ in range(32):
        if tb is None:
            break
        filename = tb.tb_frame.f_code.co_filename
        for name in _STACK_FILES:
            if filename == str(Path(_PROVIDER_ROOT) / name):
                frames.append([name, tb.tb_lineno])
                break
        tb = tb.tb_next
    if frames:
        result["cleanup_local_stack"] = json.dumps(frames[-8:], separators=(",", ":"))
    return result


def cleanup_diagnostics(
    failure: BaseException | None, *, unsettled: bool, action: str
) -> dict[str, Any]:
    """Describe observations, never infer remote cancellation from local close."""
    exception_type = next(
        (name for cls, name in _EXCEPTION_TYPES if type(failure) is cls), "unknown"
    )
    if exception_type == "CancelledError":
        reason = "cleanup_cancelled"
    elif exception_type in {
        "TimeoutError",
        "TimeoutException",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "PoolTimeout",
    }:
        reason = "cleanup_timeout"
    elif unsettled:
        reason = "cleanup_pending"
    elif exception_type == "unknown":
        reason = "unknown_exception"
    else:
        reason = "close_exception"
    result: dict[str, Any] = {
        "cleanup_diagnostic_version": 1,
        "cleanup_action": action,
        "cleanup_reason": reason,
        "cleanup_exception_type": exception_type,
        "cancellation_requested": True,
        "stream_close_state": "pending" if unsettled else "not_confirmed",
        "remote_cancellation_state": "unknown",
        "remote_settlement_state": "unknown",
    }
    if failure is not None:
        result.update(_local_exception_evidence(failure))
    if type(failure) is ModelProviderError:
        status = exception_state(failure, "status_code")
        if type(status) is int and 100 <= status <= 599:
            result["cleanup_status_code"] = status
        provider = exception_state(failure, "provider")
        code = exception_state(failure, "error_code")
        if type(provider) is str and provider in _PROVIDER_CODES:
            result["cleanup_provider"] = provider
            if type(code) is str and code in _PROVIDER_CODES[provider]:
                result["cleanup_error_code"] = code
    return result


def copy_cleanup_diagnostics(fields: dict[str, Any]) -> dict[str, Any]:
    """Validate flat primitive fields for persistence, recovery, and export."""
    if not fields.keys() >= _REQUIRED_FIELDS or fields.keys() - (
        _REQUIRED_FIELDS | _OPTIONAL_FIELDS
    ):
        raise ValueError("Provider cleanup diagnostic fields are invalid.")
    if (
        type(fields["cleanup_diagnostic_version"]) is not int
        or fields["cleanup_diagnostic_version"] != 1
    ):
        raise ValueError("Provider cleanup diagnostic version is invalid.")
    if fields["cancellation_requested"] is not True:
        raise ValueError("Provider cleanup cancellation state is invalid.")
    for name, allowed in _ENUM_FIELDS.items():
        value = fields[name]
        if type(value) is not str or value not in allowed:
            raise ValueError("Provider cleanup diagnostic classification is invalid.")
    if "cleanup_status_code" in fields:
        status = fields["cleanup_status_code"]
        if type(status) is not int or not 100 <= status <= 599:
            raise ValueError("Provider cleanup status is invalid.")
    provider = fields.get("cleanup_provider")
    if "cleanup_provider" in fields and (
        type(provider) is not str or provider not in _PROVIDER_CODES
    ):
        raise ValueError("Provider cleanup provider is invalid.")
    if "cleanup_error_code" in fields:
        code = fields["cleanup_error_code"]
        if type(code) is not str or provider is None or code not in _PROVIDER_CODES[provider]:
            raise ValueError("Provider cleanup code is invalid.")
    for name in ("cleanup_exception_message", "cleanup_cause_message"):
        if name in fields and (
            type(fields[name]) is not str or fields[name] not in {*_SAFE_MESSAGES, "redacted"}
        ):
            raise ValueError("Provider cleanup exception message is invalid.")
    if "cleanup_cause_type" in fields and (
        type(fields["cleanup_cause_type"]) is not str
        or fields["cleanup_cause_type"] not in _ENUM_FIELDS["cleanup_exception_type"]
    ):
        raise ValueError("Provider cleanup cause type is invalid.")
    if "cleanup_local_stack" in fields:
        value = fields["cleanup_local_stack"]
        if type(value) is not str or len(value) > 1024:
            raise ValueError("Provider cleanup stack is invalid.")
        try:
            frames = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Provider cleanup stack is invalid.") from exc
        if type(frames) is not list or not 1 <= len(frames) <= 8:
            raise ValueError("Provider cleanup stack is invalid.")
        for frame in frames:
            if (
                type(frame) is not list
                or len(frame) != 2
                or type(frame[0]) is not str
                or frame[0] not in _STACK_FILES
                or type(frame[1]) is not int
                or not 1 <= frame[1] <= 1_000_000
            ):
                raise ValueError("Provider cleanup stack frame is invalid.")
    for name in _IDENTITY_FIELDS & fields.keys():
        if type(fields[name]) is not str or len(fields[name]) > EXECUTION_UNIT_ID_MAX_CHARS:
            raise ValueError("Provider cleanup identity is invalid.")
        require_execution_unit_id(fields[name], name)
    return dict(fields)
