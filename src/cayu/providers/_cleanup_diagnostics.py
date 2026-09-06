"""Closed, bounded evidence owned by the local stream-close boundary."""

from __future__ import annotations

import asyncio
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
}
_PROVIDER_CODES = {
    "openai": {"rate_limit_exceeded", "server_error", "insufficient_quota"},
    "anthropic": {"rate_limit_error", "overloaded_error", "api_error"},
}


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
    for name in _IDENTITY_FIELDS & fields.keys():
        if type(fields[name]) is not str or len(fields[name]) > EXECUTION_UNIT_ID_MAX_CHARS:
            raise ValueError("Provider cleanup identity is invalid.")
        require_execution_unit_id(fields[name], name)
    return dict(fields)
