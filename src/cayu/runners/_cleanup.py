from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, cast

from cayu._exception_groups import set_exception_cause
from cayu._exception_state import exception_state, pop_exception_state, set_exception_state
from cayu.runners._diagnostics import (
    trusted_runner_error_type_name,
    trusted_runner_exception_type_name,
)

DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS = 5.0
RUNNER_COMMAND_KILL_ATTEMPTS = 2
RUNNER_CLEANUP_ARTIFACT_TYPE = "cayu.runner_cleanup.v1"
RunnerCleanupPolicy = Literal["command", "sandbox", "none"]
DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY: RunnerCleanupPolicy = "command"
DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY: RunnerCleanupPolicy = "command"
_KNOWN_CLEANUP_ADAPTERS = frozenset({"docker", "e2b", "lambda-microvm", "microsandbox"})
_KNOWN_CLEANUP_ACTIONS = frozenset({"kill_command", "kill_sandbox", "none"})
_KNOWN_CLEANUP_STATUSES = frozenset(
    {"completed", "deferred", "failed", "skipped", "timeout", "unsupported"}
)
_DEFERRED_CLEANUP_REASON = (
    "command handle is not available yet; cleanup will continue in background"
)
_RUNNER_CANCELLATION_FAILURE_ATTRIBUTE = "_cayu_runner_cancellation_failure"
_RUNNER_CANCELLATION_FAILURE_TOKEN = object()


@dataclass(frozen=True)
class RunnerCleanupResult:
    artifact: dict[str, Any]
    close_runner: bool


@dataclass(frozen=True, slots=True)
class _RunnerCancellationFailure:
    token: object
    failure: BaseException


def attach_runner_cancellation_failure(
    cancellation: asyncio.CancelledError,
    failure: BaseException,
) -> None:
    """Authenticate one sanitized runner failure carried by cancellation."""

    if not isinstance(cancellation, asyncio.CancelledError):
        raise TypeError("cancellation must be CancelledError.")
    if not isinstance(failure, BaseException):
        raise TypeError("failure must be BaseException.")
    if not set_exception_state(
        cancellation,
        _RUNNER_CANCELLATION_FAILURE_ATTRIBUTE,
        _RunnerCancellationFailure(
            token=_RUNNER_CANCELLATION_FAILURE_TOKEN,
            failure=failure,
        ),
    ):
        raise RuntimeError("Could not attach runner cancellation failure.")


def runner_cancellation_failure(
    cancellation: asyncio.CancelledError,
) -> BaseException | None:
    """Return only a failure authenticated by the runner boundary."""

    state = exception_state(cancellation, _RUNNER_CANCELLATION_FAILURE_ATTRIBUTE)
    if (
        type(state) is not _RunnerCancellationFailure
        or state.token is not _RUNNER_CANCELLATION_FAILURE_TOKEN
        or not isinstance(state.failure, BaseException)
    ):
        return None
    return state.failure


def pop_runner_cancellation_failure(
    cancellation: asyncio.CancelledError,
) -> BaseException | None:
    """Consume only a failure authenticated by the runner boundary."""

    state = pop_exception_state(
        cancellation,
        _RUNNER_CANCELLATION_FAILURE_ATTRIBUTE,
    )
    if (
        type(state) is not _RunnerCancellationFailure
        or state.token is not _RUNNER_CANCELLATION_FAILURE_TOKEN
        or not isinstance(state.failure, BaseException)
    ):
        return None
    return state.failure


def transfer_runner_cancellation_failures(
    cancellation: asyncio.CancelledError,
    child_cancellations: list[asyncio.CancelledError | None],
) -> None:
    """Carry ordered child cleanup failures across a TaskGroup cancellation."""

    failures: list[BaseException] = []
    existing = runner_cancellation_failure(cancellation)
    if existing is not None:
        failures.append(existing)
    for child_cancellation in child_cancellations:
        if child_cancellation is None:
            continue
        failure = runner_cancellation_failure(child_cancellation)
        if failure is None or any(candidate is failure for candidate in failures):
            continue
        failures.append(failure)
    if not failures:
        return
    combined = (
        failures[0]
        if len(failures) == 1
        else BaseExceptionGroup(
            "Parallel runner cleanup failures.",
            failures,
        )
    )
    attach_runner_cancellation_failure(cancellation, combined)
    set_exception_cause(cancellation, combined)


def validate_cancel_timeout(timeout_s: float | None) -> float:
    if timeout_s is None:
        return DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS
    if type(timeout_s) not in {int, float}:
        raise TypeError("Runner cancel_timeout_s must be numeric.")
    if not isfinite(timeout_s):
        raise ValueError("Runner cancel_timeout_s must be finite.")
    if timeout_s <= 0:
        raise ValueError("Runner cancel_timeout_s must be greater than zero.")
    return float(timeout_s)


def validate_runner_cleanup_policy(
    policy: RunnerCleanupPolicy,
    field_name: str,
) -> RunnerCleanupPolicy:
    if policy not in {"command", "sandbox", "none"}:
        raise ValueError(f"Runner {field_name} must be one of: command, sandbox, none.")
    return policy


async def cleanup_runner_command_with_diagnostic(
    sandbox: Any,
    *,
    handle: Any | None,
    adapter: str,
    timeout_s: float,
    policy: RunnerCleanupPolicy,
) -> RunnerCleanupResult:
    cleanup_policy = validate_runner_cleanup_policy(policy, "cleanup policy")
    if cleanup_policy == "sandbox":
        artifact = await _call_cleanup_target(
            sandbox,
            method_name="kill",
            adapter=adapter,
            action="kill_sandbox",
            timeout_s=timeout_s,
        )
        return RunnerCleanupResult(artifact=artifact, close_runner=True)

    if cleanup_policy == "command":
        if handle is None:
            artifact = _cleanup_artifact(
                adapter=adapter,
                action="kill_command",
                status="unsupported",
                timeout_s=timeout_s,
                error_message="command handle is not available",
            )
            return RunnerCleanupResult(artifact=artifact, close_runner=False)
        artifact = await _call_cleanup_target(
            handle,
            method_name="kill",
            adapter=adapter,
            action="kill_command",
            timeout_s=timeout_s,
        )
        return RunnerCleanupResult(artifact=artifact, close_runner=False)

    artifact = _cleanup_artifact(
        adapter=adapter,
        action="none",
        status="skipped",
        timeout_s=timeout_s,
    )
    return RunnerCleanupResult(artifact=artifact, close_runner=False)


async def kill_sandbox_with_diagnostic(
    sandbox: Any,
    *,
    adapter: str,
    timeout_s: float,
) -> dict[str, Any]:
    result = await cleanup_runner_command_with_diagnostic(
        sandbox,
        handle=None,
        adapter=adapter,
        timeout_s=timeout_s,
        policy="sandbox",
    )
    return result.artifact


async def _call_cleanup_target(
    target: Any,
    *,
    method_name: str,
    adapter: str,
    action: str,
    timeout_s: float,
) -> dict[str, Any]:
    cleanup = getattr(target, method_name, None)
    if cleanup is None:
        return _cleanup_artifact(
            adapter=adapter,
            action=action,
            status="unsupported",
            timeout_s=timeout_s,
        )
    try:
        result = await asyncio.wait_for(cleanup(), timeout=timeout_s)
    except TimeoutError:
        return _cleanup_artifact(
            adapter=adapter,
            action=action,
            status="timeout",
            timeout_s=timeout_s,
        )
    except Exception as exc:
        return _cleanup_artifact(
            adapter=adapter,
            action=action,
            status="failed",
            timeout_s=timeout_s,
            error=exc,
        )
    if result is False:
        return _cleanup_artifact(
            adapter=adapter,
            action=action,
            status="failed",
            timeout_s=timeout_s,
            error_message="kill returned false",
        )
    return _cleanup_artifact(
        adapter=adapter,
        action=action,
        status="completed",
        timeout_s=timeout_s,
    )


def _cleanup_artifact(
    *,
    adapter: str,
    action: str,
    status: str,
    timeout_s: float,
    error: Exception | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "type": RUNNER_CLEANUP_ARTIFACT_TYPE,
        "adapter": adapter,
        "action": action,
        "status": status,
        "timeout_s": timeout_s,
    }
    if error is not None:
        artifact["error_type"] = trusted_runner_exception_type_name(error)
    if error_message is not None:
        artifact["error"] = error_message
    return artifact


def sanitize_runner_artifacts(artifacts: object) -> list[dict[str, Any]]:
    """Project runner-owned artifacts onto fixed, typed public evidence."""

    if type(artifacts) is not list:
        return []
    sanitized: list[dict[str, Any]] = []
    for artifact in artifacts:
        safe = _sanitize_runner_artifact(artifact)
        if safe is not None:
            sanitized.append(safe)
    return sanitized


def sanitize_cancellation_runner_artifacts(cancellation: BaseException) -> None:
    """Sanitize attached runner evidence without replacing a cancellation."""

    try:
        namespace = BaseException.__dict__["__dict__"].__get__(
            cancellation,
            BaseException,
        )
    except BaseException:
        return
    if type(namespace) is not dict:
        return
    artifacts = dict.get(namespace, "artifacts")
    if artifacts is not None:
        dict.__setitem__(
            namespace,
            "artifacts",
            sanitize_runner_artifacts(artifacts),
        )


def _sanitize_runner_artifact(artifact: object) -> dict[str, Any] | None:
    if type(artifact) is not dict:
        return None
    artifact = cast("dict[str, Any]", artifact)
    artifact_type = artifact.get("type")
    if type(artifact_type) is not str or artifact_type != RUNNER_CLEANUP_ARTIFACT_TYPE:
        return None
    adapter = artifact.get("adapter")
    action = artifact.get("action")
    status = artifact.get("status")
    timeout_s = _positive_finite_float(artifact.get("timeout_s"))
    if (
        type(action) is not str
        or action not in _KNOWN_CLEANUP_ACTIONS
        or type(status) is not str
        or status not in _KNOWN_CLEANUP_STATUSES
        or timeout_s is None
    ):
        return None
    safe_adapter = (
        adapter if type(adapter) is str and adapter in _KNOWN_CLEANUP_ADAPTERS else "unknown"
    )
    safe: dict[str, Any] = {
        "type": RUNNER_CLEANUP_ARTIFACT_TYPE,
        "adapter": safe_adapter,
        "action": action,
        "status": status,
        "timeout_s": timeout_s,
    }
    error_type = artifact.get("error_type")
    trusted_error_type = trusted_runner_error_type_name(error_type)
    if trusted_error_type is not None:
        safe["error_type"] = trusted_error_type
    late_timeout = _positive_finite_float(artifact.get("late_start_cleanup_timeout_s"))
    if late_timeout is not None:
        safe["late_start_cleanup_timeout_s"] = late_timeout
    reason = artifact.get("reason")
    if type(reason) is str and reason == _DEFERRED_CLEANUP_REASON:
        safe["reason"] = _DEFERRED_CLEANUP_REASON
    error_message = artifact.get("error")
    if type(error_message) is str and error_message in {
        "command handle is not available",
        "kill returned false",
    }:
        safe["error"] = error_message
    return safe


def _positive_finite_float(value: object) -> float | None:
    if type(value) is int:
        try:
            number = float(value)
        except OverflowError:
            return None
    elif type(value) is float:
        number = value
    else:
        return None
    return number if isfinite(number) and number > 0 else None
