from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS = 5.0
RUNNER_COMMAND_KILL_ATTEMPTS = 2
RUNNER_CLEANUP_ARTIFACT_TYPE = "cayu.runner_cleanup.v1"
RunnerCleanupPolicy = Literal["command", "sandbox", "none"]
DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY: RunnerCleanupPolicy = "command"
DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY: RunnerCleanupPolicy = "command"


@dataclass(frozen=True)
class RunnerCleanupResult:
    artifact: dict[str, Any]
    close_runner: bool


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
        artifact["error_type"] = type(error).__name__
        artifact["error"] = str(error)
    if error_message is not None:
        artifact["error"] = error_message
    return artifact
