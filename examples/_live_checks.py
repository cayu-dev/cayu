from __future__ import annotations

from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def require_exec_success(result: Any, *, stdout: str | None = None, label: str = "command") -> None:
    require(
        result.exit_code == 0, f"{label} failed: exit={result.exit_code} stderr={result.stderr!r}"
    )
    require(result.timed_out is False, f"{label} unexpectedly timed out")
    if stdout is not None:
        require_equal(result.stdout, stdout, f"{label} stdout")


def require_cleanup_artifact(
    artifacts: list[dict[str, Any]],
    *,
    adapter: str,
    action: str,
    status: str = "completed",
) -> None:
    require(bool(artifacts), f"missing {adapter} cleanup artifact")
    artifact = artifacts[0]
    require_equal(artifact.get("type"), "cayu.runner_cleanup.v1", "cleanup artifact type")
    require_equal(artifact.get("adapter"), adapter, "cleanup artifact adapter")
    require_equal(artifact.get("action"), action, "cleanup artifact action")
    require_equal(artifact.get("status"), status, "cleanup artifact status")
