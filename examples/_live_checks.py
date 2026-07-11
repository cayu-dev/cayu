from __future__ import annotations

from collections.abc import Collection
from typing import Any

from cayu import Event, EventType

RUNTIME_FAILURE_EVENTS = frozenset(
    {
        EventType.MODEL_ERROR,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }
)
SESSION_TERMINAL_EVENTS = frozenset(
    {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def require_successful_terminal(
    events: list[Event],
    *,
    additional_failure_types: Collection[EventType] = (),
) -> None:
    failure_types = RUNTIME_FAILURE_EVENTS.union(additional_failure_types)
    for event in events:
        if event.type in failure_types:
            raise RuntimeError(f"runtime emitted {event.type}: {event.payload!r}")
    terminal_types = [event.type for event in events if event.type in SESSION_TERMINAL_EVENTS]
    require(
        terminal_types == [EventType.SESSION_COMPLETED],
        f"expected exactly one session.completed terminal, got {terminal_types!r}",
    )


def require_positive_model_usage(events: list[Event]) -> list[Event]:
    completed_events = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    require(bool(completed_events), "runtime did not emit model.completed")
    for event in completed_events:
        usage = event.payload.get("usage_metrics")
        if not isinstance(usage, dict):
            raise RuntimeError(f"model.completed missing normalized usage: {event.payload!r}")
        total_tokens = usage.get("total_tokens")
        require(
            isinstance(total_tokens, int) and total_tokens > 0,
            f"model.completed total_tokens is invalid: {usage!r}",
        )
    return completed_events


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
