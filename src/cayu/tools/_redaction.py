"""Invocation-aware secret-redaction helpers shared by built-in tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from cayu.core.tools import ToolContext, ToolResult
from cayu.vaults import SecretRedactor

_EMPTY_REDACTOR = SecretRedactor()
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class InvocationRedactorSnapshot:
    """One atomic view of an invocation's evolving secret registry."""

    revision: int
    redactor: SecretRedactor

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 0:
            raise TypeError("Invocation redactor revision must be a non-negative integer.")
        if not isinstance(self.redactor, SecretRedactor):
            raise TypeError("Invocation redactor snapshot must contain a SecretRedactor.")


def resolve_secret_redactor(
    provider: Callable[[], Any] | None,
) -> SecretRedactor:
    """Resolve the current invocation registry without persisting its values."""

    if provider is None:
        return _EMPTY_REDACTOR
    redactor = provider()
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("Tool context secret redactor must return a SecretRedactor.")
    return redactor


def resolve_invocation_redactor_snapshot(
    provider: Callable[[], Any],
) -> InvocationRedactorSnapshot:
    """Resolve and validate one invocation registry snapshot."""

    if not callable(provider):
        raise TypeError("Invocation redactor snapshot provider must be callable.")
    snapshot = provider()
    if type(snapshot) is not InvocationRedactorSnapshot:
        raise TypeError(
            "Invocation redactor snapshot provider must return InvocationRedactorSnapshot."
        )
    return snapshot


def active_secret_redactor(ctx: ToolContext) -> SecretRedactor:
    """Return the redactor currently authoritative for this tool invocation."""

    return resolve_secret_redactor(ctx.invocation_secret_redactor)


def active_secret_redactor_snapshot(ctx: ToolContext) -> InvocationRedactorSnapshot:
    """Return the current revision when the runtime supplies one."""

    snapshot_provider = ctx.invocation_secret_snapshot_provider
    if snapshot_provider is not None:
        return resolve_invocation_redactor_snapshot(snapshot_provider)
    redactor_provider = ctx.invocation_secret_redactor
    if redactor_provider is None:
        return InvocationRedactorSnapshot(revision=0, redactor=_EMPTY_REDACTOR)
    value = redactor_provider()
    if isinstance(value, SecretRedactor):
        # Direct ToolContext callers historically supplied only a redactor.
        # Preserve that static contract; runtime contexts provide exact
        # revision evidence through InvocationRedactorSnapshot.
        return InvocationRedactorSnapshot(revision=0, redactor=value)
    raise TypeError("Tool context secret redactor must return a SecretRedactor.")


async def await_revision_stable_secret_output(
    ctx: ToolContext,
    operation: Callable[[SecretRedactor], Awaitable[_T]],
) -> tuple[_T, InvocationRedactorSnapshot] | None:
    """Retry one read-only capture when its secret registry changes."""

    if not callable(operation):
        raise TypeError("Secret output operation must be callable.")
    snapshot = active_secret_redactor_snapshot(ctx)
    for _ in range(2):
        value = await operation(snapshot.redactor)
        try:
            current = active_secret_redactor_snapshot(ctx)
        except BaseException:
            del value
            raise
        if current.revision == snapshot.revision and current.redactor.has_same_registry(
            snapshot.redactor
        ):
            return value, current
        del value
        snapshot = current
    return None


def record_ambiguous_secret_output(
    ctx: ToolContext,
    snapshot: InvocationRedactorSnapshot,
) -> None:
    """Bind an irreversible bounded result to its capture revision."""

    if type(snapshot) is not InvocationRedactorSnapshot:
        raise TypeError("Secret output capture requires InvocationRedactorSnapshot.")
    observer = ctx.invocation_secret_capture_observer
    if observer is not None:
        observer(snapshot.revision)


def unstable_secret_redaction_result() -> ToolResult:
    """Return a fixed fail-closed result after repeated registry changes."""

    return ToolResult(
        content=(
            "Tool output was omitted because its secret-redaction scope changed "
            "repeatedly during the read."
        ),
        structured={"error": "secret_redaction_scope_unstable"},
        is_error=True,
    )
