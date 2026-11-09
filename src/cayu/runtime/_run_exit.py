"""Classify how a run, resume, recovery or task body ended.

Every exit owner asks this one table instead of ordering ``except`` clauses.
Ordered handlers miss shapes such as a ``BaseExceptionGroup`` that carries both
a delivered ``CancelledError`` and an ordinary failure, or a process-control
leaf beside caller cancellation. Classification never sanitizes or rewrites
the exception: owners settle first and then re-raise ``RunExit.error``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum

from cayu._exception_groups import exception_tree_leaves, iter_exception_tree

_CONTROL_FLOW_TYPES = (asyncio.CancelledError, GeneratorExit)


class RunExitKind(StrEnum):
    """The settlement-relevant kind of one body exit."""

    RETURNED = "returned"
    """The body returned. Its outcome is whatever it recorded."""

    FAILED = "failed"
    """An ordinary failure, or a group without delivered control."""

    CANCELLED = "cancelled"
    """Cancellation delivered to the owning task, alone or inside a group."""

    ABANDONED = "abandoned"
    """``GeneratorExit``: the consumer closed the stream at some yield."""

    PROCESS_CONTROL = "process"
    """A process-level stop anywhere in the tree.

    ``KeyboardInterrupt``, ``SystemExit``, or any other ``BaseException`` leaf
    that is neither an ``Exception`` nor cancellation or ``GeneratorExit``.
    The process is going away; its recovery owns the invocation.
    """

    FENCED = "fenced"
    """Authenticated run-fence loss. No durable write is legal."""


@dataclass(frozen=True, slots=True)
class RunExit:
    """One classified exit and the untouched evidence it was derived from."""

    kind: RunExitKind
    error: BaseException | None
    """The exact exception to re-raise after settlement, never sanitized."""

    cancellation: asyncio.CancelledError | None = None
    """The delivered cancellation, when the exit carries one."""

    supervisory: tuple[BaseException, ...] = ()
    """Untouched process-level leaves, in tree order."""

    leaves: tuple[BaseException, ...] = ()
    """A cancelled exit's non-group exceptions, in tree order."""


def delivered_cancellation(
    error: BaseException | None,
    *,
    cancellation_baseline: int = 0,
) -> asyncio.CancelledError | None:
    """Return caller cancellation carried by ``error``, if it was delivered.

    A plain ``CancelledError`` is always the owner's own control signal. Inside
    a group, a ``CancelledError`` leaf counts only when the current task has
    more pending cancellation requests than ``cancellation_baseline``; otherwise
    it is a child's internal cancellation and the group is a failure.
    """

    if isinstance(error, asyncio.CancelledError):
        return error
    if not isinstance(error, BaseExceptionGroup):
        return None
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is not None and task.cancelling() <= cancellation_baseline:
        return None
    for candidate in iter_exception_tree(error):
        if isinstance(candidate, asyncio.CancelledError):
            return candidate
    return None


def _carries_run_fence_loss(error: BaseException) -> bool:
    # Imported lazily: the marker's authority lives with interaction-transition
    # publication, which itself depends on modules that use this classifier.
    from cayu.runtime._session_finalization import _is_interaction_transition_run_fence

    return any(_is_interaction_transition_run_fence(node) for node in iter_exception_tree(error))


def classify_run_exit(
    error: BaseException | None,
    *,
    cancellation_baseline: int = 0,
) -> RunExit:
    """Classify one exit.

    Precedence: fence loss, then process control, then delivered cancellation,
    then ``GeneratorExit``, then any other failure, then a normal return.
    Groups are searched with ``iter_exception_tree``.
    """

    if error is None:
        return RunExit(RunExitKind.RETURNED, None)
    if _carries_run_fence_loss(error):
        return RunExit(RunExitKind.FENCED, error)
    leaves = exception_tree_leaves(error)
    supervisory = tuple(
        leaf
        for leaf in leaves
        if not isinstance(leaf, Exception) and not isinstance(leaf, _CONTROL_FLOW_TYPES)
    )
    cancellation = delivered_cancellation(error, cancellation_baseline=cancellation_baseline)
    if supervisory:
        return RunExit(
            RunExitKind.PROCESS_CONTROL,
            error,
            cancellation=cancellation,
            supervisory=supervisory,
        )
    if cancellation is not None:
        return RunExit(RunExitKind.CANCELLED, error, cancellation=cancellation, leaves=leaves)
    if any(isinstance(leaf, GeneratorExit) for leaf in leaves):
        return RunExit(RunExitKind.ABANDONED, error)
    return RunExit(RunExitKind.FAILED, error)


def grouped_caller_cancellation(
    error: BaseException,
    *,
    cancellation_baseline: int = 0,
) -> asyncio.CancelledError | None:
    """Return the caller's cancellation when a group settles exactly as it.

    Cleanup that receives the caller's cancellation and fails too raises the
    delivered ``CancelledError`` grouped with ordinary failures. That is still
    the caller cancelling the run: like a plain cancellation, it abandons the
    run, which recovery continues, unless an interrupt was requested. Every
    session exit handler asks this one rule. A group with ``GeneratorExit``,
    process control or fence loss, without an ordinary failure, or whose
    cancellation was not delivered settles as before.
    """

    if not isinstance(error, BaseExceptionGroup):
        return None
    run_exit = classify_run_exit(error, cancellation_baseline=cancellation_baseline)
    if run_exit.kind is not RunExitKind.CANCELLED:
        return None
    leaves = run_exit.leaves
    if any(isinstance(leaf, GeneratorExit) for leaf in leaves) or not any(
        isinstance(leaf, Exception) for leaf in leaves
    ):
        return None
    return run_exit.cancellation
