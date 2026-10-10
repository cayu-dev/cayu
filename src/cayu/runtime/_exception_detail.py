"""Bounded, redacted exception detail for failed outcomes.

``f"{type(exc).__name__}: {exc}"`` keeps only an exception group's message and
child count. The detail here also keeps the group's leaf exceptions and the
cause chains, so a failure can be diagnosed without reproducing it. Traversal
uses the base-owned exception accessors, and every message is snapshotted
through :func:`exception_diagnostic`, so hostile exception classes cannot run
code or leak registered secrets through the summary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_group_children,
    exception_suppresses_context,
)
from cayu.runtime._diagnostics import MAX_DIAGNOSTIC_UTF8_BYTES, exception_diagnostic
from cayu.vaults import SecretRedactor

MAX_EXCEPTION_DETAIL_LEAVES = 8
MAX_EXCEPTION_DETAIL_CAUSES = 4
MAX_EXCEPTION_DETAIL_NODES = 256
MAX_EXCEPTION_DETAIL_MESSAGE_UTF8_BYTES = 512
MAX_EXCEPTION_SUMMARY_UTF8_BYTES = 8 * 1024
_UNRENDERABLE_MESSAGE = "message could not be rendered"


@dataclass(frozen=True, slots=True)
class ExceptionCause:
    """One exception in a cause chain: its type name and redacted message."""

    error_type: str
    message: str

    def __str__(self) -> str:
        return _render(self.error_type, self.message)


@dataclass(frozen=True, slots=True)
class ExceptionLeaf:
    """Leaf exceptions of a group that share one type and message.

    ``count`` is how many leaves had this type and message. ``causes`` is the
    bounded cause chain of the first of them.
    """

    error_type: str
    message: str
    count: int = 1
    causes: tuple[ExceptionCause, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "causes", tuple(self.causes))

    def __str__(self) -> str:
        notes = [f"{self.count} times"] if self.count > 1 else []
        if self.causes:
            notes.append(_caused_by(self.causes))
        text = _render(self.error_type, self.message)
        return f"{text} ({'; '.join(notes)})" if notes else text


@dataclass(frozen=True, slots=True)
class ExceptionDetail:
    """Bounded, redacted description of one exception.

    - ``error_type`` and ``message`` describe the exception itself.
    - ``causes`` is its ``__cause__``/``__context__`` chain, nearest first.
    - ``leaves`` lists an exception group's non-group descendants, grouped by
      type and message in first-seen order; it is empty for plain exceptions.
    - ``omitted_leaf_count`` counts visited leaves left out by the distinct-leaf bound.
    - ``traversal_truncated`` means unvisited occurrences remain; leaf counts
      then describe only the visited prefix, not the complete group.
    """

    error_type: str
    message: str
    causes: tuple[ExceptionCause, ...] = ()
    leaves: tuple[ExceptionLeaf, ...] = ()
    omitted_leaf_count: int = 0
    traversal_truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "causes", tuple(self.causes))
        object.__setattr__(self, "leaves", tuple(self.leaves))

    def __str__(self) -> str:
        return self.summary()

    def summary(self) -> str:
        """Return ``Type: message`` followed by the cause chain and leaf errors."""

        return _render(self.error_type, self.message) + exception_detail_suffix(self)


def exception_detail(
    error: BaseException,
    *,
    redactor: SecretRedactor | None = None,
) -> ExceptionDetail:
    """Describe ``error`` with its leaf exceptions and cause chains, bounded and redacted."""

    resolved_redactor = redactor or SecretRedactor()
    error_type, message = _snapshot(
        error,
        redactor=resolved_redactor,
        max_message_bytes=MAX_DIAGNOSTIC_UTF8_BYTES,
    )
    causes = _cause_chain(error, redactor=resolved_redactor)
    if not isinstance(error, BaseExceptionGroup):
        return ExceptionDetail(error_type=error_type, message=message, causes=causes)

    grouped: dict[tuple[str, str], ExceptionLeaf] = {}
    omitted = 0
    leaves, traversal_truncated = _group_leaves(error)
    for leaf in leaves:
        leaf_type, leaf_message = _snapshot(
            leaf,
            redactor=resolved_redactor,
            max_message_bytes=MAX_EXCEPTION_DETAIL_MESSAGE_UTF8_BYTES,
        )
        key = (leaf_type, leaf_message)
        existing = grouped.get(key)
        if existing is not None:
            grouped[key] = ExceptionLeaf(
                error_type=leaf_type,
                message=leaf_message,
                count=existing.count + 1,
                causes=existing.causes,
            )
        elif len(grouped) < MAX_EXCEPTION_DETAIL_LEAVES:
            grouped[key] = ExceptionLeaf(
                error_type=leaf_type,
                message=leaf_message,
                causes=_cause_chain(leaf, redactor=resolved_redactor),
            )
        else:
            omitted += 1
    return ExceptionDetail(
        error_type=error_type,
        message=message,
        causes=causes,
        leaves=tuple(grouped.values()),
        omitted_leaf_count=omitted,
        traversal_truncated=traversal_truncated,
    )


def exception_detail_suffix(detail: ExceptionDetail) -> str:
    """Return the cause-chain and leaf-error text appended after an error message."""

    text = ""
    if detail.causes:
        text += f"; {_caused_by(detail.causes)}"
    if detail.leaves:
        text += "; leaf errors: " + ", ".join(str(leaf) for leaf in detail.leaves)
        if detail.omitted_leaf_count:
            text += f", and {detail.omitted_leaf_count} more"
    if detail.traversal_truncated:
        text += "; exception traversal truncated (leaf counts are incomplete)"
    return text


def _render(error_type: str, message: str) -> str:
    return f"{error_type}: {message}" if message else error_type


def _caused_by(causes: tuple[ExceptionCause, ...]) -> str:
    return "; ".join(f"caused by {cause}" for cause in causes)


def _snapshot(
    error: BaseException,
    *,
    redactor: SecretRedactor,
    max_message_bytes: int,
) -> tuple[str, str]:
    diagnostic = exception_diagnostic(
        error,
        empty_message=_UNRENDERABLE_MESSAGE,
        preserve_empty_message=True,
        redactor=redactor,
        max_message_bytes=max_message_bytes,
    )
    message = diagnostic.message
    if message == f"{diagnostic.error_type}: {_UNRENDERABLE_MESSAGE}":
        message = _UNRENDERABLE_MESSAGE
    return diagnostic.error_type, message


def _cause_chain(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> tuple[ExceptionCause, ...]:
    causes: list[ExceptionCause] = []
    seen = {id(error)}
    current = error
    while len(causes) < MAX_EXCEPTION_DETAIL_CAUSES:
        following = exception_cause(current)
        if following is None and not exception_suppresses_context(current):
            following = exception_context(current)
        if following is None or id(following) in seen:
            break
        seen.add(id(following))
        error_type, message = _snapshot(
            following,
            redactor=redactor,
            max_message_bytes=MAX_EXCEPTION_DETAIL_MESSAGE_UTF8_BYTES,
        )
        causes.append(ExceptionCause(error_type=error_type, message=message))
        current = following
    return tuple(causes)


def _group_leaves(error: BaseExceptionGroup) -> tuple[list[BaseException], bool]:
    leaves: list[BaseException] = []
    pending: list[Iterator[BaseException]] = [iter((error,))]
    visited = 0
    while pending and visited < MAX_EXCEPTION_DETAIL_NODES:
        candidate = next(pending[-1], None)
        if candidate is None:
            pending.pop()
            continue
        # Shared leaves and subgroups represent repeated occurrences. The node
        # budget bounds traversal even when the same object appears repeatedly.
        visited += 1
        children = (
            exception_group_children(candidate, maximum=MAX_EXCEPTION_DETAIL_NODES - visited + 1)
            if isinstance(candidate, BaseExceptionGroup)
            else None
        )
        if children is None:
            if candidate is not error:
                leaves.append(candidate)
            continue
        # Keep at most the remaining budget plus one child: that extra child
        # proves truncation without copying or validating an unbounded fanout.
        pending.append(iter(children))
    return leaves, any(next(children, None) is not None for children in pending)
