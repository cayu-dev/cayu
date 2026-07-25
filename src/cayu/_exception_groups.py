"""Safe traversal primitives for exceptions crossing extension boundaries.

Exception groups and causal links are runtime-owned state on ``BaseException``.
Extension-defined exception subclasses can override the corresponding Python
attribute accessors and group methods, so boundary code must not dispatch
through those overrides while classifying failures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

_BASE_EXCEPTION_CAUSE_DESCRIPTOR = BaseException.__dict__["__cause__"]
_BASE_EXCEPTION_CONTEXT_DESCRIPTOR = BaseException.__dict__["__context__"]
_BASE_EXCEPTION_GROUP_EXCEPTIONS_DESCRIPTOR = BaseExceptionGroup.__dict__["exceptions"]


def exception_group_children(
    error: BaseExceptionGroup,
) -> tuple[BaseException, ...] | None:
    """Return base-owned group children without invoking subclass accessors."""

    try:
        children = _BASE_EXCEPTION_GROUP_EXCEPTIONS_DESCRIPTOR.__get__(
            error,
            BaseExceptionGroup,
        )
    except BaseException:
        return None
    if type(children) is not tuple or not all(
        isinstance(child, BaseException) for child in children
    ):
        return None
    return children


def exception_cause(error: BaseException) -> BaseException | None:
    """Return the base-owned explicit cause, failing closed on corrupt state."""

    if not isinstance(error, BaseException):
        return None
    try:
        value = _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return None
    return value if isinstance(value, BaseException) else None


def exception_context(error: BaseException) -> BaseException | None:
    """Return the base-owned implicit context, failing closed on corrupt state."""

    if not isinstance(error, BaseException):
        return None
    try:
        value = _BASE_EXCEPTION_CONTEXT_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return None
    return value if isinstance(value, BaseException) else None


def set_exception_cause(
    error: BaseException,
    cause: BaseException | None,
) -> bool:
    """Set the base-owned explicit cause without invoking subclass accessors."""

    if not isinstance(error, BaseException):
        return False
    if cause is not None and not isinstance(cause, BaseException):
        return False
    try:
        _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__set__(error, cause)
    except BaseException:
        return False
    return True


def exception_tree_contains(
    error: BaseException,
    exception_types: type[BaseException] | tuple[type[BaseException], ...],
) -> bool:
    """Return whether a safely traversable exception tree contains a type."""

    return any(isinstance(candidate, exception_types) for candidate in iter_exception_tree(error))


def iter_exception_tree(error: BaseException) -> Iterator[BaseException]:
    """Yield one exception tree without extension-controlled group dispatch."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        yield candidate
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(reversed(children))


def rebuild_exception_group(
    error: BaseExceptionGroup,
    *,
    group_message: str,
    leaf_mapper: Callable[[BaseException], BaseException],
    invalid_leaf_factory: Callable[[], BaseException],
) -> BaseExceptionGroup:
    """Iteratively rebuild a group without retaining extension-owned nodes.

    ``BaseExceptionGroup.derive`` and subclass accessors are deliberately not
    used. A malformed group or a mapper failure is replaced with a fresh
    runtime-owned fallback so sanitization cannot become a new failure path.
    """

    def fresh_invalid_leaf() -> BaseException:
        try:
            fallback = invalid_leaf_factory()
        except BaseException:
            return RuntimeError("Exception group sanitization failed")
        return (
            fallback
            if isinstance(fallback, BaseException)
            else RuntimeError("Exception group sanitization failed")
        )

    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    rebuilt: dict[int, BaseException] = {}
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in rebuilt:
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            try:
                mapped = leaf_mapper(candidate)
            except BaseException:
                mapped = fresh_invalid_leaf()
            rebuilt[candidate_id] = (
                mapped if isinstance(mapped, BaseException) else fresh_invalid_leaf()
            )
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            detached_children: list[BaseException] = []
            for child in children:
                detached_child = rebuilt.get(id(child))
                detached_children.append(
                    detached_child if detached_child is not None else fresh_invalid_leaf()
                )
            if not detached_children:
                detached_children = [fresh_invalid_leaf()]
            rebuilt[candidate_id] = BaseExceptionGroup(
                group_message,
                detached_children,
            )
            continue
        if candidate_id in children_by_group:
            rebuilt[candidate_id] = BaseExceptionGroup(
                group_message,
                [fresh_invalid_leaf()],
            )
            continue
        children = exception_group_children(candidate)
        if children is None:
            rebuilt[candidate_id] = BaseExceptionGroup(
                group_message,
                [fresh_invalid_leaf()],
            )
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))

    detached = rebuilt.get(id(error))
    if isinstance(detached, BaseExceptionGroup):
        return detached
    return BaseExceptionGroup(group_message, [fresh_invalid_leaf()])


def add_exception_note_safely(error: BaseException, note: str) -> bool:
    """Attach one runtime-owned note without invoking exception overrides."""

    if not isinstance(error, BaseException) or not isinstance(note, str):
        return False
    try:
        BaseException.add_note(error, note)
    except BaseException:
        return False
    return True
