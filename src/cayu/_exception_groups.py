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
_BASE_EXCEPTION_SUPPRESS_CONTEXT_DESCRIPTOR = BaseException.__dict__["__suppress_context__"]
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


def exception_suppresses_context(error: BaseException) -> bool:
    """Return whether the base-owned implicit context is hidden from diagnostics."""

    if not isinstance(error, BaseException):
        return False
    try:
        value = _BASE_EXCEPTION_SUPPRESS_CONTEXT_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return False
    return value is True


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


def set_exception_context(
    error: BaseException,
    context: BaseException | None,
) -> bool:
    """Set the base-owned implicit context without invoking subclass accessors."""

    if not isinstance(error, BaseException):
        return False
    if context is not None and not isinstance(context, BaseException):
        return False
    try:
        _BASE_EXCEPTION_CONTEXT_DESCRIPTOR.__set__(error, context)
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
    max_nodes: int | None = None,
    truncated_leaf_factory: Callable[[], BaseException] | None = None,
) -> BaseExceptionGroup:
    """Iteratively rebuild a group without retaining extension-owned nodes.

    ``BaseExceptionGroup.derive`` and subclass accessors are deliberately not
    used. A malformed group or a mapper failure is replaced with a fresh
    runtime-owned fallback so sanitization cannot become a new failure path.
    """

    if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 1):
        raise ValueError("max_nodes must be a positive integer or None.")

    def fresh_leaf(factory: Callable[[], BaseException]) -> BaseException:
        try:
            fallback = factory()
        except BaseException:
            return RuntimeError("Exception group sanitization failed")
        return (
            fallback
            if isinstance(fallback, BaseException)
            else RuntimeError("Exception group sanitization failed")
        )

    def fresh_invalid_leaf() -> BaseException:
        return fresh_leaf(invalid_leaf_factory)

    def fresh_truncated_leaf() -> BaseException:
        return fresh_leaf(truncated_leaf_factory or invalid_leaf_factory)

    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[tuple[BaseException, ...], bool]] = {}
    rebuilt: dict[int, BaseException] = {}
    remaining_nodes = None if max_nodes is None else max_nodes - 1
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
            children, truncated = children_by_group.pop(candidate_id, ((), False))
            detached_children: list[BaseException] = []
            for child in children:
                detached_child = rebuilt.get(id(child))
                detached_children.append(
                    detached_child if detached_child is not None else fresh_invalid_leaf()
                )
            if truncated:
                detached_children.append(fresh_truncated_leaf())
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
        selected_children = children
        truncated = False
        if remaining_nodes is not None:
            selected_count = min(len(children), remaining_nodes)
            selected_children = children[:selected_count]
            remaining_nodes -= selected_count
            truncated = selected_count < len(children)
        children_by_group[candidate_id] = (selected_children, truncated)
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(selected_children))

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
