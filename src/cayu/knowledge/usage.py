"""Content-free observations for knowledge searches inside a runtime context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from cayu.events import Event, EventType

KnowledgeLookupObserver = Callable[[str, tuple[str, ...]], Awaitable[None]]
_LOOKUP_OBSERVER: ContextVar[KnowledgeLookupObserver | None] = ContextVar(
    "cayu_knowledge_lookup_observer", default=None
)


@contextmanager
def knowledge_lookup_scope(observer: KnowledgeLookupObserver | None) -> Iterator[None]:
    """Bind the runtime's observer; standalone store reads have no session to count."""
    token = _LOOKUP_OBSERVER.set(observer)
    try:
        yield
    finally:
        _LOOKUP_OBSERVER.reset(token)


async def record_knowledge_lookup(namespace: str, entry_ids: tuple[str, ...]) -> None:
    """Record each returned entry once per search, never query or document text."""
    observer = _LOOKUP_OBSERVER.get()
    if observer is not None and entry_ids:
        await observer(namespace, tuple(dict.fromkeys(entry_ids)))


def knowledge_usage_contribution(event: Event) -> tuple[str, tuple[str, ...]] | None:
    """One lookup's namespace and distinct returned IDs, shared with Cloud.

    Registration/access eligibility and timestamp aggregation belong to readers;
    this pure projection never includes query or document content.
    """
    if event.type != EventType.KNOWLEDGE_LOOKUP:
        return None
    namespace = event.payload.get("namespace")
    ids = event.payload.get("entry_ids")
    if not isinstance(namespace, str) or not isinstance(ids, list):
        return None
    return namespace, tuple(dict.fromkeys(value for value in ids if isinstance(value, str)))
