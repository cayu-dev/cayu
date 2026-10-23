"""Content-free observations for knowledge searches inside a runtime context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

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
