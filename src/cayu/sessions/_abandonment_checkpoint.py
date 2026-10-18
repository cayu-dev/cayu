"""Keep permanent abandonment authority outside ordinary checkpoint mutation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from cayu._validation import copy_durable_json_object

ABANDONED_SESSION_CHECKPOINT_KEY = "abandoned_session"

_PUBLICATION_ALLOWED: ContextVar[bool] = ContextVar(
    "abandonment_checkpoint_publication", default=False
)


@contextmanager
def abandonment_checkpoint_publication() -> Iterator[None]:
    token = _PUBLICATION_ALLOWED.set(True)
    try:
        yield
    finally:
        _PUBLICATION_ALLOWED.reset(token)


def preserve_abandonment_checkpoint(
    current: dict[str, Any] | None, replacement: dict[str, Any]
) -> None:
    key = ABANDONED_SESSION_CHECKPOINT_KEY
    if current is not None and key in current:
        # Once selected, even the abandonment owner cannot rewrite the reason
        # or the retained uncertain call identities.
        replacement[key] = copy_durable_json_object(current[key], key)
    elif not _PUBLICATION_ALLOWED.get():
        replacement.pop(key, None)
