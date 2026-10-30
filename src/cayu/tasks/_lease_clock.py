"""Whether a task store reports the clock it decides lease expiry by."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cayu.tasks.store import TaskStore

logger = logging.getLogger(__name__)
_STORES_WITHOUT_LEASE_CLOCK: set[tuple[type, str]] = set()


def reports_lease_clock(task_store: TaskStore, *, fallback: str) -> bool:
    """Whether ``task_store`` reports its lease clock; log ``fallback`` once per store class."""

    if task_store.supports_lease_clock:
        return True
    key = (type(task_store), fallback)
    if key not in _STORES_WITHOUT_LEASE_CLOCK:
        _STORES_WITHOUT_LEASE_CLOCK.add(key)
        logger.warning(
            "Task store %s doesn't report its lease clock (supports_lease_clock), so %s.",
            type(task_store).__qualname__,
            fallback,
        )
    return False
