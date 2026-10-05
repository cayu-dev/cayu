"""Attach admitted external-wait work to the existing application drain owner."""

import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any

from cayu.runtime.application_lifecycle import _admitted_entrance
from cayu.sessions._session_continuation import ContinuationUnavailable
from cayu.sessions.external_waits import ExternalWaitUnavailable

_TRACK: ContextVar[Callable[[asyncio.Task[Any]], None] | None] = ContextVar(
    "external_wait_observer", default=None
)


def external_wait_tracker():
    return _TRACK.get()


def external_wait_entrance(operation):
    @wraps(operation)
    async def observe(self, *args, **kwargs):
        token = _TRACK.set(self.app._request_coordinator.owners.track)
        try:
            return await operation(self, *args, **kwargs)
        except ContinuationUnavailable as error:
            # Observation may finish before the native owner. Keep its exact
            # responsibility pending; do not reclassify conflicts or cancellation.
            raise ExternalWaitUnavailable("External continuation remains pending.") from error
        finally:
            _TRACK.reset(token)

    return _admitted_entrance(observe)
