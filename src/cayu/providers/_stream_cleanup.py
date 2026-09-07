"""Dispatch-scoped local HTTP settlement evidence, independent of remote outcome."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cayu.providers.deadlines import ProviderStreamDeadlineEvidence


class _LocalHttpCleanupObserver:
    def __init__(
        self, publish: Callable[[ProviderStreamDeadlineEvidence, bool], Awaitable[None]]
    ) -> None:
        self._publish = publish
        self._deadline: ProviderStreamDeadlineEvidence | None = None
        self._confirmed = False
        self._closed: bool | None = None
        self._reported = False

    def expired(self, evidence: ProviderStreamDeadlineEvidence) -> None:
        if self._deadline is None:
            self._deadline = evidence

    def confirm_expiry(self) -> Coroutine[Any, Any, None] | None:
        # A terminal result accepted after cancellation is not a failed attempt.
        self._confirmed = True
        return None if self._closed is None or self._reported else self._report()

    async def closed(self, *, succeeded: bool) -> None:
        self._closed = succeeded
        await self._report()

    async def _report(self) -> None:
        if self._deadline is None or not self._confirmed or self._closed is None or self._reported:
            return
        self._reported = True
        # The controller or existing retained HTTP close task owns publication.
        # No detached task or remote recovery authority is introduced.
        await self._publish(self._deadline, self._closed)


_local_http_cleanup_observer: ContextVar[_LocalHttpCleanupObserver | None] = ContextVar(
    "provider_local_http_cleanup_observer", default=None
)
