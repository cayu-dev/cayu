"""Experimental, application-authorized session access.

These objects are trusted Python configuration, never HTTP credentials. The
handle intentionally is not a SessionStore and does not forward other methods.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from cayu._resource_access_binding import ResourceExecutionBinding
from cayu._validation import copy_label_map
from cayu.session_metadata import (
    SUMMARY_METADATA_KEY,
    SUMMARY_TONE_METADATA_KEY,
    SummaryTone,
    summary_metadata,
)
from cayu.sessions import _access_rules as session_access_rules
from cayu.sessions._access_rules import Action as Action
from cayu.sessions._access_rules import SessionAccessDenied as SessionAccessDenied
from cayu.sessions._access_rules import SessionAccessRule as SessionAccessRule
from cayu.sessions._access_rules import SessionAccessScope as SessionAccessScope
from cayu.sessions._access_rules import SessionAccessSelector as SessionAccessSelector
from cayu.sessions._access_rules import require_resource_session as require_resource_session
from cayu.sessions._access_rules import runtime_session_mutation as runtime_session_mutation
from cayu.sessions._access_rules import runtime_session_query as runtime_session_query
from cayu.sessions._access_rules import scoped_creation_binding as scoped_creation_binding
from cayu.sessions.queries import (
    SessionListResult,
    SessionQuery,
    copy_session_query,
)
from cayu.sessions.records import Session, SessionIdentity
from cayu.sessions.requests import RunRequest, copy_run_request
from cayu.sessions.store import SessionStore
from cayu.sessions.supervision_events import SupervisionEventPage, SupervisionEventQuery


class ScopedSessionAccess:
    """A narrow session handle constructed only by trusted application code.

    Resolve current policy before every operation. Resolver failures propagate
    without running a store operation. The admitted maximum never expands.
    This is not a runtime execution handle or a drop-in SessionStore adapter.
    """

    __slots__ = ("__admitted", "__binding", "__resolve", "__store")

    def __init__(
        self,
        store: SessionStore,
        *,
        admitted: SessionAccessScope,
        resolve: Callable[[], Awaitable[SessionAccessScope]],
        execution_binding: ResourceExecutionBinding | None = None,
    ) -> None:
        if (
            not isinstance(store, SessionStore)
            or type(store).__dict__.get("session_access_version") != 1
        ):
            raise NotImplementedError("Store does not support atomic scoped session access v1.")
        if type(admitted) is not SessionAccessScope or not callable(resolve):
            raise TypeError("Scoped access requires an explicit scope and trusted resolver.")
        self.__store = store
        self.__admitted = admitted
        self.__resolve = resolve
        self.__binding = execution_binding

    async def _bounds(self) -> session_access_rules._SessionAccessBounds:
        current = await self.__resolve()
        if type(current) is not SessionAccessScope:
            raise TypeError("Access resolver must return SessionAccessScope.")
        return session_access_rules._SessionAccessBounds(self.__admitted, current, self.__binding)

    async def load(self, session_id: str) -> Session:
        return await self.__store._access_load_session(await self._bounds(), session_id)

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        query = copy_session_query(query)
        return await self.__store._access_list_sessions(await self._bounds(), query)

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        labels = copy_label_map(labels, "labels", allow_reserved=False)
        return await self.__store._access_update_labels(await self._bounds(), session_id, labels)

    async def _query(self, operation):
        token = session_access_rules._query_bounds.set(await self._bounds())
        try:
            return await operation()
        finally:
            session_access_rules._query_bounds.reset(token)

    async def events(self, query=None, *, max_bytes: int = 1_048_576):
        from cayu.sessions.event_queries import copy_event_query

        query = copy_event_query(query)
        if type(max_bytes) is not int or not 1 <= max_bytes <= 4_194_304:
            raise ValueError("Event exports require a byte limit between 1 and 4194304.")
        return await self._query(
            lambda: self.__store.query_events_bounded(query, max_bytes=max_bytes)
        )

    async def usage(self, query=None, *, by_session: bool = False):
        from cayu.sessions.event_queries import copy_event_query

        query = copy_event_query(query)
        return await self._query(
            lambda: self.__store.read_usage_accounting(query, by_session=by_session)
        )

    async def costs(self, pricing, query=None, *, by_session: bool = False):
        from cayu.sessions.event_queries import copy_event_query

        query = copy_event_query(query)
        return await self._query(
            lambda: self.__store.read_cost_accounting(query, pricing, by_session=by_session)
        )

    async def supervision_events(self, query: SupervisionEventQuery) -> SupervisionEventPage:
        """Page settled supervision touches on admitted sessions; never includes content."""

        if type(query) is not SupervisionEventQuery:
            raise TypeError("Supervision event reads require a SupervisionEventQuery.")
        query = SupervisionEventQuery.model_validate(query.model_dump(mode="python"))
        return await self._query(lambda: self.__store.query_supervision_events(query))

    async def create(self, request: RunRequest, *, identity: SessionIdentity) -> Session:
        if type(request) is not RunRequest or type(identity) is not SessionIdentity:
            raise TypeError("Creation requires RunRequest and SessionIdentity.")
        request = copy_run_request(request)
        identity = identity.model_copy(deep=True)
        return await self.__store._access_create_session(await self._bounds(), request, identity)

    async def set_summary(
        self, session_id: str, summary: str, *, tone: SummaryTone = "neutral"
    ) -> Session:
        patch = summary_metadata(summary, tone)
        return await self.__store._access_set_summary(
            await self._bounds(),
            session_id,
            patch[SUMMARY_METADATA_KEY],
            patch[SUMMARY_TONE_METADATA_KEY],
        )

    async def update_metadata(self, session_id: str, metadata: dict) -> Session:
        from cayu.sessions.records import copy_session_user_metadata

        metadata = copy_session_user_metadata(metadata)
        return await self.__store._access_update_metadata(
            await self._bounds(), session_id, metadata
        )

    async def delete_session(self, session_id: str) -> None:
        await self.__store._access_delete_session(await self._bounds(), session_id)

    async def read_records(
        self,
        session_id: str,
        *,
        kind: Literal["events", "transcript", "checkpoint", "access_audit"],
        offset: int = 0,
        limit: int = 100,
        max_bytes: int = 1_048_576,
    ):
        return await self.__store._access_read_records(
            await self._bounds(), session_id, kind, offset, limit, max_bytes
        )
