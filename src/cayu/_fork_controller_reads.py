"""Host-enforced reads behind ``ForkControllerAccess``.

Every read runs under the controller's ``_ForkControllerReadBounds``, which the
store evaluates in SQL or as a predicate. Public contracts and docstrings live
on ``ForkControllerAccess``; this module only implements them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from cayu._fork_controller_errors import ForkControllerDenied
from cayu._resource_access_errors import ResourceAccessDenied
from cayu.sessions import _access_rules as session_access_rules
from cayu.sessions import _correlation as correlation
from cayu.sessions._fork_controller_bounds import _ForkControllerReadBounds
from cayu.sessions.event_queries import EventOrder, EventQuery, copy_event_query
from cayu.sessions.fork_overlays import fork_controller_agent_name
from cayu.sessions.queries import SessionListResult, SessionQuery, copy_session_query
from cayu.sessions.records import EventRecord, Session
from cayu.sessions.supervision_events import (
    SupervisionEvent,
    SupervisionEventPage,
    SupervisionEventQuery,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.fork_controllers import ForkController
    from cayu.runtime.public_authority import PublicAuthorityAliasCodec

FORK_CONTROLLER_READ_DEFAULT_BYTES = 1_048_576
FORK_CONTROLLER_READ_MAX_BYTES = 4_194_304
# Event-query fields a controller may set. Every other field carries host
# authority (budgets, workflows) or private linkage a controller never holds.
FORK_CONTROLLER_EVENT_QUERY_FIELDS = frozenset(
    {
        "session_id",
        "session_ids",
        "event_type",
        "event_types",
        "exclude_event_types",
        "agent_name",
        "environment_name",
        "tool_name",
        "since",
        "until",
        "after_sequence",
        "before_sequence",
        "limit",
        "order_by",
    }
)
UNREADABLE = "The session is not readable by this controller."
_T = TypeVar("_T")


def _single_session(query: EventQuery) -> bool:
    return query.session_id is not None or len(query.session_ids) == 1


class ForkControllerReads:
    """Reads for one verified controller; built only by ``ForkControllerAccess``."""

    __slots__ = ("_app", "_controller")

    def __init__(self, app: CayuApp, controller: ForkController) -> None:
        self._app = app
        self._controller = controller

    def _bounds(self, *, exclude_forked: bool = False) -> _ForkControllerReadBounds:
        """Build the host-enforced read limit, or fail closed when the store cannot apply it."""

        store = self._app.session_store
        if type(store).__dict__.get("supports_fork_controller_reads") is not True:
            raise NotImplementedError("The session store does not support fork-controller reads.")
        # Bounds do not compose: never replace another reader's bound with this one.
        if session_access_rules._query_bounds.get() is not None:
            raise ForkControllerDenied("Controller reads cannot run inside another read bound.")
        controller = self._controller
        names = {
            *controller.agents,
            *(fork_controller_agent_name(agent, controller.name) for agent in controller.agents),
        }
        return _ForkControllerReadBounds(
            controller=controller.name,
            agent_names=tuple(sorted(names)),
            exclude_forked=exclude_forked,
        )

    @staticmethod
    async def _bounded(
        bounds: _ForkControllerReadBounds, operation: Callable[[], Awaitable[_T]]
    ) -> _T:
        token = session_access_rules._query_bounds.set(bounds)
        try:
            return await operation()
        finally:
            session_access_rules._query_bounds.reset(token)

    async def _private_session_id(self, session_id: str) -> str | None:
        """Resolve a public or private session id; an unresolvable one is unreadable."""

        try:
            return await self._app._resolve_public_session_id(session_id)
        except ValueError:
            return None

    def _codec(self) -> PublicAuthorityAliasCodec | None:
        # Without a configured keyring there is no stable key, so correlation
        # values are withheld rather than derived from a process-local one.
        return self._app._public_authority_alias_codec

    def event_correlation_ids(self, record: EventRecord) -> dict[str, str]:
        if type(record) is not EventRecord:
            raise TypeError("record must be an EventRecord.")
        codec = self._codec()
        return {} if codec is None else correlation.event_correlation_ids(record.event, codec)

    def supervision_correlation_ids(
        self, event: SupervisionEvent
    ) -> dict[str, str | list[str] | None]:
        if type(event) is not SupervisionEvent:
            raise TypeError("event must be a SupervisionEvent.")
        codec = self._codec()
        if codec is None:
            return {}
        return correlation.supervision_correlation_ids(
            codec,
            session_id=event.session_id,
            tool_round_id=event.tool_round_id,
            tool_call_ids=event.tool_call_ids,
            model_step_id=event.model_step_id,
            model_attempt_id=event.model_attempt_id,
        )

    async def sessions(self, query: SessionQuery | None) -> SessionListResult:
        query = copy_session_query(query)
        if query.causal_budget_id is not None:
            raise ValueError("Controller session reads cannot filter by causal_budget_id.")
        bounds = self._bounds()
        return await self._app.session_store._access_list_sessions(bounds, query)

    async def session(self, session_id: str) -> Session:
        bounds = self._bounds()
        private_id = await self._private_session_id(session_id)
        if private_id is None:
            raise ForkControllerDenied(UNREADABLE)
        try:
            return await self._app.session_store._access_load_session(bounds, private_id)
        except ResourceAccessDenied:
            raise ForkControllerDenied(UNREADABLE) from None

    async def _prepared_event_query(
        self, query: EventQuery, *, exclude_forked_sessions: bool, max_bytes: int
    ) -> tuple[_ForkControllerReadBounds, EventQuery] | None:
        """Validate, bound and resolve one controller event query; ``None`` reads nothing.

        A query across sessions always reads in ascending order from an
        ``after_sequence`` (0 when omitted), so a PostgreSQL store applies its
        commit-visibility cutoff to the first page as well as to later ones.
        """

        if type(query) is not EventQuery:
            raise TypeError("Controller event reads require an EventQuery.")
        if type(exclude_forked_sessions) is not bool:
            raise TypeError("exclude_forked_sessions must be a bool.")
        if type(max_bytes) is not int or not 1 <= max_bytes <= FORK_CONTROLLER_READ_MAX_BYTES:
            raise ValueError(
                f"Event reads require a byte limit between 1 and {FORK_CONTROLLER_READ_MAX_BYTES}."
            )
        query = copy_event_query(query)
        refused = sorted(
            set(query.model_dump(exclude_defaults=True)) - FORK_CONTROLLER_EVENT_QUERY_FIELDS
        )
        if refused:
            raise ValueError("Controller event queries cannot set: " + ", ".join(refused) + ".")
        update: dict[str, object] = {}
        if not _single_session(query):
            if query.order_by is not EventOrder.SEQUENCE_ASC:
                raise ValueError(
                    "Controller event queries across sessions read in ascending sequence order."
                )
            if query.after_sequence is None:
                update["after_sequence"] = 0
        bounds = self._bounds(exclude_forked=exclude_forked_sessions)
        if query.session_id is not None:
            private_id = await self._private_session_id(query.session_id)
            if private_id is None:
                return None
            update["session_id"] = private_id
        if query.session_ids:
            resolved = [await self._private_session_id(value) for value in query.session_ids]
            private_ids = tuple(dict.fromkeys(value for value in resolved if value is not None))
            if not private_ids:
                return None
            update["session_ids"] = private_ids
        return bounds, copy_event_query(query, update=update) if update else query

    async def query_events(
        self, query: EventQuery, *, exclude_forked_sessions: bool, max_bytes: int
    ) -> list[EventRecord]:
        prepared = await self._prepared_event_query(
            query, exclude_forked_sessions=exclude_forked_sessions, max_bytes=max_bytes
        )
        if prepared is None:
            return []
        bounds, query = prepared
        store = self._app.session_store
        return await self._bounded(
            bounds, lambda: store.query_events_bounded(query, max_bytes=max_bytes)
        )

    async def query_event_page(
        self, query: EventQuery, *, exclude_forked_sessions: bool, max_bytes: int
    ) -> tuple[list[EventRecord], bool]:
        """Read up to ``query.limit`` records under ``max_bytes``, then test for more.

        Only the returned records count against the byte limit. Whether more
        records follow is an existence check past the last one, so a next record
        too large to fit with this page never makes this page fail.
        """

        prepared = await self._prepared_event_query(
            query, exclude_forked_sessions=exclude_forked_sessions, max_bytes=max_bytes
        )
        if prepared is None:
            return [], False
        bounds, query = prepared
        store = self._app.session_store

        async def read() -> tuple[list[EventRecord], bool]:
            records = await store.query_events_bounded(query, max_bytes=max_bytes)
            if len(records) < query.limit:
                return records, False
            past_last = (
                {"after_sequence": records[-1].sequence}
                if query.order_by is EventOrder.SEQUENCE_ASC
                else {"before_sequence": records[-1].sequence}
            )
            return records, await store.event_exists(
                copy_event_query(query, update={**past_last, "limit": 1})
            )

        return await self._bounded(bounds, read)

    async def supervision_page_with_anchors(
        self, query: SupervisionEventQuery
    ) -> tuple[SupervisionEventPage, dict[int, EventRecord]]:
        from cayu.sessions._supervision_projection import supervision_page_with_anchors

        if type(query) is not SupervisionEventQuery:
            raise TypeError("Supervision event reads require a SupervisionEventQuery.")
        query = SupervisionEventQuery.model_validate(query.model_dump(mode="python"))
        if query.sessions.causal_budget_id is not None:
            raise ValueError("Controller supervision reads cannot filter by causal_budget_id.")
        bounds = self._bounds()
        store = self._app.session_store
        if not store.supports_supervision_events:
            raise NotImplementedError("The session store does not support supervision events.")
        return await self._bounded(
            bounds,
            lambda: supervision_page_with_anchors(store, query, include_content=False),
        )
