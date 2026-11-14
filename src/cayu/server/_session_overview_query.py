"""Bounded public text/overview selection and query-bound continuations.

The inner cursor belongs to the SessionStore. This envelope detects accidental
query changes; it grants no authority and does not replace HTTP authorization.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu.sessions.overview import SessionOverviewStatus
from cayu.sessions.queries import SessionListResult, SessionQuery, encode_session_cursor
from cayu.sessions.records import Session
from cayu.sessions.store import SessionStore
from cayu.sessions.summary_rows import SessionOverviewFilter

_PREFIX = "sessionquery_"
# A 4096-byte opaque cursor can expand sixfold in JSON, then by 4/3 in base64.
_MAX_CURSOR_CHARS = 34_000
_INVALID = "Invalid session query cursor. Restart the session listing."
# Store pages one listing request reads at most while filling a page. Rows SQL
# cannot settle (unsettled summaries) and text matches are checked here.
_MAX_FILL_ROUNDS = 20
_MAX_SCAN_ROUNDS = 50
_SCAN_PAGE = 200

#: Reads the overviews of listed sessions: the rows kept and their overviews, in order.
OverviewReader = Callable[[list[Session]], Awaitable[tuple[list[Session], list[Any]]]]


@dataclass(frozen=True, slots=True)
class SessionOverviewSelection:
    states: frozenset[SessionOverviewStatus] = frozenset()
    include_eval: bool = True
    include_forks: bool = True
    text: str | None = None

    @property
    def active(self) -> bool:
        return self.text is not None or self.overview_active

    @property
    def overview_active(self) -> bool:
        return bool(self.states) or not self.include_eval or not self.include_forks

    async def read_page(
        self,
        store: SessionStore,
        query: SessionQuery,
        *,
        cursor: str | None,
        include: Sequence[str] | None,
        project_session: Callable[[Session], Mapping[str, Any]],
        project_title: Callable[[Session], str | None],
        overviews: OverviewReader | None = None,
    ) -> tuple[SessionListResult, str, list[Any] | None]:
        """One page of the selection, and its rows' overviews when ``overviews`` reads them.

        A store with summary rows narrows the list in SQL, and the page is filled
        from following store pages until it holds ``query.limit`` rows or the
        list ends, so only the last page is short. Without summary rows each
        store page is filtered on its own and may come back short or empty with
        a continuation, as before; the overviews are then ``None``.
        """

        if self.active and query.offset:
            raise ValueError("Text and overview filters require cursor paging without offset.")
        binding = self.binding(query, include) if self.active else ""
        native_query = SessionQuery.model_validate(
            {
                **query.model_dump(mode="python"),
                # Membership and continuation must not depend on hidden stored text.
                "q": None if self.text is not None else query.q,
                "cursor": self.unwrap(cursor, binding),
                "include_total_count": query.include_total_count and not self.active,
            }
        )
        if self.active and (overviews is not None or not self.overview_active):
            try:
                result, rows = await self._filled(
                    store, native_query, project_session, project_title, overviews
                )
            except NotImplementedError:
                pass
            else:
                return result, binding, rows
        result = await store.list_sessions(native_query)
        if self.text is not None:
            result = result.model_copy(
                update={
                    "sessions": [
                        session
                        for session in result.sessions
                        if self._matches_text(project_session(session), project_title(session))
                    ]
                }
            )
        if self.active:
            result = result.model_copy(update={"total_count": None})
        return result, binding, None

    async def _filled(
        self,
        store: SessionStore,
        query: SessionQuery,
        project_session: Callable[[Session], Mapping[str, Any]],
        project_title: Callable[[Session], str | None],
        overviews: OverviewReader | None,
    ) -> tuple[SessionListResult, list[Any] | None]:
        overview = SessionOverviewFilter(
            states=frozenset(str(state) for state in self.states),
            include_eval=self.include_eval,
            include_forks=self.include_forks,
        )
        # Text is matched on the public projection of every candidate, a larger page
        # at a time; SQL never narrows on it, so no position depends on hidden text.
        scanning = self.text is not None
        page_query = query.model_copy(
            update={
                "include_total_count": False,
                "limit": max(query.limit, _SCAN_PAGE) if scanning else query.limit,
            }
        )
        sessions: list[Session] = []
        rows: list[Any] = []
        for _round in range(_MAX_SCAN_ROUNDS if scanning else _MAX_FILL_ROUNDS):
            page = await store.list_session_overview_candidates(page_query, overview)
            visible = [
                session
                for session in page.sessions
                if self._matches_text(project_session(session), project_title(session))
            ]
            if overviews is not None and visible:
                kept, read = await overviews(visible)
            else:
                kept, read = visible, [None] * len(visible)
            for session, row in zip(kept, read, strict=True):
                sessions.append(session)
                rows.append(row)
                if len(sessions) == query.limit:
                    more = page.next_cursor is not None or session.id != page.sessions[-1].id
                    return (
                        SessionListResult(
                            sessions=sessions,
                            next_cursor=encode_session_cursor(session, query.order_by)
                            if more
                            else None,
                        ),
                        rows if overviews is not None else None,
                    )
            if page.next_cursor is None:
                break
            page_query = page_query.model_copy(update={"cursor": page.next_cursor})
        else:
            # Bounded work per request: return what was found with the continuation.
            return (
                SessionListResult(sessions=sessions, next_cursor=page.next_cursor),
                rows if overviews is not None else None,
            )
        return (
            SessionListResult(sessions=sessions, next_cursor=None),
            rows if overviews is not None else None,
        )

    def _matches_text(self, visible: Mapping[str, Any], title: str | None) -> bool:
        if self.text is None:
            return True
        values = [
            title,
            *(
                visible.get(field)
                for field in (
                    "id",
                    "agent_name",
                    "provider_name",
                    "model",
                    "environment_name",
                    "parent_session_id",
                    "causal_budget_id",
                )
            ),
        ]
        labels = visible.get("labels")
        if type(labels) is dict:
            values.extend(labels.keys())
            values.extend(labels.values())
        needle = self.text.casefold()
        return any(needle in value.casefold() for value in values if type(value) is str)

    def binding(self, query: SessionQuery, include: Sequence[str] | None) -> str:
        document = query.model_dump(
            mode="json", exclude={"cursor", "offset", "include_total_count"}
        )
        # Selectors are ANDed; states and each selector's values are sets.
        selectors = document["label_selectors"]
        for selector in selectors:
            selector["values"] = sorted(set(selector["values"]))
        document["label_selectors"] = sorted(
            {json.dumps(selector, sort_keys=True) for selector in selectors}
        )
        document["overview_state"] = sorted(self.states)
        document["include_eval"] = self.include_eval
        document["include_forks"] = self.include_forks
        document["include"] = sorted(set(include or []))
        return sha256(json.dumps(document, sort_keys=True, ensure_ascii=True).encode()).hexdigest()

    def unwrap(self, cursor: str | None, binding: str) -> str | None:
        if cursor is None:
            return None
        if not self.active:
            if cursor.startswith(_PREFIX):
                raise ValueError(_INVALID)
            return cursor
        try:
            if len(cursor) > _MAX_CURSOR_CHARS or not cursor.startswith(_PREFIX):
                raise ValueError
            encoded = cursor.removeprefix(_PREFIX).encode("ascii")
            raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
            if base64.urlsafe_b64encode(raw) != encoded:
                raise ValueError
            document = json.loads(raw)
            if (
                type(document) is not dict
                or set(document) != {"version", "query", "after"}
                or type(document["version"]) is not int
                or document["version"] != 1
                or document["query"] != binding
                or type(document["after"]) is not str
            ):
                raise ValueError
            # Apply the current store-cursor bounds without assuming its encoding.
            return SessionQuery(cursor=document["after"]).cursor
        except (ValueError, TypeError, UnicodeError, binascii.Error, RecursionError):
            raise ValueError(_INVALID) from None

    def wrap(self, cursor: str | None, binding: str) -> str | None:
        if cursor is None or not self.active:
            return cursor
        document = {"version": 1, "query": binding, "after": cursor}
        encoded = base64.urlsafe_b64encode(
            json.dumps(document, ensure_ascii=False).encode()
        ).decode()
        result = _PREFIX + encoded
        if len(result) > _MAX_CURSOR_CHARS:
            raise ValueError(_INVALID)
        return result
