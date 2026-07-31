from __future__ import annotations

from cayu.runtime import InMemorySessionStore, SessionListResult, SessionQuery


class RecordingListSessionsStore(InMemorySessionStore):
    """In-memory store that retains defensive copies of session-list queries."""

    def __init__(self) -> None:
        super().__init__()
        self.session_queries: list[SessionQuery] = []

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        copied_query = SessionQuery() if query is None else query.model_copy(deep=True)
        self.session_queries.append(copied_query)
        return await super().list_sessions(query)
