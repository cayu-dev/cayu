"""Tool-call proposal sessions can be left out of session lists and counts."""

from __future__ import annotations

import asyncio

from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)

from cayu import RunRequest
from cayu.approvals.proposals import tool_call_proposal_request
from cayu.sessions.queries import (
    TOOL_PROPOSAL_PROVIDER_NAME,
    SessionAggregateFilter,
    SessionQuery,
    session_query_from_aggregate_filter,
)
from cayu.sessions.records import SessionIdentity

__all__ = ["conformance_postgres_dsn", "session_store_case"]


async def _create(store, session_id: str, provider_name: str) -> None:
    await store.create(
        RunRequest(session_id=session_id, agent_name="writer", messages=[]),
        identity=SessionIdentity(provider_name=provider_name, model="model"),
    )


def test_lists_leave_out_tool_proposals_only_when_asked(session_store_case) -> None:
    async def scenario() -> None:
        store = await _open_store(session_store_case)
        try:
            await _create(store, "run-1", "openai")
            await _create(store, "proposal-1", TOOL_PROPOSAL_PROVIDER_NAME)
            await _create(store, "run-2", "anthropic")
            every = await store.list_sessions(SessionQuery(include_total_count=True))
            assert {s.id for s in every.sessions} == {"run-1", "proposal-1", "run-2"}
            runs = await store.list_sessions(
                SessionQuery(include_tool_proposals=False, include_total_count=True)
            )
            assert {s.id for s in runs.sessions} == {"run-1", "run-2"}
            assert runs.total_count == 2
            # Counts and rollups take the same filter through the aggregate filter.
            query = session_query_from_aggregate_filter(
                SessionAggregateFilter(include_tool_proposals=False)
            )
            assert {s.id for s in (await store.list_sessions(query)).sessions} == {
                "run-1",
                "run-2",
            }
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def test_a_proposal_is_titled_with_its_own_title_argument() -> None:
    titled = tool_call_proposal_request(
        "issue_filer", "file_issue", {"title": "  Leaked lease\n on restart ", "body": "..."}
    )
    assert titled.title == "Proposed file_issue: Leaked lease on restart"
    untitled = tool_call_proposal_request("issue_filer", "file_issue", {"body": "..."})
    assert untitled.title == "Proposed file_issue"
    escaped = tool_call_proposal_request(
        "issue_filer", "file_issue", {"title": "Color \x1b[31mred\x1b[0m\x07"}
    )
    assert escaped.title == "Proposed file_issue: Color [31mred [0m"
    named = tool_call_proposal_request("issue_filer", "file_issue", {"title": "x"}, title="Mine")
    assert named.title == "Mine"
