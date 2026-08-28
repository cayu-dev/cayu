from __future__ import annotations

import asyncio

from examples.openai_hosted_tool_search.scenario import run_scenario


def test_openai_hosted_tool_search_runs_the_credential_free_vertical() -> None:
    report = asyncio.run(run_scenario())

    assert report == {
        "api_key_required": False,
        "deferred_candidate_names": ["remember_knowledge"],
        "discovery_view_revision": 1,
        "executed_tool_names": ["remember_knowledge"],
        "loaded_tool_names": ["remember_knowledge"],
        "parallel_tool_calls": False,
        "provider_requests": 2,
        "schema_version": 1,
        "session_completed": True,
    }
