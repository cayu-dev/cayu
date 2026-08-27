from __future__ import annotations

import asyncio

from examples.openai_client_tool_search.scenario import run_scenario


def test_openai_client_tool_search_runs_the_credential_free_native_vertical() -> None:
    report = asyncio.run(run_scenario())

    assert report == {
        "api_key_required": False,
        "executed_tool_names": ["remember_knowledge"],
        "loaded_tool_names": ["remember_knowledge"],
        "provider_requests": 3,
        "schema_version": 1,
        "session_completed": True,
        "top_level_tool_counts": [1, 1, 1],
    }
