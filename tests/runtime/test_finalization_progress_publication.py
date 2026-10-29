"""Completion evidence survives a diagnostic progress publication failure."""

import asyncio

from tests.test_scout_finalization_progress import _scenario

from cayu.events import EventType
from cayu.runtime._environment_lifecycle import EnvironmentLifecycle


def test_progress_failure_preserves_completion_evidence(tmp_path, monkeypatch):
    results = []
    finalize = EnvironmentLifecycle.finalize_terminal_event

    async def capture_finalization(self, **kwargs):
        result = await finalize(self, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(EnvironmentLifecycle, "finalize_terminal_event", capture_finalization)
    asyncio.run(_scenario(tmp_path, monkeypatch))

    result = next(result for result in results if result.events)
    releases = [
        event
        for event in result.events
        if event.type == EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
        and event.payload["phase"] == "release"
    ]
    assert len(releases) == 1
    assert releases[0].payload["outcome"] == "released"
    completions = [
        event
        for event in result.events
        if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    ]
    assert len(completions) == 1
    assert {
        "final_snapshot",
        "source_publication_receipt",
        "final_git_receipt",
        "final_revision",
    } <= completions[0].payload.keys()
    assert completions[0].payload["final_snapshot"] is not None
    failures = result.event.payload["binding_finalize_publication_error"]["failures"]
    assert [failure["phase"] for failure in failures] == ["finalize_completed_progress"]
    assert failures[0]["error_type"] == "ConnectionError"
    assert "binding_finalize_error" not in result.event.payload
