from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionRejected,
    ExecutionProfileComponentClass,
    ExecutionProfileMigrationRequired,
    ExecutionProfileMismatchError,
)


def test_duplicate_run_explains_continuation_without_admitting_work() -> None:
    async def scenario() -> None:
        provider = ScriptedModelProvider([[ModelStreamEvent.completed()]])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted"))
        request = RunRequest(
            agent_name="assistant", session_id="existing", messages=[Message.text("user", "hello")]
        )
        _ = [event async for event in app.run(request)]
        with pytest.raises(ValueError) as caught:
            _ = [event async for event in app.run(request)]
        message = str(caught.value)
        assert "app.run creates a new session" in message
        assert "app.resume(ResumeRequest(" in message
        assert "pending approval/input" in message
        assert len(provider.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error_type,meaning",
    [
        (ExecutionProfileMismatchError, "Start a new session"),
        (ExecutionProfileAdoptionRejected, "adoption was rejected"),
        (ExecutionProfileMigrationRequired, "requires an execution-profile migration"),
    ],
)
def test_unknown_evidence_and_hostile_objects_are_not_inspected(error_type, meaning) -> None:
    class Hostile:
        def __getattribute__(self, name):
            raise AssertionError("secret-bearing object inspected")

        def __repr__(self):
            raise AssertionError("secret-bearing object rendered")

    error = error_type(
        session_id="session",
        expected_profile_fingerprint="a" * 64,
        candidate_profile_fingerprint="b" * 64,
        changed_component_classes=(ExecutionProfileComponentClass.EXECUTION_POLICIES,),
        expected_profile=Hostile(),
        candidate_profile=Hostile(),
    )
    assert error.differences[0].category == "other_or_unknown"
    assert error.differences[0].component_class is ExecutionProfileComponentClass.EXECUTION_POLICIES
    assert meaning in str(error)
    assert "cayu guide durable-service-tools" in str(error)
    assert "Process-local" not in str(error)
