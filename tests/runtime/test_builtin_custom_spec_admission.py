from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecutionAdmissionCandidate,
    ExecutionCapabilityEvidence,
    ExecutionExecutableEvidence,
    ExecutionToolRequirementEvidence,
    GitChangesTool,
    Message,
    ModelStreamEvent,
    Runner,
    RunRequest,
    ScriptedModelProvider,
    SearchTextTool,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolSpec,
)


def custom_spec(tool_type, requirements=()):
    return ToolSpec(
        name="custom_tool",
        description="Application description",
        input_schema=tool_type.spec.input_schema,
        execution_requirements=requirements,
    )


def executable_clause(name, executable):
    return ToolExecutionRequirement(
        name=name,
        alternatives=(ToolExecutableRequirement(executable=executable),),
    )


@pytest.mark.parametrize("tool_type,executable", [(SearchTextTool, "rg"), (GitChangesTool, "git")])
@pytest.mark.parametrize(
    "additional,missing",
    [(False, None), (False, "intrinsic"), (True, None), (True, "intrinsic"), (True, "additional")],
)
def test_custom_builtin_spec_public_admission(tool_type, executable, additional, missing):
    caller_requirements = (executable_clause("additional", "extra-program"),) if additional else ()
    supplied = custom_spec(tool_type, caller_requirements)
    tool = tool_type(spec=supplied)
    assert tool.name == supplied.name
    assert tool.description == supplied.description
    assert tool.schema == supplied.input_schema
    assert supplied.execution_requirements == caller_requirements

    class EvidenceRunner(Runner):
        default_cwd = "/workspace"

        def execution_admission_candidate(self):
            now = datetime.now(UTC)
            fingerprint = "sha256:" + "1" * 64
            absent = executable if missing == "intrinsic" else "extra-program" if missing else None
            return ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    unclaimed_reason_code="security_unclaimed",
                    environment_fingerprint=fingerprint,
                    tool_requirements=ExecutionToolRequirementEvidence(
                        environment_fingerprint=fingerprint,
                        executables=tuple(
                            ExecutionExecutableEvidence(
                                executable=name,
                                state="live_verified",
                                observed_at=now,
                                valid_until=now + timedelta(seconds=300),
                                requirement_fingerprint=ToolExecutableRequirement(
                                    executable=name
                                ).fingerprint,
                            )
                            for name in sorted((executable, "extra-program"))
                            if name != absent
                        ),
                    ),
                ),
            )

        async def exec(self, command, **kwargs):
            raise AssertionError("No tool dispatch expected")

    async def run():
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="hosted"), runner=EvidenceRunner()), default=True
        )
        app.register_agent(AgentSpec(name="agent", model="fake"), tools=[tool])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="custom_spec",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        persisted = await app.session_store.load_events("custom_spec")
        if missing is None:
            assert len(provider.requests) == 1, [
                event.payload for event in persisted if event.type is EventType.SESSION_FAILED
            ]
            assert any(event.type is EventType.SESSION_COMPLETED for event in persisted)
        else:
            assert provider.requests == []
            assert not any(
                event.type is EventType.SESSION_COMPLETED for event in (*events, *persisted)
            )
            failed = next(event for event in persisted if event.type is EventType.SESSION_FAILED)
            absent = executable if missing == "intrinsic" else "extra-program"
            assert any(
                refusal.get("tool_name") == "custom_tool" and refusal.get("executable") == absent
                for refusal in failed.payload["execution_admission"]["refusals"]
            )

    asyncio.run(run())


@pytest.mark.parametrize("tool_type", [SearchTextTool, GitChangesTool])
def test_custom_builtin_spec_cannot_remove_or_replace_intrinsic_clauses(tool_type):
    intrinsic = tool_type.spec.execution_requirements
    assert tool_type(spec=custom_spec(tool_type)).spec.execution_requirements == intrinsic
    assert (
        tool_type(spec=custom_spec(tool_type, intrinsic)).spec.execution_requirements == intrinsic
    )
    conflict = executable_clause(intrinsic[0].name, "different-program")
    with pytest.raises(ValueError, match="conflicts"):
        tool_type(spec=custom_spec(tool_type, (conflict,)))


@pytest.mark.parametrize("tool_type", [SearchTextTool, GitChangesTool])
@pytest.mark.parametrize("count", [31, 32])
def test_custom_builtin_spec_validates_merged_requirement_limit(tool_type, count):
    supplied = custom_spec(
        tool_type,
        tuple(
            executable_clause(f"extra_{index:02d}", f"program_{index}") for index in range(count)
        ),
    )
    if count == 32:
        with pytest.raises(ValueError):
            tool_type(spec=supplied)
    else:
        assert len(tool_type(spec=supplied).spec.execution_requirements) == 32
