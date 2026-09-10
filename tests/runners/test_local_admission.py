from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecutionCapabilityClaim,
    ExecutionRequirements,
    ExecutionToolRequirement,
    LocalRunner,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolRunnerCapabilityRequirement,
    ToolSpec,
)


def _requirement(executable: str, *, process: bool = False):
    return ToolExecutionRequirement(
        name="program",
        alternatives=(
            ToolExecutableRequirement(
                executable=executable,
                probe_arguments=("--version",) if process else None,
            ),
        ),
    )


@pytest.mark.parametrize("missing_first", [False, True])
def test_shared_local_runner_isolates_concurrent_public_admission_plans(tmp_path, missing_first):
    runner = LocalRunner(tmp_path)

    class DeclaredTool(Tool):
        async def run(self, ctx, args):
            raise AssertionError("No tool was requested.")

    async def run(executable, session_id):
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), runner=runner), default=True
        )
        app.register_agent(
            AgentSpec(name="agent", model="fake"),
            tools=[
                DeclaredTool(
                    ToolSpec(name="program", execution_requirements=(_requirement(executable),)),
                )
            ],
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        return events, provider

    async def exercise():
        inputs = [
            (sys.executable, "local_present"),
            (str(tmp_path / "missing-program"), "local_missing"),
        ]
        if missing_first:
            inputs.reverse()
        outcomes = await asyncio.gather(*(run(*item) for item in inputs))
        return dict(zip((item[1] for item in inputs), outcomes, strict=True))

    outcomes = asyncio.run(exercise())
    present_events, present_provider = outcomes["local_present"]
    missing_events, missing_provider = outcomes["local_missing"]
    assert len(present_provider.requests) == 1
    assert any(event.type is EventType.SESSION_COMPLETED for event in present_events)
    assert missing_provider.requests == []
    assert not any(event.type is EventType.SESSION_COMPLETED for event in missing_events)
    failed = next(event for event in missing_events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["execution_admission"]["refusals"][0]["executable"] == str(
        tmp_path / "missing-program"
    )


def test_local_availability_does_not_claim_an_unexecuted_process_probe(tmp_path):
    requirements = ExecutionRequirements(
        tool_requirements=(
            ExecutionToolRequirement(
                tool_name="program",
                requirement=_requirement(sys.executable, process=True),
            ),
        )
    )
    candidate = LocalRunner(tmp_path).execution_admission_candidate_for(requirements)
    claim = candidate.evidence.tool_requirements.executable_for(sys.executable)
    assert claim.state == "unverified"
    assert claim.observed_at is None


@pytest.mark.parametrize("changed", [False, True])
def test_public_local_native_identity_survives_executable_composition(
    tmp_path, monkeypatch, changed
):
    class NativeRunner(LocalRunner):
        native_identity = "sha256:" + "1" * 64

        def execution_admission_candidate(self):
            candidate = super().execution_admission_candidate()
            now = datetime.now(UTC)
            return candidate.model_copy(
                update={
                    "evidence": candidate.evidence.model_copy(
                        update={
                            "environment_fingerprint": self.native_identity,
                            "claims": (
                                *candidate.evidence.claims,
                                ExecutionCapabilityClaim.live_verified(
                                    "workspace_text_search_v1",
                                    observation="supported",
                                    observed_at=now,
                                    valid_until=now + timedelta(seconds=300),
                                ),
                            ),
                        }
                    )
                }
            )

    class NativeOrExecutableTool(Tool):
        async def run(self, ctx, args):
            raise AssertionError("No tool dispatch is expected.")

    async def exercise():
        runner = NativeRunner(tmp_path)
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), runner=runner), default=True
        )
        app.register_agent(
            AgentSpec(name="agent", model="fake"),
            tools=[
                NativeOrExecutableTool(
                    ToolSpec(
                        name="native_or_executable",
                        execution_requirements=(
                            ToolExecutionRequirement(
                                name="search",
                                alternatives=(
                                    ToolRunnerCapabilityRequirement(
                                        capability="workspace_text_search_v1"
                                    ),
                                    ToolExecutableRequirement(
                                        executable=str(tmp_path / "missing-program")
                                    ),
                                ),
                            ),
                        ),
                    )
                )
            ],
        )
        emit = app._event_writer.emit
        observed = []

        async def change_native_identity(event):
            persisted = await emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload.get("phase") == "final_evidence"
            ):
                observed.append(runner.native_identity)
                if changed:
                    runner.native_identity = "sha256:" + "2" * 64
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", change_native_identity)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="native_identity",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        assert observed == ["sha256:" + "1" * 64]
        persisted = await app.session_store.load_events("native_identity")
        if changed:
            assert provider.requests == []
            assert not any(
                event.type is EventType.SESSION_COMPLETED for event in (*events, *persisted)
            )
            failed = next(event for event in persisted if event.type is EventType.SESSION_FAILED)
            assert {item["code"] for item in failed.payload["execution_admission"]["refusals"]} == {
                "environment_authority_mismatch"
            }
        else:
            assert len(provider.requests) == 1
            assert any(event.type is EventType.SESSION_COMPLETED for event in persisted)

    asyncio.run(exercise())


def test_local_search_path_changes_identity_without_rewriting_absolute_executable(
    tmp_path, monkeypatch
):
    runner = LocalRunner(tmp_path)
    requirements = ExecutionRequirements(required_executables=(sys.executable,))
    monkeypatch.setenv("PATH", "relative-bin")
    first = runner.execution_admission_candidate_for(requirements)
    monkeypatch.setenv("PATH", "different-bin")
    second = runner.execution_admission_candidate_for(requirements)
    assert first.evidence.environment_fingerprint != second.evidence.environment_fingerprint
    assert first.evidence.tool_requirements.executable_for(sys.executable).state == "live_verified"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX relative PATH and symlink fixture")
def test_local_relative_path_lookup_uses_workload_root_not_controller_cwd(tmp_path, monkeypatch):
    root = tmp_path / "workload"
    controller = tmp_path / "controller"
    (root / "bin").mkdir(parents=True)
    (controller / "bin").mkdir(parents=True)
    (root / "bin" / "local-python").symlink_to(sys.executable)
    (controller / "bin" / "controller-only").symlink_to(sys.executable)
    monkeypatch.chdir(controller)
    monkeypatch.setenv("PATH", "bin")
    requirements = ExecutionRequirements(
        required_executables=("bin/local-python", "controller-only", "local-python")
    )
    candidate = LocalRunner(root).execution_admission_candidate_for(requirements)
    assert {
        claim.executable: claim.state for claim in candidate.evidence.tool_requirements.executables
    } == {
        "bin/local-python": "live_verified",
        "controller-only": "unavailable",
        "local-python": "live_verified",
    }
