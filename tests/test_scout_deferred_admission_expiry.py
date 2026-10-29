"""Deferred live admission must not use evidence aged out by its own publication."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

import cayu.environments.admission as admission_module
from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, run_to_completion
from cayu.environments import Environment, EnvironmentFactoryResult, EnvironmentSpec, SyncBinding
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionRequirements,
)
from cayu.environments.deferred import DeferredMaterialization
from cayu.evals import ScriptedModelProvider
from cayu.runners import LocalRunner
from cayu.workspaces import LocalWorkspace
from tests.core.test_deferred_environments import DeferredLocalFactory, RunnerTool, _turns
from tests.runtime.test_execution_admission_dispatch import _candidate, _EvidenceClock


@pytest.mark.parametrize("expiry_phase", ["final_evidence", "admission"])
def test_deferred_admission_rechecks_expiry_after_publication(tmp_path, monkeypatch, expiry_phase):
    monkeypatch.setattr(_EvidenceClock, "current", datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(admission_module, "datetime", _EvidenceClock)
    asyncio.run(_scenario(tmp_path, monkeypatch, expiry_phase))


async def _scenario(tmp_path, monkeypatch, expiry_phase):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "input.txt").write_text("private workspace content")

    class ExpiringRunner(LocalRunner):
        async def collect_execution_admission_candidate_for(self, requirements):
            now = _EvidenceClock.now(UTC)
            return ExecutionAdmissionCandidate(
                candidate="sandbox",
                evidence=ExecutionCapabilityEvidence(
                    subject="sandbox",
                    claims=(
                        ExecutionCapabilityClaim.live_verified(
                            "confirmed_cleanup",
                            observation="supported",
                            observed_at=now,
                            valid_until=now + timedelta(seconds=1),
                        ),
                    ),
                ),
            )

    class Factory(DeferredLocalFactory):
        def execution_admission_candidate(self, request):
            return _candidate("sandbox", state="declared")

        async def create(self, request):
            async def materialize(mode):
                self.created += 1
                return ExpiringRunner(target), SyncBinding(
                    target_workspace=LocalWorkspace(target, workspace_id="target")
                )

            deferred = DeferredMaterialization(
                materialize,
                default_cwd=str(target),
                configured_candidate=lambda requirements: _candidate("sandbox", state="declared"),
            )
            return EnvironmentFactoryResult(
                environment=Environment(
                    EnvironmentSpec(name=request.environment_name),
                    workspace=LocalWorkspace(source, workspace_id="source"),
                    runner=deferred.runner,
                    binding=deferred.binding,
                ),
                reconnect_metadata={"kind": "deferred-local"},
            )

    factory = Factory(source, target)
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider(_turns("write")), default=True)
    app.register_environment_factory(EnvironmentSpec(name="sandbox"), factory, default=True)
    app.register_agent(
        AgentSpec(name="agent", model="scripted"),
        tools=[RunnerTool()],
        execution_requirements=ExecutionRequirements.trusted(cleanup="confirmed"),
    )
    emit = app._event_writer.emit
    expired = False
    materialized_transitions = []

    async def age_evidence_during_publication(event):
        nonlocal expired
        persisted = await emit(event)
        if factory.created and event.type == EventType.ENVIRONMENT_LIFECYCLE_TRANSITION:
            materialized_transitions.append(persisted)
        if (
            not expired
            and factory.created
            and event.type == EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
            and event.payload["phase"] == expiry_phase
        ):
            _EvidenceClock.advance(seconds=2)
            expired = True
        return persisted

    monkeypatch.setattr(app._event_writer, "emit", age_evidence_during_publication)
    await run_to_completion(
        app, RunRequest(agent_name="agent", messages=[Message.text("user", "run")])
    )
    assert expired and factory.created == 1
    assert not (source / "write.txt").exists(), (
        "The command ran and its output was copied back after live admission evidence expired"
    )
    assert not (target / "input.txt").exists(), (
        "Expired live admission still allowed workspace copy-in"
    )

    admissions = [
        event for event in materialized_transitions if event.payload["phase"] == "admission"
    ]
    assert admissions[-1].payload["outcome"] == "refused"
    assert admissions[-1].payload["refusal_codes"] == ["stale_evidence"]
