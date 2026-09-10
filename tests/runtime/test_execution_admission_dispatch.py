from __future__ import annotations

import asyncio
import re
import warnings
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import cayu.environments.admission as admission_module
import cayu.runners.docker as docker_module
from cayu import (
    AgentSpec,
    CayuApp,
    CayuConfig,
    DockerCodingCommandAuthority,
    DockerCodingToolchainProfile,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentLifecyclePolicy,
    EnvironmentSpec,
    Event,
    EventType,
    ExecutionAdmissionCandidate,
    ExecutionAdmissionError,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionEnvironmentAuthority,
    ExecutionProfileBehaviorIdentity,
    ExecutionRequirements,
    ExecutionToolRequirement,
    InMemorySessionStore,
    LocalRunner,
    Message,
    NamedCheck,
    OperationsConfig,
    PostgresSessionStore,
    ProcessCommandPolicy,
    ResumeRequest,
    RunCheckTool,
    RunCommandTool,
    RunRequest,
    SearchTextTool,
    SQLiteSessionStore,
    Tool,
    ToolCapabilityCeiling,
    ToolContext,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolResult,
    ToolSpec,
    Workspace,
    WorkspaceSnapshot,
    environment_lifecycle_transition_from_event,
)
from cayu.environments import BoundWorkspace, WorkspaceBinding
from cayu.environments.factory import (
    attach_environment_factory_cleanup_settlement_task,
    register_environment_factory_cleanup_retry,
)
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
)
from cayu.runners import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    DockerImageIdentity,
    DockerRunner,
    DockerWorkloadRestrictions,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerExecutionAdmissionObserver,
)
from cayu.runtime import CheckpointCompactionContextPolicy, ModelCompactor

_DOCKER_PROBE_COMPLETION_TOKEN = re.compile(r"cayu-admission-probe-complete-[0-9a-f]{32}")


def _completed_docker_probe_result(
    args: list[str],
    *,
    stdout: str = "",
    guest_exit_code: int = 0,
) -> ExecResult:
    token = next(
        (
            match.group(0)
            for value in args
            if (match := _DOCKER_PROBE_COMPLETION_TOKEN.search(value)) is not None
        ),
        None,
    )
    if token is None:
        return ExecResult(stdout=stdout, exit_code=guest_exit_code)
    return ExecResult(
        stdout=stdout,
        stderr=f"\n{token}:{guest_exit_code}\n",
        exit_code=0,
    )


def _candidate(name: str, *, state: str) -> ExecutionAdmissionCandidate:
    claim = (
        ExecutionCapabilityClaim.declared("confirmed_cleanup")
        if state == "declared"
        else ExecutionCapabilityClaim.available("confirmed_cleanup")
    )
    return ExecutionAdmissionCandidate(
        candidate=name,
        evidence=ExecutionCapabilityEvidence(subject=name, claims=(claim,)),
    )


class _EvidenceRunner(Runner):
    def __init__(self, candidate: str) -> None:
        self.candidate = candidate

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return _candidate(self.candidate, state="available")

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        del command, cwd, env, timeout_s, stdin, output_limit_bytes
        return ExecResult()


class _NoEvidenceRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        del command, cwd, env, timeout_s, stdin, output_limit_bytes
        return ExecResult()


class _BlockingEvidenceRunner(_EvidenceRunner):
    def __init__(self, candidate: str) -> None:
        super().__init__(candidate)
        self.collection_started = asyncio.Event()

    async def collect_execution_admission_candidate(
        self,
    ) -> ExecutionAdmissionCandidate:
        self.collection_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _RenewingEvidenceRunner(_EvidenceRunner):
    def __init__(self, *, drift_field: str | None = None) -> None:
        super().__init__("hosted")
        self.drift_field = drift_field
        self.refresh_calls = 0
        self.current_candidate: ExecutionAdmissionCandidate | None = None

    @staticmethod
    def _fingerprint(value: str) -> str:
        return "sha256:" + value * 64

    def _live_candidate(
        self,
        *,
        valid_for_seconds: float,
        drift_field: str | None = None,
    ) -> ExecutionAdmissionCandidate:
        observed_at = datetime.now(UTC)
        identities = {
            "environment_fingerprint": self._fingerprint("1"),
            "image_fingerprint": self._fingerprint("2"),
            "toolchain_profile_fingerprint": self._fingerprint("3"),
        }
        if drift_field is not None:
            identities[drift_field] = self._fingerprint("4")
        return ExecutionAdmissionCandidate(
            candidate="hosted",
            evidence=ExecutionCapabilityEvidence(
                subject="hosted",
                claims=(
                    ExecutionCapabilityClaim.live_verified(
                        "confirmed_cleanup",
                        observation="supported",
                        observed_at=observed_at,
                        valid_until=observed_at + timedelta(seconds=valid_for_seconds),
                    ),
                ),
                **identities,
            ),
        )

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return self.current_candidate or super().execution_admission_candidate()

    async def collect_execution_admission_candidate(
        self,
    ) -> ExecutionAdmissionCandidate:
        self.current_candidate = self._live_candidate(valid_for_seconds=1)
        return self.current_candidate

    async def refresh_execution_admission(self) -> None:
        self.refresh_calls += 1
        self.current_candidate = self._live_candidate(
            valid_for_seconds=60,
            drift_field=self.drift_field,
        )


class _RecordingProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _HostedFactory(EnvironmentFactory):
    def __init__(
        self,
        *,
        pre_create_candidate: str | None,
        runner: Runner,
        binding: WorkspaceBinding | None = None,
    ) -> None:
        self.pre_create_candidate = pre_create_candidate
        self.runner = runner
        self.binding = binding
        self.requests: list[EnvironmentFactoryRequest] = []

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate | None:
        del request
        if self.pre_create_candidate is None:
            return None
        return _candidate(self.pre_create_candidate, state="declared")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                runner=self.runner,
                binding=self.binding,
            )
        )


class _SwitchingBinding(WorkspaceBinding):
    def __init__(self, runner: Runner, *, lifecycle: list[str] | None = None) -> None:
        self.runner = runner
        self.lifecycle = lifecycle
        self.finalize_calls = 0
        self.finalize_outcomes: list[str | None] = []
        self.source_runner: Runner | None = None

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        del session_id, agent_name, environment_name, metadata
        self.source_runner = runner
        return BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=self.runner,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        del metadata
        self.finalize_calls += 1
        self.finalize_outcomes.append(outcome)
        if self.lifecycle is not None:
            self.lifecycle.append("binding.finalize")
        if bound.runner is not None:
            await bound.runner.close()
        if self.source_runner is not None and self.source_runner is not bound.runner:
            await self.source_runner.close()
        return None


class _FailingBinding(WorkspaceBinding):
    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        **kwargs: Any,
    ) -> BoundWorkspace:
        del workspace, runner, kwargs
        raise RuntimeError("bind failed")

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        del bound, outcome, metadata
        raise AssertionError("finalize should not run after bind failure")


class _ReleasableHostedFactory(_HostedFactory):
    def __init__(
        self,
        *,
        pre_create_candidate: str,
        runner: Runner,
        binding: WorkspaceBinding,
        lifecycle: list[str],
    ) -> None:
        super().__init__(
            pre_create_candidate=pre_create_candidate,
            runner=runner,
            binding=binding,
        )
        self.lifecycle = lifecycle
        self.release_actions: list[EnvironmentFactoryReleaseAction] = []

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            self.lifecycle.append(f"factory.release:{action.value}")
            self.release_actions.append(action)
            await self.runner.close()

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                runner=self.runner,
                binding=self.binding,
            ),
            reconnect_metadata={"allocation_id": request.session_id},
            release=release,
        )


class _RecoveringFactory(EnvironmentFactory):
    def __init__(self) -> None:
        self.requests: list[EnvironmentFactoryRequest] = []
        self.first_runner = _EvidenceRunner("hosted-b")
        self.second_runner = _EvidenceRunner("hosted-a")
        self.release_actions: list[EnvironmentFactoryReleaseAction] = []

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate:
        del request
        return _candidate("hosted-a", state="declared")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        runner = self.first_runner if len(self.requests) == 1 else self.second_runner

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            self.release_actions.append(action)
            await runner.close()

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                runner=runner,
            ),
            reconnect_metadata={"allocation_id": f"allocation-{len(self.requests)}"},
            release=release,
        )


class _BindingRunnerFactory(EnvironmentFactory):
    def __init__(self, binding: WorkspaceBinding) -> None:
        self.binding = binding
        self.requests: list[EnvironmentFactoryRequest] = []

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate:
        del request
        return _candidate("hosted", state="declared")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                binding=self.binding,
            ),
            reconnect_metadata={"allocation_id": request.session_id},
        )


async def _run(app: CayuApp, session_id: str) -> list[Event]:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            )
        )
    ]


@pytest.mark.parametrize(
    "proof, admitted",
    [
        ("rg", True),
        ("rg_changed", False),
        ("native", True),
        ("native_caller_rg_missing", False),
        ("native_caller_rg_verified", True),
        ("native_changed", False),
        ("missing", False),
        ("declared", False),
        ("stale", False),
        ("wrong_probe", False),
        ("unverified", False),
    ],
)
def test_tool_alternatives_use_common_public_run_admission(
    monkeypatch: pytest.MonkeyPatch, proof: str, admitted: bool
) -> None:
    # #860 owns admission of the real public declaration. Native dispatch and
    # conformance belong to #555; this bounded evidence fixture does not claim
    # to execute a production native backend.
    executable = ToolExecutableRequirement(executable="rg")

    class ToolEvidenceRunner(_EvidenceRunner):
        capability_identity = "sha256:" + "1" * 64
        image_identity = "sha256:" + "3" * 64

        def execution_admission_candidate(self):
            now = datetime.now(UTC)
            observed = now - timedelta(seconds=400) if proof == "stale" else now
            claims = ()
            if proof in {
                "native",
                "native_changed",
                "native_caller_rg_missing",
                "native_caller_rg_verified",
            }:
                claims = (
                    ExecutionCapabilityClaim.live_verified(
                        "workspace_text_search_v1",
                        observation="supported",
                        observed_at=now,
                        valid_until=now + timedelta(seconds=60),
                    ),
                )
            elif proof == "declared":
                claims = (ExecutionCapabilityClaim.declared("workspace_text_search_v1"),)
            elif proof == "unverified":
                claims = (
                    ExecutionCapabilityClaim.unverified(
                        "workspace_text_search_v1",
                        reason_code="not_probed",
                        remediation_code="verify_support",
                    ),
                )
            executables = ()
            if proof in {"rg", "rg_changed", "stale", "wrong_probe", "native_caller_rg_verified"}:
                executables = (
                    admission_module.ExecutionExecutableEvidence(
                        executable="rg",
                        state="live_verified",
                        observed_at=observed,
                        valid_until=observed + timedelta(seconds=60),
                        requirement_fingerprint=(
                            "sha256:" + "0" * 64
                            if proof == "wrong_probe"
                            else executable.fingerprint
                        ),
                    ),
                )
            return ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    claims=claims,
                    environment_fingerprint=self.capability_identity,
                    image_fingerprint=self.image_identity,
                    unclaimed_reason_code="security_unclaimed" if not claims else None,
                    tool_requirements=admission_module.ExecutionToolRequirementEvidence(
                        environment_fingerprint=self.capability_identity,
                        image_fingerprint=self.image_identity,
                        executables=executables,
                    ),
                ),
            )

    async def run():
        provider = _RecordingProvider()
        runner = ToolEvidenceRunner("hosted")
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="hosted"), runner=runner),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SearchTextTool()],
            execution_requirements=ExecutionRequirements(
                required_executables=("rg",) if proof.startswith("native_caller_rg_") else (),
            ),
        )
        if proof in {"native_changed", "rg_changed"}:
            emit = app._event_writer.emit

            async def replace_identity_after_observation(event):
                persisted = await emit(event)
                if (
                    event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                    and event.payload.get("phase") == "final_evidence"
                ):
                    if proof == "native_changed":
                        runner.capability_identity = "sha256:" + "2" * 64
                    else:
                        runner.image_identity = "sha256:" + "4" * 64
                return persisted

            monkeypatch.setattr(app._event_writer, "emit", replace_identity_after_observation)
        events = await _run(app, "tool_alternative")
        if proof == "native_changed":
            assert runner.capability_identity == "sha256:" + "2" * 64
        elif proof == "rg_changed":
            assert runner.image_identity == "sha256:" + "4" * 64
        return events, provider

    events, provider = asyncio.run(run())
    assert bool(provider.requests) is admitted, [
        event.payload for event in events if event.type is EventType.SESSION_FAILED
    ]
    if not admitted:
        assert any(event.type is EventType.SESSION_FAILED for event in events)
        assert not any(event.type is EventType.SESSION_COMPLETED for event in events)
        assert not any(str(event.type).startswith("model.") for event in events)
        failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
        refusals = failed.payload["execution_admission"]["refusals"]
        if proof in {"native_changed", "rg_changed"}:
            assert {refusal["code"] for refusal in refusals} == {"environment_authority_mismatch"}
        elif proof == "native_caller_rg_missing":
            assert all(refusal.get("tool_name") is None for refusal in refusals)
            assert any(refusal["executable"] == "rg" for refusal in refusals)
        else:
            assert any(refusal["tool_name"] == "search_text" for refusal in refusals)


@pytest.mark.parametrize("builtin", ["named_check", "structured_command"])
@pytest.mark.parametrize("proof", ["exact", "basename", "missing"])
def test_command_builtins_require_exact_executable_evidence_before_provider(builtin, proof):
    executable = "/opt/tools/python"
    if builtin == "named_check":
        tool = RunCheckTool(
            checks=(
                NamedCheck(
                    name="tests",
                    description="Run tests.",
                    command=ExecCommand.process(executable, "-m", "unittest"),
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests", behavior_version="1", implementation_version="1"
                    ),
                ),
            ),
            command_policy=ProcessCommandPolicy(allowed_executables=(executable,)),
        )
    else:
        tool = RunCommandTool(
            toolchain_profile=DockerCodingToolchainProfile(
                profile_id="tests",
                revision="1",
                image_identity=DockerImageIdentity(
                    reference="registry.example/python@sha256:" + "b" * 64
                ),
                platform_architecture="amd64",
                command_authorities=(
                    DockerCodingCommandAuthority(
                        selector="tests",
                        description="Run tests.",
                        revision="1",
                        exposure="structured_command",
                        executable=executable,
                        fixed_arguments=("-m", "unittest"),
                    ),
                ),
            )
        )

    class CommandEvidenceRunner(_EvidenceRunner):
        def execution_admission_candidate(self):
            now = datetime.now(UTC)
            observed_executable = "python" if proof == "basename" else executable
            claims = (
                ()
                if proof == "missing"
                else (
                    admission_module.ExecutionExecutableEvidence(
                        executable=observed_executable,
                        state="live_verified",
                        observed_at=now,
                        valid_until=now + timedelta(seconds=60),
                        requirement_fingerprint=ToolExecutableRequirement(
                            executable=observed_executable
                        ).fingerprint,
                    ),
                )
            )
            return ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    unclaimed_reason_code="security_unclaimed",
                    environment_fingerprint="sha256:" + "1" * 64,
                    tool_requirements=admission_module.ExecutionToolRequirementEvidence(
                        environment_fingerprint="sha256:" + "1" * 64,
                        executables=claims,
                    ),
                ),
            )

    async def run():
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="hosted"), runner=CommandEvidenceRunner("hosted")),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
        return await _run(app, f"sess_{builtin}_{proof}"), provider

    events, provider = asyncio.run(run())
    assert len(provider.requests) == (1 if proof == "exact" else 0)
    if proof != "exact":
        assert EventType.SESSION_COMPLETED not in {event.type for event in events}
        failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
        assert any(
            refusal["tool_name"] == tool.spec.name and refusal["executable"] == executable
            for refusal in failed.payload["execution_admission"]["refusals"]
        )


@pytest.mark.parametrize("exclude_tool", [True, False])
def test_tool_ceiling_excludes_requirements_of_unavailable_tools(exclude_tool: bool) -> None:
    class UnavailableTool(Tool):
        spec = ToolSpec(
            name="unavailable",
            execution_requirements=(
                ToolExecutionRequirement(
                    name="executable",
                    alternatives=(ToolExecutableRequirement(executable="missing"),),
                ),
            ),
        )

        async def run(self, ctx, args):
            raise AssertionError("Excluded tool must not execute.")

    async def run():
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"), tools=[UnavailableTool()]
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="narrowed_tools",
                    messages=[Message.text("user", "run")],
                    tool_capability_ceiling=ToolCapabilityCeiling(
                        tool_names=() if exclude_tool else ("unavailable",)
                    ),
                )
            )
        ]
        return events, provider

    events, provider = asyncio.run(run())
    assert len(provider.requests) == (1 if exclude_tool else 0)
    assert any(event.type is EventType.SESSION_FAILED for event in events) is not exclude_tool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "case, admitted",
    [
        ("executables_below", True),
        ("executables_at", True),
        ("executables_above", False),
        ("executables_overlap", True),
        ("clauses_below", True),
        ("clauses_at", True),
        ("clauses_above", False),
        ("clauses_overlap", True),
        ("clauses_conflict", False),
    ],
)
def test_public_tool_requirement_aggregation_bounds_and_overlap(backend, case, admitted, tmp_path):
    executable_bound = case.startswith("executables_")
    caller_count = 33 if case.endswith("above") else 32
    tool_count = 31 if case.endswith("below") else 32

    def clause(index):
        return ToolExecutionRequirement(
            name=f"requirement_{index:02}",
            alternatives=(
                ToolExecutableRequirement(
                    executable=f"tool_{index:02}" if executable_bound else "shared"
                ),
            ),
        )

    class DeclaredTool(Tool):
        spec = ToolSpec(
            name="declared",
            execution_requirements=tuple(clause(index) for index in range(tool_count)),
        )

        async def run(self, ctx, args):
            raise AssertionError("The scripted provider must not dispatch a tool.")

    if executable_bound:
        caller_names = tuple(f"caller_{index:02}" for index in range(caller_count))
        if case == "executables_overlap":
            caller_names = tuple(sorted((*caller_names[:-1], "tool_00")))
        base = ExecutionRequirements(required_executables=caller_names)
    else:
        overlap = case in {"clauses_overlap", "clauses_conflict"}
        caller_clauses = [
            ExecutionToolRequirement(
                tool_name="declared" if overlap else "caller",
                requirement=clause(index),
            )
            for index in range(caller_count)
        ]
        if case == "clauses_conflict":
            caller_clauses[0] = ExecutionToolRequirement(
                tool_name="declared",
                requirement=ToolExecutionRequirement(
                    name="requirement_00",
                    alternatives=(ToolExecutableRequirement(executable="caller_only"),),
                ),
            )
        base = ExecutionRequirements(tool_requirements=tuple(caller_clauses))

    observed_requirements = []

    class AggregateRunner(_EvidenceRunner):
        def execution_admission_candidate_for(self, requirements):
            observed_requirements.append(requirements)
            now = datetime.now(UTC)
            fingerprint = "sha256:" + "1" * 64
            probes = {probe.executable: probe for probe in requirements.executable_probes()}
            return ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    unclaimed_reason_code="security_unclaimed",
                    environment_fingerprint=fingerprint,
                    tool_requirements=admission_module.ExecutionToolRequirementEvidence(
                        environment_fingerprint=fingerprint,
                        executables=tuple(
                            admission_module.ExecutionExecutableEvidence(
                                executable=name,
                                state="live_verified",
                                observed_at=now,
                                valid_until=now + timedelta(seconds=60),
                                requirement_fingerprint=(
                                    probes[name].fingerprint if name in probes else None
                                ),
                            )
                            for name in requirements.executable_names()
                        ),
                    ),
                ),
            )

        async def collect_execution_admission_candidate_for(self, requirements):
            return self.execution_admission_candidate_for(requirements)

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "aggregation.sqlite")
        )
        try:
            provider = _RecordingProvider()
            app = CayuApp(enable_logging=False, session_store=store)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(EnvironmentSpec(name="hosted"), runner=AggregateRunner("hosted")),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[DeclaredTool()],
                execution_requirements=base,
            )
            events = await _run(app, "aggregation")
            durable = await store.load_events("aggregation")
            return events, durable, provider
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    events, durable, provider = asyncio.run(run())
    assert len(provider.requests) == int(admitted)
    for recorded in (events, durable):
        kinds = {event.type for event in recorded}
        assert (EventType.SESSION_COMPLETED in kinds) is admitted
        assert (EventType.SESSION_FAILED in kinds) is not admitted
        if not admitted:
            assert EventType.MODEL_STARTED not in kinds
    if admitted:
        assert observed_requirements
        for requirements in observed_requirements:
            assert requirements.required_executables == base.required_executables
            expected_clauses = (
                tool_count
                if executable_bound or case == "clauses_overlap"
                else caller_count + tool_count
            )
            assert len(requirements.tool_requirements) == expected_clauses
            expected_names = (
                caller_count + tool_count - int(case == "executables_overlap")
                if executable_bound
                else 1
            )
            assert len(requirements.executable_names()) == expected_names
    else:
        assert observed_requirements == []


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("reconstruct", [False, True])
@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("caller_requires_rg", [False, True])
def test_resume_recomputes_tool_requirements_without_weakening_caller_policy(
    backend, reconstruct, narrow, caller_requires_rg, tmp_path
):
    def identity(name):
        return ExecutionProfileBehaviorIdentity(
            name=name, behavior_version="1", implementation_version="1"
        )

    class StableProvider(_RecordingProvider):
        @property
        def execution_profile_identity(self):
            return identity("resume-admission-provider")

    class ResumeRunner(_EvidenceRunner):
        present = True

        def __init__(self):
            super().__init__("hosted")
            self.requirements = []
            self.observers = []

        @property
        def execution_profile_identity(self):
            return identity("resume-admission-runner")

        def execution_admission_observer(self, requirements):
            observer = super().execution_admission_observer(requirements)
            self.observers.append(observer)
            return observer

        def execution_admission_candidate_for(self, requirements):
            self.requirements.append(requirements)
            now = datetime.now(UTC)
            fingerprint = "sha256:" + "1" * 64
            return ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    unclaimed_reason_code="security_unclaimed",
                    environment_fingerprint=fingerprint,
                    tool_requirements=admission_module.ExecutionToolRequirementEvidence(
                        environment_fingerprint=fingerprint,
                        executables=(
                            (
                                admission_module.ExecutionExecutableEvidence(
                                    executable="rg",
                                    state="live_verified",
                                    observed_at=now,
                                    valid_until=now + timedelta(seconds=60),
                                    requirement_fingerprint=ToolExecutableRequirement(
                                        executable="rg"
                                    ).fingerprint,
                                ),
                            )
                            if self.present
                            else ()
                        ),
                    ),
                ),
            )

        async def collect_execution_admission_candidate_for(self, requirements):
            return self.execution_admission_candidate_for(requirements)

    def composition(store, runner, provider):
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="hosted", execution_profile_identity=identity("resume-env")),
                runner=runner,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SearchTextTool()],
            execution_requirements=ExecutionRequirements(
                required_executables=("rg",) if caller_requires_rg else (),
            ),
        )
        return app

    async def run():
        database = tmp_path / "resume-admission.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
        try:
            runner, provider = ResumeRunner(), StableProvider()
            app = composition(store, runner, provider)
            initial = await _run(app, "resume-admission")
            assert initial[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 1
            assert len(runner.observers) == 1
            initial_observer = runner.observers[0]
            if reconstruct:
                if isinstance(store, SQLiteSessionStore):
                    await store.close()
                    store = SQLiteSessionStore(database)
                runner, provider = ResumeRunner(), StableProvider()
                app = composition(store, runner, provider)
            runner.present = False
            runner.requirements.clear()
            runner.observers.clear()
            provider.requests.clear()
            before = await store.load_events("resume-admission")
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="resume-admission",
                        messages=[Message.text("user", "continue")],
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=())
                        if narrow
                        else None,
                    )
                )
            ]
            durable = await store.load_events("resume-admission")
            new_events = [
                event for event in durable if event.id not in {item.id for item in before}
            ]
            return resumed, new_events, runner, provider, initial_observer
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    events, durable, runner, provider, initial_observer = asyncio.run(run())
    admitted = narrow and not caller_requires_rg
    assert len(provider.requests) == int(admitted)
    if admitted:
        assert len(runner.observers) == 1
        assert runner.observers[0] is not initial_observer
        assert runner.observers[0].runner is runner
        assert runner.observers[0].requirements.tool_requirements == ()
    assert runner.requirements
    for requirements in runner.requirements:
        assert requirements.required_executables == (("rg",) if caller_requires_rg else ())
        assert {item.tool_name for item in requirements.tool_requirements} == (
            set() if narrow else {"search_text"}
        )
    for recorded in (events, durable):
        kinds = {event.type for event in recorded}
        assert (EventType.SESSION_COMPLETED in kinds) is admitted
        assert (EventType.SESSION_FAILED in kinds) is not admitted
        if not admitted:
            assert not any(str(kind).startswith("model.") for kind in kinds)
            failed = next(event for event in recorded if event.type is EventType.SESSION_FAILED)
            refusals = failed.payload["execution_admission"]["refusals"]
            if caller_requires_rg:
                assert any(
                    item.get("tool_name") is None and item.get("executable") == "rg"
                    for item in refusals
                )
            if not narrow:
                assert any(item.get("tool_name") == "search_text" for item in refusals)


def test_documented_hosted_runner_supplies_collection_and_dispatch_snapshot():
    documentation = (Path(__file__).parents[2] / "docs/environment-factories.md").read_text()
    source = next(
        block.split("```", 1)[0]
        for block in documentation.split("```python\n")[1:]
        if "class HostedRunner(Runner):" in block.split("```", 1)[0]
    )
    namespace = {}
    exec(compile(source, "docs/environment-factories.md", "exec"), namespace)
    snapshots = []
    observations = []

    class HostedRunner(namespace["HostedRunner"]):
        exec = _EvidenceRunner.exec

        def _snapshot_exact_runtime_evidence(self, requirements):
            snapshots.append(requirements)
            now = datetime.now(UTC)
            fingerprint = "sha256:" + "1" * 64
            return ExecutionCapabilityEvidence(
                subject="acme-sandbox",
                unclaimed_reason_code="security_unclaimed",
                environment_fingerprint=fingerprint,
                tool_requirements=admission_module.ExecutionToolRequirementEvidence(
                    environment_fingerprint=fingerprint,
                    executables=(
                        admission_module.ExecutionExecutableEvidence(
                            executable="rg",
                            state="live_verified",
                            observed_at=now,
                            valid_until=now + timedelta(seconds=60),
                            requirement_fingerprint=ToolExecutableRequirement(
                                executable="rg"
                            ).fingerprint,
                        ),
                    ),
                ),
            )

        async def _observe_exact_runtime_evidence(self, requirements):
            observations.append(requirements)
            return self._snapshot_exact_runtime_evidence(requirements)

    async def run():
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="hosted"), runner=HostedRunner()), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"), tools=[SearchTextTool()]
        )
        return await _run(app, "documented-hosted-runner"), provider

    events, provider = asyncio.run(run())
    assert events[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 1
    assert len(observations) == 1
    assert len(snapshots) > len(observations)
    assert all(requirements == observations[0] for requirements in snapshots)
    assert {item.tool_name for item in observations[0].tool_requirements} == {"search_text"}


def _requirements() -> ExecutionRequirements:
    return ExecutionRequirements.trusted(
        cleanup="confirmed",
        minimum_evidence="available",
    )


def _bound_factory_app(
    *,
    max_environment_lifecycle_owners: int | None = None,
) -> tuple[
    CayuApp,
    _ReleasableHostedFactory,
    _SwitchingBinding,
    _EvidenceRunner,
    _EvidenceRunner,
    list[str],
]:
    lifecycle: list[str] = []
    source_runner = _EvidenceRunner("hosted")
    bound_runner = _EvidenceRunner("hosted")
    binding = _SwitchingBinding(bound_runner, lifecycle=lifecycle)
    factory = _ReleasableHostedFactory(
        pre_create_candidate="hosted",
        runner=source_runner,
        binding=binding,
        lifecycle=lifecycle,
    )
    app = (
        CayuApp(enable_logging=False)
        if max_environment_lifecycle_owners is None
        else CayuApp(
            enable_logging=False,
            config=CayuConfig(
                operations=OperationsConfig(
                    max_environment_lifecycle_owners=max_environment_lifecycle_owners
                )
            ),
        )
    )
    app.register_provider(_RecordingProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="hosted"),
        factory,
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        execution_requirements=_requirements(),
    )
    return app, factory, binding, source_runner, bound_runner, lifecycle


@pytest.mark.parametrize("backend", ["e2b", "lambda_microvm"])
@pytest.mark.parametrize("requires_executable", [False, True])
def test_unobserved_remote_runner_fails_closed_only_for_required_dependencies(
    backend,
    requires_executable,
):
    from types import SimpleNamespace

    from cayu.runners import E2BRunner, LambdaMicroVMRunner

    class PythonOnlyTool(Tool):
        spec = ToolSpec(name="python_only", input_schema={"type": "object"})

        async def run(self, ctx, args):
            return ToolResult(content="local Python result")

    async def scenario():
        # No SDK or endpoint methods exist: admission must not dispatch remotely
        # or infer executable support from a backend's identity.
        if backend == "e2b":
            runner = E2BRunner(SimpleNamespace(sandbox_id="admission-860"))
        else:
            runner = LambdaMicroVMRunner(
                object(),
                microvm_id="admission-860",
                endpoint="https://invalid.example",
                endpoint_transport=object(),
            )
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="remote"), runner=runner), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SearchTextTool()] if requires_executable else [PythonOnlyTool()],
        )
        events = await _run(app, f"unobserved-{backend}-{requires_executable}")
        assert len(provider.requests) == int(not requires_executable)
        assert (EventType.SESSION_COMPLETED in {event.type for event in events}) is (
            not requires_executable
        )
        if requires_executable:
            assert EventType.SESSION_FAILED in {event.type for event in events}
            assert not any(str(event.type).startswith("model.") for event in events)
        await runner.close()

    asyncio.run(scenario())


def test_static_local_runner_is_refused_for_untrusted_workload(tmp_path) -> None:
    async def run() -> tuple[list[Event], _RecordingProvider]:
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=LocalRunner(tmp_path),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.untrusted(),
        )
        return await _run(app, "sess_static_local_refused"), provider

    events, provider = asyncio.run(run())

    assert provider.requests == []
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error_type"] == "ExecutionAdmissionError"
    transitions = [
        environment_lifecycle_transition_from_event(event)
        for event in events
        if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
    ]
    assert [(item.phase.value, item.outcome.value) for item in transitions] == [
        ("selected", "observed"),
        ("preflight", "refused"),
    ]


def test_malformed_factory_candidate_is_refused_without_diagnostic_disclosure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "candidate-secret-canary"

    class SecretCanary:
        def __repr__(self) -> str:
            return secret

        def __str__(self) -> str:
            return secret

    class MalformedCandidateFactory(EnvironmentFactory):
        create_calls = 0

        def execution_admission_candidate(
            self,
            request: EnvironmentFactoryRequest,
        ) -> ExecutionAdmissionCandidate:
            del request
            candidate = _candidate("hosted", state="declared")
            object.__setattr__(candidate.evidence, "claims", (SecretCanary(),))
            return candidate

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            del request
            self.create_calls += 1
            raise AssertionError("Malformed candidate reached allocation.")

    async def run() -> tuple[list[Event], MalformedCandidateFactory]:
        factory = MalformedCandidateFactory()
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        return await _run(app, "sess_malformed_candidate"), factory

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        events, factory = asyncio.run(run())

    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["execution_admission"]["refusals"][0]["code"] == ("malformed_evidence")
    assert factory.create_calls == 0
    captured = capsys.readouterr()
    diagnostics = "\n".join(
        (
            caplog.text,
            captured.out,
            captured.err,
            *(str(item.message) for item in caught_warnings),
            repr(failed.payload),
        )
    )
    assert secret not in diagnostics


def test_generic_factory_is_admitted_through_common_dispatch() -> None:
    async def run() -> tuple[list[Event], _RecordingProvider, _HostedFactory]:
        provider = _RecordingProvider()
        factory = _HostedFactory(
            pre_create_candidate="hosted",
            runner=_EvidenceRunner("hosted"),
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, "sess_generic_admitted"), provider, factory

    events, provider, factory = asyncio.run(run())

    assert len(provider.requests) == 1
    assert len(factory.requests) == 1
    assert factory.requests[0].execution_requirements == _requirements()
    assert EventType.SESSION_COMPLETED in {event.type for event in events}
    transitions = [
        environment_lifecycle_transition_from_event(event)
        for event in events
        if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
    ]
    assert [(item.phase.value, item.outcome.value) for item in transitions] == [
        ("selected", "observed"),
        ("preflight", "accepted"),
        ("allocated", "completed"),
        ("bound", "completed"),
        ("final_evidence", "observed"),
        ("admission", "admitted"),
        ("exposure", "exposed"),
    ]


@pytest.mark.parametrize(
    "backend",
    ["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)],
)
def test_environment_transition_order_is_durable_across_session_stores(
    backend: str,
    tmp_path,
    request: pytest.FixtureRequest,
) -> None:
    postgres_dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run() -> list[tuple[str, str]]:
        if backend == "memory":
            store = InMemorySessionStore()
        elif backend == "sqlite":
            store = SQLiteSessionStore(tmp_path / "environment-transitions.sqlite")
        else:
            from cayu.storage.migrations import SchemaMode

            assert postgres_dsn is not None
            store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            provider = _RecordingProvider()
            app = CayuApp(enable_logging=False, session_store=store)
            app.register_provider(provider, default=True)
            app.register_environment_factory(
                EnvironmentSpec(name="hosted"),
                _HostedFactory(
                    pre_create_candidate="hosted",
                    runner=_EvidenceRunner("hosted"),
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                execution_requirements=_requirements(),
            )
            session_id = f"sess_durable_environment_transitions_{backend}"
            await _run(app, session_id)
            durable_events = await store.load_events(session_id)
            return [
                (transition.phase.value, transition.outcome.value)
                for event in durable_events
                if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                for transition in (environment_lifecycle_transition_from_event(event),)
            ]
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    assert asyncio.run(run()) == [
        ("selected", "observed"),
        ("preflight", "accepted"),
        ("allocated", "completed"),
        ("bound", "completed"),
        ("final_evidence", "observed"),
        ("admission", "admitted"),
        ("exposure", "exposed"),
    ]


@pytest.mark.parametrize("copy_runner_authority", [False, True])
def test_final_runner_requires_exact_factory_environment_authority(
    copy_runner_authority: bool,
) -> None:
    authority = ExecutionEnvironmentAuthority(
        identity="factory_exact_authority",
        profile_identity="factory_profile_v1",
    )
    runner_authority = authority.model_copy() if copy_runner_authority else authority
    assert (runner_authority is authority) is not copy_runner_authority

    class AuthorityRunner(_EvidenceRunner):
        def execution_environment_authority(self) -> ExecutionEnvironmentAuthority:
            return runner_authority

    class AuthorityFactory(_HostedFactory):
        def __init__(self) -> None:
            super().__init__(
                pre_create_candidate="hosted",
                runner=AuthorityRunner("hosted"),
            )
            self.release_actions: list[EnvironmentFactoryReleaseAction] = []

        def execution_environment_authority(self) -> ExecutionEnvironmentAuthority:
            return authority

        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            self.requests.append(request)

            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                self.release_actions.append(action)
                await self.runner.close()

            return EnvironmentFactoryResult(
                environment=Environment(
                    EnvironmentSpec(name=request.environment_name),
                    runner=self.runner,
                ),
                release=release,
            )

    async def run() -> tuple[list[Event], _RecordingProvider, AuthorityFactory]:
        provider = _RecordingProvider()
        factory = AuthorityFactory()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, f"sess_exact_authority_{copy_runner_authority}"), provider, factory

    events, provider, factory = asyncio.run(run())

    if not copy_runner_authority:
        assert len(provider.requests) == 1
        assert factory.release_actions == []
        assert EventType.SESSION_COMPLETED in {event.type for event in events}
        return

    assert provider.requests == []
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["execution_admission"]["refusals"][0]["code"] == (
        "environment_authority_mismatch"
    )


def test_expired_exposure_is_refused_again_at_actual_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.environments.admission as admission_module

    class AdmissionClock(datetime):
        current = datetime.now(UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current if tz is None else cls.current.astimezone(tz)

    monkeypatch.setattr(admission_module, "datetime", AdmissionClock)

    class ExpiringEvidenceRunner(_EvidenceRunner):
        current_candidate: ExecutionAdmissionCandidate | None = None

        def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
            return self.current_candidate or super().execution_admission_candidate()

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            observed_at = AdmissionClock.now(UTC)
            self.current_candidate = ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    claims=(
                        ExecutionCapabilityClaim.live_verified(
                            "confirmed_cleanup",
                            observation="supported",
                            observed_at=observed_at,
                            valid_until=observed_at + timedelta(seconds=1),
                        ),
                    ),
                ),
            )
            return self.current_candidate

    class DispatchDelayedProvider(_RecordingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.delay_next_dispatch = False
            self.dispatch_delay_applied = False

        @property
        def provider_operation_mode(self):
            if self.delay_next_dispatch:
                self.delay_next_dispatch = False
                self.dispatch_delay_applied = True
                # Expire proof at this exact boundary, not during CI setup.
                AdmissionClock.current += timedelta(seconds=2)
            return super().provider_operation_mode

    async def run() -> tuple[list[Event], _RecordingProvider]:
        provider = DispatchDelayedProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(
                pre_create_candidate="hosted",
                runner=ExpiringEvidenceRunner("hosted"),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def arm_delay_after_exposure(event: Event) -> Event:
            persisted = await original_emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "exposure"
            ):
                provider.delay_next_dispatch = True
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", arm_delay_after_exposure)
        return await _run(app, "sess_expired_before_provider_dispatch"), provider

    events, provider = asyncio.run(run())

    assert provider.requests == []
    assert provider.dispatch_delay_applied is True
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["execution_admission"]["refusals"][0]["code"] == "stale_evidence"
    phases = [
        event.payload["phase"]
        for event in events
        if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
    ]
    assert phases[-1] == "exposure"


@pytest.mark.parametrize(
    "change_phase",
    [
        None,
        "after_collection",
        "during_collection",
        "unsupported",
        "unverified",
        "missing",
        "malformed",
    ],
)
def test_public_docker_native_identity_invalidates_cached_tool_evidence(
    monkeypatch: pytest.MonkeyPatch, change_phase: str | None
) -> None:
    class NativeDockerRunner(DockerRunner):
        native_identity = "sha256:" + "1" * 64
        native_state = "live_verified"

        def execution_capability_evidence(self):
            base = super().execution_capability_evidence()
            now = datetime.now(UTC)
            claim = ExecutionCapabilityClaim.live_verified(
                "workspace_text_search_v1",
                observation="supported",
                observed_at=now,
                valid_until=now + timedelta(seconds=300),
            )
            if self.native_state in {"unsupported", "unverified"}:
                claim = getattr(ExecutionCapabilityClaim, self.native_state)(
                    "workspace_text_search_v1",
                    reason_code="native_withdrawn",
                    remediation_code="verify_native_support",
                )
            elif self.native_state == "malformed":
                claim = claim.model_copy(update={"state": "invalid-state"})
            return base.model_copy(
                update={
                    "environment_fingerprint": self.native_identity,
                    "claims": (
                        *base.claims,
                        *((claim,) if self.native_state != "missing" else ()),
                    ),
                }
            )

    runner = NativeDockerRunner(
        "native",
        image="test-image",
        default_cwd="/workspace",
        close_action="none",
        docker_path="/usr/bin/docker",
        credential_mode="trusted_tool",
        allow_raw_secret_env=False,
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
        _container_id="a" * 64,
    )
    probe_calls = []

    async def inspect(*args, **kwargs):
        return {
            "Id": runner.container_id,
            "Image": "sha256:" + "b" * 64,
            "State": {"Running": True},
        }

    async def probe(*args, **kwargs):
        probe_calls.append(kwargs["required_executables"])
        if change_phase == "during_collection":
            runner.native_identity = "sha256:" + "2" * 64
        return (("rg", False),)

    monkeypatch.setattr(docker_module, "_inspect_strict_container", inspect)
    monkeypatch.setattr(docker_module, "_probe_container_executables", probe)

    async def run():
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="docker"), runner=runner), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"), tools=[SearchTextTool()]
        )
        emit = app._event_writer.emit
        observed = []

        async def replace_after_collection(event):
            persisted = await emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload.get("phase") == "final_evidence"
            ):
                observed.append(runner.native_identity)
                if change_phase == "after_collection":
                    runner.native_identity = "sha256:" + "2" * 64
                elif change_phase in {"unsupported", "unverified", "missing", "malformed"}:
                    runner.native_state = change_phase
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", replace_after_collection)
        events = await _run(app, "docker_native_identity")
        persisted = await app.session_store.load_events("docker_native_identity")
        assert probe_calls == [("rg",)]
        if change_phase != "during_collection":
            assert observed == ["sha256:" + "1" * 64]
        if change_phase is None:
            assert len(provider.requests) == 1
            assert any(event.type is EventType.SESSION_COMPLETED for event in persisted)
        else:
            assert provider.requests == []
            assert not any(
                event.type is EventType.SESSION_COMPLETED for event in (*events, *persisted)
            )
            failed = next(event for event in persisted if event.type is EventType.SESSION_FAILED)
            assert failed.payload["execution_admission"]["refusals"]
            if change_phase in {"unsupported", "unverified", "missing"}:
                assert runner.native_identity == "sha256:" + "1" * 64
                expected = {
                    "unsupported": "unsupported_capability",
                    "unverified": "unverified_capability",
                    "missing": "missing_capability",
                }[change_phase]
                assert any(
                    refusal["code"] == expected
                    and refusal.get("capability") == "workspace_text_search_v1"
                    for refusal in failed.payload["execution_admission"]["refusals"]
                )

    asyncio.run(run())


def test_docker_snapshot_preserves_cached_executable_observation_lifetime(monkeypatch):
    runner = DockerRunner(
        "cached",
        image="test-image",
        default_cwd="/workspace",
        close_action="none",
        docker_path="/usr/bin/docker",
        credential_mode="trusted_tool",
        allow_raw_secret_env=False,
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
        _container_id="a" * 64,
    )
    requirements = ExecutionRequirements(required_executables=("rg",))
    observer = runner.execution_admission_observer(requirements)
    probe_calls = []

    async def inspect(*args, **kwargs):
        return {
            "Id": runner.container_id,
            "Image": "sha256:" + "b" * 64,
            "State": {"Running": True},
        }

    async def probe(*args, **kwargs):
        probe_calls.append(kwargs["required_executables"])
        return (("rg", True),)

    monkeypatch.setattr(docker_module, "_inspect_strict_container", inspect)
    monkeypatch.setattr(docker_module, "_probe_container_executables", probe)

    async def run():
        collected = await observer.collect()
        original = collected.evidence.tool_requirements
        expired_at = original.executables[0].valid_until + timedelta(seconds=1)

        class LaterClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return expired_at.replace(tzinfo=None) if tz is None else expired_at.astimezone(tz)

        monkeypatch.setattr(docker_module, "datetime", LaterClock)
        for _ in range(2):
            snapshot = observer.snapshot()
            assert snapshot is not collected
            assert snapshot.evidence.tool_requirements == original
            decision = admission_module.evaluate_execution_admission(
                candidate=snapshot.candidate,
                requirements=requirements,
                evidence=snapshot.evidence,
                now=expired_at,
            )
            assert decision.status == "refused"
            assert {refusal.code for refusal in decision.refusals} == {"stale_evidence"}
        assert probe_calls == [("rg",)]

    asyncio.run(run())


def test_docker_tool_observer_composes_native_identity_into_fingerprint():
    runner = DockerRunner(
        "native",
        image="test-image",
        default_cwd="/workspace",
        close_action="none",
        docker_path="/usr/bin/docker",
        credential_mode="trusted_tool",
        allow_raw_secret_env=False,
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
        _container_id="a" * 64,
    )
    observer = runner.execution_admission_observer(
        ExecutionRequirements(required_executables=("rg",))
    )
    base = runner.execution_capability_evidence()
    candidates = [
        observer._candidate(
            base.model_copy(update={"environment_fingerprint": "sha256:" + digit * 64}),
            image_id="sha256:" + "b" * 64,
        )
        for digit in ("1", "2")
    ]
    assert (
        candidates[0].evidence.environment_fingerprint
        != candidates[1].evidence.environment_fingerprint
    )
    for candidate in candidates:
        assert (
            candidate.evidence.tool_requirements.environment_fingerprint
            == candidate.evidence.environment_fingerprint
        )


def test_slow_docker_final_evidence_is_refused_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProbeClock(datetime):
        current = datetime(2026, 1, 1, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            value = cls.current
            return value.replace(tzinfo=None) if tz is None else value.astimezone(tz)

    container_id = "a" * 64
    image_id = "sha256:" + "b" * 64
    image_reference = "registry.example/cayu@sha256:" + "c" * 64
    restrictions = DockerWorkloadRestrictions()
    evidence = docker_module._DockerRuntimeEvidence(
        container_id=container_id,
        image_id=image_id,
        image_reference=image_reference,
        network_mode="none",
        default_cwd="/workspace",
        runtime=None,
        seccomp_profile_sha256=None,
        restrictions=restrictions,
        image_identity=DockerImageIdentity(reference=image_reference),
        toolchain_profile_fingerprint=None,
        required_executables=(),
        executable_availability=(),
        immutable_input_mounts=(),
        observed_at=ProbeClock.current,
        valid_until=ProbeClock.current + timedelta(seconds=300),
    )
    runner = DockerRunner(
        "strict",
        image=image_reference,
        default_cwd="/workspace",
        close_action="none",
        docker_path="/usr/bin/docker",
        credential_mode="trusted_tool",
        allow_raw_secret_env=False,
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
        _container_id=container_id,
        _runtime_evidence=evidence,
    )
    probe_cycles = 0

    async def inspect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {}

    def verify(*args: Any, **kwargs: Any) -> tuple[str, str]:
        del args, kwargs
        return image_id, image_reference

    async def slow_mount_probe(*args: Any, **kwargs: Any) -> None:
        nonlocal probe_cycles
        del args, kwargs
        probe_cycles += 1
        ProbeClock.current += timedelta(seconds=301)

    async def executable_probe(*args: Any, **kwargs: Any) -> tuple[tuple[str, bool], ...]:
        del args, kwargs
        return ()

    monkeypatch.setattr(docker_module, "datetime", ProbeClock)
    monkeypatch.setattr(admission_module, "datetime", ProbeClock)
    monkeypatch.setattr(docker_module, "_inspect_strict_container", inspect)
    monkeypatch.setattr(docker_module, "_verify_strict_container_inspection", verify)
    monkeypatch.setattr(docker_module, "_probe_immutable_input_mounts", slow_mount_probe)
    monkeypatch.setattr(docker_module, "_probe_strict_container", executable_probe)

    async def run() -> tuple[list[Event], _RecordingProvider]:
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="docker"),
            _HostedFactory(pre_create_candidate="docker", runner=runner),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        return await _run(app, "sess_slow_docker_final_evidence"), provider

    events, provider = asyncio.run(run())

    assert probe_cycles == 1
    assert provider.requests == []
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["execution_admission"]["refusals"][0]["code"] == "stale_evidence"


@pytest.mark.parametrize("failure_mode", ["cancel", "nonzero_result", "timeout", "transport"])
@pytest.mark.parametrize("runner_kind", ["strict", "tool_observer", "egress_owned"])
def test_docker_final_probe_settles_guest_before_factory_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_mode: str,
    runner_kind: str,
) -> None:
    container_id = "a" * 64
    image_id = "sha256:" + "b" * 64
    image_reference = "registry.example/cayu@sha256:" + "c" * 64
    restrictions = DockerWorkloadRestrictions()
    observed_at = datetime.now(UTC)
    evidence = docker_module._DockerRuntimeEvidence(
        container_id=container_id,
        image_id=image_id,
        image_reference=image_reference,
        network_mode="none",
        default_cwd="/workspace",
        runtime=None,
        seccomp_profile_sha256=None,
        restrictions=restrictions,
        image_identity=DockerImageIdentity(reference=image_reference),
        toolchain_profile_fingerprint=None,
        required_executables=(),
        executable_availability=(),
        immutable_input_mounts=(),
        observed_at=observed_at,
        valid_until=observed_at + timedelta(seconds=300),
    )
    from tests.egress.test_docker_reconnect import setup as setup_reconnect

    from cayu.egress._docker_reconnect import DockerEgressReconnectError, _OwnedDockerRunner

    claim = None
    if runner_kind == "egress_owned":
        reconnect_docker, reconnect_adapter, manager, identity = setup_reconnect(tmp_path)
        claim = manager.claim(identity["allocation_id"])
        claim.read()
    runner_type = _OwnedDockerRunner if runner_kind == "egress_owned" else DockerRunner
    runner = runner_type(
        "strict",
        image=image_reference,
        default_cwd="/workspace",
        close_action="none",
        docker_path="/usr/bin/docker",
        credential_mode="trusted_tool",
        allow_raw_secret_env=False,
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
        _container_id=container_id,
        _runtime_evidence=evidence if runner_kind == "strict" else None,
        env_overlay={"CAYU_TEST_PRIVATE": "private-probe-860"} if runner_kind != "strict" else None,
        _env_overlay_secret_values_present=runner_kind != "strict",
    )
    if claim is not None:
        runner._owner, runner._manager = claim, manager
        claim.runner = runner
    probe_env_files = []
    probe_dispatched = asyncio.Event()
    probe_finished = asyncio.Event()
    cleanup_dispatched = asyncio.Event()
    allow_cleanup = asyncio.Event()
    guest_active = False

    async def inspect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"Id": container_id, "Image": image_id, "State": {"Running": True}}

    def verify(*args: Any, **kwargs: Any) -> tuple[str, str]:
        del args, kwargs
        return image_id, image_reference

    async def dispatch(command, **kwargs):
        nonlocal guest_active
        args = command.argv[1:]
        if any("read pid process_group" in value for value in args):
            cleanup_dispatched.set()
            await allow_cleanup.wait()
            guest_active = False
            probe_finished.set()
            return ExecResult()
        probe_marker = "id -u" if runner_kind == "strict" else "name=$1"
        if args[0] == "exec" and any(probe_marker in value for value in args):
            if runner_kind != "strict":
                path = Path(args[args.index("--env-file") + 1])
                probe_env_files.append(path)
                assert path.exists()
                assert "private-probe-860" not in repr(command.argv)
                assert "private-probe-860" not in kwargs["output_redactor"].redact_text(
                    "private-probe-860"
                )
                assert "private-probe-860" not in repr(kwargs["env"])
            guest_active = True
            probe_dispatched.set()
            if failure_mode == "cancel":
                await probe_finished.wait()
            elif failure_mode == "timeout":
                return ExecResult(exit_code=1, timed_out=True)
            elif failure_mode == "transport":
                raise ConnectionError("Docker probe transport disconnected")
            else:
                return ExecResult(exit_code=1, stderr="Docker stream disconnected")
            guest_active = False
            return _completed_docker_probe_result(args, stdout=restrictions.user)
        if args[0] == "exec":
            return _completed_docker_probe_result(args)
        return ExecResult()

    monkeypatch.setattr(docker_module, "_inspect_strict_container", inspect)
    monkeypatch.setattr(docker_module, "_verify_strict_container_inspection", verify)
    monkeypatch.setattr(docker_module, "run_subprocess", dispatch)
    monkeypatch.setattr(docker_module, "_require_docker", lambda path=None: "/usr/bin/docker")

    class ReleasingDockerFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.release_actions: list[EnvironmentFactoryReleaseAction] = []
            self.release_guest_states: list[bool] = []

        def execution_admission_candidate(
            self,
            request: EnvironmentFactoryRequest,
        ) -> ExecutionAdmissionCandidate:
            if runner_kind != "strict":
                return runner.execution_admission_candidate_for(request.execution_requirements)
            del request
            return _candidate("docker", state="declared")

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                self.release_actions.append(action)
                self.release_guest_states.append(guest_active)
                if claim is not None:
                    claim.close()

            return EnvironmentFactoryResult(
                environment=Environment(
                    EnvironmentSpec(name=request.environment_name),
                    runner=runner,
                ),
                reconnect_metadata={"container_id": container_id},
                release=release,
            )

    async def run() -> tuple[asyncio.Task[list[Event]], ReleasingDockerFactory]:
        provider = _RecordingProvider()
        factory = ReleasingDockerFactory()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="docker"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SearchTextTool()] if runner_kind != "strict" else [],
            execution_requirements=(
                ExecutionRequirements.trusted()
                if runner_kind != "strict"
                else ExecutionRequirements.trusted(
                    cleanup="confirmed",
                    minimum_evidence="live_verified",
                )
            ),
        )
        task = asyncio.create_task(_run(app, "sess_cancel_docker_final_probe"))
        await asyncio.wait_for(probe_dispatched.wait(), timeout=10)
        if failure_mode == "cancel":
            task.cancel("cancel Docker final evidence probe")
            assert task.cancelling() == 1
        await asyncio.wait_for(cleanup_dispatched.wait(), timeout=10)
        assert guest_active is True
        assert factory.release_actions == []
        assert provider.requests == []
        if runner_kind != "strict" and failure_mode == "cancel":
            # Caller cancellation must not unlink the still-running CLI's file.
            assert probe_env_files and all(path.exists() for path in probe_env_files)
        try:
            with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
                runner._ensure_exec_open()
            with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
                await DockerRunner.reconnect_strict(
                    "strict",
                    container_id=container_id,
                    image_identity=DockerImageIdentity(reference=image_reference),
                    workload_restrictions=restrictions,
                    docker_path="/usr/bin/docker",
                )
            if claim is not None:
                journal_before = dict(claim.journal)
                with pytest.raises(DockerEgressReconnectError, match="ownership_conflict"):
                    await reconnect_adapter.prepare_reconnect(
                        session_id=identity["session_id"],
                        environment_name=identity["environment_name"],
                        grants=(),
                        broker=None,
                        reconnect_metadata=identity,
                    )
                assert claim.journal == journal_before
                assert not claim.closed
                assert all(args[:2] == ["context", "inspect"] for args in reconnect_docker.commands)
        finally:
            allow_cleanup.set()
        if failure_mode == "cancel":
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert raised.value.args == ("cancel Docker final evidence probe",)
            assert task.cancelled() is True
        else:
            events = await task
            assert any(event.type is EventType.SESSION_FAILED for event in events)
            assert task.cancelled() is False
        assert factory.release_actions
        assert set(factory.release_actions) == {
            EnvironmentFactoryReleaseAction.PRESERVE
            if failure_mode == "cancel"
            else EnvironmentFactoryReleaseAction.DISCARD
        }
        assert factory.release_guest_states
        assert not any(factory.release_guest_states)
        assert all(not path.exists() for path in probe_env_files)
        if claim is not None:
            assert claim.closed
            replacement_claim = manager.claim(identity["allocation_id"])
            replacement_claim.read()
            assert replacement_claim.journal == journal_before
            replacement_claim.close()
        assert provider.requests == []
        return task, factory

    task, factory = asyncio.run(run())

    assert task.cancelled() is (failure_mode == "cancel")
    assert factory.release_actions


def test_expired_exposure_refuses_model_authored_tool_before_its_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExpiringEvidenceRunner(_EvidenceRunner):
        current_candidate: ExecutionAdmissionCandidate | None = None

        def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
            return self.current_candidate or super().execution_admission_candidate()

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            observed_at = datetime.now(UTC)
            self.current_candidate = ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    claims=(
                        ExecutionCapabilityClaim.live_verified(
                            "confirmed_cleanup",
                            observation="supported",
                            observed_at=observed_at,
                            valid_until=observed_at + timedelta(seconds=1),
                        ),
                    ),
                ),
            )
            return self.current_candidate

    class ToolProvider(ModelProvider):
        name = "tool-provider"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.tool_call(
                id="call_after_expiry",
                name="effect",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    class EffectTool(Tool):
        spec = ToolSpec(
            name="effect",
            description="Record an externally visible effect.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(content="effect completed")

    async def run() -> tuple[list[Event], ToolProvider, EffectTool]:
        provider = ToolProvider()
        tool = EffectTool()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(
                pre_create_candidate="hosted",
                runner=ExpiringEvidenceRunner("hosted"),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def delay_after_tool_intent(event: Event) -> Event:
            persisted = await original_emit(event)
            if event.type is EventType.TOOL_CALL_STARTED:
                await asyncio.sleep(1.1)
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", delay_after_tool_intent)
        return await _run(app, "sess_expired_before_tool_dispatch"), provider, tool

    events, provider, tool = asyncio.run(run())

    assert len(provider.requests) == 1
    assert tool.calls == 0
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["execution_admission"]["refusals"][0]["code"] == "stale_evidence"


def test_expired_exposure_refuses_background_provider_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExpiringEvidenceRunner(_EvidenceRunner):
        current_candidate: ExecutionAdmissionCandidate | None = None

        def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
            return self.current_candidate or super().execution_admission_candidate()

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            observed_at = datetime.now(UTC)
            self.current_candidate = ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    claims=(
                        ExecutionCapabilityClaim.live_verified(
                            "confirmed_cleanup",
                            observation="supported",
                            observed_at=observed_at,
                            valid_until=observed_at + timedelta(seconds=2),
                        ),
                    ),
                ),
            )
            return self.current_candidate

    class RecordingAdapter(ProviderOperationAdapter):
        def __init__(self) -> None:
            self.start_calls = 0

        async def start(
            self,
            request: ProviderOperationStartRequest,
        ) -> ProviderOperationConnection:
            del request
            self.start_calls += 1
            raise AssertionError("expired exposure reached background provider dispatch")

        async def retrieve(
            self,
            state: ProviderOperationState,
        ) -> ProviderOperationSnapshot:
            del state
            raise AssertionError("undispatched operation cannot be retrieved")

        async def reconnect(
            self,
            state: ProviderOperationState,
        ) -> ProviderOperationConnection:
            del state
            raise AssertionError("undispatched operation cannot be reconnected")

    class BackgroundProvider(ModelProvider):
        name = "background-provider"

        def __init__(self) -> None:
            self.adapter = RecordingAdapter()

        @property
        def provider_operation_mode(self) -> ProviderOperationMode:
            return ProviderOperationMode.BACKGROUND

        @property
        def provider_operations(self) -> ProviderOperationAdapter:
            return self.adapter

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            raise AssertionError("background provider used synchronous stream")

    async def run() -> tuple[list[Event], BackgroundProvider]:
        provider = BackgroundProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(
                pre_create_candidate="hosted",
                runner=ExpiringEvidenceRunner("hosted"),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def expire_after_durable_start_intent(event: Event) -> Event:
            persisted = await original_emit(event)
            if event.type is EventType.PROVIDER_OPERATION_STARTING:
                await asyncio.sleep(2.1)
            return persisted

        monkeypatch.setattr(
            app._event_writer,
            "emit",
            expire_after_durable_start_intent,
        )
        return await _run(app, "sess_expired_before_background_start"), provider

    events, provider = asyncio.run(run())

    assert provider.adapter.start_calls == 0
    assert EventType.PROVIDER_OPERATION_STARTING in {event.type for event in events}
    model_error = next(event for event in events if event.type is EventType.MODEL_ERROR)
    assert model_error.payload["error_type"] == "ExecutionAdmissionError"
    assert model_error.payload["execution_admission"]["refusals"][0]["code"] == ("stale_evidence")
    # Once the durable start intent is visible, restart-safe recovery must
    # conservatively retain it even though this process proves no adapter call.
    interrupted = next(event for event in events if event.type is EventType.SESSION_INTERRUPTED)
    assert interrupted.payload == {
        "interruption_type": "provider_operation_unavailable",
        "recovery_reason": "ambiguous_submission",
        "duplicate_request_risk": True,
    }


@pytest.mark.parametrize(
    "drift_field",
    [None, "environment_fingerprint", "image_fingerprint", "toolchain_profile_fingerprint"],
)
def test_runtime_renews_expired_evidence_without_changing_exact_environment_identity(
    monkeypatch: pytest.MonkeyPatch,
    drift_field: str | None,
) -> None:
    class DispatchDelayedProvider(_RecordingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.delay_next_dispatch = False

        @property
        def provider_operation_mode(self):
            if self.delay_next_dispatch:
                self.delay_next_dispatch = False
                import time

                time.sleep(1.1)
            return super().provider_operation_mode

    async def run() -> tuple[list[Event], DispatchDelayedProvider, _RenewingEvidenceRunner]:
        provider = DispatchDelayedProvider()
        runner = _RenewingEvidenceRunner(drift_field=drift_field)
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(pre_create_candidate="hosted", runner=runner),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def arm_delay_after_exposure(event: Event) -> Event:
            persisted = await original_emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "exposure"
            ):
                provider.delay_next_dispatch = True
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", arm_delay_after_exposure)
        events = await _run(app, f"sess_renewed_provider_{drift_field}")
        return events, provider, runner

    events, provider, runner = asyncio.run(run())

    assert runner.refresh_calls == 1
    if drift_field is None:
        assert len(provider.requests) == 1
        assert EventType.SESSION_COMPLETED in {event.type for event in events}
    else:
        assert provider.requests == []
        failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
        assert failed.payload["execution_admission"]["refusals"][0]["code"] == (
            "environment_authority_mismatch"
        )


@pytest.mark.parametrize("renewed_proof", ["verified", "unavailable", "wrong_probe"])
def test_search_text_renews_stale_executable_with_missing_native_alternative(
    monkeypatch: pytest.MonkeyPatch,
    renewed_proof: str,
) -> None:
    class SearchRunner(_RenewingEvidenceRunner):
        def _live_candidate(self, *, valid_for_seconds, drift_field=None):
            candidate = super()._live_candidate(
                valid_for_seconds=valid_for_seconds, drift_field=drift_field
            )
            evidence = candidate.evidence
            assert evidence is not None
            now = datetime.now(UTC)
            proof = renewed_proof if self.refresh_calls else "verified"
            executable = admission_module.ExecutionExecutableEvidence(
                executable="rg",
                state="unavailable" if proof == "unavailable" else "live_verified",
                observed_at=now if proof != "unavailable" else None,
                valid_until=(
                    now + timedelta(seconds=valid_for_seconds) if proof != "unavailable" else None
                ),
                reason_code="executable_unavailable" if proof == "unavailable" else None,
                remediation_code="install_executable" if proof == "unavailable" else None,
                requirement_fingerprint=(
                    "sha256:" + "0" * 64
                    if proof == "wrong_probe"
                    else ToolExecutableRequirement(executable="rg").fingerprint
                ),
            )
            return candidate.model_copy(
                update={
                    "evidence": evidence.model_copy(
                        update={
                            "tool_requirements": admission_module.ExecutionToolRequirementEvidence(
                                environment_fingerprint=evidence.environment_fingerprint,
                                image_fingerprint=evidence.image_fingerprint,
                                executables=(executable,),
                            )
                        }
                    )
                }
            )

    async def run():
        provider = _RecordingProvider()
        runner = SearchRunner()
        runner.current_candidate = runner._live_candidate(valid_for_seconds=60)
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="hosted"), runner=runner), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"), tools=[SearchTextTool()]
        )
        original_emit = app._event_writer.emit

        async def expire_after_exposure(event):
            persisted = await original_emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "exposure"
            ):
                await asyncio.sleep(1.1)
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", expire_after_exposure)
        events = await _run(app, f"sess_search_renewal_{renewed_proof}")
        return events, provider, runner

    events, provider, runner = asyncio.run(run())
    assert runner.refresh_calls == 1, [(event.type, event.payload) for event in events]
    if renewed_proof == "verified":
        assert len(provider.requests) == 1
        assert EventType.SESSION_COMPLETED in {event.type for event in events}
    else:
        assert provider.requests == []
        assert EventType.SESSION_COMPLETED not in {event.type for event in events}
        failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
        assert any(
            refusal["tool_name"] == "search_text" and refusal["executable"] == "rg"
            for refusal in failed.payload["execution_admission"]["refusals"]
        )


@pytest.mark.parametrize("mismatch", ["runner", "requirements", "type"])
def test_public_admission_rejects_mismatched_observer_before_provider(mismatch):
    class MismatchedRunner(_EvidenceRunner):
        def execution_admission_observer(self, requirements):
            if mismatch == "type":
                return object()
            return RunnerExecutionAdmissionObserver(
                _EvidenceRunner("hosted") if mismatch == "runner" else self,
                ExecutionRequirements.trusted() if mismatch == "requirements" else requirements,
            )

    async def run():
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(pre_create_candidate="hosted", runner=MismatchedRunner("hosted")),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(cleanup="confirmed"),
        )
        return await _run(app, f"observer_mismatch_{mismatch}"), provider

    events, provider = asyncio.run(run())
    assert provider.requests == []
    assert EventType.SESSION_COMPLETED not in {event.type for event in events}
    assert EventType.SESSION_FAILED in {event.type for event in events}


def test_shared_runner_preserves_distinct_observer_state_through_public_renewal(monkeypatch):
    class Observer(RunnerExecutionAdmissionObserver):
        def __post_init__(self):
            super().__post_init__()
            object.__setattr__(self, "state", {"candidate": None, "collections": 0, "refreshes": 0})

        def snapshot(self):
            assert self.state["candidate"] is not None
            return self.state["candidate"]

        async def collect(self):
            self.state["collections"] += 1
            self.state["candidate"] = self.runner._live_candidate(valid_for_seconds=1)
            return self.snapshot()

        async def refresh(self):
            self.state["refreshes"] += 1
            self.state["candidate"] = self.runner._live_candidate(valid_for_seconds=60)

    class SharedRunner(_RenewingEvidenceRunner):
        def __init__(self):
            super().__init__()
            self.observers = []

        def execution_admission_observer(self, requirements):
            observer = Observer(self, requirements)
            self.observers.append(observer)
            return observer

        async def refresh_execution_admission(self):
            raise AssertionError("Renewal must use the request-scoped observer.")

        async def collect_execution_admission_candidate(self):
            raise AssertionError("Collection must use the request-scoped observer.")

    async def exercise():
        runner = SharedRunner()

        async def run(index):
            provider = _RecordingProvider()
            app = CayuApp(enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment_factory(
                EnvironmentSpec(name="hosted"),
                _HostedFactory(pre_create_candidate="hosted", runner=runner),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                execution_requirements=ExecutionRequirements.trusted(
                    cleanup="confirmed", minimum_evidence="live_verified"
                ),
            )
            original_emit = app._event_writer.emit

            async def expire_after_exposure(event):
                persisted = await original_emit(event)
                if (
                    event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                    and event.payload["phase"] == "exposure"
                ):
                    await asyncio.sleep(1.1)
                return persisted

            monkeypatch.setattr(app._event_writer, "emit", expire_after_exposure)
            return await _run(app, f"observer_session_{index}"), provider

        return await asyncio.gather(run(0), run(1)), runner

    outcomes, runner = asyncio.run(exercise())
    assert len(runner.observers) == 2
    assert runner.observers[0] is not runner.observers[1]
    assert runner.observers[0].state is not runner.observers[1].state
    for observer in runner.observers:
        assert observer.state["collections"] == 1
        assert observer.state["refreshes"] == 1
    for events, provider in outcomes:
        assert len(provider.requests) == 1
        assert EventType.SESSION_COMPLETED in {event.type for event in events}


def test_runtime_renews_expired_evidence_at_exact_tool_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ToolProvider(ModelProvider):
        name = "renewing-tool-provider"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_after_renewal",
                    name="effect",
                    arguments={},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class EffectTool(Tool):
        spec = ToolSpec(
            name="effect",
            description="Record an externally visible effect.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(content="effect completed")

    async def run() -> tuple[list[Event], ToolProvider, EffectTool, _RenewingEvidenceRunner]:
        provider = ToolProvider()
        tool = EffectTool()
        runner = _RenewingEvidenceRunner()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(pre_create_candidate="hosted", runner=runner),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def delay_after_tool_intent(event: Event) -> Event:
            persisted = await original_emit(event)
            if event.type is EventType.TOOL_CALL_STARTED:
                await asyncio.sleep(1.1)
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", delay_after_tool_intent)
        events = await _run(app, "sess_renewed_tool_dispatch")
        return events, provider, tool, runner

    events, provider, tool, runner = asyncio.run(run())

    assert len(provider.requests) == 2
    assert tool.calls == 1
    assert runner.refresh_calls == 1
    assert EventType.SESSION_COMPLETED in {event.type for event in events}


@pytest.mark.parametrize("delay_seam", ["billing", "provider_child"])
@pytest.mark.parametrize("renewal_outcome", ["verified", "identity_drift", "cancel"])
def test_runtime_renews_expired_evidence_at_compaction_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    delay_seam: str,
    renewal_outcome: str,
) -> None:
    refreshes_at_dispatch: list[int] = []

    class CompactionRunner(_RenewingEvidenceRunner):
        def __init__(self) -> None:
            super().__init__(
                drift_field="image_fingerprint" if renewal_outcome == "identity_drift" else None
            )
            self.refresh_started = asyncio.Event()

        def _live_candidate(self, *, valid_for_seconds, drift_field=None):
            candidate = super()._live_candidate(
                valid_for_seconds=valid_for_seconds, drift_field=drift_field
            )
            evidence = candidate.evidence
            assert evidence is not None
            now = datetime.now(UTC)
            return candidate.model_copy(
                update={
                    "evidence": evidence.model_copy(
                        update={
                            "tool_requirements": admission_module.ExecutionToolRequirementEvidence(
                                environment_fingerprint=evidence.environment_fingerprint,
                                image_fingerprint=evidence.image_fingerprint,
                                executables=(
                                    admission_module.ExecutionExecutableEvidence(
                                        executable="rg",
                                        state="live_verified",
                                        observed_at=now,
                                        valid_until=now + timedelta(seconds=valid_for_seconds),
                                        requirement_fingerprint=ToolExecutableRequirement(
                                            executable="rg"
                                        ).fingerprint,
                                    ),
                                ),
                            )
                        }
                    )
                }
            )

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            self.current_candidate = self._live_candidate(valid_for_seconds=2)
            return self.current_candidate

        async def refresh_execution_admission(self) -> None:
            if renewal_outcome == "cancel":
                self.refresh_calls += 1
                self.refresh_started.set()
                await asyncio.Event().wait()
            await super().refresh_execution_admission()

    class CompactionProvider(_RecordingProvider):
        async def billing_identity_for_request(self, request: ModelRequest) -> None:
            del request
            if delay_seam == "billing":
                await asyncio.sleep(2.1)
            return None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            refreshes_at_dispatch.append(runner.refresh_calls)
            self.requests.append(request)
            yield ModelStreamEvent.text_delta("compacted context")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    runner = CompactionRunner()
    runner.current_candidate = runner._live_candidate(valid_for_seconds=60)
    original_ensure_future = asyncio.ensure_future

    def delayed_provider_child(operation, *, loop=None):
        code = getattr(operation, "cr_code", None)
        if (
            delay_seam == "provider_child"
            and code is not None
            and code.co_qualname.endswith(
                "_await_owned_compaction_provider_stream.<locals>.capture_provider_abandonment"
            )
        ):

            async def delayed():
                await asyncio.sleep(2.1)
                return await operation

            return original_ensure_future(delayed(), loop=loop)
        return original_ensure_future(operation, loop=loop)

    monkeypatch.setattr(asyncio, "ensure_future", delayed_provider_child)

    async def run() -> tuple[
        list[Event],
        _RecordingProvider,
        CompactionProvider,
        CompactionRunner,
    ]:
        provider = _RecordingProvider()
        compaction_provider = CompactionProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="hosted"), runner=runner),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SearchTextTool()],
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="compaction-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        events: list[Event] = []

        async def consume_run() -> None:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_renewed_compaction_dispatch",
                    messages=[
                        Message.text("user", "old request"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current request"),
                    ],
                )
            ):
                events.append(event)

        task = asyncio.create_task(consume_run())
        if renewal_outcome == "cancel":
            await asyncio.wait_for(runner.refresh_started.wait(), timeout=10)
            task.cancel("cancel compaction admission renewal")
            assert task.cancelling() == 1
            try:
                await task
            except asyncio.CancelledError as exc:
                assert exc.args == ("cancel compaction admission renewal",)
            else:
                pytest.fail("Public run swallowed compaction admission cancellation")
            assert task.cancelled()
        else:
            await task
        return events, provider, compaction_provider, runner

    events, provider, compaction_provider, runner = asyncio.run(run())

    assert runner.refresh_calls == 1
    if renewal_outcome != "verified":
        assert refreshes_at_dispatch == []
        assert compaction_provider.requests == []
        assert provider.requests == []
        if renewal_outcome == "identity_drift":
            assert EventType.SESSION_FAILED in {event.type for event in events}
        assert EventType.SESSION_COMPLETED not in {event.type for event in events}
        return
    assert refreshes_at_dispatch == [1]
    assert len(compaction_provider.requests) == 1
    assert len(provider.requests) == 1
    assert EventType.SESSION_COMPLETED in {event.type for event in events}


def test_environment_admission_renewal_preserves_real_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingRenewalRunner(_RenewingEvidenceRunner):
        def __init__(self) -> None:
            super().__init__()
            self.refresh_started = asyncio.Event()

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            self.current_candidate = self._live_candidate(valid_for_seconds=2)
            return self.current_candidate

        async def refresh_execution_admission(self) -> None:
            self.refresh_calls += 1
            self.refresh_started.set()
            await asyncio.Event().wait()

    async def run() -> tuple[asyncio.Task[list[Event]], _RecordingProvider, BlockingRenewalRunner]:
        provider = _RecordingProvider()
        runner = BlockingRenewalRunner()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(pre_create_candidate="hosted", runner=runner),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def expire_after_exposure(event: Event) -> Event:
            persisted = await original_emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "exposure"
            ):
                await asyncio.sleep(2.1)
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", expire_after_exposure)
        task = asyncio.create_task(_run(app, "sess_cancel_admission_renewal"))
        await asyncio.wait_for(runner.refresh_started.wait(), timeout=10)
        task.cancel("cancel environment admission renewal")
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("cancel environment admission renewal",)
        assert task.cancelled() is True
        return task, provider, runner

    task, provider, runner = asyncio.run(run())

    assert task.cancelled() is True
    assert provider.requests == []
    assert runner.refresh_calls == 1


@pytest.mark.parametrize("failure", ["cancel", "timeout", "cancel_settlement", "cancel_lock"])
def test_cancelled_environment_admission_renewal_fences_binding_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from cayu.runtime import _environment_exposure as exposure_module
    from cayu.runtime import _environment_lifecycle as lifecycle_module

    monkeypatch.setattr(exposure_module, "_EXPOSURE_ADMISSION_SETTLEMENT_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(
        lifecycle_module, "_PRE_EXPOSURE_ADMISSION_SETTLEMENT_TIMEOUT_SECONDS", 0.03
    )
    settlement_started = asyncio.Event()
    original_wait = exposure_module._await_exposure_admission_settlement

    async def observe_wait(exposure, **kwargs):
        if exposure.admission.settlement_task is not None and not kwargs.get("background", False):
            settlement_started.set()
        return await original_wait(exposure, **kwargs)

    monkeypatch.setattr(exposure_module, "_await_exposure_admission_settlement", observe_wait)

    class TransferringCancellationRunner(_RenewingEvidenceRunner):
        def __init__(self) -> None:
            super().__init__()
            self.refresh_started = asyncio.Event()
            self.allow_settlement = asyncio.Event()
            self.settled = False
            self.cleanup_overtook_settlement = False

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            self.current_candidate = self._live_candidate(valid_for_seconds=2)
            return self.current_candidate

        async def refresh_execution_admission(self) -> None:
            self.refresh_calls += 1

            async def settle_dispatched_probe() -> None:
                await self.allow_settlement.wait()
                self.settled = True

            settlement = asyncio.create_task(settle_dispatched_probe())
            self.refresh_started.set()
            try:
                async with asyncio.timeout(0.03):
                    await asyncio.Event().wait()
            except BaseException as cancellation:
                attach_environment_factory_cleanup_settlement_task(
                    cancellation,
                    settlement,
                )
                raise

        async def close(self) -> None:
            self.cleanup_overtook_settlement = not self.settled
            await super().close()

    async def run() -> tuple[
        asyncio.Task[list[Event]],
        _RecordingProvider,
        TransferringCancellationRunner,
        _SwitchingBinding,
    ]:
        provider = _RecordingProvider()
        source_runner = _EvidenceRunner("hosted")
        renewal_runner = TransferringCancellationRunner()
        binding = _SwitchingBinding(renewal_runner)
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted",
            runner=source_runner,
            binding=binding,
            lifecycle=[],
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def expire_after_exposure(event: Event) -> Event:
            persisted = await original_emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "exposure"
            ):
                await asyncio.sleep(2.1)
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", expire_after_exposure)
        lock_holders = []
        original_terminal_wait = lifecycle_module.await_environment_exposure_settlement

        async def hold_terminal_lock(registered_environment):
            if failure == "cancel_lock" and not lock_holders:
                exposure = registered_environment.environment_exposure
                locked = asyncio.Event()

                async def hold():
                    async with exposure.admission.renewal_lock:
                        locked.set()
                        await renewal_runner.allow_settlement.wait()

                lock_holders.append(asyncio.create_task(hold()))
                await locked.wait()
            return await original_terminal_wait(registered_environment)

        monkeypatch.setattr(
            lifecycle_module, "await_environment_exposure_settlement", hold_terminal_lock
        )
        task = asyncio.create_task(_run(app, "sess_cancelled_renewal_settlement"))
        await asyncio.wait_for(renewal_runner.refresh_started.wait(), timeout=10)
        try:
            if failure == "cancel_settlement":
                await asyncio.wait_for(settlement_started.wait(), timeout=5)
            if failure != "timeout":
                task.cancel("cancel renewal with transferred owner")
                assert task.cancelling() == 1
            done, _ = await asyncio.wait((task,), timeout=2)
            assert task in done, "Caller must exit while settlement is still blocked"
            if failure != "timeout":
                with pytest.raises(asyncio.CancelledError) as raised:
                    await task
                assert raised.value.args == ("cancel renewal with transferred owner",)
                assert task.cancelled() is True
                assert task.cancelling() == 1
            else:
                events = await task
                assert any(event.type is EventType.SESSION_FAILED for event in events)
            assert binding.finalize_calls == 0
            assert not renewal_runner.settled
            assert factory.release_actions == []
            owner = app._environment_lifecycle._active_environment_setups[
                "sess_cancelled_renewal_settlement"
            ]
            assert owner.admission_settlement_task is not None
            assert not owner.admission_settlement_task.done()
            assert owner.cleanup_ready_for_retry
            with pytest.raises(RuntimeError, match="incomplete environment cleanup"):
                app._environment_lifecycle._require_no_retained_cleanup_for_session(
                    "sess_cancelled_renewal_settlement"
                )
            assert not await app.drain_environment_cleanups(timeout_s=0.01)
            persisted = await app.session_store.load_events("sess_cancelled_renewal_settlement")
            assert not any(event.type is EventType.SESSION_COMPLETED for event in persisted)
        finally:
            renewal_runner.allow_settlement.set()
            await asyncio.gather(*lock_holders)
            await asyncio.gather(task, return_exceptions=True)
            assert await app.drain_environment_cleanups(timeout_s=5)
            # Binding adoption transferred release ownership; cleanup belongs
            # to its finalizer, not the factory's unclaimed-result callback.
            assert factory.release_actions == []
            assert binding.finalize_calls == 1
            assert (
                "sess_cancelled_renewal_settlement"
                not in app._environment_lifecycle._active_environment_setups
            )
        return task, provider, renewal_runner, binding

    task, provider, runner, binding = asyncio.run(run())

    assert task.cancelled() is (failure != "timeout")
    assert provider.requests == []
    assert runner.refresh_calls == 1
    assert runner.settled is True
    assert runner.cleanup_overtook_settlement is False
    assert binding.finalize_calls == 1


def test_environment_admission_renewal_rejects_runner_generated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunnerGeneratedCancellation(_RenewingEvidenceRunner):
        cancel_snapshot = False

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            self.current_candidate = self._live_candidate(valid_for_seconds=2)
            return self.current_candidate

        def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
            if self.cancel_snapshot:
                raise asyncio.CancelledError("runner-generated cancellation")
            return super().execution_admission_candidate()

        async def refresh_execution_admission(self) -> None:
            self.refresh_calls += 1
            self.cancel_snapshot = True

    async def run() -> tuple[list[Event], _RecordingProvider, RunnerGeneratedCancellation]:
        provider = _RecordingProvider()
        runner = RunnerGeneratedCancellation()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(pre_create_candidate="hosted", runner=runner),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def expire_after_exposure(event: Event) -> Event:
            persisted = await original_emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "exposure"
            ):
                await asyncio.sleep(2.1)
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", expire_after_exposure)
        return await _run(app, "sess_runner_generated_renewal_cancellation"), provider, runner

    events, provider, runner = asyncio.run(run())

    assert runner.refresh_calls == 1
    assert provider.requests == []
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["execution_admission"]["refusals"][0]["code"] == (
        "missing_final_evidence"
    )


def test_environment_admission_renewal_reconciles_acknowledgement_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AcknowledgementLossRunner(_RenewingEvidenceRunner):
        def __init__(self) -> None:
            super().__init__()
            self.refresh_started = asyncio.Event()
            self.allow_settlement = asyncio.Event()
            self.settled = False

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            self.current_candidate = self._live_candidate(valid_for_seconds=2)
            return self.current_candidate

        async def refresh_execution_admission(self) -> None:
            self.refresh_calls += 1

            async def settle_dispatched_probe() -> None:
                await self.allow_settlement.wait()
                self.current_candidate = self._live_candidate(valid_for_seconds=60)
                self.settled = True

            settlement = asyncio.create_task(settle_dispatched_probe())
            failure = ConnectionError("renewal acknowledgement lost")
            attach_environment_factory_cleanup_settlement_task(failure, settlement)
            self.refresh_started.set()
            raise failure

    async def run() -> tuple[list[Event], _RecordingProvider, AcknowledgementLossRunner]:
        provider = _RecordingProvider()
        runner = AcknowledgementLossRunner()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(pre_create_candidate="hosted", runner=runner),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def expire_after_exposure(event: Event) -> Event:
            persisted = await original_emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "exposure"
            ):
                await asyncio.sleep(2.1)
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", expire_after_exposure)
        task = asyncio.create_task(_run(app, "sess_renewal_ack_loss"))
        await asyncio.wait_for(runner.refresh_started.wait(), timeout=10)
        await asyncio.sleep(0)
        assert task.done() is False
        assert provider.requests == []
        runner.allow_settlement.set()
        events = await asyncio.wait_for(task, timeout=10)
        return events, provider, runner

    events, provider, runner = asyncio.run(run())

    assert runner.refresh_calls == 1
    assert runner.settled is True
    assert len(provider.requests) == 1
    assert EventType.SESSION_COMPLETED in {event.type for event in events}


@pytest.mark.parametrize("initial_settlement_fails", [False, True])
def test_environment_admission_renewal_settles_before_binding_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    initial_settlement_fails: bool,
) -> None:
    class DeferredRenewalRunner(_RenewingEvidenceRunner):
        def __init__(self) -> None:
            super().__init__()
            self.refresh_started = asyncio.Event()
            self.allow_settlement = asyncio.Event()
            self.settled = False
            self.retry_calls = 0
            self.cleanup_overtook_settlement = False

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            self.current_candidate = self._live_candidate(valid_for_seconds=2)
            return self.current_candidate

        async def refresh_execution_admission(self) -> None:
            self.refresh_calls += 1
            if initial_settlement_fails:
                # Publishing fresh evidence does not prove that the opaque
                # probe which produced it reached a safe terminal boundary.
                self.current_candidate = self._live_candidate(valid_for_seconds=60)

            async def settle_dispatched_probe() -> None:
                await self.allow_settlement.wait()
                if initial_settlement_fails:
                    raise RuntimeError("initial renewal settlement failed")
                self.settled = True

            def retry_dispatched_probe() -> asyncio.Task[None]:
                self.retry_calls += 1

                async def settle_retry() -> None:
                    self.settled = True

                return asyncio.create_task(settle_retry())

            settlement = asyncio.create_task(settle_dispatched_probe())
            if initial_settlement_fails:
                register_environment_factory_cleanup_retry(
                    settlement,
                    retry_dispatched_probe,
                )
            failure = RuntimeError("renewal probe acknowledgement lost")
            attach_environment_factory_cleanup_settlement_task(failure, settlement)
            self.refresh_started.set()
            raise failure

        async def close(self) -> None:
            self.cleanup_overtook_settlement = not self.settled
            await super().close()

    async def run() -> tuple[
        list[Event],
        _RecordingProvider,
        DeferredRenewalRunner,
        _SwitchingBinding,
    ]:
        provider = _RecordingProvider()
        source_runner = _EvidenceRunner("hosted")
        renewal_runner = DeferredRenewalRunner()
        binding = _SwitchingBinding(renewal_runner)
        lifecycle: list[str] = []
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted",
            runner=source_runner,
            binding=binding,
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.trusted(
                cleanup="confirmed",
                minimum_evidence="live_verified",
            ),
        )
        original_emit = app._event_writer.emit

        async def expire_after_exposure(event: Event) -> Event:
            persisted = await original_emit(event)
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "exposure"
            ):
                await asyncio.sleep(2.1)
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", expire_after_exposure)
        task = asyncio.create_task(_run(app, "sess_renewal_settlement_fence"))
        await asyncio.wait_for(renewal_runner.refresh_started.wait(), timeout=10)
        await asyncio.sleep(0)
        assert task.done() is False
        assert binding.finalize_calls == 0
        renewal_runner.allow_settlement.set()
        events = await asyncio.wait_for(task, timeout=10)
        return events, provider, renewal_runner, binding

    events, provider, runner, binding = asyncio.run(run())

    assert provider.requests == []
    assert runner.refresh_calls == 1
    assert runner.retry_calls == int(initial_settlement_fails)
    assert runner.settled is True
    assert runner.cleanup_overtook_settlement is False
    assert binding.finalize_calls == 1
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["execution_admission"]["refusals"][0]["code"] == (
        "missing_final_evidence" if initial_settlement_fails else "stale_evidence"
    )


@pytest.mark.parametrize("lost_phase", ["admission", "exposure"])
def test_exposure_publication_acknowledgement_loss_releases_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    lost_phase: str,
) -> None:
    class ReleasableFactory(_HostedFactory):
        def __init__(self) -> None:
            super().__init__(
                pre_create_candidate="hosted",
                runner=_EvidenceRunner("hosted"),
            )
            self.release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            self.requests.append(request)

            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                self.release_actions.append(action)
                await self.runner.close()

            return EnvironmentFactoryResult(
                environment=Environment(
                    EnvironmentSpec(name=request.environment_name),
                    runner=self.runner,
                ),
                reconnect_metadata={"allocation_id": request.session_id},
                release=release,
            )

    async def run() -> tuple[list[Event], list[Event], _RecordingProvider, ReleasableFactory]:
        provider = _RecordingProvider()
        factory = ReleasableFactory()
        store = InMemorySessionStore()
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        original_emit = app._event_writer.emit
        failed_once = False

        async def commit_then_lose_acknowledgement(event: Event) -> Event:
            nonlocal failed_once
            persisted = await original_emit(event)
            if (
                not failed_once
                and event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == lost_phase
            ):
                failed_once = True
                raise ConnectionError(f"{lost_phase} publication acknowledgement lost")
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", commit_then_lose_acknowledgement)
        session_id = f"sess_{lost_phase}_ack_loss"
        emitted = await _run(app, session_id)
        durable = await store.load_events(session_id)
        return emitted, durable, provider, factory

    events, durable_events, provider, factory = asyncio.run(run())

    assert provider.requests == []
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    assert factory.runner.is_closed is True
    assert EventType.SESSION_FAILED in {event.type for event in events}
    durable_transitions = [
        environment_lifecycle_transition_from_event(event)
        for event in durable_events
        if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
    ]
    assert [(item.phase.value, item.outcome.value) for item in durable_transitions[-2:]] == [
        (lost_phase, "admitted" if lost_phase == "admission" else "exposed"),
        ("release", "released"),
    ]


def test_factory_binding_can_supply_the_admitted_runner() -> None:
    async def run() -> tuple[list[Event], _RecordingProvider, _BindingRunnerFactory]:
        provider = _RecordingProvider()
        factory = _BindingRunnerFactory(_SwitchingBinding(_EvidenceRunner("hosted")))
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, "sess_binding_supplies_runner"), provider, factory

    events, provider, factory = asyncio.run(run())

    assert len(factory.requests) == 1
    assert len(provider.requests) == 1
    assert EventType.SESSION_COMPLETED in {event.type for event in events}


def test_generic_factory_without_evidence_is_refused_before_create() -> None:
    async def run() -> tuple[list[Event], _RecordingProvider, _HostedFactory]:
        provider = _RecordingProvider()
        factory = _HostedFactory(
            pre_create_candidate=None,
            runner=_EvidenceRunner("hosted"),
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, "sess_generic_precreate_refused"), provider, factory

    events, provider, factory = asyncio.run(run())

    assert factory.requests == []
    assert provider.requests == []
    assert EventType.ENVIRONMENT_FACTORY_FAILED in {event.type for event in events}


def test_binding_cannot_switch_the_admitted_execution_candidate() -> None:
    async def run() -> tuple[
        list[Event],
        _RecordingProvider,
        _SwitchingBinding,
        _ReleasableHostedFactory,
        _EvidenceRunner,
        _EvidenceRunner,
        list[str],
        dict[str, Any],
    ]:
        provider = _RecordingProvider()
        lifecycle: list[str] = []
        original_runner = _EvidenceRunner("hosted-a")
        replacement_runner = _EvidenceRunner("hosted-b")
        binding = _SwitchingBinding(replacement_runner, lifecycle=lifecycle)
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted-a",
            runner=original_runner,
            binding=binding,
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        events = await _run(app, "sess_binding_candidate_switch")
        checkpoint = await app.session_store.load_checkpoint("sess_binding_candidate_switch")
        assert checkpoint is not None
        return (
            events,
            provider,
            binding,
            factory,
            original_runner,
            replacement_runner,
            lifecycle,
            checkpoint,
        )

    (
        events,
        provider,
        binding,
        factory,
        original_runner,
        replacement_runner,
        lifecycle,
        checkpoint,
    ) = asyncio.run(run())

    assert provider.requests == []
    assert binding.finalize_calls == 1
    assert binding.finalize_outcomes == ["interrupted"]
    assert factory.release_actions == []
    assert lifecycle == ["binding.finalize"]
    assert original_runner.is_closed is True
    assert replacement_runner.is_closed is True
    assert checkpoint["environment_factory_reconnect"] == {
        "hosted": {"allocation_id": "sess_binding_candidate_switch"}
    }
    assert checkpoint["environment_factory_allocation_owner"] == {
        "hosted": "sess_binding_candidate_switch"
    }
    finalize_started = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED
    )
    assert finalize_started.payload["outcome"] == "failed"
    assert finalize_started.payload["terminal_outcome"] == "failed"
    assert finalize_started.payload["factory_allocation_action"] == "preserve"
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error_type"] == "ExecutionAdmissionError"
    assert "environment_factory_release" not in failed.payload


@pytest.mark.parametrize("result_present", [False, True])
@pytest.mark.parametrize("publication_failure", ["none", "error", "cancel"])
@pytest.mark.parametrize("group_source", ["none", "primary", "progress", "primary_cancel"])
def test_factory_failure_progress_preserves_real_caller_cancellation(
    monkeypatch,
    result_present,
    publication_failure,
    group_source,
):
    import sys

    primary = RuntimeError("admission probe transport failed")
    if group_source == "primary":
        primary = ExceptionGroup(
            "primary probe failures",
            [primary, ExceptionGroup("nested", [ValueError("probe detail")])],
        )
    secondary = RuntimeError("factory failure event persistence failed")
    progress_detail = ExceptionGroup("progress detail", [RuntimeError("diagnostic failure")])
    releases = []
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    later_publication_started = asyncio.Event()
    progress_cancellations = []
    progress_errors = []
    observed_primary = []

    class FailingFactory(_HostedFactory):
        async def create(self, request):
            if result_present:

                async def release(action):
                    releases.append(action)
                    release_started.set()
                    await allow_release.wait()

                return EnvironmentFactoryResult(
                    environment=Environment(EnvironmentSpec(name="hosted"), runner=self.runner),
                    release=release,
                )
            raise primary

    async def scenario():
        nonlocal primary
        if group_source == "primary_cancel":
            child_started = asyncio.Event()

            async def failed_child():
                child_started.set()
                await asyncio.Event().wait()

            child = asyncio.create_task(failed_child())
            await child_started.wait()
            child.cancel("earlier child cancellation")
            try:
                await child
            except asyncio.CancelledError as earlier:
                primary = BaseExceptionGroup("primary failures", [primary, earlier])
            assert child.cancelled() and child.cancelling() == 1
        factory = FailingFactory(pre_create_candidate="hosted", runner=_EvidenceRunner("hosted"))
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted", lifecycle_policy=EnvironmentLifecyclePolicy()),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        progress_started = asyncio.Event()
        original_emit = app._event_writer.emit

        async def block_failure_progress(event):
            if result_present and event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED:
                raise primary
            if event.type is EventType.ENVIRONMENT_FACTORY_FAILED:
                if publication_failure == "error":
                    raise secondary
                if publication_failure == "cancel":
                    later_publication_started.set()
                    await asyncio.Event().wait()
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_PROGRESS
                and event.payload["operation"] == "factory"
                and event.payload["status"] == "failed"
            ):
                observed_primary.append(sys.exception())
                progress_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    progress_cancellations.append(cancellation)
                    if group_source == "progress":
                        aggregate = BaseExceptionGroup(
                            "progress failures", [cancellation, progress_detail]
                        )
                        progress_errors.append(aggregate)
                        raise aggregate from None
                    progress_errors.append(cancellation)
                    raise
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_failure_progress)
        task = asyncio.create_task(_run(app, "factory-failure-progress-cancel"))
        await asyncio.wait_for(progress_started.wait(), timeout=10)
        accepted_primary = observed_primary[0]
        assert isinstance(accepted_primary, BaseException)
        if group_source == "primary_cancel" and not result_present:
            # The opaque factory boundary safely reconstructs cancellation
            # groups before the lifecycle accepts them. Preserve that exact
            # accepted object, not the provider-owned pre-boundary wrapper.
            assert isinstance(accepted_primary, BaseExceptionGroup)
            assert len(accepted_primary.exceptions) == 2
            assert isinstance(accepted_primary.exceptions[0], RuntimeError)
            assert isinstance(accepted_primary.exceptions[1], asyncio.CancelledError)
        else:
            assert accepted_primary is primary
        task.cancel("cancel failure progress")
        assert task.cancelling() == 1
        if result_present:
            try:
                await asyncio.wait_for(release_started.wait(), timeout=10)
                assert not task.done()
                assert len(releases) == 1
            finally:
                allow_release.set()
        if publication_failure == "cancel":
            await asyncio.wait_for(later_publication_started.wait(), timeout=10)
            task.cancel("cancel later failure publication")
            assert task.cancelling() == 2
        try:
            await task
        except asyncio.CancelledError as cancellation:
            if publication_failure == "cancel":
                assert cancellation.args == ("cancel later failure publication",)
                assert isinstance(cancellation.__cause__, BaseExceptionGroup)
                assert cancellation.__cause__.exceptions == (accepted_primary, progress_errors[0])
            elif publication_failure == "error":
                assert cancellation.args == ("cancel failure progress",)
                assert isinstance(cancellation.__cause__, BaseExceptionGroup)
                assert cancellation.__cause__.exceptions == (
                    (accepted_primary, progress_detail, secondary)
                    if group_source == "progress"
                    else (accepted_primary, secondary)
                )
            else:
                assert cancellation.args == ("cancel failure progress",)
                if group_source == "progress":
                    assert isinstance(cancellation.__cause__, BaseExceptionGroup)
                    assert cancellation.__cause__.exceptions == (accepted_primary, progress_detail)
                else:
                    assert cancellation.__cause__ is accepted_primary
        else:
            pytest.fail("Factory failure progress swallowed caller cancellation")
        assert task.cancelled()
        assert provider.requests == []
        assert len(releases) == int(result_present)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "publication_failure,release_fails",
    [("none", False), ("error", False), ("cancel", False), ("none", True)],
)
@pytest.mark.parametrize("grouped_progress", [False, True])
def test_binding_failure_progress_preserves_real_caller_cancellation(
    monkeypatch, publication_failure, release_fails, grouped_progress
):
    async def scenario():
        primary = RuntimeError("binding preparation failed")
        progress_started = asyncio.Event()
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        releases = []
        publication_started = asyncio.Event()
        secondary = RuntimeError("binding failure publication failed")
        release_error = RuntimeError("factory release failed")
        detail = ExceptionGroup("progress detail", [ValueError("diagnostic failure")])
        progress_errors = []

        class FailingBinding(_FailingBinding):
            async def bind(self, workspace, runner, **kwargs):
                raise primary

        class Factory(_HostedFactory):
            async def create(self, request):
                async def release(action):
                    releases.append(action)
                    release_started.set()
                    await allow_release.wait()
                    await self.runner.close()
                    if release_fails:
                        raise release_error

                return EnvironmentFactoryResult(
                    environment=Environment(
                        EnvironmentSpec(name="hosted"),
                        runner=self.runner,
                        binding=FailingBinding(),
                    ),
                    release=release,
                )

        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted", lifecycle_policy=EnvironmentLifecyclePolicy()),
            Factory(pre_create_candidate="hosted", runner=_EvidenceRunner("hosted")),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        original_emit = app._event_writer.emit

        async def block_failure_progress(event):
            if event.type is EventType.ENVIRONMENT_BINDING_FAILED:
                if publication_failure == "error":
                    raise secondary
                if publication_failure == "cancel":
                    publication_started.set()
                    await asyncio.Event().wait()
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_PROGRESS
                and event.payload["operation"] == "binding"
                and event.payload["status"] == "failed"
            ):
                progress_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    if grouped_progress:
                        aggregate = BaseExceptionGroup("progress failures", [cancellation, detail])
                        progress_errors.append(aggregate)
                        raise aggregate from None
                    progress_errors.append(cancellation)
                    raise
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_failure_progress)
        task = asyncio.create_task(_run(app, "binding-failure-progress-cancel"))
        await asyncio.wait_for(progress_started.wait(), timeout=10)
        task.cancel("cancel binding failure progress")
        assert task.cancelling() == 1
        try:
            await asyncio.wait_for(release_started.wait(), timeout=10)
            assert not task.done()
            assert len(releases) == 1
        finally:
            allow_release.set()
        if publication_failure == "cancel":
            await asyncio.wait_for(publication_started.wait(), timeout=10)
            task.cancel("cancel later binding publication")
            assert task.cancelling() == 2
        try:
            await task
        except asyncio.CancelledError as cancellation:
            if publication_failure == "cancel":
                assert cancellation.args == ("cancel later binding publication",)
                assert isinstance(cancellation.__cause__, BaseExceptionGroup)
                assert cancellation.__cause__.exceptions == (primary, progress_errors[0])
            else:
                assert cancellation.args == ("cancel binding failure progress",)
                expected = [primary]
                if grouped_progress:
                    expected.append(detail)
                if release_fails:
                    expected.append(release_error)
                if publication_failure == "error":
                    expected.append(secondary)
                if len(expected) == 1:
                    assert cancellation.__cause__ is primary
                else:
                    assert isinstance(cancellation.__cause__, BaseExceptionGroup)
                    assert cancellation.__cause__.exceptions == tuple(expected)
        else:
            pytest.fail("Binding failure progress swallowed caller cancellation")
        assert task.cancelled()
        assert task.cancelling() == (2 if publication_failure == "cancel" else 1)
        assert releases == [EnvironmentFactoryReleaseAction.PRESERVE]
        assert provider.requests == []
        if release_fails:
            from cayu.core.runtime_authority import SessionRunFenced

            with pytest.raises(SessionRunFenced, match="previous invocation still owns"):
                async for _event in app.resume(
                    ResumeRequest(
                        session_id="binding-failure-progress-cancel",
                        messages=[Message.text("user", "retry")],
                    )
                ):
                    pass
            assert releases == [EnvironmentFactoryReleaseAction.PRESERVE]
            assert provider.requests == []

    asyncio.run(scenario())


def test_bind_failure_releases_unadopted_factory_result_once() -> None:
    async def run() -> tuple[
        list[Event],
        _RecordingProvider,
        _ReleasableHostedFactory,
        _EvidenceRunner,
        list[str],
        dict[str, Any],
    ]:
        provider = _RecordingProvider()
        lifecycle: list[str] = []
        runner = _EvidenceRunner("hosted")
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted",
            runner=runner,
            binding=_FailingBinding(),
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        events = await _run(app, "sess_bind_release_once")
        checkpoint = await app.session_store.load_checkpoint("sess_bind_release_once")
        assert checkpoint is not None
        return events, provider, factory, runner, lifecycle, checkpoint

    events, provider, factory, runner, lifecycle, checkpoint = asyncio.run(run())

    assert provider.requests == []
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert lifecycle == ["factory.release:preserve"]
    assert runner.is_closed is True
    assert checkpoint["environment_factory_allocation_owner"] == {
        "hosted": "sess_bind_release_once"
    }
    binding_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FAILED
    )
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    expected_release = {
        "action": "preserve",
        "callback_provided": True,
        "completed": True,
    }
    assert binding_failed.payload["environment_factory_release"] == expected_release
    assert failed.payload["environment_factory_release"] == expected_release


def test_bind_cancellation_releases_unadopted_factory_result_once() -> None:
    class _BlockingBinding(WorkspaceBinding):
        def __init__(self, started: asyncio.Event) -> None:
            self.started = started

        async def bind(
            self,
            workspace: Workspace | None,
            runner: Runner | None,
            **kwargs: Any,
        ) -> BoundWorkspace:
            del workspace, runner, kwargs
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled bind unexpectedly resumed")

        async def finalize(
            self,
            bound: BoundWorkspace,
            *,
            outcome: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> WorkspaceSnapshot | None:
            del bound, outcome, metadata
            raise AssertionError("finalize should not run after bind cancellation")

    async def run() -> tuple[
        _ReleasableHostedFactory,
        _EvidenceRunner,
        list[str],
        dict[str, Any],
        str,
        list[Event],
    ]:
        provider = _RecordingProvider()
        lifecycle: list[str] = []
        runner = _EvidenceRunner("hosted")
        bind_started = asyncio.Event()
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted",
            runner=runner,
            binding=_BlockingBinding(bind_started),
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        run_task = asyncio.create_task(_run(app, "sess_bind_release_cancel"))
        await asyncio.wait_for(bind_started.wait(), timeout=10)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        checkpoint = await app.session_store.load_checkpoint("sess_bind_release_cancel")
        assert checkpoint is not None
        session = await app.session_store.load("sess_bind_release_cancel")
        assert session is not None
        events = await app.session_store.load_events("sess_bind_release_cancel")
        assert provider.requests == []
        return factory, runner, lifecycle, checkpoint, session.status.value, events

    factory, runner, lifecycle, checkpoint, status, events = asyncio.run(run())

    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert lifecycle == ["factory.release:preserve"]
    assert runner.is_closed is True
    assert status == "interrupted"
    interrupted = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1
    assert interrupted[0].payload["abandoned"] is True
    assert checkpoint["environment_factory_allocation_owner"] == {
        "hosted": "sess_bind_release_cancel"
    }


@pytest.mark.parametrize("release_fails", [False, True])
@pytest.mark.parametrize("cancel_terminal_progress", [False, True])
def test_cancellation_during_successful_factory_release_does_not_redispatch(
    monkeypatch, release_fails, cancel_terminal_progress
) -> None:
    binding_error = RuntimeError("binding failed before cancelled release")
    release_error = RuntimeError("release callback failed after cancellation")

    class ExactFailingBinding(_FailingBinding):
        async def bind(self, workspace, runner, **kwargs):
            raise binding_error

    class BlockingReleaseFactory(_ReleasableHostedFactory):
        def __init__(self) -> None:
            super().__init__(
                pre_create_candidate="hosted",
                runner=_EvidenceRunner("hosted"),
                binding=ExactFailingBinding(),
                lifecycle=[],
            )
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            self.requests.append(request)

            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                self.lifecycle.append(f"factory.release:{action.value}")
                self.release_actions.append(action)
                self.release_started.set()
                await self.allow_release.wait()
                await self.runner.close()
                if release_fails:
                    raise release_error

            return EnvironmentFactoryResult(
                environment=Environment(
                    EnvironmentSpec(name=request.environment_name),
                    runner=self.runner,
                    binding=self.binding,
                ),
                reconnect_metadata={"allocation_id": request.session_id},
                release=release,
            )

    async def run() -> tuple[asyncio.Task[list[Event]], BlockingReleaseFactory]:
        factory = BlockingReleaseFactory()
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted", lifecycle_policy=EnvironmentLifecyclePolicy()),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        session_id = "sess_cancel_during_successful_factory_release"
        terminal_progress_started = asyncio.Event()
        original_emit = app._event_writer.emit

        async def block_terminal_progress(event):
            if (
                cancel_terminal_progress
                and event.type is EventType.ENVIRONMENT_LIFECYCLE_PROGRESS
                and event.payload["operation"] == "release"
                and event.payload["status"] in {"completed", "retained", "failed"}
            ):
                terminal_progress_started.set()
                await asyncio.Event().wait()
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_terminal_progress)
        task = asyncio.create_task(_run(app, session_id))
        await asyncio.wait_for(factory.release_started.wait(), timeout=10)
        assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
        task.cancel("cancel while factory release is running")
        assert task.cancelling() == 1
        await asyncio.sleep(0)
        assert task.done() is False
        factory.allow_release.set()
        if cancel_terminal_progress:
            await asyncio.wait_for(terminal_progress_started.wait(), timeout=10)
            task.cancel("cancel release terminal progress")
            assert task.cancelling() == 2
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == (
            "cancel release terminal progress"
            if cancel_terminal_progress
            else "cancel while factory release is running",
        )
        if cancel_terminal_progress:
            assert isinstance(raised.value.__cause__, BaseExceptionGroup)
            leaves = []
            pending = [raised.value.__cause__]
            while pending:
                current = pending.pop()
                if isinstance(current, BaseExceptionGroup):
                    pending.extend(reversed(current.exceptions))
                else:
                    leaves.append(current)
            assert leaves[0] is binding_error
            assert isinstance(leaves[1], asyncio.CancelledError)
            assert leaves[1].args == ("cancel while factory release is running",)
            assert leaves[2:] == ([release_error] if release_fails else [])
        elif release_fails:
            assert isinstance(raised.value.__cause__, BaseExceptionGroup)
            assert raised.value.__cause__.exceptions == (binding_error, release_error)
        assert task.cancelled() is True
        assert task.cancelling() == (2 if cancel_terminal_progress else 1)
        assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
        assert factory.lifecycle == ["factory.release:preserve"]
        events = await app.session_store.load_events(session_id)
        release_transitions = [
            event
            for event in events
            if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
            and event.payload["phase"] == "release"
        ]
        if not cancel_terminal_progress:
            assert release_transitions
            assert release_transitions[-1].payload["outcome"] == (
                "deferred" if release_fails else "released"
            )
        return task, factory

    task, factory = asyncio.run(run())

    assert task.cancelled() is True
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]


def test_cancelled_factory_release_timeout_retains_owner_until_settlement():
    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()
        completed = asyncio.Event()
        actions = []

        class Factory(_HostedFactory):
            async def create(self, request):
                self.requests.append(request)

                async def release(action):
                    actions.append(action)
                    started.set()
                    await finish.wait()
                    await self.runner.close()
                    completed.set()

                return EnvironmentFactoryResult(
                    environment=Environment(
                        EnvironmentSpec(name="hosted"),
                        runner=self.runner,
                        binding=_FailingBinding(),
                    ),
                    release=release,
                    release_timeout_s=0.1,
                )

        factory = Factory(pre_create_candidate="hosted", runner=_EvidenceRunner("hosted"))
        provider = _RecordingProvider()
        app = CayuApp(
            enable_logging=False,
            config=CayuConfig(operations=OperationsConfig(max_environment_lifecycle_owners=1)),
        )
        app.register_provider(provider, default=True)
        app.register_environment_factory(EnvironmentSpec(name="hosted"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = asyncio.create_task(_run(app, "cancel-release-timeout"))
        try:
            await asyncio.wait_for(started.wait(), timeout=10)
            task.cancel("cancel release before deadline")
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, timeout=10)
            assert raised.value.args == ("cancel release before deadline",)
            assert task.cancelled() and task.cancelling() == 1
            assert isinstance(raised.value.__cause__, BaseExceptionGroup)
            assert isinstance(raised.value.__cause__.exceptions[-1], TimeoutError)
            assert not completed.is_set()
            assert "cancel-release-timeout" in (
                app._environment_lifecycle._deferred_factory_cleanup_tasks
            )
            competing = await _run(app, "cancel-release-timeout-contender")
            assert any(event.type is EventType.SESSION_FAILED for event in competing)
            assert len(factory.requests) == 1
            assert len(actions) == 1
            assert provider.requests == []
        finally:
            finish.set()
            assert await app.drain_environment_cleanups(timeout_s=1) is True
        assert completed.is_set()
        assert actions == [EnvironmentFactoryReleaseAction.PRESERVE]
        assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}
        assert app._environment_lifecycle._pending_environment_owner_admissions == set()

    asyncio.run(scenario())


def test_abandoned_factory_result_is_released_before_binding() -> None:
    async def run() -> tuple[
        _ReleasableHostedFactory,
        _EvidenceRunner,
        list[str],
        str,
        str,
    ]:
        app, factory, _binding, runner, _bound_runner, lifecycle = _bound_factory_app(
            max_environment_lifecycle_owners=1,
        )
        stream: Any = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_abandoned_before_bind",
                messages=[Message.text("user", "run")],
            )
        )
        async for event in stream:
            if event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED:
                break
        environment_lifecycle = app._environment_lifecycle
        assert set(environment_lifecycle._active_environment_setups) == {
            "sess_factory_abandoned_before_bind"
        }
        assert environment_lifecycle._pending_environment_owner_admissions == set()
        await stream.aclose()
        assert environment_lifecycle._active_environment_setups == {}
        assert environment_lifecycle._pending_environment_owner_admissions == set()
        session = await app.session_store.load("sess_factory_abandoned_before_bind")
        assert session is not None
        abandonment_lifecycle = list(lifecycle)
        second = await _run(app, "sess_factory_after_abandoned_before_bind")
        return (
            factory,
            runner,
            abandonment_lifecycle,
            session.status.value,
            second[-1].type,
        )

    factory, runner, lifecycle, status, second_terminal = asyncio.run(run())

    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert lifecycle == ["factory.release:preserve"]
    assert runner.is_closed is True
    assert status == "interrupted"
    assert second_terminal is EventType.SESSION_COMPLETED


def test_active_factory_setups_consume_one_capacity_slot_each() -> None:
    async def advance_to_factory_completion(stream: AsyncIterator[Event]) -> None:
        async for event in stream:
            if event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED:
                return
        raise AssertionError("Environment factory did not complete.")

    async def run() -> tuple[int, dict[str, Any], set[str]]:
        app, factory, _binding, _runner, _bound_runner, _lifecycle = _bound_factory_app(
            max_environment_lifecycle_owners=2,
        )
        first: Any = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_capacity_first",
                messages=[Message.text("user", "run")],
            )
        )
        second: Any = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_capacity_second",
                messages=[Message.text("user", "run")],
            )
        )
        await advance_to_factory_completion(first)
        environment_lifecycle = app._environment_lifecycle
        assert set(environment_lifecycle._active_environment_setups) == {
            "sess_factory_capacity_first"
        }
        assert environment_lifecycle._pending_environment_owner_admissions == set()

        await advance_to_factory_completion(second)
        active_at_capacity = dict(environment_lifecycle._active_environment_setups)
        pending_at_capacity = set(environment_lifecycle._pending_environment_owner_admissions)
        await second.aclose()
        await first.aclose()
        assert environment_lifecycle._active_environment_setups == {}
        assert environment_lifecycle._pending_environment_owner_admissions == set()
        return len(factory.requests), active_at_capacity, pending_at_capacity

    factory_calls, active_at_capacity, pending_at_capacity = asyncio.run(run())

    assert factory_calls == 2
    assert set(active_at_capacity) == {
        "sess_factory_capacity_first",
        "sess_factory_capacity_second",
    }
    assert pending_at_capacity == set()


@pytest.mark.parametrize(
    ("cancel_at", "cancel_cleanup"), [("started", False), ("failed", False), ("started", True)]
)
def test_aborted_setup_cleanup_progress_preserves_caller_cancellation(
    monkeypatch, cancel_at, cancel_cleanup
):
    async def scenario():
        binding_error = RuntimeError("binding completion publication failed")
        preflight_error = RuntimeError("setup failure preflight read failed")
        cleanup_error = RuntimeError("aborted binding cleanup failed")
        publication_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        publishers = []
        progress_events = []
        binding_publication_failed = False

        class Binding(_SwitchingBinding):
            async def finalize(self, bound, *, outcome=None, metadata=None):
                await super().finalize(bound, outcome=outcome, metadata=metadata)
                if cancel_cleanup:
                    cleanup_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError as cancellation:
                        raise BaseExceptionGroup(
                            "Cleanup failed during a new cancellation.",
                            [cleanup_error, cancellation],
                        ) from None
                raise cleanup_error

        runner = _EvidenceRunner("hosted")
        binding = Binding(runner)
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted", lifecycle_policy=EnvironmentLifecyclePolicy()),
            _HostedFactory(pre_create_candidate="hosted", runner=runner, binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        emit = app._event_writer.emit

        async def fail_then_block(event):
            nonlocal binding_publication_failed
            if event.type is EventType.ENVIRONMENT_LIFECYCLE_PROGRESS:
                progress_events.append((event.payload["operation"], event.payload["status"]))
            if event.type is EventType.ENVIRONMENT_BINDING_COMPLETED:
                binding_publication_failed = True
                raise binding_error
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_PROGRESS
                and event.payload["operation"] == "retained_cleanup"
                and event.payload["status"] == cancel_at
            ):
                publishers.append(asyncio.current_task())
                publication_started.set()
                await asyncio.Event().wait()
            return await emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_then_block)

        query_events = app.session_store.query_events

        async def fail_failure_preflight(query):
            if binding_publication_failed and query.event_types == (EventType.INTERACTION_STARTED,):
                raise preflight_error
            return await query_events(query)

        monkeypatch.setattr(app.session_store, "query_events", fail_failure_preflight)
        task = asyncio.create_task(_run(app, "abort-cleanup-progress"))
        try:
            publication_waiter = asyncio.create_task(publication_started.wait())
            done, _ = await asyncio.wait(
                (publication_waiter, task), timeout=10, return_when=asyncio.FIRST_COMPLETED
            )
            if publication_waiter not in done:
                publication_waiter.cancel()
                await asyncio.gather(publication_waiter, return_exceptions=True)
            assert publication_started.is_set(), "\n".join(map(str, progress_events))
            assert publishers == [task]
            assert task.cancelling() == 0 and not task.done()
            assert binding.finalize_calls == (0 if cancel_at == "started" else 1)
            task.cancel("cancel aborted setup diagnostics")
            assert task.cancelling() == 1
            if cancel_cleanup:
                await asyncio.wait_for(cleanup_started.wait(), timeout=10)
                assert not task.done() and task.cancelling() == 1
                task.cancel("cancel aborted setup cleanup")
                assert task.cancelling() == 2
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert raised.value.args == (
                "cancel aborted setup cleanup"
                if cancel_cleanup
                else "cancel aborted setup diagnostics",
            )
            assert isinstance(raised.value.__cause__, BaseExceptionGroup)
            if cancel_cleanup:
                leaves = []
                pending = [raised.value.__cause__]
                while pending:
                    current = pending.pop()
                    if isinstance(current, BaseExceptionGroup):
                        pending.extend(reversed(current.exceptions))
                    else:
                        leaves.append(current)
                assert leaves[0] is binding_error
                assert isinstance(leaves[1], asyncio.CancelledError)
                assert leaves[1].args == ("cancel aborted setup diagnostics",)
                assert leaves[2:] == [cleanup_error]
            else:
                assert raised.value.__cause__.exceptions == (binding_error, cleanup_error)
            assert binding_error.__cause__ is preflight_error
            assert task.cancelled() and task.cancelling() == (2 if cancel_cleanup else 1)
            assert provider.requests == []
            assert binding.finalize_calls == 1
            assert runner.is_closed
            assert (
                "abort-cleanup-progress"
                not in app._environment_lifecycle._active_environment_setups
            )
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_binding_completion_publication_failure_finalizes_adopted_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[_SwitchingBinding, _EvidenceRunner, _EvidenceRunner, list[str]]:
        app, _factory, binding, source_runner, bound_runner, lifecycle = _bound_factory_app()
        original_emit = app._event_writer.emit

        async def fail_binding_completion(event: Event) -> Event:
            if event.type is EventType.ENVIRONMENT_BINDING_COMPLETED:
                raise RuntimeError("binding completion publication failed")
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_binding_completion)
        await _run(app, "sess_binding_completion_publication_failed")
        return binding, source_runner, bound_runner, lifecycle

    binding, source_runner, bound_runner, lifecycle = asyncio.run(run())

    assert binding.finalize_calls == 1
    assert binding.finalize_outcomes == ["interrupted"]
    assert lifecycle == ["binding.finalize"]
    assert source_runner.is_closed is True
    assert bound_runner.is_closed is True


def test_finalize_started_publication_failure_does_not_skip_binding_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[_SwitchingBinding, _EvidenceRunner, _EvidenceRunner]:
        app, _factory, binding, source_runner, bound_runner, _lifecycle = _bound_factory_app()
        original_emit = app._event_writer.emit

        async def fail_finalize_started(event: Event) -> Event:
            if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED:
                raise RuntimeError("finalize start publication failed")
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_finalize_started)
        events = await _run(app, "sess_finalize_start_publication_failed")
        terminal = next(event for event in events if event.type is EventType.SESSION_COMPLETED)
        assert terminal.payload["binding_finalize_publication_error"]["failures"] == [
            {
                "phase": "finalize_started_event",
                "error": "finalize start publication failed",
                "error_type": "RuntimeError",
            }
        ]
        return binding, source_runner, bound_runner

    binding, source_runner, bound_runner = asyncio.run(run())

    assert binding.finalize_calls == 1
    assert source_runner.is_closed is True
    assert bound_runner.is_closed is True


def test_finalize_completed_publication_failure_preserves_final_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[_SwitchingBinding, _EvidenceRunner, _EvidenceRunner]:
        app, _factory, binding, source_runner, bound_runner, _lifecycle = _bound_factory_app()
        original_emit = app._event_writer.emit

        async def fail_finalize_completed(event: Event) -> Event:
            if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED:
                raise ConnectionError("final revision publication acknowledgement lost")
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_finalize_completed)
        events = await _run(app, "sess_finalize_completion_publication_failed")
        terminal = next(event for event in events if event.type is EventType.SESSION_COMPLETED)
        assert terminal.payload["binding_finalize_publication_error"]["failures"] == [
            {
                "phase": "finalize_completed_event",
                "error": "final revision publication acknowledgement lost",
                "error_type": "ConnectionError",
            }
        ]
        final_revision = terminal.payload["final_revision"]
        assert final_revision == {
            "workspace_id": final_revision["workspace_id"],
            "observer": final_revision["observer"],
            "status": "truncated",
            "revision": None,
            "head_revision": None,
            "branch": None,
            "path_scope": "complete",
            "total_paths": 0,
            "detail_code": "final_revision_secret_scope_unavailable",
            "finalization_delta": {
                "attribution_confidence": "unattributed_finalization_change",
                "status": "truncated",
                "before_revision": None,
                "after_revision": None,
                "paths": [],
                "retained_paths": 0,
                "total_paths": 0,
                "truncated": True,
                "head_changed": False,
                "branch_changed": False,
                "detail_code": "finalization_delta_secret_scope_unavailable",
            },
        }
        assert final_revision["workspace_id"].startswith("cayu_authority_v1.")
        assert final_revision["observer"].startswith("cayu_authority_v1.")
        assert not any(
            event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED for event in events
        )
        return binding, source_runner, bound_runner

    binding, source_runner, bound_runner = asyncio.run(run())

    assert binding.finalize_calls == 1
    assert source_runner.is_closed is True
    assert bound_runner.is_closed is True


def test_binding_completion_publication_cancellation_finalizes_adopted_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[_SwitchingBinding, _EvidenceRunner, _EvidenceRunner]:
        app, _factory, binding, source_runner, bound_runner, _lifecycle = _bound_factory_app()
        publication_started = asyncio.Event()
        original_emit = app._event_writer.emit

        async def block_binding_completion(event: Event) -> Event:
            if event.type is EventType.ENVIRONMENT_BINDING_COMPLETED:
                publication_started.set()
                await asyncio.Event().wait()
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_binding_completion)
        run_task = asyncio.create_task(_run(app, "sess_binding_completion_publication_cancelled"))
        await asyncio.wait_for(publication_started.wait(), timeout=10)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        return binding, source_runner, bound_runner

    binding, source_runner, bound_runner = asyncio.run(run())

    assert binding.finalize_calls == 1
    assert binding.finalize_outcomes == ["interrupted"]
    assert source_runner.is_closed is True
    assert bound_runner.is_closed is True


@pytest.mark.parametrize("cancellation_phase", ["release", "failed_progress"])
def test_release_publication_cancellation_after_finalize_failure_remains_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    cancellation_phase: str,
) -> None:
    finalization_error = RuntimeError("binding finalize failed")

    class FailingFinalizeBinding(_SwitchingBinding):
        async def finalize(
            self,
            bound: BoundWorkspace,
            *,
            outcome: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> WorkspaceSnapshot | None:
            del bound, metadata
            self.finalize_calls += 1
            self.finalize_outcomes.append(outcome)
            if self.lifecycle is not None:
                self.lifecycle.append("binding.finalize")
            raise finalization_error

    async def run() -> tuple[asyncio.Task[list[Event]], list[Event]]:
        lifecycle: list[str] = []
        source_runner = _EvidenceRunner("hosted")
        binding = FailingFinalizeBinding(
            _EvidenceRunner("hosted"),
            lifecycle=lifecycle,
        )
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted",
            runner=source_runner,
            binding=binding,
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted", lifecycle_policy=EnvironmentLifecyclePolicy()),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        release_publication_started = asyncio.Event()
        original_emit = app._event_writer.emit

        async def block_failed_release(event: Event) -> Event:
            if (
                cancellation_phase == "release"
                and event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "release"
            ) or (
                cancellation_phase == "failed_progress"
                and event.type is EventType.ENVIRONMENT_LIFECYCLE_PROGRESS
                and event.payload["operation"] == "finalization"
                and event.payload["status"] == "failed"
            ):
                release_publication_started.set()
                await asyncio.Event().wait()
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_failed_release)
        task = asyncio.create_task(
            _run(app, "sess_release_publication_cancel_after_finalize_failure")
        )
        await asyncio.wait_for(release_publication_started.wait(), timeout=10)
        durable_events = await app.session_store.load_events(
            "sess_release_publication_cancel_after_finalize_failure"
        )
        assert sum(
            event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in durable_events
        ) == (1 if cancellation_phase == "release" else 0)
        task.cancel("cancel failed binding release publication")
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("cancel failed binding release publication",)
        assert raised.value.__cause__ is finalization_error
        assert task.cancelled() is True
        assert task.cancelling() == 1
        durable_events = await app.session_store.load_events(
            "sess_release_publication_cancel_after_finalize_failure"
        )
        assert (
            sum(
                event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
                for event in durable_events
            )
            == 1
        )
        assert binding.finalize_calls == 1
        return task, durable_events

    task, durable_events = asyncio.run(run())

    assert task.cancelled() is True
    failed = next(
        event
        for event in durable_events
        if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )
    assert failed.payload["error"] == "binding finalize failed"
    assert failed.payload["error_type"] == "RuntimeError"


def test_factory_release_publication_cancellation_after_start_failure_remains_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding_error = RuntimeError("binding failed before factory release")
    release_start_error = RuntimeError("release start publication failed")

    class ExactFailingBinding(_FailingBinding):
        async def bind(
            self,
            workspace: Workspace | None,
            runner: Runner | None,
            **kwargs: Any,
        ) -> BoundWorkspace:
            del workspace, runner, kwargs
            raise binding_error

    async def run() -> tuple[asyncio.Task[list[Event]], _ReleasableHostedFactory]:
        lifecycle: list[str] = []
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted",
            runner=_EvidenceRunner("hosted"),
            binding=ExactFailingBinding(),
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(
                name="hosted",
                lifecycle_policy=EnvironmentLifecyclePolicy(
                    progress_min_interval_seconds=0.0,
                ),
            ),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        release_publication_started = asyncio.Event()
        original_emit = app._event_writer.emit
        release_transition_blocked = False

        async def fail_start_then_block_release(event: Event) -> Event:
            nonlocal release_transition_blocked
            if (
                event.type is EventType.ENVIRONMENT_LIFECYCLE_PROGRESS
                and event.payload["operation"] == "release"
                and event.payload["status"] == "started"
            ):
                raise release_start_error
            if (
                not release_transition_blocked
                and event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload["phase"] == "release"
            ):
                release_transition_blocked = True
                release_publication_started.set()
                await asyncio.Event().wait()
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_start_then_block_release)
        task = asyncio.create_task(_run(app, "sess_factory_release_cancel_after_start_failure"))
        await asyncio.wait_for(release_publication_started.wait(), timeout=10)
        assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
        task.cancel("cancel factory release publication")
        assert task.cancelling() == 1
        try:
            await task
        except asyncio.CancelledError as cancellation:
            assert cancellation.args == ("cancel factory release publication",)
            prior_failures = cancellation.__cause__
            assert isinstance(prior_failures, BaseExceptionGroup)
            assert prior_failures.exceptions == (binding_error, release_start_error)
        else:
            pytest.fail("Factory release publication swallowed caller cancellation.")
        assert task.cancelled() is True
        assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
        assert lifecycle == ["factory.release:preserve"]
        return task, factory

    task, factory = asyncio.run(run())

    assert task.cancelled() is True
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]


def test_factory_candidate_switch_is_refused_and_discarded_before_checkpoint() -> None:
    async def run() -> tuple[
        list[Event],
        _RecordingProvider,
        _EvidenceRunner,
        dict[str, Any] | None,
    ]:
        provider = _RecordingProvider()
        runner = _EvidenceRunner("hosted-b")
        factory = _HostedFactory(
            pre_create_candidate="hosted-a",
            runner=runner,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        events = await _run(app, "sess_factory_candidate_switch")
        checkpoint = await app.session_store.load_checkpoint("sess_factory_candidate_switch")
        return events, provider, runner, checkpoint

    events, provider, runner, checkpoint = asyncio.run(run())

    assert provider.requests == []
    assert runner.is_closed is True
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["environment_factory_release"] == {
        "action": "discard",
        "callback_provided": False,
        "completed": True,
    }
    transitions = [
        environment_lifecycle_transition_from_event(event)
        for event in events
        if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
    ]
    assert [(item.phase.value, item.outcome.value) for item in transitions] == [
        ("selected", "observed"),
        ("preflight", "accepted"),
        ("allocated", "completed"),
        ("bound", "completed"),
        ("final_evidence", "observed"),
        ("admission", "refused"),
        ("release", "released"),
    ]
    assert checkpoint is None or "environment_factory_reconnect" not in checkpoint
    assert checkpoint is None or "environment_factory_allocation_owner" not in checkpoint


def test_explicit_resume_reenters_lifecycle_after_rejected_allocation_is_discarded() -> None:
    async def run() -> tuple[
        list[Event],
        list[Event],
        list[Event],
        _RecordingProvider,
        _RecoveringFactory,
    ]:
        provider = _RecordingProvider()
        factory = _RecoveringFactory()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        initial_events = await _run(app, "sess_factory_recreate")
        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="sess_factory_recreate",
                    messages=[Message.text("user", "retry")],
                )
            )
        ]
        retried_events = await _run(app, "sess_factory_recreate_retry")
        return initial_events, resumed_events, retried_events, provider, factory

    initial_events, resumed_events, retried_events, provider, factory = asyncio.run(run())

    assert EventType.SESSION_FAILED in {event.type for event in initial_events}
    assert EventType.SESSION_COMPLETED in {event.type for event in resumed_events}
    assert EventType.SESSION_COMPLETED in {event.type for event in retried_events}
    assert [request.operation for request in factory.requests] == [
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.CREATE,
    ]
    assert [request.reconnect_metadata for request in factory.requests] == [{}, {}, {}]
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    assert factory.first_runner.is_closed is True
    assert len(provider.requests) == 2


def test_rejected_allocation_release_precedes_acknowledged_reconnect_retirement() -> None:
    class _CommitThenLoseRetirementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, factory: _RecoveringFactory) -> None:
            super().__init__()
            self.factory = factory
            self.lost_acknowledgement = False

        async def transform_checkpoint(  # type: ignore[no-untyped-def]
            self, session_id, checkpoint_transform
        ) -> None:
            await super().transform_checkpoint(session_id, checkpoint_transform)
            checkpoint = await self.load_checkpoint(session_id)
            retired = (checkpoint or {}).get("environment_factory_retired_disposals", {})
            if retired.get("hosted", {}).get("reason") == "admission_refusal":
                assert self.factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
                assert self.factory.first_runner.is_closed is True
                if not self.lost_acknowledgement:
                    self.lost_acknowledgement = True
                    raise ConnectionError("retirement acknowledgement lost")

    async def run() -> tuple[list[Event], _RecoveringFactory, dict[str, Any]]:
        factory = _RecoveringFactory()
        store = _CommitThenLoseRetirementStore(factory)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        events = await _run(app, "sess_rejection_retirement_ack_loss")
        checkpoint = await app.session_store.load_checkpoint("sess_rejection_retirement_ack_loss")
        assert checkpoint is not None
        return events, factory, checkpoint

    events, factory, checkpoint = asyncio.run(run())

    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error_type"] == "ExecutionAdmissionError"
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    assert "environment_factory_reconnect" not in checkpoint
    assert "environment_factory_allocation_owner" not in checkpoint
    assert checkpoint["environment_factory_retired_disposals"]["hosted"] == {
        "reason": "admission_refusal",
        "reconnect_metadata": {"allocation_id": "allocation-1"},
    }


def test_cancellation_during_rejection_retirement_preserves_task_cancellation() -> None:
    class _BlockAfterRetirementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.retirement_committed = asyncio.Event()

        async def transform_checkpoint(  # type: ignore[no-untyped-def]
            self, session_id, checkpoint_transform
        ) -> None:
            await super().transform_checkpoint(session_id, checkpoint_transform)
            checkpoint = await self.load_checkpoint(session_id)
            retired = (checkpoint or {}).get("environment_factory_retired_disposals", {})
            if retired.get("hosted", {}).get("reason") == "admission_refusal":
                self.retirement_committed.set()
                await asyncio.Event().wait()

    async def run() -> tuple[_RecoveringFactory, dict[str, Any]]:
        factory = _RecoveringFactory()
        store = _BlockAfterRetirementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        run_task = asyncio.create_task(
            _run(app, "sess_rejection_retirement_cancel"),
        )
        await asyncio.wait_for(store.retirement_committed.wait(), timeout=10)
        run_task.cancel("stop after retirement commit")
        assert run_task.cancelling() == 1
        cancellation_caught = False
        try:
            await run_task
        except asyncio.CancelledError as cancellation:
            cancellation_caught = True
            assert cancellation.args == ("stop after retirement commit",)
            assert isinstance(cancellation.__cause__, ExecutionAdmissionError)
        assert cancellation_caught is True
        assert run_task.cancelled() is True
        checkpoint = await app.session_store.load_checkpoint("sess_rejection_retirement_cancel")
        assert checkpoint is not None
        return factory, checkpoint

    factory, checkpoint = asyncio.run(run())

    assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    assert factory.first_runner.is_closed is True
    assert "environment_factory_reconnect" not in checkpoint
    assert "environment_factory_allocation_owner" not in checkpoint


def test_final_evidence_collection_cancellation_releases_before_propagation() -> None:
    async def run() -> tuple[asyncio.Task[list[Event]], _RecoveringFactory]:
        runner = _BlockingEvidenceRunner("hosted-b")
        factory = _RecoveringFactory()
        factory.first_runner = runner
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        run_task = asyncio.create_task(
            _run(app, "sess_final_evidence_collection_cancel"),
        )
        await asyncio.wait_for(runner.collection_started.wait(), timeout=10)
        run_task.cancel("stop final evidence collection")
        assert run_task.cancelling() == 1
        cancellation_caught = False
        try:
            await run_task
        except asyncio.CancelledError as cancellation:
            cancellation_caught = True
            assert cancellation.args == ("stop final evidence collection",)
        assert cancellation_caught is True
        assert run_task.cancelled() is True
        return run_task, factory

    run_task, factory = asyncio.run(run())

    assert run_task.cancelled() is True
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert factory.first_runner.is_closed is True


@pytest.mark.parametrize("lost_phase", [None, "final_evidence", "admission"])
def test_final_evidence_failure_settles_dispatched_probe_before_release(
    lost_phase: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeferredProbeFailureRunner(_EvidenceRunner):
        def __init__(self) -> None:
            super().__init__("hosted-b")
            self.collection_started = asyncio.Event()
            self.allow_probe_settlement = asyncio.Event()
            self.probe_settled = asyncio.Event()
            self.release_overtook_probe = False

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            async def settle_dispatched_probe() -> None:
                await self.allow_probe_settlement.wait()
                self.probe_settled.set()

            settlement = asyncio.create_task(settle_dispatched_probe())
            error = RuntimeError("final evidence probe failed")
            attach_environment_factory_cleanup_settlement_task(error, settlement)
            self.collection_started.set()
            raise error

        async def close(self) -> None:
            self.release_overtook_probe = not self.probe_settled.is_set()
            await super().close()

    async def run() -> tuple[list[Event], _RecoveringFactory, DeferredProbeFailureRunner]:
        runner = DeferredProbeFailureRunner()
        factory = _RecoveringFactory()
        factory.first_runner = runner
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        if lost_phase is not None:
            original_emit = app._event_writer.emit
            acknowledgement_lost = False

            async def commit_then_lose_acknowledgement(event: Event) -> Event:
                nonlocal acknowledgement_lost
                persisted = await original_emit(event)
                if (
                    not acknowledgement_lost
                    and event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                    and event.payload["phase"] == lost_phase
                ):
                    acknowledgement_lost = True
                    raise ConnectionError(f"{lost_phase} acknowledgement lost")
                return persisted

            monkeypatch.setattr(
                app._event_writer,
                "emit",
                commit_then_lose_acknowledgement,
            )
        run_task = asyncio.create_task(
            _run(app, f"sess_final_evidence_probe_settlement_{lost_phase}")
        )
        await asyncio.wait_for(runner.collection_started.wait(), timeout=10)
        await asyncio.sleep(0)
        assert run_task.done() is False
        assert factory.release_actions == []
        runner.allow_probe_settlement.set()
        return await asyncio.wait_for(run_task, timeout=10), factory, runner

    events, factory, runner = asyncio.run(run())

    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    if lost_phase is None:
        assert failed.payload["error_type"] == "ExecutionAdmissionError"
        assert failed.payload["execution_admission"]["refusals"][0]["code"] == (
            "missing_final_evidence"
        )
    else:
        assert failed.payload["error_type"] == "ConnectionError"
    assert runner.probe_settled.is_set()
    assert runner.release_overtook_probe is False
    assert runner.is_closed is True
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD]


@pytest.mark.parametrize("release_callback", [True, False])
def test_final_evidence_probe_retry_retains_full_release_sequence(
    release_callback: bool,
) -> None:
    class RetryableProbeFailureRunner(_EvidenceRunner):
        def __init__(self) -> None:
            super().__init__("hosted-b")
            self.probe_settled = False
            self.retry_calls = 0
            self.release_overtook_probe = False

        def retry_probe_settlement(self) -> asyncio.Task[None]:
            self.retry_calls += 1

            async def settle() -> None:
                self.probe_settled = True

            return asyncio.create_task(settle())

        async def collect_execution_admission_candidate(
            self,
        ) -> ExecutionAdmissionCandidate:
            async def fail_initial_settlement() -> None:
                raise RuntimeError("probe settlement failed")

            settlement = asyncio.create_task(fail_initial_settlement())
            register_environment_factory_cleanup_retry(
                settlement,
                self.retry_probe_settlement,
            )
            error = RuntimeError("final evidence probe failed")
            attach_environment_factory_cleanup_settlement_task(error, settlement)
            raise error

        async def close(self) -> None:
            self.release_overtook_probe = not self.probe_settled
            await super().close()

    class ProbeFactory(_RecoveringFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            if release_callback:
                return await super().create(request)
            self.requests.append(request)
            runner = self.first_runner if len(self.requests) == 1 else self.second_runner
            return EnvironmentFactoryResult(
                environment=Environment(
                    EnvironmentSpec(name=request.environment_name),
                    runner=runner,
                ),
                reconnect_metadata={"allocation_id": f"allocation-{len(self.requests)}"},
            )

    async def run() -> tuple[list[Event], ProbeFactory, RetryableProbeFailureRunner, bool]:
        runner = RetryableProbeFailureRunner()
        factory = ProbeFactory()
        factory.first_runner = runner
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        events = await _run(app, "sess_final_evidence_probe_retry")
        assert factory.release_actions == []
        drained = await app.drain_environment_cleanups(timeout_s=10)
        return events, factory, runner, drained

    events, factory, runner, drained = asyncio.run(run())

    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error_type"] == "ExecutionAdmissionError"
    assert drained is True
    assert runner.retry_calls == 1
    assert runner.release_overtook_probe is False
    assert runner.is_closed is True
    assert factory.release_actions == (
        [EnvironmentFactoryReleaseAction.DISCARD] if release_callback else []
    )


def test_missing_final_runner_evidence_returns_structured_refusal() -> None:
    async def run() -> list[Event]:
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(
                pre_create_candidate="hosted",
                runner=_NoEvidenceRunner(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, "sess_missing_final_evidence")

    events = asyncio.run(run())

    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error_type"] == "ExecutionAdmissionError"
    decision = failed.payload["execution_admission"]
    assert decision["status"] == "refused"
    assert decision["candidate"] == "hosted"
    assert decision["stage"] == "pre_exposure"
    assert decision["refusals"] == [
        {
            "code": "missing_final_evidence",
            "tool_name": None,
            "requirement_name": None,
            "capability": None,
            "executable": None,
            "required_state": None,
            "observed_state": "mismatched",
            "reason_code": None,
            "remediation_code": None,
        }
    ]
