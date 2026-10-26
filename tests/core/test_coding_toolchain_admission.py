"""Command tools admit a neutral toolchain profile through per-candidate evidence policies."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from cayu import ToolContext
from cayu.environments import (
    CodingCommandAuthority,
    CodingDependencyInput,
    CodingToolchainEvidencePolicy,
    CodingToolchainProfile,
    DockerCodingToolchainProfile,
    coding_toolchain_evidence_policy,
    coding_toolchain_runner_admission_failure,
    ensure_coding_toolchain_runner_admission,
    register_coding_toolchain_evidence_policy,
    run_coding_toolchain_admission_probes,
)
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionExecutableEvidence,
    ExecutionRequirements,
    ExecutionToolRequirementEvidence,
    evaluate_execution_admission,
)
from cayu.runners import DockerImageIdentity, ExecCommand, ExecResult, Runner
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    NamedCheck,
    RunCheckTool,
    RunCommandTool,
    StructuredCommandToolPolicy,
    ToolExecutableRequirement,
)
from cayu.workspaces import LocalWorkspace

_LOCK = b"locked\n"


def _profile() -> CodingToolchainProfile:
    return CodingToolchainProfile(
        profile_id="neutral-python",
        revision="1",
        command_authorities=(
            CodingCommandAuthority(
                selector="focused-test",
                revision="1",
                description="Run one focused test.",
                exposure="structured_command",
                executable="/opt/tools/pytest",
                fixed_arguments=("-q",),
                allow_positional_arguments=True,
                positional_arguments_are_paths=True,
                positional_path_prefixes=("tests",),
                positional_path_suffixes=(".py",),
                max_arguments=2,
                allowed_exit_codes=(0, 1),
            ),
            CodingCommandAuthority(
                selector="test",
                revision="1",
                description="Run the project tests.",
                exposure="named_check",
                executable="/opt/tools/pytest",
                fixed_arguments=("-q", "tests"),
                max_arguments=0,
            ),
        ),
        dependency_inputs=(
            CodingDependencyInput(
                path="uv.lock", content_sha256="sha256:" + sha256(_LOCK).hexdigest()
            ),
        ),
    )


def _docker(profile: CodingToolchainProfile) -> DockerCodingToolchainProfile:
    return DockerCodingToolchainProfile.from_toolchain(
        profile,
        image_identity=DockerImageIdentity(reference="registry.example/python@sha256:" + "b" * 64),
        platform_architecture="amd64",
    )


def _claim(capability: str, state: str, now: datetime) -> ExecutionCapabilityClaim:
    if state == "live_verified":
        return ExecutionCapabilityClaim.live_verified(
            capability,
            observation="denied" if capability == "deny_by_default_network" else "supported",
            observed_at=now,
            valid_until=now + timedelta(minutes=1),
        )
    if state == "available":
        return ExecutionCapabilityClaim.available(capability)
    if state == "declared":
        return ExecutionCapabilityClaim.declared(capability)
    return ExecutionCapabilityClaim.unsupported(
        capability, reason_code="unsupported", remediation_code="select_other"
    )


class _Runner:
    """A runner whose admission evidence each test sets explicitly."""

    def __init__(
        self,
        profile: CodingToolchainProfile,
        *,
        candidate: str = "cayu-cloud-sandbox",
        network: str = "available",
        executables_observed_at: datetime | None = None,
        fingerprint: str | None = None,
    ) -> None:
        self.profile = profile
        self.candidate_name = candidate
        self.network = network
        self.fingerprint = profile.fingerprint if fingerprint is None else fingerprint
        self.observed_at = executables_observed_at or datetime.now(UTC)
        self.refreshes = 0
        self.commands: list[tuple[ExecCommand, dict]] = []

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        environment = "sha256:" + "c" * 64
        now = self.observed_at
        return ExecutionAdmissionCandidate(
            candidate=self.candidate_name,
            evidence=ExecutionCapabilityEvidence(
                subject=self.candidate_name,
                environment_fingerprint=environment,
                toolchain_profile_fingerprint=self.fingerprint,
                claims=(
                    _claim("deny_by_default_network", self.network, datetime.now(UTC)),
                    ExecutionCapabilityClaim.unsupported(
                        "untrusted_code_isolation",
                        reason_code="unsupported",
                        remediation_code="select_other",
                    ),
                ),
                tool_requirements=ExecutionToolRequirementEvidence(
                    environment_fingerprint=environment,
                    executables=tuple(
                        ExecutionExecutableEvidence(
                            executable=executable,
                            state="live_verified",
                            requirement_fingerprint=ToolExecutableRequirement(
                                executable=executable
                            ).fingerprint,
                            observed_at=now,
                            valid_until=now + timedelta(minutes=1),
                        )
                        for executable in self.profile.required_executables
                    ),
                ),
            ),
        )

    async def refresh_execution_admission(self) -> None:
        self.refreshes += 1
        self.observed_at = datetime.now(UTC)

    def resolve_cwd(self, cwd=None):
        return "/workspace" if cwd is None else f"/workspace/{cwd}"

    def preflight_exec(self, command, **kwargs):
        del command, kwargs

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return ExecResult(stdout="1 passed\n", exit_code=0)


def _context(runner: object, workspace: LocalWorkspace) -> ToolContext:
    return ToolContext(
        session_id="session",
        agent_name="agent",
        environment_name="coding",
        workspace_id=workspace.id,
        workspace=workspace,
        runner=runner,
    )


def _workspace(tmp_path: Path) -> LocalWorkspace:
    (tmp_path / "uv.lock").write_bytes(_LOCK)
    return LocalWorkspace(tmp_path, workspace_id="workspace")


class _ExactPolicy(CommandPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test.exact", behavior_version="1", implementation_version="1"
        )

    async def evaluate(self, ctx, request) -> CommandPolicyResult:
        del ctx
        exact = tuple(request.command.argv or ()) == ("/opt/tools/pytest", "-q", "tests")
        return CommandPolicyResult(
            decision=CommandPolicyDecision.ALLOW if exact else CommandPolicyDecision.DENY
        )


def _check_tool(profile: CodingToolchainProfile) -> RunCheckTool:
    authority = profile.command_authority("test")
    assert authority is not None
    return RunCheckTool(
        checks=[
            NamedCheck(
                name="test",
                description=authority.description,
                command=ExecCommand.process(*authority.command_argv()),
                timeout_s=authority.timeout_seconds,
                max_output_bytes=authority.max_output_bytes,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="test.check", behavior_version="1", implementation_version="1"
                ),
            )
        ],
        command_policy=_ExactPolicy(),
        toolchain_profile=profile,
    )


def test_neutral_profile_runs_commands_and_checks_on_a_brokered_candidate(tmp_path: Path) -> None:
    profile = _profile()
    workspace = _workspace(tmp_path)
    runner = _Runner(profile)
    command = RunCommandTool(toolchain_profile=profile)
    check = _check_tool(profile)

    ran = asyncio.run(
        command.run(
            _context(runner, workspace), {"selector": "focused-test", "args": ["tests/t.py"]}
        )
    )
    checked = asyncio.run(check.run(_context(runner, workspace), {"check": "test"}))

    assert ran.is_error is False, ran.content
    assert ran.structured["status"] == "succeeded"
    assert ran.structured["toolchain_profile_fingerprint"] == profile.fingerprint
    assert checked.structured["status"] == "passed", checked.content
    assert [tuple(item.argv or ()) for item, _ in runner.commands] == [
        ("/opt/tools/pytest", "-q", "tests/t.py"),
        ("/opt/tools/pytest", "-q", "tests"),
    ]
    assert type(command.toolchain_profile) is CodingToolchainProfile


def test_brokered_lower_bar_does_not_reach_docker_runners(tmp_path: Path) -> None:
    profile = _profile()
    workspace = _workspace(tmp_path)
    # A Docker runner whose network claim is only "available" is refused for a
    # neutral profile, exactly as it would be for a Docker-typed one.
    docker_runner = _Runner(profile, candidate="docker", network="available")
    refused = asyncio.run(
        RunCommandTool(toolchain_profile=profile).run(
            _context(docker_runner, workspace), {"selector": "focused-test", "args": []}
        )
    )
    assert refused.structured["error"] == "toolchain_network_admission_missing"
    assert refused.content == "Structured commands require the exact admitted toolchain."
    assert docker_runner.commands == []

    live_docker = _Runner(profile, candidate="docker", network="live_verified")
    assert coding_toolchain_runner_admission_failure(live_docker, profile=profile) is None

    # A Docker-typed profile never accepts a brokered runner at all.
    docker_profile = _docker(profile)
    brokered = _Runner(docker_profile, candidate="cayu-cloud-sandbox", network="available")
    docker_refused = asyncio.run(
        RunCommandTool(toolchain_profile=docker_profile).run(
            _context(brokered, workspace), {"selector": "focused-test", "args": []}
        )
    )
    assert docker_refused.structured["error"] == "docker_admission_mismatch"
    assert docker_refused.content == (
        "Structured commands require the exact admitted Docker toolchain."
    )


@pytest.mark.parametrize(
    ("runner_options", "code"),
    [
        ({"candidate": "unknown-sandbox"}, "toolchain_admission_mismatch"),
        ({"network": "declared"}, "toolchain_network_admission_missing"),
        ({"network": "unsupported"}, "toolchain_network_admission_missing"),
        ({"fingerprint": "sha256:" + "f" * 64}, "toolchain_profile_drift"),
    ],
)
def test_neutral_admission_refuses_with_stable_codes(runner_options: dict, code: str) -> None:
    profile = _profile()

    assert (
        coding_toolchain_runner_admission_failure(
            _Runner(profile, **runner_options), profile=profile
        )
        == code
    )
    assert coding_toolchain_runner_admission_failure(object(), profile=profile) == (
        "toolchain_admission_unavailable"
    )
    with pytest.raises(TypeError, match="exact CodingToolchainProfile"):
        coding_toolchain_runner_admission_failure(_Runner(profile), profile=_docker(profile))


def test_stale_executable_evidence_is_renewed_once() -> None:
    profile = _profile()
    runner = _Runner(profile, executables_observed_at=datetime.now(UTC) - timedelta(minutes=5))

    assert coding_toolchain_runner_admission_failure(runner, profile=profile) == (
        "toolchain_admission_stale"
    )
    assert asyncio.run(ensure_coding_toolchain_runner_admission(runner, profile=profile)) is None
    assert runner.refreshes == 1


def test_neutral_admission_never_makes_a_runner_untrusted_capable() -> None:
    profile = _profile()
    runner = _Runner(profile)

    # Admitted for the trusted toolchain, while untrusted admission still fails
    # on the runner's own evidence.
    assert coding_toolchain_runner_admission_failure(runner, profile=profile) is None
    untrusted = evaluate_execution_admission(
        candidate="cayu-cloud-sandbox",
        requirements=ExecutionRequirements.untrusted(),
        evidence=runner.execution_admission_candidate().evidence,
    )
    assert untrusted.status == "refused"


def test_evidence_policies_are_data_and_never_lower_a_registered_bar() -> None:
    assert coding_toolchain_evidence_policy("docker") == CodingToolchainEvidencePolicy(
        "docker", "live_verified"
    )
    assert coding_toolchain_evidence_policy("cayu-cloud-sandbox") == (
        CodingToolchainEvidencePolicy("cayu-cloud-sandbox", "available")
    )
    with pytest.raises(ValueError, match="can't be replaced"):
        register_coding_toolchain_evidence_policy(
            CodingToolchainEvidencePolicy("docker", "available")
        )
    with pytest.raises(ValueError, match="'available' or 'live_verified'"):
        CodingToolchainEvidencePolicy("test-declared-sandbox", "declared")

    policy = CodingToolchainEvidencePolicy("test-registered-sandbox", "live_verified")
    register_coding_toolchain_evidence_policy(policy)
    register_coding_toolchain_evidence_policy(policy)
    assert coding_toolchain_evidence_policy("test-registered-sandbox") == policy
    with pytest.raises(ValueError, match="different evidence policy"):
        register_coding_toolchain_evidence_policy(
            CodingToolchainEvidencePolicy("test-registered-sandbox", "available")
        )
    profile = _profile()
    runner = _Runner(profile, candidate="test-registered-sandbox", network="available")
    assert coding_toolchain_runner_admission_failure(runner, profile=profile) == (
        "toolchain_network_admission_missing"
    )


def test_neutral_tool_identity_differs_from_the_docker_typed_one() -> None:
    profile = _profile()
    docker_profile = _docker(profile)

    neutral_material = RunCommandTool(toolchain_profile=profile)._execution_profile_material()
    docker_material = RunCommandTool(toolchain_profile=docker_profile)._execution_profile_material()
    assert neutral_material["toolchain_admission"] == "cayu.coding_toolchain_evidence_policy.v1"
    assert "toolchain_admission" not in docker_material
    assert "toolchain_admission" in _check_tool(profile)._execution_profile_material()
    assert StructuredCommandToolPolicy(toolchain_profile=profile).execution_profile_identity != (
        StructuredCommandToolPolicy(toolchain_profile=docker_profile).execution_profile_identity
    )


def test_cancelling_admission_probes_cancels_instead_of_failing_the_toolchain() -> None:
    class _Hanging(Runner):
        async def exec(self, command: ExecCommand, **kwargs: object) -> ExecResult:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def scenario() -> asyncio.Task[None]:
        task = asyncio.create_task(run_coding_toolchain_admission_probes(_Hanging(), _profile()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task

    assert asyncio.run(scenario()).cancelled()
