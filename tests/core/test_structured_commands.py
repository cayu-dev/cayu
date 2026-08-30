from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from tests._session_provenance import fixture_session_invocation

import cayu.tools.structured_commands as structured_commands
from cayu import (
    AgentSpec,
    DockerCodingCommandAuthority,
    DockerCodingDependencyInput,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    ExecResult,
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionExecutableEvidence,
    ExecutionToolRequirementEvidence,
    LocalWorkspace,
    RunCommandTool,
    SecretRedactor,
    StructuredCommandToolPolicy,
    ToolContext,
    ToolPolicyDecision,
    ToolPolicyRequest,
)
from cayu.core.tools import DurableToolRecoveryAuthority, _bind_runtime_tool_invocation_authority
from cayu.runtime import Session, SessionStatus
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._resources import InvocationWorkspaceHandle


def _digest(content: bytes) -> str:
    return "sha256:" + sha256(content).hexdigest()


def _profile(lock_content: bytes = b"locked\n") -> DockerCodingToolchainProfile:
    return DockerCodingToolchainProfile(
        profile_id="python-existing-repo",
        revision="1",
        image_identity=DockerImageIdentity(
            reference="registry.example/python@sha256:" + ("b" * 64),
        ),
        platform_architecture="amd64",
        command_authorities=(
            DockerCodingCommandAuthority(
                selector="focused-test",
                revision="1",
                description="Run one focused Python test.",
                exposure="structured_command",
                executable="/opt/tools/pytest",
                fixed_arguments=("-q",),
                allow_positional_arguments=True,
                positional_arguments_are_paths=True,
                positional_path_prefixes=("tests",),
                positional_path_suffixes=(".py",),
                allow_pytest_node_ids=True,
                max_arguments=2,
                allowed_exit_codes=(0, 1),
            ),
        ),
        dependency_inputs=(
            DockerCodingDependencyInput(
                path="uv.lock",
                content_sha256=_digest(lock_content),
            ),
        ),
    )


def _rust_profile(lock_content: bytes = b"locked\n") -> DockerCodingToolchainProfile:
    return DockerCodingToolchainProfile(
        profile_id="rust-existing-repo",
        revision="1",
        image_identity=DockerImageIdentity(
            reference="registry.example/rust@sha256:" + ("d" * 64),
        ),
        platform_architecture="amd64",
        command_authorities=(
            DockerCodingCommandAuthority(
                selector="integration-test",
                revision="1",
                description="Run one admitted Rust integration test.",
                exposure="structured_command",
                executable="/usr/local/cargo/bin/cargo",
                fixed_arguments=("test", "--test"),
                allowed_literals=("api", "storage"),
                allow_positional_arguments=True,
                min_arguments=1,
                max_arguments=1,
                allowed_exit_codes=(0, 101),
            ),
        ),
        dependency_inputs=(
            DockerCodingDependencyInput(
                path="Cargo.lock",
                content_sha256=_digest(lock_content),
            ),
        ),
    )


class _AdmittedRunner:
    def __init__(
        self,
        profile: DockerCodingToolchainProfile,
        *,
        result: ExecResult | None = None,
        admission_observed_at: datetime | None = None,
    ) -> None:
        now = admission_observed_at or datetime.now(UTC)
        environment_fingerprint = "sha256:" + ("c" * 64)
        executable_evidence = tuple(
            ExecutionExecutableEvidence(
                executable=executable,
                state="live_verified",
                observed_at=now,
                valid_until=now + timedelta(minutes=1),
            )
            for executable in profile.required_executables
        )
        self.candidate = ExecutionAdmissionCandidate(
            candidate="docker",
            evidence=ExecutionCapabilityEvidence(
                subject="docker",
                environment_fingerprint=environment_fingerprint,
                image_fingerprint=profile.image_identity.fingerprint,
                toolchain_profile_fingerprint=profile.fingerprint,
                claims=(
                    ExecutionCapabilityClaim.live_verified(
                        "deny_by_default_network",
                        observation="denied",
                        observed_at=now,
                        valid_until=now + timedelta(minutes=1),
                    ),
                ),
                tool_requirements=ExecutionToolRequirementEvidence(
                    environment_fingerprint=environment_fingerprint,
                    image_fingerprint=profile.image_identity.fingerprint,
                    executables=executable_evidence,
                ),
            ),
        )
        self.commands = []
        self.durable_operations: list[dict[str, object]] = []
        self.result = result or ExecResult(stdout="1 passed\n", exit_code=0)

    def execution_admission_candidate(self):
        return self.candidate

    def resolve_cwd(self, cwd=None):
        return "/workspace" if cwd is None else f"/workspace/{cwd}"

    def preflight_exec(self, command, **kwargs):
        del command, kwargs

    def durable_resource_identity(self):
        return "sha256:" + ("e" * 64)

    def bind_durable_command_operation(self, identity):
        self.durable_operations.append(identity)

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return self.result


class _MutatingAdmittedRunner(_AdmittedRunner):
    def __init__(
        self,
        profile: DockerCodingToolchainProfile,
        *,
        root: Path,
        mutation_path: str,
        content: bytes = b"generated\n",
    ) -> None:
        super().__init__(profile)
        self.root = root
        self.mutation_path = mutation_path
        self.content = content

    async def exec(self, command, **kwargs):
        destination = self.root / self.mutation_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.content)
        return await super().exec(command, **kwargs)


class _ModeMutatingAdmittedRunner(_AdmittedRunner):
    def __init__(self, profile: DockerCodingToolchainProfile, *, path: Path) -> None:
        super().__init__(profile)
        self.path = path

    async def exec(self, command, **kwargs):
        self.path.chmod(self.path.stat().st_mode | 0o100)
        return await super().exec(command, **kwargs)


class _SymlinkMutatingAdmittedRunner(_AdmittedRunner):
    def __init__(self, profile: DockerCodingToolchainProfile, *, root: Path) -> None:
        super().__init__(profile)
        self.root = root

    async def exec(self, command, **kwargs):
        os.symlink("missing-target", self.root / "generated-link")
        return await super().exec(command, **kwargs)


class _SecretMutatingAdmittedRunner(_AdmittedRunner):
    def __init__(
        self,
        profile: DockerCodingToolchainProfile,
        *,
        path: Path,
        replacement: bytes,
    ) -> None:
        super().__init__(profile)
        self.path = path
        self.replacement = replacement

    async def exec(self, command, **kwargs):
        self.path.write_bytes(self.replacement)
        return await super().exec(command, **kwargs)


class _ForcedCommandInterruption(BaseException):
    pass


class _InterruptingAdmittedRunner(_AdmittedRunner):
    def __init__(self, profile: DockerCodingToolchainProfile, *, root: Path) -> None:
        super().__init__(profile)
        self.root = root

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        (self.root / "interrupted-output.txt").write_bytes(b"possibly complete\n")
        raise _ForcedCommandInterruption


class _AdmissionExpiringWorkspace(LocalWorkspace):
    def __init__(self, root: Path, *, runner: _AdmittedRunner) -> None:
        super().__init__(root, workspace_id="workspace")
        self.runner = runner

    async def list_git_entries(self, *, limit: int):
        result = await super().list_git_entries(limit=limit)
        self.runner.candidate = _AdmittedRunner(
            _profile(),
            admission_observed_at=datetime.now(UTC) - timedelta(minutes=5),
        ).candidate
        return result


def _run(tool: RunCommandTool, runner: object, workspace: LocalWorkspace, args: dict):
    return asyncio.run(
        tool.run(
            ToolContext(
                session_id="session",
                agent_name="agent",
                environment_name="coding",
                workspace_id=workspace.id,
                workspace=workspace,
                runner=runner,
            ),
            args,
        )
    )


def _policy_request(args: dict) -> ToolPolicyRequest:
    return ToolPolicyRequest(
        session=Session(
            id="session",
            agent_name="agent",
            provider_name="fake",
            model="fake-model",
            invocation=fixture_session_invocation("session"),
            status=SessionStatus.RUNNING,
        ),
        agent=AgentSpec(name="agent", model="fake-model"),
        tool_name="run_command",
        tool_call_id="call-1",
        arguments=args,
        environment_name="coding",
        workspace_id="workspace",
    )


def test_structured_command_executes_only_resolved_profile_argv(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _AdmittedRunner(profile)
    tool = RunCommandTool(toolchain_profile=profile)

    result = _run(
        tool,
        runner,
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py::test_ok"]},
    )

    assert result.is_error is False
    assert result.structured["status"] == "succeeded"
    assert result.structured["toolchain_profile_id"] == profile.profile_id
    assert result.structured["toolchain_image_fingerprint"] == (profile.image_identity.fingerprint)
    command, kwargs = runner.commands[0]
    assert command.kind == "process"
    assert command.shell is None
    assert command.argv == [
        "/opt/tools/pytest",
        "-q",
        "tests/test_unit.py::test_ok",
    ]
    assert kwargs["env"] is None
    assert kwargs["stdin"] is None


def test_structured_command_runtime_workspace_facade_preserves_git_mode(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    raw_workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    snapshot = InvocationRedactorSnapshot(revision=0, redactor=SecretRedactor())
    workspace = InvocationWorkspaceHandle(
        raw_workspace,
        redactor_snapshot_provider=lambda: snapshot,
        capture_observer=lambda _revision: None,
    )
    profile = _profile()
    runner = _AdmittedRunner(profile)

    result = asyncio.run(
        RunCommandTool(toolchain_profile=profile).run(
            ToolContext(
                session_id="session",
                agent_name="agent",
                environment_name="coding",
                workspace_id=workspace.id,
                workspace=workspace,
                runner=runner,
            ),
            {"selector": "focused-test", "args": ["tests/test_unit.py"]},
        )
    )

    assert result.structured["status"] == "succeeded"
    assert runner.commands


def test_structured_command_manifest_uses_raw_digest_through_redacted_facade(
    tmp_path: Path,
) -> None:
    first_secret = "first-secret-value"
    second_secret = "other-secret-value"
    assert len(first_secret) == len(second_secret)
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text(first_secret, encoding="utf-8")
    raw_workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    snapshot = InvocationRedactorSnapshot(
        revision=0,
        redactor=SecretRedactor((first_secret, second_secret)),
    )
    workspace = InvocationWorkspaceHandle(
        raw_workspace,
        redactor_snapshot_provider=lambda: snapshot,
        capture_observer=lambda _revision: None,
    )
    profile = _profile()
    runner = _SecretMutatingAdmittedRunner(
        profile,
        path=secret_path,
        replacement=second_secret.encode("utf-8"),
    )

    result = asyncio.run(
        RunCommandTool(toolchain_profile=profile).run(
            ToolContext(
                session_id="session",
                agent_name="agent",
                environment_name="coding",
                workspace_id=workspace.id,
                workspace=workspace,
                runner=runner,
            ),
            {"selector": "focused-test", "args": ["tests/test_unit.py"]},
        )
    )

    mutation = result.structured["workspace_mutation_evidence"]
    assert result.structured["status"] == "failed"
    assert result.structured["error"] == "unexpected_workspace_mutation"
    assert mutation["changed_path_count"] == 1
    assert mutation["unexpected_path_count"] == 1


def test_structured_command_executes_custom_non_python_profile(tmp_path: Path) -> None:
    (tmp_path / "Cargo.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _rust_profile()
    runner = _AdmittedRunner(profile)

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        runner,
        workspace,
        {"selector": "integration-test", "args": ["api"]},
    )

    assert result.is_error is False
    command, _ = runner.commands[0]
    assert command.argv == [
        "/usr/local/cargo/bin/cargo",
        "test",
        "--test",
        "api",
    ]
    assert result.structured["toolchain_profile_id"] == "rust-existing-repo"


def test_structured_command_refuses_stale_dependencies_without_dispatch(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"changed\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _AdmittedRunner(profile)

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        runner,
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert result.is_error is True
    assert result.structured["status"] == "stale_toolchain"
    assert result.structured["error"] == "dependency_inputs_changed"
    assert result.structured["dependency_path_count"] == 1
    assert result.structured["dependency_paths_fingerprint"].startswith("sha256:")
    assert "uv.lock" not in str(result.structured)
    assert runner.commands == []


def test_structured_command_refuses_expired_admission_without_dispatch(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _AdmittedRunner(
        profile,
        admission_observed_at=datetime.now(UTC) - timedelta(minutes=5),
    )

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        runner,
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert result.is_error is True
    assert result.structured["error"] == "docker_admission_stale"
    assert runner.commands == []


def test_structured_command_refuses_same_image_different_profile_without_dispatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    admitted_profile = _profile()
    requested_profile = admitted_profile.model_copy(update={"revision": "2"})
    runner = _AdmittedRunner(admitted_profile)

    result = _run(
        RunCommandTool(toolchain_profile=requested_profile),
        runner,
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert requested_profile.image_identity == admitted_profile.image_identity
    assert requested_profile.required_executables == admitted_profile.required_executables
    assert result.structured["error"] == "toolchain_profile_drift"
    assert runner.commands == []


def test_structured_command_rechecks_admission_after_manifest_before_dispatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    profile = _profile()
    runner = _AdmittedRunner(profile)
    workspace = _AdmissionExpiringWorkspace(tmp_path, runner=runner)

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        runner,
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert result.structured["error"] == "docker_admission_stale"
    assert runner.commands == []


def test_structured_command_refuses_non_docker_runner(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _AdmittedRunner(profile)
    runner.candidate = None

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        runner,
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert result.is_error is True
    assert result.structured["error"] == "docker_admission_unavailable"


def test_admitted_nonzero_is_authoritative_but_unexpected_exit_is_error(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()

    admitted = _run(
        RunCommandTool(toolchain_profile=profile),
        _AdmittedRunner(profile, result=ExecResult(stderr="failed\n", exit_code=1)),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )
    unexpected = _run(
        RunCommandTool(toolchain_profile=profile),
        _AdmittedRunner(profile, result=ExecResult(stderr="usage\n", exit_code=2)),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert admitted.structured["status"] == "nonzero"
    assert admitted.structured["exit_code_admitted"] is True
    assert admitted.is_error is False
    assert unexpected.structured["status"] == "nonzero"
    assert unexpected.structured["exit_code_admitted"] is False
    assert unexpected.is_error is True


def test_truncated_projection_without_artifact_is_explicitly_partial(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    authority = (
        _profile()
        .structured_command_authorities[0]
        .model_copy(update={"max_model_output_bytes": 256})
    )
    profile = _profile().model_copy(update={"command_authorities": (authority,)})
    result = _run(
        RunCommandTool(toolchain_profile=profile),
        _AdmittedRunner(profile, result=ExecResult(stdout="x" * 1024, exit_code=0)),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert result.structured["status"] == "partial"
    assert result.structured["stdout_projection_truncated"] is True
    assert result.structured["output_artifact_status"] == "unavailable"
    assert result.structured["output_collection_complete"] is True
    assert result.structured["output_publication_complete"] is False
    assert result.is_error is True


def test_complete_tool_result_respects_profile_publication_ceiling(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    result = _run(
        RunCommandTool(toolchain_profile=profile),
        _AdmittedRunner(
            profile,
            result=ExecResult(
                stdout="x" * 16_384,
                stderr="y" * 16_384,
                exit_code=0,
            ),
        ),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert len(result.model_dump_json().encode("utf-8")) <= (profile.result_publication_max_bytes)
    assert result.structured["result_publication_ceiling_applied"] is True
    assert result.structured["stdout_projection_truncated"] is True
    assert result.structured["stderr_projection_truncated"] is True
    assert result.structured["output_artifact_status"] == "unavailable"
    assert result.structured["output_publication_complete"] is False
    assert result.structured["status"] == "partial"
    assert result.is_error is True


def test_read_only_selector_rejects_observed_workspace_mutation(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _MutatingAdmittedRunner(
        profile,
        root=tmp_path,
        mutation_path="generated/result.txt",
    )

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        runner,
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    mutation = result.structured["workspace_mutation_evidence"]
    assert result.structured["status"] == "failed"
    assert result.structured["process_status"] == "succeeded"
    assert result.structured["error"] == "unexpected_workspace_mutation"
    assert mutation["complete"] is True
    assert mutation["scope_admitted"] is False
    assert mutation["changed_path_count"] == 1
    assert mutation["unexpected_path_count"] == 1
    assert "generated/result.txt" not in str(mutation)
    assert result.is_error is True


def test_read_only_selector_rejects_mode_only_workspace_mutation(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    test_path = tmp_path / "tests" / "test_unit.py"
    test_path.parent.mkdir()
    test_path.write_bytes(b"def test_ok(): pass\n")
    test_path.chmod(0o644)
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        _ModeMutatingAdmittedRunner(profile, path=test_path),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    mutation = result.structured["workspace_mutation_evidence"]
    assert os.stat(test_path).st_mode & 0o100
    assert result.structured["status"] == "failed"
    assert result.structured["error"] == "unexpected_workspace_mutation"
    assert mutation["changed_path_count"] == 1
    assert mutation["scope_admitted"] is False
    assert result.is_error is True


def test_read_only_selector_rejects_symlink_workspace_mutation(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        _SymlinkMutatingAdmittedRunner(profile, root=tmp_path),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    mutation = result.structured["workspace_mutation_evidence"]
    assert (tmp_path / "generated-link").is_symlink()
    assert result.structured["status"] == "failed"
    assert result.structured["error"] == "unexpected_workspace_mutation"
    assert mutation["complete"] is True
    assert mutation["changed_path_count"] == 1
    assert mutation["unexpected_path_count"] == 1


def test_structured_command_durable_recovery_never_replays_forced_interruption(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _InterruptingAdmittedRunner(profile, root=tmp_path)
    tool = RunCommandTool(toolchain_profile=profile)
    arguments = {"selector": "focused-test", "args": ["tests/test_unit.py"]}
    records: dict[str, dict[str, object]] = {}

    async def load_operation(storage_key: str):
        return records.get(storage_key)

    async def compare_and_set_operation(storage_key, expected, desired, secondary):
        assert records.get(storage_key) == expected
        assert secondary == {}
        records[storage_key] = desired
        return desired

    ctx = ToolContext(
        session_id="session",
        agent_name="agent",
        environment_name="coding",
        workspace_id="workspace",
        idempotency_key="tool-invocation",
        workspace=workspace,
        runner=runner,
    )
    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id="task",
        parent_run_epoch=3,
        model_step_id="mstep_00000000000000000000000000000000",
        model_attempt_id="mattempt_00000000000000000000000000000000",
        tool_round_id="tround_00000000000000000000000000000000",
        tool_call_id="call",
        tool_name="run_command",
        idempotency_key="tool-invocation",
        effective_arguments=arguments,
        execution_profile_fingerprint="profile",
        environment_allocation_fingerprint="allocation",
        load_durable_operation=load_operation,
        compare_and_set_durable_operation=compare_and_set_operation,
        seal_durable_output=lambda record: record,
        secret_publication_sealer=lambda: None,
    )

    with pytest.raises(_ForcedCommandInterruption):
        asyncio.run(tool.run(ctx, arguments))

    journal = next(iter(records.values()))
    assert journal["state"] == "dispatching"
    assert len(runner.commands) == 1
    recovery_authority = DurableToolRecoveryAuthority(
        agent_name="agent",
        environment_name="coding",
        workspace=workspace,
        artifact_reader=None,
        compare_and_set_operation=compare_and_set_operation,
    )
    recovered = asyncio.run(
        tool.reconcile_durable_tool_call(
            parent_session_id="session",
            parent_run_epoch=3,
            execution_profile_fingerprint="profile",
            environment_name="coding",
            environment_allocation_fingerprint="allocation",
            model_step_id="mstep_00000000000000000000000000000000",
            model_attempt_id="mattempt_00000000000000000000000000000000",
            tool_round_id="tround_00000000000000000000000000000000",
            tool_call_id="call",
            idempotency_key="tool-invocation",
            arguments=arguments,
            started=True,
            load_operation=load_operation,
            recovery_authority=recovery_authority,
        )
    )

    assert recovered is not None
    assert recovered.structured["status"] == "ambiguous"
    assert recovered.structured["error"] == "command_acknowledgement_lost"
    assert recovered.structured["replayed"] is False
    assert recovered.structured["workspace_identity_matches"] is True
    assert recovered.structured["workspace_mutation_evidence"]["workspace_changed"] is True
    assert len(runner.commands) == 1


@pytest.mark.parametrize("recovery_source", ("journal", "runner"))
def test_structured_command_recovery_projects_checkpointed_runner_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_source: str,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _AdmittedRunner(profile)
    tool = RunCommandTool(toolchain_profile=profile)
    arguments = {"selector": "focused-test", "args": ["tests/test_unit.py"]}
    records: dict[str, dict[str, object]] = {}

    async def load_operation(storage_key: str):
        return records.get(storage_key)

    async def compare_and_set_operation(storage_key, expected, desired, secondary):
        assert records.get(storage_key) == expected
        assert secondary == {}
        records[storage_key] = desired
        return desired

    ctx = ToolContext(
        session_id="checkpoint-session",
        agent_name="agent",
        environment_name="coding",
        workspace_id="workspace",
        idempotency_key="checkpoint-invocation",
        workspace=workspace,
        runner=runner,
    )
    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id="task",
        parent_run_epoch=5,
        model_step_id="mstep_20000000000000000000000000000000",
        model_attempt_id="mattempt_20000000000000000000000000000000",
        tool_round_id="tround_20000000000000000000000000000000",
        tool_call_id="checkpoint-call",
        tool_name="run_command",
        idempotency_key="checkpoint-invocation",
        effective_arguments=arguments,
        execution_profile_fingerprint="profile",
        environment_allocation_fingerprint="allocation",
        load_durable_operation=load_operation,
        compare_and_set_durable_operation=compare_and_set_operation,
        seal_durable_output=lambda record: record,
        secret_publication_sealer=lambda: None,
    )
    original_project_result = RunCommandTool._project_result
    interrupt_projection = True

    async def project_or_interrupt(self, *args, **kwargs):
        if interrupt_projection:
            raise _ForcedCommandInterruption
        return await original_project_result(self, *args, **kwargs)

    monkeypatch.setattr(RunCommandTool, "_project_result", project_or_interrupt)
    with pytest.raises(_ForcedCommandInterruption):
        asyncio.run(tool.run(ctx, arguments))

    journal = next(iter(records.values()))
    assert journal["state"] == "dispatching"
    assert journal["runner_terminal_result"] is not None
    operation = journal["runner_operation"]
    assert operation["runner_resource_identity"] == runner.durable_resource_identity()
    assert operation["process_identity"].startswith("sha256:")
    assert operation["output_identity"].startswith("sha256:")
    assert operation["artifact_identity"].startswith("sha256:")
    assert operation["cleanup_identity"].startswith("sha256:")
    observations: list[dict[str, object]] = []

    async def reconcile_runner_operation(identity):
        observations.append(identity)
        return {
            "identity": identity,
            "state": "terminal",
            "result": runner.result.model_dump(mode="json"),
        }

    if recovery_source == "runner":
        journal["runner_terminal_result"] = None
        journal["runner_terminal_timing"] = None
        journal["runner_terminal_identity"] = None
    interrupt_projection = False
    recovery_authority = DurableToolRecoveryAuthority(
        agent_name="agent",
        environment_name="coding",
        workspace=workspace,
        artifact_reader=None,
        compare_and_set_operation=compare_and_set_operation,
        runner_resource_identity=runner.durable_resource_identity(),
        reconcile_runner_operation=(
            reconcile_runner_operation if recovery_source == "runner" else None
        ),
    )
    recovered = asyncio.run(
        tool.reconcile_durable_tool_call(
            parent_session_id="checkpoint-session",
            parent_run_epoch=5,
            execution_profile_fingerprint="profile",
            environment_name="coding",
            environment_allocation_fingerprint="allocation",
            model_step_id="mstep_20000000000000000000000000000000",
            model_attempt_id="mattempt_20000000000000000000000000000000",
            tool_round_id="tround_20000000000000000000000000000000",
            tool_call_id="checkpoint-call",
            idempotency_key="checkpoint-invocation",
            arguments=arguments,
            started=True,
            load_operation=load_operation,
            recovery_authority=recovery_authority,
        )
    )

    assert recovered is not None
    assert recovered.structured["status"] == "succeeded"
    assert recovered.structured["process_status"] == "succeeded"
    assert recovered.structured["dispatch"] == "runner_terminal_evidence"
    assert recovered.structured["recovered"] is True
    assert recovered.structured["replayed"] is False
    assert observations == ([operation] if recovery_source == "runner" else [])
    assert next(iter(records.values()))["state"] == "terminal"
    assert len(runner.commands) == 1


def test_structured_command_durable_recovery_reconstructs_terminal_result(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _AdmittedRunner(profile)
    tool = RunCommandTool(toolchain_profile=profile)
    arguments = {"selector": "focused-test", "args": ["tests/test_unit.py"]}
    records: dict[str, dict[str, object]] = {}

    async def load_operation(storage_key: str):
        return records.get(storage_key)

    async def compare_and_set_operation(storage_key, expected, desired, secondary):
        assert records.get(storage_key) == expected
        assert secondary == {}
        records[storage_key] = desired
        return desired

    ctx = ToolContext(
        session_id="terminal-session",
        agent_name="agent",
        environment_name="coding",
        workspace_id="workspace",
        idempotency_key="terminal-invocation",
        workspace=workspace,
        runner=runner,
    )
    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id="task",
        parent_run_epoch=7,
        model_step_id="mstep_10000000000000000000000000000000",
        model_attempt_id="mattempt_10000000000000000000000000000000",
        tool_round_id="tround_10000000000000000000000000000000",
        tool_call_id="terminal-call",
        tool_name="run_command",
        idempotency_key="terminal-invocation",
        effective_arguments=arguments,
        execution_profile_fingerprint="profile",
        environment_allocation_fingerprint="allocation",
        load_durable_operation=load_operation,
        compare_and_set_durable_operation=compare_and_set_operation,
        seal_durable_output=lambda record: record,
        secret_publication_sealer=lambda: None,
    )

    completed = asyncio.run(tool.run(ctx, arguments))
    recovered = asyncio.run(
        tool.reconcile_durable_tool_call(
            parent_session_id="terminal-session",
            parent_run_epoch=7,
            execution_profile_fingerprint="profile",
            environment_name="coding",
            environment_allocation_fingerprint="allocation",
            model_step_id="mstep_10000000000000000000000000000000",
            model_attempt_id="mattempt_10000000000000000000000000000000",
            tool_round_id="tround_10000000000000000000000000000000",
            tool_call_id="terminal-call",
            idempotency_key="terminal-invocation",
            arguments=arguments,
            started=True,
            load_operation=load_operation,
        )
    )

    assert next(iter(records.values()))["state"] == "terminal"
    assert recovered is not None
    assert recovered.structured["status"] == completed.structured["status"]
    assert recovered.structured["output_sha256"] == completed.structured["output_sha256"]
    assert recovered.structured["recovered"] is True
    assert recovered.structured["replayed"] is False
    assert recovered.structured["dispatch"] == "terminal_evidence"
    assert len(runner.commands) == 1


def test_structured_command_refuses_success_with_uncertain_cleanup(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    result = _run(
        RunCommandTool(toolchain_profile=profile),
        _AdmittedRunner(
            profile,
            result=ExecResult(
                exit_code=0,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "docker",
                        "action": "kill_command",
                        "status": "failed",
                    }
                ],
            ),
        ),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert result.structured["process_status"] == "succeeded"
    assert result.structured["status"] == "ambiguous"
    assert result.structured["error"] == "workspace_cleanup_uncertain"
    assert result.structured["cleanup_uncertain"] is True
    assert result.is_error is True


def test_mutating_selector_enforces_declared_path_scope(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    authority = (
        _profile()
        .structured_command_authorities[0]
        .model_copy(
            update={
                "effect": "workspace_mutating",
                "mutation_path_prefixes": ("generated",),
            }
        )
    )
    profile = _profile().model_copy(update={"command_authorities": (authority,)})

    admitted = _run(
        RunCommandTool(toolchain_profile=profile),
        _MutatingAdmittedRunner(
            profile,
            root=tmp_path,
            mutation_path="generated/result.txt",
        ),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )
    outside_scope = _run(
        RunCommandTool(toolchain_profile=profile),
        _MutatingAdmittedRunner(
            profile,
            root=tmp_path,
            mutation_path="outside.txt",
        ),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert admitted.structured["status"] == "succeeded"
    assert admitted.structured["workspace_mutation_evidence"]["scope_admitted"] is True
    assert admitted.is_error is False
    assert outside_scope.structured["status"] == "failed"
    assert outside_scope.structured["error"] == "unexpected_workspace_mutation"
    assert outside_scope.structured["workspace_mutation_evidence"]["scope_admitted"] is False
    assert outside_scope.is_error is True


def test_incomplete_post_command_manifest_is_explicitly_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    monkeypatch.setattr(structured_commands, "_COMMAND_MANIFEST_MAX_FILE_BYTES", 8)
    runner = _MutatingAdmittedRunner(
        profile,
        root=tmp_path,
        mutation_path="oversized.txt",
        content=b"x" * 9,
    )

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        runner,
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    mutation = result.structured["workspace_mutation_evidence"]
    assert result.structured["status"] == "partial"
    assert result.structured["process_status"] == "succeeded"
    assert result.structured["error"] == "workspace_observation_incomplete"
    assert mutation["complete"] is False
    assert mutation["scope_admitted"] is False
    assert mutation["after_manifest_fingerprint"] is None
    assert result.is_error is True


def test_structured_command_rejects_shell_and_path_escape_before_dispatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    profile = _profile()
    runner = _AdmittedRunner(profile)
    tool = RunCommandTool(toolchain_profile=profile)

    for argument in ("../outside.py", "tests/*.py", "tests/test.py;whoami"):
        result = _run(
            tool,
            runner,
            workspace,
            {"selector": "focused-test", "args": [argument]},
        )
        assert result.is_error is True
        assert result.structured["error"] == "invalid_arguments"
    assert runner.commands == []


def test_structured_command_tool_policy_resolves_safe_receipt_metadata() -> None:
    profile = _profile()
    policy = StructuredCommandToolPolicy(toolchain_profile=profile)

    result = asyncio.run(
        policy.authorize(
            _policy_request({"selector": "focused-test", "args": ["tests/test_unit.py::test_ok"]})
        )
    )

    assert result.decision is ToolPolicyDecision.ALLOW
    assert result.metadata["selector"] == "focused-test"
    assert result.metadata["argv_sha256"].startswith("sha256:")
    assert "tests/test_unit.py" not in str(result.metadata)
    assert result.metadata["toolchain_profile_fingerprint"] == profile.fingerprint


def test_structured_command_tool_policy_requires_durable_selector_approval() -> None:
    profile = _profile().model_copy(
        update={
            "command_authorities": (
                _profile()
                .structured_command_authorities[0]
                .model_copy(
                    update={
                        "approval": "required",
                        "approval_expires_in_seconds": 300,
                    }
                ),
            )
        }
    )
    policy = StructuredCommandToolPolicy(toolchain_profile=profile)

    result = asyncio.run(
        policy.authorize(
            _policy_request({"selector": "focused-test", "args": ["tests/test_unit.py"]})
        )
    )

    assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL
    assert result.approval_expires_in_seconds == 300


def test_run_command_schema_is_per_profile_and_arguments_are_quarantined() -> None:
    first = RunCommandTool(toolchain_profile=_profile())
    second_profile = _profile().model_copy(
        update={
            "command_authorities": (
                _profile()
                .structured_command_authorities[0]
                .model_copy(update={"selector": "other"}),
            )
        }
    )
    second = RunCommandTool(toolchain_profile=second_profile)

    assert first.schema["properties"]["selector"]["enum"] == ["focused-test"]
    assert second.schema["properties"]["selector"]["enum"] == ["other"]
    assert first._publish_arguments is False


def test_structured_command_quarantines_exact_working_directory(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked\n")
    (tmp_path / "tests").mkdir()
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace")
    authority = (
        _profile()
        .structured_command_authorities[0]
        .model_copy(
            update={
                "default_working_directory": "tests",
                "allowed_working_directories": ("tests",),
            }
        )
    )
    profile = _profile().model_copy(update={"command_authorities": (authority,)})

    result = _run(
        RunCommandTool(toolchain_profile=profile),
        _AdmittedRunner(profile),
        workspace,
        {"selector": "focused-test", "args": ["tests/test_unit.py"]},
    )

    assert result.is_error is False
    assert "working_directory" not in result.structured
    assert result.structured["working_directory_sha256"].startswith("sha256:")
