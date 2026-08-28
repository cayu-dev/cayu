"""Bounded trusted-code Docker environment with conflict-aware workspace sync."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cayu._validation import canonical_durable_json_bytes
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.credentials import CredentialMode
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionExecutableEvidence,
    ExecutionToolRequirementEvidence,
    evaluate_execution_admission,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import BoundWorkspace, SyncBinding, WorkspaceSnapshot
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.runners import ExecCommand, Runner
from cayu.runners.docker import DockerRunner, validate_docker_seccomp_profile
from cayu.runners.docker_workload import DockerImageIdentity, DockerWorkloadRestrictions
from cayu.workspaces import LocalWorkspace, RunnerWorkspace, Workspace

DOCKER_CODING_PROTECTED_DIRECTORY_NAMES = (".cayu", ".git", ".runtime")
_DOCKER_CODING_RUNTIME_EXECUTABLES = ("git", "python3", "rm", "sh", "sleep")
_GIT_ENV_REMOVE = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_EXTERNAL_DIFF",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_WORK_TREE",
)


class DockerWorkspaceTransferLimits(BaseModel):
    """Finite copy-in/copy-back limits for one coding container."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_files: int = Field(default=10_000, ge=1, le=100_000)
    max_file_bytes: int = Field(
        default=8 * 1024 * 1024,
        ge=1,
        le=64 * 1024 * 1024,
    )
    max_total_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1,
        le=1024 * 1024 * 1024,
    )
    max_archive_bytes: int = Field(
        default=128 * 1024 * 1024,
        ge=1024,
        le=2 * 1024 * 1024 * 1024,
    )


class DockerCodingWorkspaceBinding(SyncBinding):
    """Sync a projected host tree and establish an ephemeral guest Git baseline."""

    def __init__(
        self,
        *,
        target_workspace: RunnerWorkspace,
        limits: DockerWorkspaceTransferLimits,
        path: str = "/workspace",
    ) -> None:
        if not isinstance(target_workspace, RunnerWorkspace):
            raise TypeError("Docker coding target_workspace must be a RunnerWorkspace.")
        if not isinstance(limits, DockerWorkspaceTransferLimits):
            raise TypeError("Docker coding limits must be DockerWorkspaceTransferLimits.")
        expected_exclusions = frozenset(DOCKER_CODING_PROTECTED_DIRECTORY_NAMES)
        if frozenset(target_workspace.excluded_directory_names) != expected_exclusions:
            raise ValueError(
                "Docker coding target workspace must exclude .cayu, .git, and .runtime."
            )
        self._docker_target = target_workspace
        super().__init__(
            target_workspace=target_workspace,
            path=path,
            max_files=limits.max_files,
            max_file_bytes=limits.max_file_bytes,
            max_total_bytes=limits.max_total_bytes,
            max_archive_bytes=limits.max_archive_bytes,
            clean_target="always",
            sync_back="always",
            delete_missing=True,
            source_conflict_policy="require_revision",
        )

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
        if not isinstance(workspace, LocalWorkspace):
            raise TypeError(
                "Docker coding requires a LocalWorkspace source so protected host paths "
                "can be excluded before traversal."
            )
        if not isinstance(runner, DockerRunner) or not self._docker_target.is_bound_to_runner(
            runner
        ):
            raise ValueError("Docker coding binding requires its exact DockerRunner.")
        protected_source = LocalWorkspace(
            workspace.root,
            workspace_id=workspace.id,
            excluded_directory_names=_merge_protected_directory_names(
                workspace.excluded_directory_names
            ),
        )
        bound = await super().bind(
            protected_source,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        try:
            await _initialize_ephemeral_git_baseline(runner)
        except BaseException:
            self.abandon(bound)
            raise
        return bound

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        return await super().finalize(bound, outcome=outcome, metadata=metadata)


class DockerCodingEnvironmentFactory(EnvironmentFactory):
    """Create exact, non-networked Docker environments for explicitly trusted code."""

    def __init__(
        self,
        *,
        source_workspace: LocalWorkspace,
        image_identity: DockerImageIdentity,
        restrictions: DockerWorkloadRestrictions | None = None,
        required_executables: tuple[str, ...] = (),
        transfer_limits: DockerWorkspaceTransferLimits | None = None,
        runtime: str | None = None,
        seccomp_profile: str | None = None,
        docker_path: str | None = None,
    ) -> None:
        if not isinstance(source_workspace, LocalWorkspace):
            raise TypeError("source_workspace must be LocalWorkspace.")
        if not isinstance(image_identity, DockerImageIdentity):
            raise TypeError("image_identity must be DockerImageIdentity.")
        if restrictions is not None and not isinstance(restrictions, DockerWorkloadRestrictions):
            raise TypeError("restrictions must be DockerWorkloadRestrictions or None.")
        if transfer_limits is not None and not isinstance(
            transfer_limits, DockerWorkspaceTransferLimits
        ):
            raise TypeError("transfer_limits must be DockerWorkspaceTransferLimits or None.")
        self.source_workspace = source_workspace
        self.image_identity = DockerImageIdentity.model_validate(
            image_identity.model_dump(mode="python")
        )
        self.restrictions = DockerWorkloadRestrictions.model_validate(
            (restrictions or DockerWorkloadRestrictions()).model_dump(mode="python")
        )
        if isinstance(required_executables, str | bytes):
            raise TypeError("required_executables must be an iterable of strings.")
        required = tuple(sorted(set((*required_executables, *_DOCKER_CODING_RUNTIME_EXECUTABLES))))
        if any(type(value) is not str or not value.strip() for value in required):
            raise ValueError("required_executables must contain nonblank strings.")
        if len(required) > 64:
            raise ValueError("required_executables must contain at most 64 entries.")
        self.required_executables = required
        self.transfer_limits = transfer_limits or DockerWorkspaceTransferLimits()
        self.runtime = runtime
        self.seccomp_profile = validate_docker_seccomp_profile(seccomp_profile)
        self.docker_path = docker_path
        self._seccomp_sha256 = _read_seccomp_sha256(seccomp_profile)
        self._configuration_fingerprint = _docker_coding_configuration_fingerprint(
            image_identity=self.image_identity,
            restrictions=self.restrictions,
            required_executables=self.required_executables,
            transfer_limits=self.transfer_limits,
            runtime=self.runtime,
            seccomp_sha256=self._seccomp_sha256,
        )
        self._profile_identity = ExecutionProfileBehaviorIdentity(
            name="cayu.docker_coding_environment",
            behavior_version="1",
            implementation_version=self._configuration_fingerprint,
        )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._profile_identity

    def construction_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return self._configured_candidate()

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate:
        if not isinstance(request, EnvironmentFactoryRequest):
            raise TypeError("Docker coding admission requires EnvironmentFactoryRequest.")
        return self._configured_candidate()

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self._validate_request(request)
        effective_requirements = request.execution_requirements.model_copy(
            update={
                "required_executables": tuple(
                    sorted(
                        set(request.execution_requirements.required_executables).union(
                            self.required_executables
                        )
                    )
                )
            }
        )
        evaluate_execution_admission(
            candidate="docker",
            requirements=effective_requirements,
            evidence=self._configured_candidate().evidence,
            stage="pre_create",
        ).require_admitted()
        if _read_seccomp_sha256(self.seccomp_profile) != self._seccomp_sha256:
            raise RuntimeError("Docker coding seccomp profile changed after factory admission.")

        runner = await DockerRunner.create(
            f"cayu-coding-{uuid4().hex}",
            image=self.image_identity.reference,
            runtime=self.runtime,
            default_cwd="/workspace",
            close_action="remove",
            docker_path=self.docker_path,
            replace=False,
            cancellation_cleanup="sandbox",
            timeout_cleanup="sandbox",
            credential_mode=CredentialMode.TRUSTED_TOOL,
            allow_raw_secret_env=False,
            network="none",
            seccomp_profile=self.seccomp_profile,
            image_identity=self.image_identity,
            workload_restrictions=self.restrictions,
            required_executables=self.required_executables,
        )
        try:
            final_candidate = runner.execution_admission_candidate()
            final_evidence = final_candidate.evidence
            if (
                final_evidence.environment_fingerprint is None
                or final_evidence.image_fingerprint != self.image_identity.fingerprint
                or final_evidence.tool_requirements is None
                or tuple(claim.executable for claim in final_evidence.tool_requirements.executables)
                != self.required_executables
            ):
                raise RuntimeError(
                    "Docker coding runner did not produce exact final environment evidence."
                )
            final_decision = evaluate_execution_admission(
                candidate=final_candidate.candidate,
                requirements=effective_requirements,
                evidence=final_evidence,
                stage="pre_exposure",
            ).require_admitted()
            if final_decision.evidence is None:
                raise RuntimeError("Docker coding admission returned no final evidence.")
            workspace = RunnerWorkspace(
                runner,
                cwd=None,
                workspace_id=f"docker:{runner.container_id}:workspace",
                python_executable="python3",
                default_read_limit_bytes=self.transfer_limits.max_file_bytes,
                default_list_limit=self.transfer_limits.max_files,
                excluded_directory_names=DOCKER_CODING_PROTECTED_DIRECTORY_NAMES,
            )
            binding = DockerCodingWorkspaceBinding(
                target_workspace=workspace,
                limits=self.transfer_limits,
            )
            evidence_metadata = final_decision.evidence.to_metadata()
            metadata = {
                "kind": "docker_coding",
                "container_id": runner.container_id,
                "image_fingerprint": self.image_identity.fingerprint,
                "configuration_fingerprint": self._configuration_fingerprint,
                "execution_capabilities": evidence_metadata,
                "execution_requirements": effective_requirements.model_dump(mode="json"),
                "protected_directory_names": list(DOCKER_CODING_PROTECTED_DIRECTORY_NAMES),
            }
            environment = Environment(
                EnvironmentSpec(
                    name=request.environment_name,
                    metadata=metadata,
                    execution_profile_identity=self.execution_profile_identity,
                ),
                workspace=self.source_workspace,
                runner=runner,
                binding=binding,
            )

            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                del action
                await runner.close()

            return EnvironmentFactoryResult(
                environment=environment,
                metadata=metadata,
                release=release,
            )
        except BaseException as original:
            try:
                await runner.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "Docker coding creation and exact container cleanup both failed.",
                    [original, cleanup_error],
                ) from None
            raise

    def _validate_request(self, request: EnvironmentFactoryRequest) -> None:
        if not isinstance(request, EnvironmentFactoryRequest):
            raise TypeError("Docker coding create requires EnvironmentFactoryRequest.")
        if request.operation is not EnvironmentFactoryOperation.CREATE:
            raise ValueError("Docker coding environments do not support reconnect.")

    def _configured_candidate(self) -> ExecutionAdmissionCandidate:
        environment_fingerprint = self._configuration_fingerprint
        image_fingerprint = self.image_identity.fingerprint
        if self.restrictions.supports_strict_privilege_evidence:
            privilege_claims = (
                ExecutionCapabilityClaim.declared("guest_privilege_containment"),
                ExecutionCapabilityClaim.declared("unprivileged_guest"),
            )
        else:
            privilege_claims = (
                ExecutionCapabilityClaim.unsupported(
                    "guest_privilege_containment",
                    reason_code="docker_privilege_restrictions_weakened",
                    remediation_code="use_verified_docker_restrictions",
                ),
                ExecutionCapabilityClaim.unsupported(
                    "unprivileged_guest",
                    reason_code="docker_privilege_restrictions_weakened",
                    remediation_code="use_verified_docker_restrictions",
                ),
            )
        unsupported = (
            ExecutionCapabilityClaim.unsupported(
                "untrusted_code_isolation",
                reason_code="docker_untrusted_isolation_unsupported",
                remediation_code="select_untrusted_isolation",
            ),
            ExecutionCapabilityClaim.unsupported(
                "brokered_egress",
                reason_code="docker_network_disabled",
                remediation_code="select_brokered_egress",
            ),
            ExecutionCapabilityClaim.unsupported(
                "read_only_host_inputs",
                reason_code="docker_host_inputs_not_mounted",
                remediation_code="use_workspace_sync",
            ),
            ExecutionCapabilityClaim.unsupported(
                "reconnect",
                reason_code="docker_reconnect_unsupported",
                remediation_code="select_reconnectable_execution",
            ),
        )
        evidence = ExecutionCapabilityEvidence(
            subject="docker",
            environment_fingerprint=environment_fingerprint,
            image_fingerprint=image_fingerprint,
            claims=(
                ExecutionCapabilityClaim.declared("real_credential_non_possession"),
                ExecutionCapabilityClaim.declared("deny_by_default_network"),
                *privilege_claims,
                ExecutionCapabilityClaim.declared("host_filesystem_isolation"),
                ExecutionCapabilityClaim.declared("confirmed_cancellation"),
                ExecutionCapabilityClaim.declared("confirmed_cleanup"),
                *unsupported,
            ),
            tool_requirements=ExecutionToolRequirementEvidence(
                environment_fingerprint=environment_fingerprint,
                image_fingerprint=image_fingerprint,
                executables=tuple(
                    ExecutionExecutableEvidence(
                        executable=executable,
                        state="declared",
                    )
                    for executable in self.required_executables
                ),
            ),
        )
        return ExecutionAdmissionCandidate(candidate="docker", evidence=evidence)


async def _initialize_ephemeral_git_baseline(runner: DockerRunner) -> None:
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_ASKPASS": "/bin/false",
        "GIT_CONFIG_COUNT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "PAGER": "cat",
    }
    commands = (
        ExecCommand.process("rm", "-rf", "--", ".git"),
        ExecCommand.process("git", "init", "-q"),
        ExecCommand.process("git", "config", "--local", "core.hooksPath", "/dev/null"),
        ExecCommand.process("git", "config", "--local", "core.fsmonitor", "false"),
        ExecCommand.process("git", "config", "--local", "core.pager", "cat"),
        ExecCommand.process("git", "config", "--local", "pager.branch", "false"),
        ExecCommand.process("git", "config", "--local", "pager.diff", "false"),
        ExecCommand.process("git", "config", "--local", "pager.status", "false"),
        ExecCommand.process("git", "config", "--local", "credential.helper", "/bin/false"),
        ExecCommand.process("git", "config", "--local", "credential.interactive", "never"),
        ExecCommand.process("git", "config", "--local", "commit.gpgSign", "false"),
        ExecCommand.process("git", "config", "--local", "tag.gpgSign", "false"),
        ExecCommand.process("git", "config", "--local", "protocol.allow", "never"),
        ExecCommand.process("git", "add", "-A", "--", "."),
        ExecCommand.process(
            "git",
            "-c",
            "user.name=Cayu Runtime",
            "-c",
            "user.email=runtime@cayu.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            "--allow-empty",
            "--no-gpg-sign",
            "--no-verify",
            "-m",
            "Cayu workspace baseline",
        ),
    )
    for command in commands:
        result = await runner.exec_system(
            command,
            env=environment,
            env_remove=_GIT_ENV_REMOVE,
            timeout_s=60,
            output_limit_bytes=64 * 1024,
        )
        if result.exit_code != 0 or result.timed_out:
            raise RuntimeError("Docker coding could not establish its ephemeral Git baseline.")


def _read_seccomp_sha256(path: str | None) -> str | None:
    if path is None:
        return None
    return sha256(Path(path).read_bytes()).hexdigest()


def _merge_protected_directory_names(existing: tuple[str, ...]) -> tuple[str, ...]:
    names = {name.casefold(): name for name in existing}
    for protected in DOCKER_CODING_PROTECTED_DIRECTORY_NAMES:
        names[protected.casefold()] = protected
    return tuple(names[key] for key in sorted(names))


def _docker_coding_configuration_fingerprint(
    *,
    image_identity: DockerImageIdentity,
    restrictions: DockerWorkloadRestrictions,
    required_executables: tuple[str, ...],
    transfer_limits: DockerWorkspaceTransferLimits,
    runtime: str | None,
    seccomp_sha256: str | None,
) -> str:
    material = {
        "schema": "cayu.docker_coding_environment.v1",
        "image_identity": image_identity.model_dump(mode="json"),
        "restrictions": restrictions.model_dump(mode="json"),
        "required_executables": list(required_executables),
        "transfer_limits": transfer_limits.model_dump(mode="json"),
        "runtime": runtime,
        "seccomp_sha256": seccomp_sha256,
        "network": "none",
        "default_cwd": "/workspace",
        "host_mounts": False,
        "credential_mode": CredentialMode.TRUSTED_TOOL.value,
        "protected_directory_names": list(DOCKER_CODING_PROTECTED_DIRECTORY_NAMES),
        "sync_back": "revision_aware",
        "guest_git_baseline": "ephemeral",
    }
    return (
        "sha256:"
        + sha256(canonical_durable_json_bytes(material, "docker_coding_configuration")).hexdigest()
    )
