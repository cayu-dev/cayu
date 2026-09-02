"""Bounded trusted-code Docker environment with conflict-aware workspace sync."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cayu._coding_product_authority import (
    CODING_PRODUCT_FINAL_GIT_MAX_CHANGES,
    CodingProductSourceCopyAuthority,
    is_final_git_result_envelope,
    source_copy_authority_from_metadata,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    thaw_json_value,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import ToolContext
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
from cayu.environments.docker_toolchains import (
    DockerCodingToolchainError,
    DockerCodingToolchainProfile,
    legacy_docker_coding_toolchain_profile,
    verify_local_docker_coding_toolchain_dependencies,
)
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
from cayu.workspaces.base import matches_list_pattern
from cayu.workspaces.revisions import (
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
    compare_workspace_revisions,
    observe_deterministic_workspace,
)

DOCKER_CODING_PROTECTED_DIRECTORY_NAMES = (".cayu", ".git", ".runtime")
_DOCKER_CODING_RUNTIME_EXECUTABLES = ("git", "python3", "rm", "sh", "sleep")
_GIT_HASH_PATH_CHUNK_BYTES = 24 * 1024
_GIT_HASH_PATH_CHUNK_COUNT = 512
_GIT_HASH_OUTPUT_BYTES = 64 * 1024
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


@dataclass(frozen=True, slots=True)
class _EphemeralGitBaseline:
    head_revision: str
    staged_entries_sha256: str
    tracked_flags_sha256: str
    configuration_sha256: str


@dataclass(frozen=True, slots=True)
class _DockerCodingBindAuthority:
    session_id: str
    source: CodingProductSourceCopyAuthority | None
    workspace_baseline: WorkspaceRevisionObservation | None
    git: _EphemeralGitBaseline
    git_transformed_baseline_paths: frozenset[str]


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
        source_copy_authority: CodingProductSourceCopyAuthority | None = None,
        path: str = "/workspace",
    ) -> None:
        if not isinstance(target_workspace, RunnerWorkspace):
            raise TypeError("Docker coding target_workspace must be a RunnerWorkspace.")
        if not isinstance(limits, DockerWorkspaceTransferLimits):
            raise TypeError("Docker coding limits must be DockerWorkspaceTransferLimits.")
        if source_copy_authority is not None and type(source_copy_authority) is not (
            CodingProductSourceCopyAuthority
        ):
            raise TypeError(
                "source_copy_authority must be CodingProductSourceCopyAuthority or None."
            )
        expected_exclusions = frozenset(DOCKER_CODING_PROTECTED_DIRECTORY_NAMES)
        if not expected_exclusions.issubset(target_workspace.excluded_directory_names):
            raise ValueError(
                "Docker coding target workspace must exclude .cayu, .git, and .runtime."
            )
        self._docker_target = target_workspace
        self._source_copy_authority = source_copy_authority
        self._coding_authority_lock = threading.Lock()
        self._coding_authorities: dict[str, _DockerCodingBindAuthority] = {}
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
            preserve_git_modes=True,
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
            excluded_path_patterns=workspace.excluded_path_patterns,
        )
        if frozenset(self._docker_target.excluded_directory_names) != frozenset(
            protected_source.excluded_directory_names
        ) or frozenset(
            pattern.casefold() for pattern in self._docker_target.excluded_path_patterns
        ) != frozenset(pattern.casefold() for pattern in protected_source.excluded_path_patterns):
            raise ValueError("Docker coding source and target path projections must match exactly.")
        bound = await super().bind(
            protected_source,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        try:
            copied_workspace = bound.workspace
            if not isinstance(copied_workspace, RunnerWorkspace):
                raise RuntimeError("Docker coding binding lost its target workspace.")
            authority = self._source_copy_authority
            copied: WorkspaceRevisionObservation | None = None
            if authority is not None:
                if workspace.id != authority.source_workspace_id:
                    raise RuntimeError(
                        "Docker coding source workspace conflicts with product authority."
                    )
                copied = await observe_deterministic_workspace(
                    copied_workspace,
                    observer="cayu-coding-product-source",
                    limits=authority.observation_limits,
                )
                if (
                    copied.status is not WorkspaceRevisionObservationStatus.SUPPORTED
                    or copied.path_scope != "complete"
                    or copied.revision != authority.baseline_revision
                ):
                    raise RuntimeError(
                        "Docker coding copy-in conflicts with the admitted source revision."
                    )
            git_baseline = await _initialize_ephemeral_git_baseline(runner)
            transformed_baseline_paths = frozenset()
            if copied is not None:
                transformed = await _git_paths_with_transformed_bytes(
                    runner,
                    paths=_observable_file_paths(copied),
                )
                if transformed is None:
                    raise RuntimeError(
                        "Docker coding could not bind raw source bytes to its Git baseline."
                    )
                transformed_baseline_paths = transformed
            state_key = bound.state_key
            if state_key is None:
                raise RuntimeError("Docker coding binding lost its sync generation authority.")
            with self._coding_authority_lock:
                if state_key in self._coding_authorities:
                    raise RuntimeError("Docker coding generated duplicate bind authority.")
                self._coding_authorities[state_key] = _DockerCodingBindAuthority(
                    session_id=session_id,
                    source=authority,
                    workspace_baseline=copied,
                    git=git_baseline,
                    git_transformed_baseline_paths=transformed_baseline_paths,
                )
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
        authority = self._bind_authority(bound)
        final_git_evidence = (
            None
            if authority.source is None
            else await _capture_final_git_evidence(bound, authority)
        )
        await _require_no_publishable_ignored_paths(bound)
        snapshot = await super().finalize(bound, outcome=outcome, metadata=metadata)
        if snapshot is None:
            self._discard_bind_authority(bound)
            return None
        self._discard_bind_authority(bound)
        if final_git_evidence is None:
            return snapshot
        return replace(
            snapshot,
            metadata={**snapshot.metadata, "final_git_evidence": final_git_evidence},
        )

    def abandon(self, bound: BoundWorkspace) -> bool:
        abandoned = super().abandon(bound)
        if abandoned:
            self._discard_bind_authority(bound)
        return abandoned

    def _bind_authority(self, bound: BoundWorkspace) -> _DockerCodingBindAuthority:
        state_key = bound.state_key
        if state_key is None:
            raise RuntimeError("Docker coding finalization lost its bind generation.")
        with self._coding_authority_lock:
            authority = self._coding_authorities.get(state_key)
        if authority is None:
            raise RuntimeError("Docker coding finalization lost its admitted authority.")
        return authority

    def _discard_bind_authority(self, bound: BoundWorkspace) -> None:
        state_key = bound.state_key
        if state_key is None:
            return
        with self._coding_authority_lock:
            self._coding_authorities.pop(state_key, None)


class DockerCodingEnvironmentFactory(EnvironmentFactory):
    """Create exact, non-networked Docker environments for explicitly trusted code."""

    def __init__(
        self,
        *,
        source_workspace: LocalWorkspace,
        image_identity: DockerImageIdentity | None = None,
        toolchain_profile: DockerCodingToolchainProfile | None = None,
        restrictions: DockerWorkloadRestrictions | None = None,
        required_executables: tuple[str, ...] = (),
        transfer_limits: DockerWorkspaceTransferLimits | None = None,
        runtime: str | None = None,
        seccomp_profile: str | None = None,
        docker_path: str | None = None,
    ) -> None:
        if not isinstance(source_workspace, LocalWorkspace):
            raise TypeError("source_workspace must be LocalWorkspace.")
        if image_identity is not None and not isinstance(image_identity, DockerImageIdentity):
            raise TypeError("image_identity must be DockerImageIdentity or None.")
        if (
            toolchain_profile is not None
            and type(toolchain_profile) is not DockerCodingToolchainProfile
        ):
            raise TypeError(
                "toolchain_profile must be an exact DockerCodingToolchainProfile or None."
            )
        if image_identity is None and toolchain_profile is None:
            raise ValueError("image_identity or toolchain_profile is required.")
        if restrictions is not None and not isinstance(restrictions, DockerWorkloadRestrictions):
            raise TypeError("restrictions must be DockerWorkloadRestrictions or None.")
        if transfer_limits is not None and not isinstance(
            transfer_limits, DockerWorkspaceTransferLimits
        ):
            raise TypeError("transfer_limits must be DockerWorkspaceTransferLimits or None.")
        self.source_workspace = source_workspace
        selected_restrictions = restrictions or (
            DockerWorkloadRestrictions()
            if toolchain_profile is None
            else toolchain_profile.restrictions
        )
        self.restrictions = DockerWorkloadRestrictions.model_validate(
            selected_restrictions.model_dump(mode="python")
        )
        selected_image = image_identity or (
            None if toolchain_profile is None else toolchain_profile.image_identity
        )
        if selected_image is None:  # pragma: no cover - guarded above
            raise AssertionError("Docker coding image selection was lost.")
        self.image_identity = DockerImageIdentity.model_validate(
            selected_image.model_dump(mode="python")
        )
        if toolchain_profile is not None:
            owned_profile = DockerCodingToolchainProfile.model_validate(
                toolchain_profile.model_dump(mode="python", by_alias=True)
            )
            if owned_profile.image_identity != self.image_identity:
                raise ValueError("toolchain_profile image_identity must match image_identity.")
            if owned_profile.restrictions != self.restrictions:
                raise ValueError("toolchain_profile restrictions must match restrictions.")
        else:
            owned_profile = None
        self._explicit_toolchain_profile = owned_profile is not None
        if isinstance(required_executables, str | bytes):
            raise TypeError("required_executables must be an iterable of strings.")
        profile_executables = () if owned_profile is None else owned_profile.required_executables
        required = tuple(
            sorted(
                set(
                    (
                        *required_executables,
                        *profile_executables,
                        *_DOCKER_CODING_RUNTIME_EXECUTABLES,
                    )
                )
            )
        )
        if any(type(value) is not str or not value.strip() for value in required):
            raise ValueError("required_executables must contain nonblank strings.")
        if len(required) > 64:
            raise ValueError("required_executables must contain at most 64 entries.")
        self.required_executables = required
        self.toolchain_profile = owned_profile or legacy_docker_coding_toolchain_profile(
            image_identity=self.image_identity,
            restrictions=self.restrictions,
            required_executables=(),
        )
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
            toolchain_profile_fingerprint=self.toolchain_profile.fingerprint,
            source_excluded_directory_names=self.source_workspace.excluded_directory_names,
            source_excluded_path_patterns=self.source_workspace.excluded_path_patterns,
        )
        self._profile_identity = ExecutionProfileBehaviorIdentity(
            name="cayu.docker_coding_environment",
            behavior_version="9",
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

    def create_workspace_binding(
        self,
        request: EnvironmentFactoryRequest,
        *,
        target_workspace: RunnerWorkspace,
    ) -> DockerCodingWorkspaceBinding:
        """Construct the exact request-bound publication seam for an admitted runner."""

        self._validate_request(request)
        source_copy_authority = source_copy_authority_from_metadata(request.metadata)
        if (
            source_copy_authority is not None
            and source_copy_authority.source_workspace_id != self.source_workspace.id
        ):
            raise RuntimeError(
                "Docker coding source-copy authority names another source workspace."
            )
        return DockerCodingWorkspaceBinding(
            target_workspace=target_workspace,
            limits=self.transfer_limits,
            source_copy_authority=source_copy_authority,
        )

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
        verify_local_docker_coding_toolchain_dependencies(
            self.toolchain_profile,
            self.source_workspace.root,
        )
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
            toolchain_profile_fingerprint=self.toolchain_profile.fingerprint,
        )
        try:
            final_candidate = runner.execution_admission_candidate()
            final_evidence = final_candidate.evidence
            if (
                final_evidence.environment_fingerprint is None
                or final_evidence.image_fingerprint != self.image_identity.fingerprint
                or final_evidence.toolchain_profile_fingerprint
                != self.toolchain_profile.fingerprint
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
            await _run_toolchain_admission_probes(
                runner,
                self.toolchain_profile,
                verify_platform=self._explicit_toolchain_profile,
            )
            workspace = RunnerWorkspace(
                runner,
                cwd=None,
                workspace_id=f"docker:{runner.container_id}:workspace",
                python_executable="python3",
                default_read_limit_bytes=self.transfer_limits.max_file_bytes,
                default_list_limit=self.transfer_limits.max_files,
                excluded_directory_names=_merge_protected_directory_names(
                    self.source_workspace.excluded_directory_names
                ),
                excluded_path_patterns=self.source_workspace.excluded_path_patterns,
            )
            binding = self.create_workspace_binding(
                request,
                target_workspace=workspace,
            )
            evidence_metadata = final_decision.evidence.to_metadata()
            metadata = {
                "kind": "docker_coding",
                "container_id": runner.container_id,
                "image_fingerprint": self.image_identity.fingerprint,
                "configuration_fingerprint": self._configuration_fingerprint,
                **self.toolchain_profile.evidence(),
                "toolchain_command_authorities": [
                    {
                        "selector": authority.selector,
                        "revision": authority.revision,
                        "exposure": authority.exposure,
                        "fingerprint": authority.fingerprint,
                    }
                    for authority in self.toolchain_profile.command_authorities
                ],
                "execution_capabilities": evidence_metadata,
                "execution_requirements": effective_requirements.model_dump(mode="json"),
                "protected_directory_names": list(DOCKER_CODING_PROTECTED_DIRECTORY_NAMES),
                "source_excluded_directory_names": list(
                    self.source_workspace.excluded_directory_names
                ),
                "source_excluded_path_patterns": list(self.source_workspace.excluded_path_patterns),
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
            toolchain_profile_fingerprint=self.toolchain_profile.fingerprint,
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


async def _initialize_ephemeral_git_baseline(
    runner: DockerRunner,
) -> _EphemeralGitBaseline:
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
    if not await _git_filter_configuration_is_safe(runner):
        raise RuntimeError("Docker coding ephemeral Git configuration is unsafe.")
    return await _ephemeral_git_baseline(runner)


async def _ephemeral_git_baseline(runner: DockerRunner) -> _EphemeralGitBaseline:
    async def capture(*argv: str, output_limit_bytes: int = 16 * 1024 * 1024) -> str:
        result = await runner.exec_system(
            ExecCommand.process("git", *argv),
            env={
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_CONFIG_COUNT": "0",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            },
            env_remove=_GIT_ENV_REMOVE,
            timeout_s=60,
            output_limit_bytes=output_limit_bytes,
        )
        if (
            result.exit_code != 0
            or result.timed_out
            or result.cancelled
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            raise RuntimeError("Docker coding could not verify its ephemeral Git baseline.")
        return result.stdout

    head_revision = (
        await capture("rev-parse", "--verify", "HEAD", output_limit_bytes=1024)
    ).strip()
    staged_entries = await capture("ls-files", "--stage", "-z", "--")
    tracked_flags = await capture("ls-files", "-v", "-z", "--")
    configuration = await capture("config", "--includes", "--null", "--list")
    return _EphemeralGitBaseline(
        head_revision=head_revision,
        staged_entries_sha256="sha256:" + sha256(staged_entries.encode("utf-8")).hexdigest(),
        tracked_flags_sha256="sha256:" + sha256(tracked_flags.encode("utf-8")).hexdigest(),
        configuration_sha256="sha256:" + sha256(configuration.encode("utf-8")).hexdigest(),
    )


async def _capture_final_git_evidence(
    bound: BoundWorkspace,
    authority: _DockerCodingBindAuthority,
) -> dict[str, object]:
    # Import lazily because the public tools package also imports environment
    # adapters while the module graph is being initialized.
    from cayu.tools.git import (
        MAX_GIT_CHANGES_RESULT_BYTES,
        GitChangesTool,
    )

    runner = bound.runner
    workspace = bound.workspace
    source = authority.source
    if (
        not isinstance(runner, DockerRunner)
        or not isinstance(workspace, RunnerWorkspace)
        or source is None
    ):
        raise TypeError("Docker coding final Git capture lost its admitted authority.")
    before_identity = await _ephemeral_git_baseline(runner)
    if before_identity != authority.git:
        raise RuntimeError("Docker coding ephemeral Git authority changed during execution.")
    before = await observe_deterministic_workspace(
        workspace,
        observer="cayu-coding-product-source",
        limits=source.observation_limits,
    )
    if (
        before.status is not WorkspaceRevisionObservationStatus.SUPPORTED
        or before.path_scope != "complete"
        or before.revision is None
    ):
        raise RuntimeError("Docker coding could not observe its final workspace revision.")
    context = ToolContext(
        session_id=authority.session_id,
        environment_name="coding",
        workspace=workspace,
        runner=runner,
    )
    captured: dict[str, dict[str, object]] = {}
    tool = GitChangesTool()
    for mode in ("status", "summary", "diff"):
        result = await tool.run(
            context,
            {
                "mode": mode,
                "scope": "all",
                "limit": CODING_PRODUCT_FINAL_GIT_MAX_CHANGES,
                "max_result_bytes": MAX_GIT_CHANGES_RESULT_BYTES,
            },
        )
        structured = thaw_json_value(result.structured)
        if result.is_error or not is_final_git_result_envelope(structured, mode=mode):
            detail = (
                structured.get("error")
                if type(structured) is dict and type(structured.get("error")) is str
                else "invalid_result_shape"
            )
            raise RuntimeError(f"Docker coding final Git {mode} evidence is incomplete ({detail}).")
        assert type(structured) is dict
        captured[mode] = {
            "structured": copy_durable_json_object(
                structured,
                f"final_git_{mode}",
            ),
            **({"content": result.content} if mode == "diff" else {}),
        }
    if not _final_git_evidence_covers_workspace_delta(
        baseline=authority.workspace_baseline,
        final=before,
        status=captured["status"]["structured"],
        summary=captured["summary"]["structured"],
        diff=captured["diff"]["structured"],
    ) or not await _git_diff_preserves_workspace_bytes(
        runner,
        baseline=authority.workspace_baseline,
        final=before,
        transformed_baseline_paths=authority.git_transformed_baseline_paths,
    ):
        _mark_final_git_diff_incomplete(captured["diff"]["structured"])
    after = await observe_deterministic_workspace(
        workspace,
        observer="cayu-coding-product-source",
        limits=source.observation_limits,
    )
    after_identity = await _ephemeral_git_baseline(runner)
    if after != before or after_identity != authority.git:
        raise RuntimeError("Docker coding workspace changed during final Git capture.")
    return {
        "request_fingerprint": source.request_fingerprint,
        "source_workspace_id": source.source_workspace_id,
        "baseline_revision": source.baseline_revision,
        "workspace_revision": before.revision,
        **captured,
    }


def _final_git_evidence_covers_workspace_delta(
    *,
    baseline: WorkspaceRevisionObservation | None,
    final: WorkspaceRevisionObservation,
    status: object,
    summary: object,
    diff: object,
) -> bool:
    """Require Git evidence to cover every byte-level workspace path change."""

    if baseline is None:
        return False
    delta = compare_workspace_revisions(baseline, final)
    if delta.status not in {
        WorkspaceRevisionDeltaStatus.CHANGED,
        WorkspaceRevisionDeltaStatus.NO_CHANGE,
    }:
        return False
    status_entries = _final_git_change_identities(status)
    summary_entries = _final_git_change_identities(summary)
    diff_entries = _final_git_change_identities(diff)
    if (
        status_entries is None
        or status_entries != summary_entries
        or status_entries != diff_entries
    ):
        return False
    workspace_paths = {
        path
        for change in delta.paths
        for path in (change.path, change.renamed_from)
        if path is not None
    }
    git_paths = {
        path
        for path, _index, _worktree, original_path in status_entries
        for path in (path, original_path)
        if path is not None
    }
    if workspace_paths != git_paths or type(summary) is not dict:
        return False
    summary = cast("dict[str, Any]", summary)
    changes = summary.get("changes")
    if type(changes) is not list:
        return False
    return all(
        type(change) is dict
        and (
            change.get("count_kind") == "untracked"
            if change.get("index") == "?"
            else change.get("count_kind") in {"text", "binary"}
        )
        for change in changes
    )


def _final_git_change_identities(
    structured: object,
) -> tuple[tuple[str, str, str, str | None], ...] | None:
    if type(structured) is not dict:
        return None
    structured = cast("dict[str, Any]", structured)
    changes = structured.get("changes")
    if type(changes) is not list:
        return None
    entries: list[tuple[str, str, str, str | None]] = []
    for change in changes:
        if type(change) is not dict:
            return None
        path = change.get("path")
        index = change.get("index")
        worktree = change.get("worktree")
        original_path = change.get("original_path")
        if (
            type(path) is not str
            or not path
            or type(index) is not str
            or len(index) != 1
            or type(worktree) is not str
            or len(worktree) != 1
            or (original_path is not None and type(original_path) is not str)
            or original_path == ""
        ):
            return None
        entries.append((path, index, worktree, original_path))
    return tuple(entries)


async def _git_diff_preserves_workspace_bytes(
    runner: DockerRunner,
    *,
    baseline: WorkspaceRevisionObservation | None,
    final: WorkspaceRevisionObservation,
    transformed_baseline_paths: frozenset[str],
) -> bool:
    """Require Git's real pre/post images to equal the copied raw source bytes."""

    if baseline is None:
        return False
    delta = compare_workspace_revisions(baseline, final)
    if delta.status not in {
        WorkspaceRevisionDeltaStatus.CHANGED,
        WorkspaceRevisionDeltaStatus.NO_CHANGE,
    }:
        return False
    changed_paths = {
        path
        for change in delta.paths
        for path in (change.path, change.renamed_from)
        if path is not None
    }
    if not changed_paths:
        return True
    if changed_paths.intersection(transformed_baseline_paths):
        return False
    final_paths = tuple(sorted(changed_paths.intersection(_observable_file_paths(final))))
    transformed_final_paths = await _git_paths_with_transformed_bytes(
        runner,
        paths=final_paths,
    )
    return transformed_final_paths == frozenset()


def _observable_file_paths(observation: WorkspaceRevisionObservation) -> tuple[str, ...]:
    return tuple(
        sorted(
            entry.path
            for entry in observation.paths
            if entry.kind in {"file", "symlink"} and entry.content_sha256 is not None
        )
    )


def _git_hash_path_chunks(paths: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    current_bytes = 0
    for path in paths:
        path_bytes = len(path.encode("utf-8")) + 1
        if current and (
            len(current) >= _GIT_HASH_PATH_CHUNK_COUNT
            or current_bytes + path_bytes > _GIT_HASH_PATH_CHUNK_BYTES
        ):
            chunks.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += path_bytes
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


async def _git_paths_with_transformed_bytes(
    runner: DockerRunner,
    *,
    paths: tuple[str, ...],
) -> frozenset[str] | None:
    if not paths:
        return frozenset()
    if not await _git_filter_configuration_is_safe(runner):
        return None
    transformed: set[str] = set()
    for chunk in _git_hash_path_chunks(paths):
        raw = await _git_hash_paths(runner, paths=chunk, filtered=False)
        filtered = await _git_hash_paths(runner, paths=chunk, filtered=True)
        if raw is None or filtered is None:
            return None
        transformed.update(
            path
            for path, raw_object, filtered_object in zip(
                chunk,
                raw,
                filtered,
                strict=True,
            )
            if raw_object != filtered_object
        )
    return frozenset(transformed)


async def _git_filter_configuration_is_safe(runner: DockerRunner) -> bool:
    configured_filters = await runner.exec_system(
        ExecCommand.process(
            "git",
            "config",
            "--includes",
            "--null",
            "--name-only",
            "--get-regexp",
            r"^(include([iI][fF])?\..*|filter\..*\.(clean|smudge|process))$",
        ),
        env={
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        },
        env_remove=_GIT_ENV_REMOVE,
        timeout_s=60,
        output_limit_bytes=_GIT_HASH_OUTPUT_BYTES,
    )
    return not (
        configured_filters.exit_code not in {0, 1}
        or configured_filters.timed_out
        or configured_filters.cancelled
        or configured_filters.stdout_truncated
        or configured_filters.stderr_truncated
        or bool(configured_filters.stdout)
        or configured_filters.exit_code == 0
    )


async def _git_hash_paths(
    runner: DockerRunner,
    *,
    paths: tuple[str, ...],
    filtered: bool,
) -> tuple[str, ...] | None:
    result = await runner.exec_system(
        ExecCommand.process(
            "git",
            "hash-object",
            "--filters" if filtered else "--no-filters",
            "--",
            *paths,
        ),
        env={
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        },
        env_remove=_GIT_ENV_REMOVE,
        timeout_s=60,
        output_limit_bytes=_GIT_HASH_OUTPUT_BYTES,
    )
    if (
        result.exit_code != 0
        or result.timed_out
        or result.cancelled
        or result.stdout_truncated
        or result.stderr_truncated
    ):
        return None
    object_ids = tuple(result.stdout.splitlines())
    if len(object_ids) != len(paths) or any(
        len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
        for value in object_ids
    ):
        return None
    return object_ids


def _mark_final_git_diff_incomplete(structured: object) -> None:
    if type(structured) is not dict:
        raise RuntimeError("Docker coding final Git diff evidence lost its trusted shape.")
    structured = cast("dict[str, Any]", structured)
    reasons = structured.get("truncation_reasons")
    if type(reasons) is not list:
        raise RuntimeError("Docker coding final Git diff evidence lost its trusted shape.")
    reason = "workspace_delta_unrepresented"
    if reason not in reasons:
        reasons.append(reason)
    structured["truncated"] = True
    if not is_final_git_result_envelope(structured, mode="diff"):
        raise RuntimeError("Docker coding final Git diff evidence lost its trusted shape.")


async def _require_no_publishable_ignored_paths(bound: BoundWorkspace) -> None:
    """Refuse copy-back when Git would hide a path in the published source scope."""

    runner = bound.runner
    workspace = bound.workspace
    if not isinstance(runner, DockerRunner) or not isinstance(workspace, RunnerWorkspace):
        raise TypeError("Docker coding publication lost its admitted runner workspace.")
    result = await runner.exec_system(
        ExecCommand.process(
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--directory",
            "-z",
            "--",
        ),
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        },
        env_remove=_GIT_ENV_REMOVE,
        timeout_s=60,
        output_limit_bytes=1024 * 1024,
    )
    if (
        result.exit_code != 0
        or result.timed_out
        or result.cancelled
        or result.stdout_truncated
        or result.stderr_truncated
    ):
        raise RuntimeError(
            "Docker coding could not prove that publication excludes Git-ignored paths."
        )
    excluded = {name.rstrip(" .").casefold() for name in workspace.excluded_directory_names}
    ignored_paths = tuple(path for path in result.stdout.split("\0") if path)
    publishable_ignored_paths = tuple(
        path
        for path in ignored_paths
        if not (
            any(
                part.rstrip(" .").casefold() in excluded
                for part in path.rstrip("/").replace("\\", "/").split("/")
                if part
            )
            or _path_matches_projection_pattern(
                path,
                workspace.excluded_path_patterns,
            )
        )
    )
    if publishable_ignored_paths:
        raise RuntimeError(
            "Docker coding refused publication because Git would omit a source path."
        )


def _path_matches_projection_pattern(path: str, patterns: tuple[str, ...]) -> bool:
    normalized_parts = tuple(
        part.rstrip(" .").casefold()
        for part in path.rstrip("/").replace("\\", "/").split("/")
        if part
    )
    return any(
        matches_list_pattern("/".join(normalized_parts[:end]), pattern.casefold())
        for pattern in patterns
        for end in range(1, len(normalized_parts) + 1)
    )


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
    toolchain_profile_fingerprint: str,
    source_excluded_directory_names: tuple[str, ...],
    source_excluded_path_patterns: tuple[str, ...],
) -> str:
    material = {
        "schema": "cayu.docker_coding_environment.v1",
        "image_identity": image_identity.model_dump(mode="json"),
        "restrictions": restrictions.model_dump(mode="json"),
        "required_executables": list(required_executables),
        "transfer_limits": transfer_limits.model_dump(mode="json"),
        "runtime": runtime,
        "seccomp_sha256": seccomp_sha256,
        "toolchain_profile_fingerprint": toolchain_profile_fingerprint,
        "network": "none",
        "default_cwd": "/workspace",
        "host_mounts": False,
        "credential_mode": CredentialMode.TRUSTED_TOOL.value,
        "protected_directory_names": list(DOCKER_CODING_PROTECTED_DIRECTORY_NAMES),
        "source_excluded_directory_names": list(source_excluded_directory_names),
        "source_excluded_path_patterns": list(source_excluded_path_patterns),
        "sync_back": "revision_aware",
        "guest_git_baseline": "ephemeral",
    }
    return (
        "sha256:"
        + sha256(canonical_durable_json_bytes(material, "docker_coding_configuration")).hexdigest()
    )


async def _run_toolchain_admission_probes(
    runner: DockerRunner,
    profile: DockerCodingToolchainProfile,
    *,
    verify_platform: bool,
) -> None:
    """Run bounded probes only after exact final-container admission."""

    if verify_platform:
        expected_uid, expected_gid = profile.runtime_user.split(":", 1)
        try:
            platform_result = await runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    (
                        "import os,platform,sys; "
                        "expected_cwd,uid,gid=sys.argv[1:4]; support=sys.argv[4:]; "
                        "ok=(os.getcwd()==expected_cwd and os.getuid()==int(uid) and "
                        "os.getgid()==int(gid) and all(os.path.exists(path) and "
                        "os.access(path,os.R_OK) for path in support)); "
                        "probe=os.path.join(expected_cwd,'.cayu-toolchain-write-probe'); "
                        "fd=os.open(probe,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600) if ok else -1; "
                        "os.close(fd) if fd>=0 else None; "
                        "os.unlink(probe) if fd>=0 else None; "
                        "sys.exit(73) if not ok else None; "
                        "machine={'x86_64':'amd64','aarch64':'arm64'}.get("
                        "platform.machine().lower(),platform.machine().lower()); "
                        "print(platform.system().lower()+'/'+machine)"
                    ),
                    profile.working_directory,
                    expected_uid,
                    expected_gid,
                    *profile.read_only_support_paths,
                ),
                cwd=profile.working_directory,
                env=None,
                timeout_s=10,
                stdin=None,
                output_limit_bytes=4096,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise DockerCodingToolchainError(
                "platform_probe_unavailable",
                "Docker coding toolchain platform could not be verified.",
            ) from None
        expected_platform = f"{profile.platform_os}/{profile.platform_architecture}\n"
        if (
            platform_result.exit_code != 0
            or platform_result.timed_out
            or platform_result.cancelled
            or platform_result.stdout_truncated
            or platform_result.stderr_truncated
            or platform_result.stdout != expected_platform
        ):
            raise DockerCodingToolchainError(
                "platform_mismatch",
                "Docker coding toolchain platform does not match its admitted profile.",
            )

    for probe in profile.admission_probes:
        try:
            result = await runner.exec(
                ExecCommand.process(*probe.argv),
                cwd=profile.working_directory,
                env=None,
                timeout_s=probe.timeout_seconds,
                stdin=None,
                output_limit_bytes=probe.max_output_bytes,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise DockerCodingToolchainError(
                "admission_probe_unavailable",
                "Docker coding toolchain admission probe could not settle.",
            ) from None
        if result.timed_out:
            raise DockerCodingToolchainError(
                "admission_probe_timeout",
                "Docker coding toolchain admission probe timed out.",
            )
        if (
            result.cancelled
            or result.stdout_truncated
            or result.stderr_truncated
            or result.exit_code not in probe.expected_exit_codes
        ):
            raise DockerCodingToolchainError(
                "admission_probe_mismatch",
                "Docker coding toolchain admission probe did not match its declaration.",
            )
        if probe.stdout_sha256 is not None:
            observed = "sha256:" + sha256(result.stdout.encode("utf-8")).hexdigest()
            if observed != probe.stdout_sha256:
                raise DockerCodingToolchainError(
                    "admission_probe_mismatch",
                    "Docker coding toolchain admission probe did not match its declaration.",
                )
