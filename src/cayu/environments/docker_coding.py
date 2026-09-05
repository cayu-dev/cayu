"""Bounded trusted-code Docker environment with conflict-aware workspace sync."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

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
    require_durable_clean_nonblank,
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
    verify_local_docker_coding_toolchain_dependencies,
)
from cayu.environments.factory import (
    EnvironmentAllocationContext,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.immutable_inputs import (
    DockerImmutableInputMount,
    ImmutableInputAdapterCapability,
    ImmutableInputAttachment,
    ImmutableInputStore,
    LocalImmutableInput,
    docker_immutable_input_capability,
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


@dataclass(slots=True)
class _ImmutableInputFinalizeState:
    snapshot: WorkspaceSnapshot | None
    runner_closed: bool = False


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


def _validate_immutable_inputs(
    inputs: Sequence[LocalImmutableInput],
    *,
    store: ImmutableInputStore | None,
    runtime_compatibility_fingerprint: str | None,
) -> tuple[LocalImmutableInput, ...]:
    if isinstance(inputs, str | bytes):
        raise TypeError("immutable_inputs must be a sequence of LocalImmutableInput values.")
    values = tuple(inputs)
    if len(values) > 32:
        raise ValueError("Docker coding supports at most 32 immutable inputs.")
    if any(type(value) is not LocalImmutableInput for value in values):
        raise TypeError("immutable_inputs must contain exact LocalImmutableInput values.")
    if store is not None and not isinstance(store, ImmutableInputStore):
        raise TypeError("immutable_input_store must be ImmutableInputStore or None.")
    if values and store is None:
        raise ValueError("immutable_input_store is required when immutable_inputs are configured.")
    if runtime_compatibility_fingerprint is not None and (
        type(runtime_compatibility_fingerprint) is not str
        or len(runtime_compatibility_fingerprint) != 71
        or not runtime_compatibility_fingerprint.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in runtime_compatibility_fingerprint.removeprefix("sha256:")
        )
    ):
        raise ValueError(
            "immutable_input_runtime_compatibility_fingerprint must be a lowercase SHA-256."
        )
    if values and runtime_compatibility_fingerprint is None:
        raise ValueError(
            "immutable_input_runtime_compatibility_fingerprint is required with immutable inputs."
        )
    for value in values:
        if value.projection.runtime_compatibility_fingerprint != runtime_compatibility_fingerprint:
            raise ValueError("Immutable input runtime compatibility identity does not match.")
        if store is not None and (
            store.root.is_relative_to(value.root) or value.root.is_relative_to(store.root)
        ):
            raise ValueError("Immutable input source and managed store must not overlap.")
    ordered = tuple(sorted(values, key=lambda value: value.projection.target_path))
    targets = tuple(value.projection.target_path for value in ordered)
    if len(targets) != len(set(targets)):
        raise ValueError("Immutable input target paths must be unique.")
    for index, target in enumerate(targets):
        if any(
            other.startswith(target.rstrip("/") + "/") or target.startswith(other.rstrip("/") + "/")
            for other in targets[index + 1 :]
        ):
            raise ValueError("Immutable input target paths must not overlap.")
    return ordered


def _immutable_input_attachment_id(
    request: EnvironmentFactoryRequest,
    source: LocalImmutableInput,
) -> str:
    return _immutable_input_attachment_id_from_owner(
        session_id=request.session_id,
        environment_name=request.environment_name,
        source=source,
    )


def _immutable_input_attachment_id_from_owner(
    *,
    session_id: str,
    environment_name: str,
    source: LocalImmutableInput,
) -> str:
    material = {
        "schema_version": 1,
        "session_id": session_id,
        "environment_name": environment_name,
        "projection_fingerprint": source.projection.fingerprint,
    }
    return (
        "docker:"
        + sha256(
            canonical_durable_json_bytes(material, "docker_immutable_input_attachment")
        ).hexdigest()
    )


def _docker_coding_container_name(
    request: EnvironmentFactoryRequest,
    *,
    configuration_fingerprint: str,
    allocation_id: str | None = None,
) -> str:
    if allocation_id is not None:
        return f"cayu-coding-{allocation_id}"
    material = {
        "schema_version": 1,
        "session_id": request.session_id,
        "environment_name": request.environment_name,
        "configuration_fingerprint": configuration_fingerprint,
    }
    return (
        "cayu-coding-"
        + sha256(canonical_durable_json_bytes(material, "docker_coding_container_name")).hexdigest()
    )


async def _release_immutable_input_attachments(
    store: ImmutableInputStore,
    attachments: tuple[ImmutableInputAttachment, ...],
) -> None:
    failures: list[BaseException] = []
    for attachment in attachments:
        try:
            await store.release(attachment.attachment_id)
        except BaseException as error:
            failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(
            "Docker immutable input releases failed.",
            failures,
        )


class DockerCodingWorkspaceBinding(SyncBinding):
    """Sync a projected host tree and establish an ephemeral guest Git baseline."""

    def __init__(
        self,
        *,
        target_workspace: RunnerWorkspace,
        limits: DockerWorkspaceTransferLimits,
        source_copy_authority: CodingProductSourceCopyAuthority | None = None,
        immutable_input_store: ImmutableInputStore | None = None,
        immutable_input_attachments: Sequence[ImmutableInputAttachment] = (),
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
        attachments = tuple(immutable_input_attachments)
        if any(type(value) is not ImmutableInputAttachment for value in attachments):
            raise TypeError(
                "immutable_input_attachments must contain exact ImmutableInputAttachment values."
            )
        if attachments and not isinstance(immutable_input_store, ImmutableInputStore):
            raise ValueError("immutable_input_store is required with immutable input attachments.")
        expected_exclusions = frozenset(DOCKER_CODING_PROTECTED_DIRECTORY_NAMES)
        if not expected_exclusions.issubset(target_workspace.excluded_directory_names):
            raise ValueError(
                "Docker coding target workspace must exclude .cayu, .git, and .runtime."
            )
        self._docker_target = target_workspace
        self._source_copy_authority = source_copy_authority
        self._immutable_input_store = immutable_input_store
        self._immutable_input_attachments = attachments
        self._coding_authority_lock = threading.Lock()
        self._coding_authorities: dict[str, _DockerCodingBindAuthority] = {}
        self._immutable_finalize_states: dict[str, _ImmutableInputFinalizeState] = {}
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

    def _completion_finalization_recovery_state(
        self,
        bound: BoundWorkspace,
    ) -> dict[str, Any] | None:
        sync_state = super()._completion_finalization_recovery_state(bound)
        if sync_state is None:  # pragma: no cover - SyncBinding is completion-critical.
            raise AssertionError("Docker coding recovery lost its sync state.")
        authority = self._bind_authority(bound)
        return copy_durable_json_object(
            {
                "version": 1,
                "kind": "docker_coding_sync_binding",
                "sync_binding": sync_state,
                "authority": {
                    "session_id": authority.session_id,
                    "source": (
                        None
                        if authority.source is None
                        else authority.source.model_dump(mode="json")
                    ),
                    "workspace_baseline": (
                        None
                        if authority.workspace_baseline is None
                        else authority.workspace_baseline.model_dump(mode="json")
                    ),
                    "git": {
                        "head_revision": authority.git.head_revision,
                        "staged_entries_sha256": authority.git.staged_entries_sha256,
                        "tracked_flags_sha256": authority.git.tracked_flags_sha256,
                        "configuration_sha256": authority.git.configuration_sha256,
                    },
                    "git_transformed_baseline_paths": sorted(
                        authority.git_transformed_baseline_paths
                    ),
                },
            },
            "Docker coding completion finalization recovery state",
        )

    async def _recover_completion_finalization(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None,
        environment_name: str | None,
        recovery_state: dict[str, Any],
    ) -> BoundWorkspace:
        if not isinstance(workspace, LocalWorkspace):
            raise TypeError("Docker coding recovery requires a LocalWorkspace source.")
        if not isinstance(runner, DockerRunner) or not self._docker_target.is_bound_to_runner(
            runner
        ):
            raise ValueError("Docker coding recovery requires its exact DockerRunner.")
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
            raise ValueError("Docker coding recovery path projections changed.")

        state = copy_durable_json_object(
            recovery_state,
            "Docker coding completion finalization recovery state",
        )
        if state.get("version") != 1 or state.get("kind") != "docker_coding_sync_binding":
            raise ValueError("Docker coding recovery state has an unsupported format.")
        raw_sync_state = state.get("sync_binding")
        raw_authority = state.get("authority")
        if type(raw_sync_state) is not dict or type(raw_authority) is not dict:
            raise ValueError("Docker coding recovery state is incomplete.")
        sync_state = copy_durable_json_object(raw_sync_state, "Docker coding sync state")
        authority_state = copy_durable_json_object(
            raw_authority,
            "Docker coding bind authority",
        )
        if set(authority_state) != {
            "session_id",
            "source",
            "workspace_baseline",
            "git",
            "git_transformed_baseline_paths",
        }:
            raise ValueError("Docker coding bind authority has unsupported fields.")
        recovered_session_id = require_durable_clean_nonblank(
            cast("str", authority_state.get("session_id")),
            "Docker coding recovery session id",
        )
        if recovered_session_id != session_id:
            raise RuntimeError("Docker coding recovery authority belongs to another session.")
        raw_source = authority_state.get("source")
        source = (
            None
            if raw_source is None
            else CodingProductSourceCopyAuthority.model_validate(raw_source)
        )
        if source != self._source_copy_authority:
            raise RuntimeError("Docker coding source-copy authority changed during recovery.")
        raw_workspace_baseline = authority_state.get("workspace_baseline")
        workspace_baseline = (
            None
            if raw_workspace_baseline is None
            else WorkspaceRevisionObservation.model_validate(raw_workspace_baseline)
        )
        if (source is None) != (workspace_baseline is None):
            raise ValueError("Docker coding recovery source authority is incomplete.")
        if workspace_baseline is not None:
            if source is None:  # pragma: no cover - paired authority checked above.
                raise AssertionError("Docker coding recovery source authority disappeared.")
            if (
                workspace_baseline.status is not WorkspaceRevisionObservationStatus.SUPPORTED
                or workspace_baseline.path_scope != "complete"
                or workspace_baseline.revision != source.baseline_revision
                or workspace_baseline.identity.workspace_id != self._docker_target.id
            ):
                raise RuntimeError("Docker coding recovery baseline conflicts with its authority.")
        raw_git = authority_state.get("git")
        if type(raw_git) is not dict or set(raw_git) != {
            "head_revision",
            "staged_entries_sha256",
            "tracked_flags_sha256",
            "configuration_sha256",
        }:
            raise ValueError("Docker coding recovery Git authority is malformed.")
        git = _EphemeralGitBaseline(
            head_revision=require_durable_clean_nonblank(
                raw_git.get("head_revision"),
                "Docker coding recovery Git head",
            ),
            staged_entries_sha256=require_durable_clean_nonblank(
                raw_git.get("staged_entries_sha256"),
                "Docker coding recovery staged-entry identity",
            ),
            tracked_flags_sha256=require_durable_clean_nonblank(
                raw_git.get("tracked_flags_sha256"),
                "Docker coding recovery tracked-flag identity",
            ),
            configuration_sha256=require_durable_clean_nonblank(
                raw_git.get("configuration_sha256"),
                "Docker coding recovery Git configuration identity",
            ),
        )
        raw_transformed_paths = authority_state.get("git_transformed_baseline_paths")
        if type(raw_transformed_paths) is not list or len(raw_transformed_paths) > self.max_files:
            raise ValueError("Docker coding transformed-path authority must be bounded.")
        transformed_paths = tuple(
            require_durable_clean_nonblank(path, "Docker coding transformed path")
            for path in raw_transformed_paths
        )
        if tuple(sorted(set(transformed_paths))) != transformed_paths:
            raise ValueError("Docker coding transformed paths must be sorted and unique.")
        baseline_paths = (
            frozenset()
            if workspace_baseline is None
            else frozenset(entry.path for entry in workspace_baseline.paths)
        )
        if not set(transformed_paths).issubset(baseline_paths):
            raise ValueError("Docker coding transformed paths exceed the recovered baseline.")
        recovered_authority = _DockerCodingBindAuthority(
            session_id=session_id,
            source=source,
            workspace_baseline=workspace_baseline,
            git=git,
            git_transformed_baseline_paths=frozenset(transformed_paths),
        )

        bound: BoundWorkspace | None = None
        try:
            bound = await super()._recover_completion_finalization(
                protected_source,
                runner,
                session_id=session_id,
                agent_name=agent_name,
                environment_name=environment_name,
                recovery_state=sync_state,
            )
            state_key = bound.state_key
            if state_key is None:
                raise RuntimeError("Docker coding recovery lost its sync generation.")
            with self._coding_authority_lock:
                existing = self._coding_authorities.get(state_key)
                if existing is not None and existing != recovered_authority:
                    raise RuntimeError("Docker coding recovery authority changed.")
                self._coding_authorities[state_key] = recovered_authority
            return bound
        except BaseException:
            if bound is not None:
                with self._coding_authority_lock:
                    retained = self._coding_authorities.get(bound.state_key or "")
                if retained is None:
                    self.abandon(bound)
            raise

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        state_key = bound.state_key
        if state_key is None:
            raise RuntimeError("Docker coding finalization lost its sync generation.")
        with self._coding_authority_lock:
            cleanup_state = self._immutable_finalize_states.get(state_key)
        if cleanup_state is None:
            if self._immutable_input_attachments:
                self._defer_finalize_release(bound)
            authority = self._bind_authority(bound)
            final_git_evidence = (
                None
                if authority.source is None
                else await _capture_final_git_evidence(bound, authority)
            )
            await _require_no_publishable_ignored_paths(bound)
            snapshot = await super().finalize(bound, outcome=outcome, metadata=metadata)
            final_snapshot = (
                snapshot
                if snapshot is None or final_git_evidence is None
                else replace(
                    snapshot,
                    metadata={**snapshot.metadata, "final_git_evidence": final_git_evidence},
                )
            )
            if not self._immutable_input_attachments:
                self._discard_bind_authority(bound)
                return final_snapshot
            cleanup_state = _ImmutableInputFinalizeState(snapshot=final_snapshot)
            with self._coding_authority_lock:
                existing = self._immutable_finalize_states.setdefault(
                    state_key,
                    cleanup_state,
                )
            if existing is not cleanup_state:  # pragma: no cover - finalize is generation-owned
                raise RuntimeError("Docker immutable input finalization raced its owner.")

        runner = bound.runner
        if not isinstance(runner, DockerRunner):  # pragma: no cover - bind invariant
            raise AssertionError("Docker immutable input cleanup lost its exact runner.")
        store = self._immutable_input_store
        if store is None:  # pragma: no cover - constructor invariant
            raise AssertionError("Docker immutable input cleanup lost its store.")
        if not cleanup_state.runner_closed:
            container_id = runner.container_id
            if container_id is None:  # pragma: no cover - strict runner invariant
                raise AssertionError("Docker immutable input cleanup lost its container id.")
            await store.mark_container_closing(
                self._immutable_input_attachments,
                container_id=container_id,
            )
            await runner.close()
            with self._coding_authority_lock:
                retained_state = self._immutable_finalize_states.get(state_key)
                if retained_state is not cleanup_state:
                    raise RuntimeError("Docker immutable input cleanup lost its retry state.")
                cleanup_state.runner_closed = True
        await _release_immutable_input_attachments(
            store,
            self._immutable_input_attachments,
        )
        if not super().abandon(bound):  # pragma: no cover - SyncBinding contract
            raise RuntimeError("Docker immutable input cleanup retained sync ownership.")
        with self._coding_authority_lock:
            self._immutable_finalize_states.pop(state_key, None)
        self._discard_bind_authority(bound)
        return cleanup_state.snapshot

    def abandon(self, bound: BoundWorkspace) -> bool:
        state_key = bound.state_key
        if state_key is not None:
            with self._coding_authority_lock:
                if state_key in self._immutable_finalize_states:
                    return False
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
        toolchain_profile: DockerCodingToolchainProfile,
        transfer_limits: DockerWorkspaceTransferLimits | None = None,
        runtime: str | None = None,
        seccomp_profile: str | None = None,
        docker_path: str | None = None,
        immutable_inputs: Sequence[LocalImmutableInput] = (),
        immutable_input_store: ImmutableInputStore | None = None,
        immutable_input_runtime_compatibility_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(source_workspace, LocalWorkspace):
            raise TypeError("source_workspace must be LocalWorkspace.")
        if type(toolchain_profile) is not DockerCodingToolchainProfile:
            raise TypeError("toolchain_profile must be an exact DockerCodingToolchainProfile.")
        if transfer_limits is not None and not isinstance(
            transfer_limits, DockerWorkspaceTransferLimits
        ):
            raise TypeError("transfer_limits must be DockerWorkspaceTransferLimits or None.")
        self.source_workspace = source_workspace
        self.toolchain_profile = DockerCodingToolchainProfile.model_validate(
            toolchain_profile.model_dump(mode="python", by_alias=True)
        )
        self.restrictions = self.toolchain_profile.restrictions
        self.image_identity = self.toolchain_profile.image_identity
        self.required_executables = tuple(
            sorted(
                {
                    *self.toolchain_profile.required_executables,
                    *_DOCKER_CODING_RUNTIME_EXECUTABLES,
                }
            )
        )
        self.transfer_limits = transfer_limits or DockerWorkspaceTransferLimits()
        self.runtime = runtime
        self.seccomp_profile = validate_docker_seccomp_profile(seccomp_profile)
        self.docker_path = docker_path
        self._seccomp_sha256 = _read_seccomp_sha256(seccomp_profile)
        self.immutable_inputs = _validate_immutable_inputs(
            immutable_inputs,
            store=immutable_input_store,
            runtime_compatibility_fingerprint=(immutable_input_runtime_compatibility_fingerprint),
        )
        self.immutable_input_store = immutable_input_store
        self.immutable_input_runtime_compatibility_fingerprint = (
            immutable_input_runtime_compatibility_fingerprint
        )
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
            immutable_input_projection_fingerprints=tuple(
                item.projection.fingerprint for item in self.immutable_inputs
            ),
            immutable_input_runtime_compatibility_fingerprint=(
                self.immutable_input_runtime_compatibility_fingerprint
            ),
        )
        self._profile_identity = ExecutionProfileBehaviorIdentity(
            name="cayu.docker_coding_environment",
            behavior_version="12",
            implementation_version=self._configuration_fingerprint,
        )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._profile_identity

    @property
    def immutable_input_capability(self) -> ImmutableInputAdapterCapability:
        """Describe the concrete read-only projection mechanism used by this factory."""

        return docker_immutable_input_capability()

    def construction_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return self._configured_candidate()

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate:
        if not isinstance(request, EnvironmentFactoryRequest):
            raise TypeError("Docker coding admission requires EnvironmentFactoryRequest.")
        self._validate_request(request)
        return self._configured_candidate()

    def create_workspace_binding(
        self,
        request: EnvironmentFactoryRequest,
        *,
        target_workspace: RunnerWorkspace,
        immutable_input_attachments: Sequence[ImmutableInputAttachment] = (),
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
            immutable_input_store=self.immutable_input_store,
            immutable_input_attachments=immutable_input_attachments,
        )

    def allocation_scope(
        self,
        request: EnvironmentFactoryRequest,
    ) -> EnvironmentAllocationScope | None:
        self._validate_request(request)
        if request.operation is not EnvironmentFactoryOperation.CREATE:
            return None
        return EnvironmentAllocationScope(
            provider="docker",
            adapter_generation="cayu.docker_coding.v11",
        )

    async def create_recoverable(
        self,
        request: EnvironmentFactoryRequest,
        allocation: EnvironmentAllocationContext,
    ) -> EnvironmentFactoryResult:
        self._validate_request(request)
        if not isinstance(allocation, EnvironmentAllocationContext):
            raise TypeError("Recoverable Docker coding requires an allocation context.")
        if request.operation is not EnvironmentFactoryOperation.CREATE:
            raise ValueError("Recoverable Docker coding only accepts create operations.")
        expected_name = _docker_coding_container_name(
            request,
            configuration_fingerprint=self._configuration_fingerprint,
            allocation_id=allocation.intent.allocation_id,
        )
        expected_metadata = {
            "container_name": expected_name,
            "configuration_fingerprint": self._configuration_fingerprint,
        }
        if allocation.state is EnvironmentAllocationState.UNPREPARED:
            await allocation.prepare(expected_metadata)
        if allocation.intent.provider_metadata != expected_metadata:
            raise RuntimeError("Docker coding allocation intent changed after preparation.")
        if allocation.state is EnvironmentAllocationState.REAPING:
            await self._reap_interrupted_allocation(allocation, container_name=expected_name)
            raise RuntimeError("A reaping Docker coding allocation cannot be replaced.")
        if allocation.state is EnvironmentAllocationState.REAPED:
            raise RuntimeError("A reaped Docker coding allocation cannot be replaced.")
        return await self._create(request, allocation=allocation)

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        return await self._create(request, allocation=None)

    async def _create(
        self,
        request: EnvironmentFactoryRequest,
        *,
        allocation: EnvironmentAllocationContext | None,
    ) -> EnvironmentFactoryResult:
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

        attachments: tuple[ImmutableInputAttachment, ...] = ()
        runner: DockerRunner | None = None
        try:
            if request.operation is EnvironmentFactoryOperation.RECONNECT:
                await self._reconcile_interrupted_immutable_cleanup(request)
            attachments = await self._attach_immutable_inputs(request)
            immutable_mounts = tuple(attachment.docker_mount() for attachment in attachments)
            if request.operation is EnvironmentFactoryOperation.CREATE:
                if allocation is not None:
                    if allocation.state is EnvironmentAllocationState.PREPARED:
                        await allocation.mark_dispatched()
                    if allocation.state is EnvironmentAllocationState.ACKNOWLEDGED:
                        acknowledged = allocation.acknowledged_reconnect_metadata
                        if acknowledged is None:
                            raise RuntimeError(
                                "Docker coding allocation acknowledgement disappeared."
                            )
                        container_id = cast("str", acknowledged.get("container_id"))
                        runner = await self._reconnect_runner(
                            container_id,
                            immutable_mounts=immutable_mounts,
                        )
                    elif allocation.state is EnvironmentAllocationState.DISPATCHED:
                        container_name = cast(
                            "str",
                            allocation.intent.provider_metadata.get("container_name"),
                        )
                        runner = await self._create_or_recover_runner(
                            container_name,
                            immutable_mounts=immutable_mounts,
                        )
                    else:
                        raise RuntimeError("Docker coding allocation is not dispatchable.")
                else:
                    runner = await self._create_or_recover_runner(
                        _docker_coding_container_name(
                            request,
                            configuration_fingerprint=self._configuration_fingerprint,
                        ),
                        immutable_mounts=immutable_mounts,
                    )
            else:
                container_id = cast("str", request.reconnect_metadata.get("container_id"))
                runner = await self._reconnect_runner(
                    container_id,
                    immutable_mounts=immutable_mounts,
                )
            exact_container_id = runner.container_id
            if exact_container_id is None:
                raise RuntimeError("Docker coding runner lost its exact container identity.")
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
                or (
                    bool(self.immutable_inputs)
                    and (
                        final_evidence.claim_for("read_only_host_inputs") is None
                        or final_evidence.claim_for("read_only_host_inputs").state
                        != "live_verified"
                    )
                )
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
            )
            workspace = RunnerWorkspace(
                runner,
                cwd=None,
                workspace_id=f"docker:{exact_container_id}:workspace",
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
                immutable_input_attachments=attachments,
            )
            evidence_metadata = final_decision.evidence.to_metadata()
            metadata = {
                "kind": "docker_coding",
                "container_id": exact_container_id,
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
                "immutable_inputs": [
                    {
                        "attachment_id": attachment.attachment_id,
                        "projection_fingerprint": attachment.projection.fingerprint,
                        "content_root": attachment.projection.content_root,
                        "target_path": attachment.projection.target_path,
                        "logical_bytes": attachment.projection.logical_bytes,
                        "reused": attachment.reused,
                    }
                    for attachment in attachments
                ],
                "immutable_input_capability": self.immutable_input_capability.model_dump(
                    mode="json"
                ),
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
            reconnect_metadata = _docker_coding_reconnect_metadata(
                container_id=exact_container_id,
                configuration_fingerprint=self._configuration_fingerprint,
                image_fingerprint=self.image_identity.fingerprint,
                toolchain_profile_fingerprint=self.toolchain_profile.fingerprint,
            )
            if allocation is not None:
                if allocation.state is EnvironmentAllocationState.DISPATCHED:
                    await allocation.acknowledge(reconnect_metadata)
                elif allocation.acknowledged_reconnect_metadata != reconnect_metadata:
                    raise RuntimeError(
                        "Recovered Docker coding allocation changed its durable identity."
                    )

            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                if action is EnvironmentFactoryReleaseAction.DISCARD:
                    await runner.close()
                    await self._release_immutable_inputs(attachments)
                    if (
                        allocation is not None
                        and allocation.state is EnvironmentAllocationState.REAPING
                    ):
                        await allocation.mark_reaped()

            return EnvironmentFactoryResult(
                environment=environment,
                metadata=metadata,
                reconnect_metadata=reconnect_metadata,
                release=release,
            )
        except BaseException as original:
            if request.operation is EnvironmentFactoryOperation.RECONNECT or allocation is not None:
                raise
            cleanup_error: BaseException | None = None
            if runner is not None:
                try:
                    await runner.close()
                except BaseException as error:
                    cleanup_error = error
            if cleanup_error is None:
                try:
                    await self._release_immutable_inputs(attachments)
                except BaseException as error:
                    cleanup_error = error
            if cleanup_error is not None:
                raise BaseExceptionGroup(
                    "Docker coding creation and resource cleanup both failed.",
                    [original, cleanup_error],
                ) from None
            raise

    async def _create_or_recover_runner(
        self,
        container_name: str,
        *,
        immutable_mounts: tuple[DockerImmutableInputMount, ...],
    ) -> DockerRunner:
        existing_id = await DockerRunner.resolve_container_id(
            container_name,
            docker_path=self.docker_path,
        )
        if existing_id is not None:
            return await self._reconnect_runner(
                existing_id,
                immutable_mounts=immutable_mounts,
            )
        try:
            return await DockerRunner.create(
                container_name,
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
                immutable_input_mounts=immutable_mounts,
            )
        except Exception:
            recovered_id = await DockerRunner.resolve_container_id(
                container_name,
                docker_path=self.docker_path,
            )
            if recovered_id is None:
                raise
            return await self._reconnect_runner(
                recovered_id,
                immutable_mounts=immutable_mounts,
            )

    async def _reconnect_runner(
        self,
        container_id: str,
        *,
        immutable_mounts: tuple[DockerImmutableInputMount, ...],
    ) -> DockerRunner:
        return await DockerRunner.reconnect_strict(
            f"cayu-coding-reconnect-{container_id[:12]}",
            container_id=container_id,
            image_identity=self.image_identity,
            workload_restrictions=self.restrictions,
            default_cwd="/workspace",
            runtime=self.runtime,
            seccomp_profile=self.seccomp_profile,
            docker_path=self.docker_path,
            required_executables=self.required_executables,
            toolchain_profile_fingerprint=self.toolchain_profile.fingerprint,
            immutable_input_mounts=immutable_mounts,
        )

    async def _reconcile_interrupted_immutable_cleanup(
        self,
        request: EnvironmentFactoryRequest,
    ) -> None:
        store = self.immutable_input_store
        if store is None or not self.immutable_inputs:
            return
        attachment_ids = tuple(
            _immutable_input_attachment_id(request, source) for source in self.immutable_inputs
        )
        closing_container_id = await store.interrupted_cleanup_container_id(attachment_ids)
        if closing_container_id is None:
            return
        expected_container_id = cast("str", request.reconnect_metadata.get("container_id"))
        if closing_container_id != expected_container_id:
            raise RuntimeError("Immutable input cleanup names another Docker container.")
        container_exists = await DockerRunner.container_exists(
            closing_container_id,
            docker_path=self.docker_path,
        )
        reactivated = await store.reconcile_interrupted_container_cleanup(
            attachment_ids,
            container_id=closing_container_id,
            container_exists=container_exists,
        )
        if not reactivated:
            raise RuntimeError(
                "Docker immutable input cleanup completed after the container disappeared."
            )

    async def _reap_interrupted_allocation(
        self,
        allocation: EnvironmentAllocationContext,
        *,
        container_name: str,
    ) -> None:
        container_id = await DockerRunner.resolve_container_id(
            container_name,
            docker_path=self.docker_path,
        )
        if container_id is not None:
            runner = DockerRunner(
                container_name,
                close_action="remove",
                docker_path=self.docker_path,
                _container_id=container_id,
            )
            await runner.close()
        store = self.immutable_input_store
        if store is not None:
            for source in self.immutable_inputs:
                await store.release(
                    _immutable_input_attachment_id_from_owner(
                        session_id=allocation.intent.session_id,
                        environment_name=allocation.intent.environment_name,
                        source=source,
                    )
                )
        await allocation.mark_reaped()

    async def _attach_immutable_inputs(
        self,
        request: EnvironmentFactoryRequest,
    ) -> tuple[ImmutableInputAttachment, ...]:
        if not self.immutable_inputs:
            return ()
        store = self.immutable_input_store
        if store is None:  # pragma: no cover - constructor invariant
            raise AssertionError("Docker immutable input store was not retained.")
        attached: list[ImmutableInputAttachment] = []
        try:
            for source in self.immutable_inputs:
                attached.append(
                    await store.attach(
                        source,
                        attachment_id=_immutable_input_attachment_id(request, source),
                        owner_id=request.session_id,
                    )
                )
        except BaseException as original:
            try:
                await self._release_immutable_inputs(tuple(attached))
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "Docker immutable input attachment and rollback both failed.",
                    [original, cleanup_error],
                ) from None
            raise
        return tuple(attached)

    async def _release_immutable_inputs(
        self,
        attachments: tuple[ImmutableInputAttachment, ...],
    ) -> None:
        store = self.immutable_input_store
        if store is None:
            if attachments:  # pragma: no cover - constructor invariant
                raise AssertionError("Docker immutable input store was not retained.")
            return
        await _release_immutable_input_attachments(store, attachments)

    def _validate_request(self, request: EnvironmentFactoryRequest) -> None:
        if not isinstance(request, EnvironmentFactoryRequest):
            raise TypeError("Docker coding create requires EnvironmentFactoryRequest.")
        if request.operation is EnvironmentFactoryOperation.CREATE:
            if request.reconnect_metadata:
                raise ValueError("Docker coding creation forbids reconnect metadata.")
            return
        container_id = request.reconnect_metadata.get("container_id")
        if (
            type(container_id) is not str
            or len(container_id) != 64
            or any(character not in "0123456789abcdef" for character in container_id)
        ):
            raise ValueError("Docker coding reconnect requires one full lowercase container ID.")
        expected = _docker_coding_reconnect_metadata(
            container_id=container_id,
            configuration_fingerprint=self._configuration_fingerprint,
            image_fingerprint=self.image_identity.fingerprint,
            toolchain_profile_fingerprint=self.toolchain_profile.fingerprint,
        )
        if request.reconnect_metadata != expected:
            raise ValueError("Docker coding reconnect metadata does not match this factory.")

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
        read_only_input_claim = (
            ExecutionCapabilityClaim.declared("read_only_host_inputs")
            if self.immutable_inputs
            else ExecutionCapabilityClaim.unsupported(
                "read_only_host_inputs",
                reason_code="docker_host_inputs_not_mounted",
                remediation_code="configure_immutable_input_projection",
            )
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
                ExecutionCapabilityClaim.declared("reconnect"),
                read_only_input_claim,
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
    immutable_input_projection_fingerprints: tuple[str, ...],
    immutable_input_runtime_compatibility_fingerprint: str | None,
) -> str:
    material = {
        "schema": "cayu.docker_coding_environment.v2",
        "image_identity": image_identity.model_dump(mode="json"),
        "restrictions": restrictions.model_dump(mode="json"),
        "required_executables": list(required_executables),
        "transfer_limits": transfer_limits.model_dump(mode="json"),
        "runtime": runtime,
        "seccomp_sha256": seccomp_sha256,
        "toolchain_profile_fingerprint": toolchain_profile_fingerprint,
        "network": "none",
        "default_cwd": "/workspace",
        "host_mounts": (
            "verified_read_only_immutable_inputs"
            if immutable_input_projection_fingerprints
            else False
        ),
        "immutable_input_projection_fingerprints": list(immutable_input_projection_fingerprints),
        "immutable_input_runtime_compatibility_fingerprint": (
            immutable_input_runtime_compatibility_fingerprint
        ),
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


def _docker_coding_reconnect_metadata(
    *,
    container_id: str,
    configuration_fingerprint: str,
    image_fingerprint: str,
    toolchain_profile_fingerprint: str,
) -> dict[str, Any]:
    identity = {
        "version": 1,
        "kind": "docker_coding",
        "container_id": container_id,
        "configuration_fingerprint": configuration_fingerprint,
        "image_fingerprint": image_fingerprint,
        "toolchain_profile_fingerprint": toolchain_profile_fingerprint,
    }
    return {
        **identity,
        "allocation_fingerprint": sha256(
            canonical_durable_json_bytes(identity, "docker_coding_reconnect")
        ).hexdigest(),
    }


async def _run_toolchain_admission_probes(
    runner: DockerRunner,
    profile: DockerCodingToolchainProfile,
) -> None:
    """Run bounded probes only after exact final-container admission."""

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
