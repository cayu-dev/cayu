"""Workspace binding contracts for bridging storage and compute."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import io
import json
import ntpath
import shutil
import tarfile
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal, NoReturn, TypeVar, cast
from urllib.parse import urlsplit
from uuid import uuid4

from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_group_children,
)
from cayu._task_wait import await_shielded_task_outcome, unexpected_child_cancellation_error
from cayu._validation import (
    copy_durable_json_object,
    copy_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.environments._sync_staging import (
    DEFAULT_SYNC_BINDING_STAGING_CAPACITY,
    SyncBindingStagingCapacity,
    SyncBindingStagingCapacityError,
    SyncBindingStagingSnapshot,
    _CapacityLease,
    _SealedTarArchive,
)
from cayu.runners import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, ExecCommand, LocalRunner, Runner
from cayu.workspaces import (
    BoundedTarReader,
    BoundedTarStreamReader,
    LocalWorkspace,
    RunnerWorkspace,
    TarStreamWriter,
    TarWriter,
    Workspace,
    WorkspaceGitMode,
    WorkspaceGitModeMutator,
    WorkspaceIdentity,
    WorkspaceMutationResult,
    WorkspacePathRevision,
    WorkspaceReadResult,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
    WorkspaceWriterIsolationEvidence,
)
from cayu.workspaces._tar import tar_archive_size_bound
from cayu.workspaces.revisions import (
    observe_deterministic_workspace,
    unsupported_workspace_revision,
)


@dataclass(frozen=True)
class SyncBindingContext:
    """Context passed to a SyncBinding target workspace factory."""

    source_workspace: Workspace
    runner: Runner | None = None
    session_id: str | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_workspace, Workspace):
            raise TypeError("SyncBindingContext source_workspace must be a Workspace.")
        if self.runner is not None and not isinstance(self.runner, Runner):
            raise TypeError("SyncBindingContext runner must be a Runner or None.")
        if self.session_id is not None:
            object.__setattr__(
                self,
                "session_id",
                require_clean_nonblank(self.session_id, "session_id"),
            )
        if self.agent_name is not None:
            object.__setattr__(
                self,
                "agent_name",
                require_clean_nonblank(self.agent_name, "agent_name"),
            )
        if self.environment_name is not None:
            object.__setattr__(
                self,
                "environment_name",
                require_clean_nonblank(self.environment_name, "environment_name"),
            )
        if type(self.metadata) is not dict:
            raise TypeError("SyncBindingContext metadata must be a dict.")
        object.__setattr__(
            self,
            "metadata",
            copy_durable_json_object(self.metadata, "metadata"),
        )


@dataclass(frozen=True)
class _SyncSourceRevision:
    path: str
    revision: str
    git_mode: WorkspaceGitMode | None = None


@dataclass(frozen=True)
class _SyncBindingState:
    source_paths: tuple[str, ...]
    target_baseline_paths: tuple[str, ...]
    source_revisions: tuple[_SyncSourceRevision, ...]
    source_resource_key: tuple[object, ...]
    target_id: str
    target_resource_key: tuple[object, ...]
    phase: Literal["active", "finalizing"] = "active"
    defer_finalize_release: bool = False
    recovered: bool = False


@dataclass(frozen=True, slots=True)
class _SyncSourceObservation:
    revisions: tuple[_SyncSourceRevision, ...]
    content_sha256: str
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class _SyncArchivePolicy:
    format_version: str
    paths: tuple[str, ...]
    pattern: str
    max_files: int
    max_file_bytes: int | None
    max_total_bytes: int | None
    max_archive_bytes: int | None
    clean_target: SyncTargetCleanPolicy
    sync_back: SyncBackPolicy
    delete_missing: bool
    source_conflict_policy: SyncSourceConflictPolicy
    preserve_git_modes: bool
    source_tar_policy_identity: tuple[object, ...]
    target_tar_policy_identity: tuple[object, ...]


class _BoundedSyncArchiveWriter:
    """Limit an adapter-owned tar stream before it can grow the private spool."""

    def __init__(self, destination: BinaryIO, *, max_bytes: int) -> None:
        self._destination = destination
        self._max_bytes = max_bytes
        self.bytes_written = 0

    def write(self, content: bytes) -> int:
        if type(content) is not bytes:
            raise TypeError("SyncBinding tar stream writes must be bytes.")
        next_size = self.bytes_written + len(content)
        if next_size > self._max_bytes:
            raise RuntimeError(
                "SyncBinding tar stream exceeds its admitted archive reservation: "
                f"required>{self._max_bytes}."
            )
        written = self._destination.write(content)
        if written is None:
            written = len(content)
        if type(written) is not int or written != len(content):
            raise OSError("SyncBinding private archive spool accepted only part of a write.")
        self.bytes_written = next_size
        return written

    def flush(self) -> None:
        self._destination.flush()


DEFAULT_SYNC_MAX_TOTAL_BYTES = 64 * 1024 * 1024
DEFAULT_SYNC_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_SYNC_GIT_MODE_TAR_OWNER = "cayu.git-mode.v1"


SYNC_FINAL_METADATA_KEYS = frozenset(
    {
        "target_workspace_id",
        "outcome",
        "copied_files",
        "copied_bytes",
        "deleted_files",
    }
)


SyncTargetWorkspaceProvisioner = Callable[[], None | Awaitable[None]]


@dataclass(frozen=True)
class SyncTargetWorkspacePlan:
    """A resolved target identity plus setup that runs after ownership admission.

    ``workspace`` must already expose its stable, quiescent resource identity.
    ``provision`` may attach, clean, or otherwise mutate that target and is
    invoked only after the source and target resource keys are owned by the
    binding generation. Allocation and failed-bind cleanup remain owned by the
    surrounding environment factory lifecycle. Constructing the in-process
    workspace wrapper needed to identify an already-owned target is allowed.
    """

    workspace: Workspace
    provision: SyncTargetWorkspaceProvisioner | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace):
            raise TypeError("SyncTargetWorkspacePlan workspace must be a Workspace.")
        if self.provision is not None and not callable(self.provision):
            raise TypeError("SyncTargetWorkspacePlan provision must be callable or None.")


SyncTargetWorkspacePlanFactory = Callable[
    [SyncBindingContext],
    SyncTargetWorkspacePlan | Awaitable[SyncTargetWorkspacePlan],
]
SyncTargetCleanPolicy = Literal["always", "never"]
SyncBackPolicy = Literal["always", "on_success", "never"]
SyncSourceConflictPolicy = Literal["overwrite", "require_revision"]


class SyncBindingSourceConflictError(RuntimeError):
    """Copy-back refused because the durable source changed concurrently."""

    def __init__(self, path: str, *, applied_paths: tuple[str, ...] = ()) -> None:
        self.path = require_clean_nonblank(path, "path")
        self.applied_paths = tuple(applied_paths)
        detail = (
            "before any copy-back mutation"
            if not applied_paths
            else f"after {len(applied_paths)} copy-back mutations"
        )
        super().__init__(f"SyncBinding source revision conflict for {path!r} {detail}.")


GIT_REPOSITORY_METADATA_KEY = "git_repository"
_GIT_OBSERVATION_ENV_REMOVE = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM",
    "GIT_DIR",
    "GIT_EXTERNAL_DIFF",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_REDIRECT_STDERR",
    "GIT_TRACE",
    "GIT_TRACE_CURL",
    "GIT_TRACE_CURL_NO_DATA",
    "GIT_TRACE_FSMONITOR",
    "GIT_TRACE_PACKET",
    "GIT_TRACE_PACK_ACCESS",
    "GIT_TRACE_PERFORMANCE",
    "GIT_TRACE_REFS",
    "GIT_TRACE_SETUP",
    "GIT_TRACE_SHALLOW",
    "GIT_TRACE2",
    "GIT_TRACE2_BRIEF",
    "GIT_TRACE2_CONFIG_PARAMS",
    "GIT_TRACE2_DST_DEBUG",
    "GIT_TRACE2_EVENT",
    "GIT_TRACE2_PERF",
    "GIT_WORK_TREE",
)

SYNC_DISTINCT_WORKSPACES_ERROR = "SyncBinding source and target workspaces must be different."

_MutationResultT = TypeVar("_MutationResultT")

_SYNC_RESOURCE_OWNERS_LOCK = threading.Lock()
_SYNC_RESOURCE_OWNERS: dict[tuple[object, ...], str] = {}
_SYNC_PROVISIONAL_SOURCES: dict[str, tuple[object, ...]] = {}
_BASE_EXCEPTION_SUPPRESS_CONTEXT_DESCRIPTOR = BaseException.__dict__["__suppress_context__"]


@dataclass(slots=True, repr=False)
class _BindingFailureOwnership:
    """Exact process-local reservations retained for outer lifecycle cleanup."""

    source_resource_key: tuple[object, ...]
    target_resource_key: tuple[object, ...]
    generation: str
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        _release_sync_resources(
            self.source_resource_key,
            self.target_resource_key,
            generation=self.generation,
        )
        self.released = True


@dataclass(slots=True)
class _EnvironmentLifecycleBindAttempt:
    """Exact failed-generation release authority for one private bind."""

    release_failed_reservations: Callable[[], None] | None = field(
        default=None,
        repr=False,
    )

    def retain(
        self,
        release_failed_reservations: Callable[[], None] | None,
    ) -> None:
        self.release_failed_reservations = release_failed_reservations


def _reserve_sync_factory_source(
    source: Workspace,
    *,
    resource_key: tuple[object, ...],
    generation: str,
) -> None:
    """Reserve a factory bind's source before the factory can provision a target."""

    with _SYNC_RESOURCE_OWNERS_LOCK:
        if resource_key in _SYNC_RESOURCE_OWNERS:
            raise ValueError(
                f"SyncBinding source workspace {source.id!r} is already bound by an active session."
            )
        _SYNC_RESOURCE_OWNERS[resource_key] = generation
        _SYNC_PROVISIONAL_SOURCES[generation] = resource_key


def _promote_sync_factory_resources(
    target: Workspace,
    *,
    source_resource_key: tuple[object, ...],
    target_resource_key: tuple[object, ...],
    generation: str,
) -> None:
    """Atomically promote one provisional source claim to the complete resource pair."""

    with _SYNC_RESOURCE_OWNERS_LOCK:
        if _SYNC_RESOURCE_OWNERS.get(source_resource_key) != generation:
            raise RuntimeError("SyncBinding lost provisional source ownership.")
        target_owner = _SYNC_RESOURCE_OWNERS.get(target_resource_key)
        if target_owner is not None:
            if _SYNC_PROVISIONAL_SOURCES.get(target_owner) == target_resource_key:
                # The target is another factory bind's provisionally owned source. Yield this
                # generation's source atomically with rejecting its promotion. For an opposite
                # A->B / B->A race, the other generation can therefore promote instead of both
                # contenders observing occupied targets and then releasing too late.
                if _SYNC_RESOURCE_OWNERS.get(source_resource_key) == generation:
                    del _SYNC_RESOURCE_OWNERS[source_resource_key]
                _SYNC_PROVISIONAL_SOURCES.pop(generation, None)
            raise ValueError(
                f"SyncBinding target workspace {target.id!r} is already bound by an active session."
            )
        _SYNC_RESOURCE_OWNERS[target_resource_key] = generation
        _SYNC_PROVISIONAL_SOURCES.pop(generation, None)


def _reserve_sync_resources(
    source: Workspace,
    target: Workspace,
    *,
    source_resource_key: tuple[object, ...],
    target_resource_key: tuple[object, ...],
    generation: str,
) -> None:
    """Atomically reserve both process-local resources for an exact generation."""

    resources = (
        ("source", source, source_resource_key),
        ("target", target, target_resource_key),
    )
    with _SYNC_RESOURCE_OWNERS_LOCK:
        for role, workspace, resource_key in resources:
            if resource_key in _SYNC_RESOURCE_OWNERS:
                raise ValueError(
                    f"SyncBinding {role} workspace {workspace.id!r} is already bound "
                    "by an active session."
                )
        for _, _, resource_key in resources:
            _SYNC_RESOURCE_OWNERS[resource_key] = generation


def _recover_sync_resources(
    source: Workspace,
    target: Workspace,
    *,
    source_resource_key: tuple[object, ...],
    target_resource_key: tuple[object, ...],
    generation: str,
) -> None:
    """Restore one process-local reservation from durable exact-generation state."""

    resources = (
        ("source", source, source_resource_key),
        ("target", target, target_resource_key),
    )
    with _SYNC_RESOURCE_OWNERS_LOCK:
        for role, workspace, resource_key in resources:
            owner = _SYNC_RESOURCE_OWNERS.get(resource_key)
            if owner is not None and owner != generation:
                raise ValueError(
                    f"SyncBinding {role} workspace {workspace.id!r} is already bound "
                    "by a different active session."
                )
        for _, _, resource_key in resources:
            _SYNC_RESOURCE_OWNERS[resource_key] = generation


def _release_sync_resources(
    source_resource_key: tuple[object, ...],
    target_resource_key: tuple[object, ...],
    *,
    generation: str,
) -> None:
    """Release both resources still owned by the exact generation."""

    with _SYNC_RESOURCE_OWNERS_LOCK:
        for resource_key in (source_resource_key, target_resource_key):
            if _SYNC_RESOURCE_OWNERS.get(resource_key) == generation:
                del _SYNC_RESOURCE_OWNERS[resource_key]
        _SYNC_PROVISIONAL_SOURCES.pop(generation, None)


def _release_sync_resource(
    resource_key: tuple[object, ...],
    *,
    generation: str,
) -> None:
    """Release one provisional resource still owned by the exact generation."""

    with _SYNC_RESOURCE_OWNERS_LOCK:
        if _SYNC_RESOURCE_OWNERS.get(resource_key) == generation:
            del _SYNC_RESOURCE_OWNERS[resource_key]
        if _SYNC_PROVISIONAL_SOURCES.get(generation) == resource_key:
            del _SYNC_PROVISIONAL_SOURCES[generation]


def _sync_resources_are_owned_by(
    source_resource_key: tuple[object, ...],
    target_resource_key: tuple[object, ...],
    *,
    generation: str,
) -> bool:
    with _SYNC_RESOURCE_OWNERS_LOCK:
        return all(
            _SYNC_RESOURCE_OWNERS.get(resource_key) == generation
            for resource_key in (source_resource_key, target_resource_key)
        )


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Serializable identity for a concrete workspace version."""

    snapshot_id: str
    workspace_id: str | None = None
    version: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_id",
            require_clean_nonblank(self.snapshot_id, "snapshot_id"),
        )
        if self.workspace_id is not None:
            object.__setattr__(
                self,
                "workspace_id",
                require_clean_nonblank(self.workspace_id, "workspace_id"),
            )
        if self.version is not None:
            object.__setattr__(
                self,
                "version",
                require_clean_nonblank(self.version, "version"),
            )
        if self.source is not None:
            object.__setattr__(
                self,
                "source",
                require_clean_nonblank(self.source, "source"),
            )
        if type(self.metadata) is not dict:
            raise TypeError("WorkspaceSnapshot metadata must be a dict.")
        object.__setattr__(
            self,
            "metadata",
            copy_durable_json_object(self.metadata, "metadata"),
        )


@dataclass(frozen=True)
class BoundWorkspace:
    """Result of binding a workspace to a runner for one session.

    ``path`` names where the workspace is visible from the runner's point of
    view, when the binding has such a path. ``metadata`` carries binding-owned
    state such as mount ids, sandbox refs, branch names, or sync tokens.
    ``snapshot`` identifies the concrete workspace version bound for the session
    when the binding backend can provide one. ``state_key`` is runtime-private
    and is not included in binding event payloads.
    """

    workspace: Workspace | None = None
    source_workspace: Workspace | None = None
    runner: Runner | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot: WorkspaceSnapshot | None = None
    state_key: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.workspace is not None and not isinstance(self.workspace, Workspace):
            raise TypeError("BoundWorkspace workspace must be a Workspace or None.")
        if self.source_workspace is not None and not isinstance(
            self.source_workspace,
            Workspace,
        ):
            raise TypeError("BoundWorkspace source_workspace must be a Workspace or None.")
        if self.runner is not None and not isinstance(self.runner, Runner):
            raise TypeError("BoundWorkspace runner must be a Runner or None.")
        if self.path is not None and type(self.path) is not str:
            raise TypeError("BoundWorkspace path must be a string or None.")
        if self.path is not None and not self.path.strip():
            raise ValueError("BoundWorkspace path cannot be blank.")
        if type(self.metadata) is not dict:
            raise TypeError("BoundWorkspace metadata must be a dict.")
        object.__setattr__(
            self,
            "metadata",
            copy_durable_json_object(self.metadata, "metadata"),
        )
        if self.snapshot is not None and type(self.snapshot) is not WorkspaceSnapshot:
            raise TypeError("BoundWorkspace snapshot must be a WorkspaceSnapshot or None.")
        if self.state_key is not None:
            object.__setattr__(
                self,
                "state_key",
                require_clean_nonblank(self.state_key, "state_key"),
            )
        object.__setattr__(self, "snapshot", copy_workspace_snapshot(self.snapshot))


class WorkspaceBinding(ABC):
    """Bridge between durable workspace storage and runner execution.

    ``bind`` makes a workspace available to a runner for one session. ``finalize``
    is called when the session lifecycle ends, so implementations can sync,
    persist, discard, or unmount according to the session outcome.
    """

    @abstractmethod
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
        """Make the workspace available to the runner."""

    @abstractmethod
    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        """Clean up or persist the binding after the session ends."""

    async def observe_revision(self, bound: BoundWorkspace) -> WorkspaceRevisionObservation:
        """Observe bounded workspace state when this binding supports it.

        The compatibility default is an explicit unsupported result.  It must
        not reinterpret per-file optimistic revision tokens or binding
        snapshots as restorable workspace revisions.
        """

        if type(bound) is not BoundWorkspace:
            raise TypeError("Workspace revision observation requires a BoundWorkspace.")
        workspace = bound.workspace or bound.source_workspace
        if workspace is None:
            workspace_id = "workspace-unavailable"
        else:
            workspace_id = require_clean_nonblank(workspace.id, "workspace.id")
        return unsupported_workspace_revision(
            workspace_id=workspace_id,
            observer=type(self).__name__,
        )

    def observe_writer_isolation(
        self,
        bound: BoundWorkspace,
    ) -> WorkspaceWriterIsolationEvidence:
        """Report inspectable writer-isolation evidence for one mutation window.

        The compatibility default is deliberately unknown. Serial tool
        dispatch, a process-local lock, or a stable workspace identity does not
        prove that users, sibling processes, or background services cannot
        mutate the same resource.
        """

        if type(bound) is not BoundWorkspace:
            raise TypeError("Workspace writer-isolation observation requires a BoundWorkspace.")
        return WorkspaceWriterIsolationEvidence()

    async def _bind_for_environment_lifecycle(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        _attempt: _EnvironmentLifecycleBindAttempt | None = None,
    ) -> BoundWorkspace:
        """Bind and return any exact cleanup authority outside diagnostics.

        Stateless and self-cleaning bindings use the public entrance unchanged.
        A binding that must retain process-local exclusion until an outer
        factory release settles can override this private lifecycle seam.
        """
        attempt = _attempt or _EnvironmentLifecycleBindAttempt()
        try:
            bound = await self.bind(
                workspace,
                runner,
                session_id=session_id,
                agent_name=agent_name,
                environment_name=environment_name,
                metadata=metadata,
            )
        except BaseException:
            raise
        attempt.retain(
            lambda: self._release_failed_outer_bind(bound),
        )
        return bound

    def _release_failed_outer_bind(self, bound: BoundWorkspace) -> None:
        """Retire a successful inner bind after a surrounding bind layer fails."""

        if self.abandon(bound) is False:
            raise RuntimeError("Workspace binding could not retire a failed outer bind.")

    def abandon(self, bound: BoundWorkspace) -> bool:
        """Release process-local retry state when finalization will not run again.

        Most bindings do not retain retry state, so the compatibility default is
        a validated no-op. Stateful bindings should override this method and
        release only state owned by ``bound``. Return ``False`` only when the
        binding must retain its lifecycle owner for a later cleanup attempt.
        """

        if type(bound) is not BoundWorkspace:
            raise TypeError("WorkspaceBinding abandon requires a BoundWorkspace.")
        return True

    def _defer_finalize_release(self, bound: BoundWorkspace) -> None:
        """Keep retry ownership after successful finalization until abandonment.

        Composite lifecycle owners use this before finalization when another
        resource must become quiescent before the binding can be released.
        Stateless bindings inherit the validated no-op.
        """

        if type(bound) is not BoundWorkspace:
            raise TypeError("WorkspaceBinding deferred release requires a BoundWorkspace.")

    def _requires_mutation_quiescence(self, bound: BoundWorkspace) -> bool:
        """Whether releasing ``bound`` requires positive external mutation quiescence."""

        if type(bound) is not BoundWorkspace:
            raise TypeError("WorkspaceBinding quiescence query requires a BoundWorkspace.")
        return False

    def _completion_requires_successful_finalization(self, bound: BoundWorkspace) -> bool:
        """Whether completed work depends on this binding persisting its output."""

        if type(bound) is not BoundWorkspace:
            raise TypeError("WorkspaceBinding completion query requires a BoundWorkspace.")
        return False

    def _completion_finalization_recovery_state(
        self,
        bound: BoundWorkspace,
    ) -> dict[str, Any] | None:
        """Return bounded private state needed to retry completion finalization."""

        if type(bound) is not BoundWorkspace:
            raise TypeError("WorkspaceBinding recovery query requires a BoundWorkspace.")
        return None

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
        """Restore a bound owner without repeating its mutating bind operation."""

        del workspace, runner, session_id, agent_name, environment_name, recovery_state
        raise RuntimeError(
            f"{type(self).__name__} does not support durable completion finalization recovery."
        )


class NativeBinding(WorkspaceBinding):
    """Binding for backends where workspace and runner already share state.

    The configured workspace and runner are passed through unchanged. Runner-
    specific bindings can later expose richer mount/copy behavior without
    changing the environment contract.
    """

    def __init__(self, *, default_path: str | None = None) -> None:
        if default_path is not None:
            if type(default_path) is not str:
                raise TypeError("NativeBinding default_path must be a string or None.")
            if not default_path.strip():
                raise ValueError("NativeBinding default_path cannot be blank.")
        self._default_path = default_path

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
        copied_metadata = _validate_bind_request(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )

        return BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=runner,
            path=self._default_path,
            metadata=copied_metadata,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        _validate_finalize_request(bound, outcome=outcome, metadata=metadata)
        return None


def _copy_workspace_revision_observation_limits(
    limits: WorkspaceRevisionObservationLimits | None,
) -> WorkspaceRevisionObservationLimits:
    if limits is None:
        return WorkspaceRevisionObservationLimits()
    if type(limits) is not WorkspaceRevisionObservationLimits:
        raise TypeError(
            "Workspace observation limits must be a WorkspaceRevisionObservationLimits instance."
        )
    values = {
        "max_paths": limits.max_paths,
        "max_path_bytes": limits.max_path_bytes,
        "max_file_bytes": limits.max_file_bytes,
        "max_total_file_bytes": limits.max_total_file_bytes,
        "max_manifest_bytes": limits.max_manifest_bytes,
    }
    if any(type(value) is not int for value in values.values()):
        raise TypeError("Workspace observation limit fields must be integers.")
    return WorkspaceRevisionObservationLimits(**values)


class DeterministicWorkspaceBinding(NativeBinding):
    """Backend-neutral revision observer for conformance and simple workspaces."""

    def __init__(
        self,
        *,
        default_path: str | None = None,
        observation_limits: WorkspaceRevisionObservationLimits | None = None,
    ) -> None:
        super().__init__(default_path=default_path)
        self.observation_limits = _copy_workspace_revision_observation_limits(observation_limits)

    async def observe_revision(self, bound: BoundWorkspace) -> WorkspaceRevisionObservation:
        if type(bound) is not BoundWorkspace:
            raise TypeError("Workspace revision observation requires a BoundWorkspace.")
        workspace = bound.workspace or bound.source_workspace
        if workspace is None:
            return await super().observe_revision(bound)
        limits = _copy_workspace_revision_observation_limits(self.observation_limits)
        return await observe_deterministic_workspace(
            workspace,
            observer=type(self).__name__,
            limits=limits,
        )


class NoWorkspaceBinding(WorkspaceBinding):
    """Binding for agents that intentionally expose no workspace to the runner."""

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
        copied_metadata = _validate_bind_request(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        return BoundWorkspace(
            workspace=None,
            source_workspace=workspace,
            runner=runner,
            path=None,
            metadata=copied_metadata,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        _validate_finalize_request(bound, outcome=outcome, metadata=metadata)
        return None


class GitRepositoryBinding(WorkspaceBinding):
    """Ensure a workspace contains a checked-out Git repository.

    The binding creates or updates the repository before the model sees the
    workspace. It records commit/dirty metadata, but it never commits, pushes,
    or creates pull requests; those remain explicit app/tool workflows.

    ``fetch_refspecs`` fetches refs the default clone/fetch does not — most
    commonly a pull-request head, which lives under ``refs/pull/N/head`` and is
    not covered by the default ``refs/heads/*`` refspec. To review PR #123, pass
    ``fetch_refspecs=["+refs/pull/123/head:refs/heads/pr-123"]`` together with
    ``ref="pr-123"``.
    """

    def __init__(
        self,
        *,
        repo_url: str,
        ref: str | None = None,
        remote_name: str = "origin",
        path: str | None = None,
        git_executable: str = "git",
        fetch: bool = True,
        fetch_refspecs: list[str] | None = None,
        require_clean: bool = True,
        verify_remote_url: bool = True,
        timeout_s: int | None = 120,
        output_limit_bytes: int = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
        observation_limits: WorkspaceRevisionObservationLimits | None = None,
    ) -> None:
        self.repo_url = _validate_git_repo_url(repo_url)
        self.ref = _validate_git_value(ref, "ref") if ref is not None else None
        self.remote_name = _validate_git_value(remote_name, "remote_name")
        self.path = require_clean_nonblank(path, "path") if path is not None else None
        self.git_executable = _validate_git_value(git_executable, "git_executable")
        if type(fetch) is not bool:
            raise TypeError("GitRepositoryBinding fetch must be a bool.")
        if type(require_clean) is not bool:
            raise TypeError("GitRepositoryBinding require_clean must be a bool.")
        if type(verify_remote_url) is not bool:
            raise TypeError("GitRepositoryBinding verify_remote_url must be a bool.")
        self.fetch = fetch
        if fetch_refspecs is not None and not isinstance(fetch_refspecs, list):
            raise TypeError("GitRepositoryBinding fetch_refspecs must be a list of strings.")
        if fetch_refspecs and not fetch:
            raise ValueError("GitRepositoryBinding fetch_refspecs requires fetch=True.")
        self.fetch_refspecs = (
            [_validate_git_value(spec, "fetch_refspecs") for spec in fetch_refspecs]
            if fetch_refspecs is not None
            else None
        )
        self.require_clean = require_clean
        self.verify_remote_url = verify_remote_url
        self.timeout_s = _validate_optional_timeout(timeout_s, "timeout_s")
        self.output_limit_bytes = _validate_positive_int(
            output_limit_bytes,
            "output_limit_bytes",
            owner="GitRepositoryBinding",
        )
        self.observation_limits = _copy_workspace_revision_observation_limits(observation_limits)

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
        request_metadata = _validate_bind_request(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        if workspace is None:
            raise ValueError("GitRepositoryBinding requires a workspace.")
        _reject_reserved_metadata(request_metadata, GIT_REPOSITORY_METADATA_KEY)
        executor = _git_executor_for_workspace(
            workspace,
            git_executable=self.git_executable,
            timeout_s=self.timeout_s,
            output_limit_bytes=self.output_limit_bytes,
        )

        inside_work_tree = await executor.is_work_tree()
        if inside_work_tree:
            await self._prepare_existing_repository(executor)
        else:
            await _require_empty_workspace_for_git_clone(
                workspace,
                timeout_s=self.timeout_s,
                output_limit_bytes=self.output_limit_bytes,
            )
            try:
                await executor.run("clone", self.repo_url, ".")
            except BaseException:
                # A clone that dies mid-transfer — an ordinary failure OR a cancellation/interrupt —
                # leaves partial artifacts that are non-empty AND not a valid work tree, so every
                # later bind would raise (neither empty nor a work tree): a permanent brick from one
                # transient failure. Reset the workspace to the empty state it was just verified to
                # be in so a retry can clone cleanly, then re-raise (propagating the cancellation).
                await _reset_workspace_after_failed_clone(
                    workspace,
                    timeout_s=self.timeout_s,
                    output_limit_bytes=self.output_limit_bytes,
                )
                raise

        await self._fetch_extra_refspecs(executor)
        if self.ref is not None:
            await self._checkout_configured_ref(executor)
        commit, branch = await _git_snapshot_identity(executor)
        dirty = await executor.is_dirty()
        git_metadata = {
            "repo_url": self.repo_url,
            "remote_name": self.remote_name,
            "ref": self.ref,
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "fetch": self.fetch,
            "fetch_refspecs": self.fetch_refspecs,
            "require_clean": self.require_clean,
            "verify_remote_url": self.verify_remote_url,
        }
        bound_metadata = {
            **request_metadata,
            GIT_REPOSITORY_METADATA_KEY: git_metadata,
        }
        return BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=runner,
            path=self.path,
            metadata=bound_metadata,
            snapshot=WorkspaceSnapshot(
                snapshot_id=f"git-bind:{session_id}:{_git_snapshot_version_label(commit)}",
                workspace_id=workspace.id,
                version=commit,
                source="git",
                metadata=git_metadata,
            ),
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        finalize_metadata = _validate_finalize_request(bound, outcome=outcome, metadata=metadata)
        _reject_reserved_metadata(finalize_metadata, GIT_REPOSITORY_METADATA_KEY)
        if bound.workspace is None:
            raise ValueError("GitRepositoryBinding finalize requires a bound workspace.")
        bind_metadata = bound.metadata.get(GIT_REPOSITORY_METADATA_KEY)
        if type(bind_metadata) is not dict:
            raise ValueError("GitRepositoryBinding bound metadata is missing git repository state.")
        executor = _git_executor_for_workspace(
            bound.workspace,
            git_executable=self.git_executable,
            timeout_s=self.timeout_s,
            output_limit_bytes=self.output_limit_bytes,
        )
        if not await executor.is_work_tree():
            raise ValueError("GitRepositoryBinding finalize requires a Git work tree.")
        commit, branch = await _git_snapshot_identity(executor)
        dirty = await executor.is_dirty()
        git_metadata = {
            **copy_json_value(bind_metadata, GIT_REPOSITORY_METADATA_KEY),
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "outcome": outcome,
        }
        return WorkspaceSnapshot(
            snapshot_id=(
                f"git-final:{bound.workspace.id}:"
                f"{_git_snapshot_version_label(commit)}:{outcome or 'unknown'}"
            ),
            workspace_id=bound.workspace.id,
            version=commit,
            source="git",
            metadata={
                **finalize_metadata,
                GIT_REPOSITORY_METADATA_KEY: git_metadata,
            },
        )

    async def observe_revision(self, bound: BoundWorkspace) -> WorkspaceRevisionObservation:
        """Observe Git HEAD, branch, index, and worktree state without mutating Git."""

        if type(bound) is not BoundWorkspace:
            raise TypeError("Workspace revision observation requires a BoundWorkspace.")
        workspace = bound.workspace or bound.source_workspace
        if workspace is None:
            return await super().observe_revision(bound)
        identity = WorkspaceIdentity(
            workspace_id=workspace.id,
            observer=type(self).__name__,
        )
        try:
            limits = _copy_workspace_revision_observation_limits(self.observation_limits)
            executor = _git_executor_for_workspace(
                workspace,
                git_executable=self.git_executable,
                timeout_s=self.timeout_s,
                output_limit_bytes=max(
                    self.output_limit_bytes,
                    limits.max_manifest_bytes,
                ),
            )
            work_tree_result = await executor.capture("rev-parse", "--is-inside-work-tree")
            if (
                work_tree_result.exit_code != 0
                or work_tree_result.stdout_truncated
                or work_tree_result.stderr_truncated
                or work_tree_result.stdout.strip() != "true"
            ):
                return WorkspaceRevisionObservation(
                    identity=identity,
                    status=WorkspaceRevisionObservationStatus.FAILED,
                    detail_code="git_worktree_unavailable",
                )
            return await _observe_git_workspace_revision(
                workspace=workspace,
                executor=executor,
                identity=identity,
                limits=limits,
            )
        except Exception:
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.FAILED,
                detail_code="git_observation_failed",
            )

    async def _prepare_existing_repository(self, executor: _GitWorkspaceExecutor) -> None:
        if self.verify_remote_url:
            current_url = await executor.stdout("remote", "get-url", self.remote_name)
            if current_url != self.repo_url:
                raise ValueError(
                    "GitRepositoryBinding existing repository remote URL does not match "
                    f"configured repo_url for {self.remote_name!r}."
                )
        if self.require_clean and await executor.is_dirty():
            raise ValueError("GitRepositoryBinding refuses to bind a dirty repository.")
        if self.fetch:
            await executor.run("fetch", "--prune", self.remote_name)

    async def _fetch_extra_refspecs(self, executor: _GitWorkspaceExecutor) -> None:
        if not self.fetch or not self.fetch_refspecs:
            return
        await executor.run("fetch", self.remote_name, *self.fetch_refspecs)

    async def _checkout_configured_ref(self, executor: _GitWorkspaceExecutor) -> None:
        if self.ref is None:
            return
        await executor.run("checkout", self.ref)
        fetched_ref = f"refs/remotes/{self.remote_name}/{self.ref}"
        if self.fetch and await executor.ref_exists(fetched_ref):
            await executor.run("merge", "--ff-only", fetched_ref)


@dataclass(slots=True, repr=False)
class _SyncBindingLifecycleBindAuthority:
    """Context-owned authority for one runtime-owned public bind dispatch."""

    binding: object
    active: bool = True
    base_invoked: bool = False
    base_task: asyncio.Task[Any] | None = None
    bound: BoundWorkspace | None = None
    failure_ownership: _BindingFailureOwnership | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def claim_base_bind(self, binding: object) -> bool:
        """Claim this live dispatch once, including from an inherited child context."""

        if binding is not self.binding:
            return False
        with self.lock:
            if not self.active:
                raise RuntimeError(
                    "SyncBinding lifecycle dispatch ended before its delegated base bind started."
                )
            if self.base_invoked:
                raise RuntimeError(
                    "SyncBinding lifecycle dispatch may invoke the base bind only once."
                )
            self.base_invoked = True
            self.base_task = asyncio.current_task()
            return True

    def delegated_base_task(self) -> asyncio.Task[Any] | None:
        """Return a claimed base task only when it differs from the lifecycle caller."""

        with self.lock:
            task = self.base_task
        return None if task is asyncio.current_task() else task

    def deactivate(self) -> None:
        """Prevent detached child contexts from consuming stale lifecycle authority."""

        with self.lock:
            self.active = False


async def _settle_delegated_sync_bind(
    authority: _SyncBindingLifecycleBindAuthority,
) -> tuple[BaseException | None, asyncio.CancelledError | None]:
    """Settle a child-task base bind before its lifecycle authority can escape."""

    task = authority.delegated_base_task()
    if task is None:
        return None, None
    # The outer environment-operation boundary owns classification and
    # redelivery of an already-observed caller cancellation. Passing that
    # signal into the settlement helper would consume its Task.cancelling()
    # request before the boundary can distinguish it from child cancellation.
    outcome = await await_shielded_task_outcome(task)
    return outcome.error, outcome.cancellation


def _timeout_represents_delegated_cancellation(error: BaseException) -> bool:
    """Return whether a timeout positively carries its wrapper cancellation."""

    return type(error) is TimeoutError and isinstance(
        exception_cause(error),
        asyncio.CancelledError,
    )


def _redeliver_settlement_cancellation(
    cancellation: asyncio.CancelledError,
) -> None:
    """Restore one caller request consumed while settling a delegated bind."""

    current_task = asyncio.current_task()
    if current_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("SyncBinding could not redeliver caller cancellation.")
    cancellation_args = cancellation.args
    if not cancellation_args:
        current_task.cancel()
    else:
        current_task.cancel(cancellation_args[0])


def _exception_graph_contains_identity(
    error: BaseException,
    expected: BaseException,
) -> bool:
    """Return whether one runtime-readable exception graph already carries a failure."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if candidate is expected:
            return True
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(children)
        cause = exception_cause(candidate)
        if cause is not None:
            pending.append(cause)
            continue
        try:
            suppress_context = _BASE_EXCEPTION_SUPPRESS_CONTEXT_DESCRIPTOR.__get__(
                candidate,
                BaseException,
            )
        except BaseException:
            suppress_context = True
        if not suppress_context:
            context = exception_context(candidate)
            if context is not None:
                pending.append(context)
    return False


def _without_delegated_cancellation(
    error: BaseException,
) -> BaseException | None:
    """Remove child cancellation already represented by the lifecycle caller.

    A wrapper such as ``asyncio.wait_for`` can propagate the caller's
    cancellation into the delegated base task.  If that task then reports an
    aggregate containing both the propagated cancellation and a real mutation
    failure, the public wrapper and delegated task expose the same cancellation
    at two levels.  Rebuild only the affected groups and retain every
    non-cancellation leaf in its original order and nesting.
    """

    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    filtered: dict[int, BaseException | None] = {}
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in filtered:
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            filtered[candidate_id] = (
                None if isinstance(candidate, asyncio.CancelledError) else candidate
            )
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            retained = [filtered.get(id(child)) for child in children]
            retained_children = [child for child in retained if child is not None]
            if not retained_children:
                filtered[candidate_id] = None
            elif len(retained_children) == len(children) and all(
                retained_child is child
                for retained_child, child in zip(retained_children, children, strict=True)
            ):
                filtered[candidate_id] = candidate
            else:
                filtered[candidate_id] = BaseExceptionGroup(
                    "SyncBinding delegated base bind failures after caller cancellation.",
                    retained_children,
                )
            continue
        children = exception_group_children(candidate)
        if children is None:
            # Fail closed for an unreadable extension-defined group. The
            # lifecycle boundary must not discard unknown failure evidence.
            filtered[candidate_id] = candidate
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))

    return filtered.get(id(error), error)


_SYNC_BINDING_LIFECYCLE_AUTHORITY: ContextVar[_SyncBindingLifecycleBindAuthority | None] = (
    ContextVar("cayu_sync_binding_lifecycle_authority", default=None)
)


class SyncBinding(WorkspaceBinding):
    """Copy a durable workspace into a bound workspace and sync changes back.

    ``workspace`` passed to ``bind`` is the durable source. ``target_workspace``
    or ``target_workspace_plan_factory`` identifies the workspace visible to tools
    during the run, typically a sandbox filesystem wrapper. A factory must only
    resolve a stable, quiescent target identity. Setup that mutates the target is
    returned as ``SyncTargetWorkspacePlan.provision`` so it runs after ownership
    admission. External allocation remains owned by the surrounding environment
    factory lifecycle; constructing a wrapper for an already-owned target is
    identity resolution rather than provisioning. The target workspace should be dedicated to this binding
    because the default clean policy deletes files in the target before copying
    source files in. Every active generation
    process-locally owns both source and target by their authoritative
    ``Workspace.resource_key`` values. Concurrent binds through the same or
    different ``SyncBinding`` instances are rejected when either resource would
    overlap in any source or target role.

    File copies use one bulk tar transfer per direction when either workspace
    implements the explicit ``BoundedTarReader`` or ``TarWriter`` capability
    (RunnerWorkspace implements both). Bounded generic transfers are staged
    before destination writes. ``max_total_bytes`` bounds logical file bytes,
    while ``max_archive_bytes`` independently bounds raw tar bytes; pass
    ``None`` for a limit to opt out of that per-transfer bound. The aggregate
    ``staging_capacity`` remains authoritative for every transfer.
    Per-bind state is keyed by an opaque owner generation. Both resources remain
    reserved until that exact generation finalizes successfully or is explicitly
    abandoned; elapsed time is not evidence that a live binding released them.
    Direct callers whose lifecycle will not invoke ``finalize`` must call
    ``abandon`` with the matching bound workspace. Runtime-managed completed
    outcomes persist a bounded private copy of the generation state before
    finalization so incomplete-session recovery can reconnect the same target
    and retry revision-aware copy-back without repeating bind or workload work.
    """

    def __init__(
        self,
        *,
        target_workspace: Workspace | None = None,
        target_workspace_plan_factory: SyncTargetWorkspacePlanFactory | None = None,
        target_workspace_factory: Callable[[SyncBindingContext], object] | None = None,
        path: str | None = None,
        pattern: str = "**/*",
        max_files: int = 10_000,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = DEFAULT_SYNC_MAX_TOTAL_BYTES,
        max_archive_bytes: int | None = DEFAULT_SYNC_MAX_ARCHIVE_BYTES,
        clean_target: SyncTargetCleanPolicy = "always",
        sync_back: SyncBackPolicy = "always",
        delete_missing: bool = True,
        source_conflict_policy: SyncSourceConflictPolicy = "overwrite",
        preserve_git_modes: bool = False,
        staging_capacity: SyncBindingStagingCapacity | None = None,
    ) -> None:
        if target_workspace is not None and not isinstance(target_workspace, Workspace):
            raise TypeError("SyncBinding target_workspace must be a Workspace or None.")
        if target_workspace_plan_factory is not None and not callable(
            target_workspace_plan_factory
        ):
            raise TypeError("SyncBinding target_workspace_plan_factory must be callable or None.")
        if target_workspace_factory is not None:
            if not callable(target_workspace_factory):
                raise TypeError("SyncBinding target_workspace_factory must be callable or None.")
            raise ValueError(
                "SyncBinding target_workspace_factory is unsafe because it can mutate a target "
                "before ownership admission; use target_workspace_plan_factory returning "
                "SyncTargetWorkspacePlan."
            )
        if target_workspace is not None and target_workspace_plan_factory is not None:
            raise ValueError(
                "SyncBinding accepts either target_workspace or "
                "target_workspace_plan_factory, not both."
            )
        if path is not None:
            require_clean_nonblank(path, "path")
        self.target_workspace = target_workspace
        self.target_workspace_plan_factory = target_workspace_plan_factory
        self.path = path
        self.pattern = require_clean_nonblank(pattern, "pattern")
        self.max_files = _validate_positive_int(max_files, "max_files")
        self.max_file_bytes = _validate_optional_positive_int(max_file_bytes, "max_file_bytes")
        self.max_total_bytes = _validate_optional_positive_int(
            max_total_bytes,
            "max_total_bytes",
        )
        self.max_archive_bytes = _validate_optional_positive_int(
            max_archive_bytes,
            "max_archive_bytes",
        )
        self.clean_target = _validate_clean_policy(clean_target)
        self.sync_back = _validate_sync_back_policy(sync_back)
        if type(delete_missing) is not bool:
            raise TypeError("SyncBinding delete_missing must be a bool.")
        self.delete_missing = delete_missing
        self.source_conflict_policy = _validate_source_conflict_policy(source_conflict_policy)
        if type(preserve_git_modes) is not bool:
            raise TypeError("SyncBinding preserve_git_modes must be a bool.")
        self.preserve_git_modes = preserve_git_modes
        if staging_capacity is not None and not isinstance(
            staging_capacity, SyncBindingStagingCapacity
        ):
            raise TypeError(
                "SyncBinding staging_capacity must be a SyncBindingStagingCapacity or None."
            )
        self.staging_capacity = staging_capacity or DEFAULT_SYNC_BINDING_STAGING_CAPACITY
        if self.source_conflict_policy == "require_revision" and (
            self.max_file_bytes is None or self.max_total_bytes is None
        ):
            raise ValueError(
                "SyncBinding revision-aware copy-back requires finite max_file_bytes "
                "and max_total_bytes."
            )
        self._state_lock = threading.Lock()
        self._states: dict[str, _SyncBindingState] = {}

    def _archive_policy(
        self,
        paths: tuple[str, ...],
        *,
        source: Workspace,
        target: Workspace,
    ) -> _SyncArchivePolicy | None:
        source_tar_policy_identity = source.tar_copy_policy_identity()
        target_tar_policy_identity = target.tar_copy_policy_identity()
        if source_tar_policy_identity is None or target_tar_policy_identity is None:
            return None
        for owner, identity in (
            ("source", source_tar_policy_identity),
            ("target", target_tar_policy_identity),
        ):
            try:
                hash(identity)
            except TypeError:
                raise TypeError(
                    f"SyncBinding {owner} tar_copy_policy_identity must be hashable."
                ) from None
        return _SyncArchivePolicy(
            format_version="cayu-sync-tar-v1",
            paths=paths,
            pattern=self.pattern,
            max_files=self.max_files,
            max_file_bytes=self.max_file_bytes,
            max_total_bytes=self.max_total_bytes,
            max_archive_bytes=self.max_archive_bytes,
            clean_target=self.clean_target,
            sync_back=self.sync_back,
            delete_missing=self.delete_missing,
            source_conflict_policy=self.source_conflict_policy,
            preserve_git_modes=self.preserve_git_modes,
            source_tar_policy_identity=source_tar_policy_identity,
            target_tar_policy_identity=target_tar_policy_identity,
        )

    def staging_snapshot(self) -> SyncBindingStagingSnapshot:
        """Return content-free process-local staging capacity state."""

        return self.staging_capacity.snapshot()

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
        authority = _SYNC_BINDING_LIFECYCLE_AUTHORITY.get()
        retain_failed_ownership = False if authority is None else authority.claim_base_bind(self)
        try:
            bound = await self._bind(
                workspace,
                runner,
                session_id=session_id,
                agent_name=agent_name,
                environment_name=environment_name,
                metadata=metadata,
                retain_failed_ownership=retain_failed_ownership,
                lifecycle_authority=authority if retain_failed_ownership else None,
            )
        except BaseException:
            raise
        if retain_failed_ownership and authority is not None:
            authority.bound = bound
        return bound

    async def _bind_for_environment_lifecycle(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        _attempt: _EnvironmentLifecycleBindAttempt | None = None,
    ) -> BoundWorkspace:
        attempt = _attempt or _EnvironmentLifecycleBindAttempt()
        authority = _SyncBindingLifecycleBindAuthority(binding=self)
        authority_token = _SYNC_BINDING_LIFECYCLE_AUTHORITY.set(authority)
        current_task = asyncio.current_task()
        cancellation_baseline = 0 if current_task is None else current_task.cancelling()
        cancellation_pending_at_entry = bool(
            current_task is not None and getattr(current_task, "_must_cancel", False)
        )
        try:
            try:
                bound = await self.bind(
                    workspace,
                    runner,
                    session_id=session_id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    metadata=metadata,
                )
            except BaseException as public_bind_error:
                delegated_error, settlement_cancellation = await _settle_delegated_sync_bind(
                    authority
                )
                if (
                    isinstance(public_bind_error, asyncio.CancelledError)
                    and delegated_error is not None
                ):
                    delegated_error = _without_delegated_cancellation(delegated_error)
                elif isinstance(
                    delegated_error, asyncio.CancelledError
                ) and _timeout_represents_delegated_cancellation(public_bind_error):
                    delegated_error = None
                elif isinstance(delegated_error, asyncio.CancelledError):
                    delegated_error = unexpected_child_cancellation_error(
                        delegated_error,
                        operation="SyncBinding delegated base bind",
                    )
                secondary_failures = tuple(
                    error
                    for error in (delegated_error, settlement_cancellation)
                    if error is not None
                    and error is not public_bind_error
                    and not _exception_graph_contains_identity(public_bind_error, error)
                    and not (
                        isinstance(public_bind_error, asyncio.CancelledError)
                        and isinstance(error, asyncio.CancelledError)
                    )
                )
                if settlement_cancellation is not None:
                    _redeliver_settlement_cancellation(settlement_cancellation)
                if secondary_failures:
                    raise BaseExceptionGroup(
                        "SyncBinding override and delegated base bind both failed.",
                        [public_bind_error, *secondary_failures],
                    ) from None
                raise
            delegated_error, settlement_cancellation = await _settle_delegated_sync_bind(authority)
            if isinstance(delegated_error, asyncio.CancelledError):
                delegated_error = unexpected_child_cancellation_error(
                    delegated_error,
                    operation="SyncBinding delegated base bind",
                )
            if settlement_cancellation is not None:
                _redeliver_settlement_cancellation(settlement_cancellation)
                if delegated_error is not None:
                    raise BaseExceptionGroup(
                        "SyncBinding delegated base bind failed after caller cancellation.",
                        [settlement_cancellation, delegated_error],
                    ) from delegated_error
                raise settlement_cancellation
            if delegated_error is not None:
                raise delegated_error
            if authority.failure_ownership is not None:
                suppressed_error = RuntimeError(
                    "SyncBinding override suppressed a failed base bind."
                )
                raise suppressed_error
            if authority.bound is None:
                raise RuntimeError(
                    "SyncBinding lifecycle override returned before its authoritative "
                    "base bind completed."
                )
            self._validate_lifecycle_bound_result(
                returned=bound,
                authoritative=authority.bound,
            )
            if current_task is not None and (
                current_task.cancelling() > cancellation_baseline or cancellation_pending_at_entry
            ):
                raise asyncio.CancelledError(getattr(current_task, "_cancel_message", None))
            attempt.retain(
                lambda: self._release_failed_outer_bind(bound),
            )
            return bound
        except BaseException as bind_error:
            ownership = authority.failure_ownership
            if ownership is None and authority.bound is not None:
                ownership = self._transfer_bound_failure_ownership(authority.bound, bind_error)
            attempt.retain(
                None if ownership is None else ownership.release,
            )
            raise
        finally:
            authority.deactivate()
            _SYNC_BINDING_LIFECYCLE_AUTHORITY.reset(authority_token)

    async def _bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None,
        environment_name: str | None,
        metadata: dict[str, Any] | None,
        retain_failed_ownership: bool,
        lifecycle_authority: _SyncBindingLifecycleBindAuthority | None,
    ) -> BoundWorkspace:
        request_metadata = _validate_bind_request(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        if workspace is None:
            raise ValueError("SyncBinding requires a source workspace.")
        if "sync_binding" in request_metadata:
            raise ValueError("SyncBinding metadata key 'sync_binding' is reserved.")
        context = SyncBindingContext(
            source_workspace=workspace,
            runner=runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=request_metadata,
        )
        state_key = uuid4().hex
        source_resource_key: tuple[object, ...] | None = None
        target_resource_key: tuple[object, ...] | None = None
        target: Workspace | None = None
        target_provision: SyncTargetWorkspaceProvisioner | None = None
        factory_source_claimed = self.target_workspace_plan_factory is not None
        try:
            if factory_source_claimed:
                source_resource_key = _validated_workspace_resource_key(workspace)
                _reserve_sync_factory_source(
                    workspace,
                    resource_key=source_resource_key,
                    generation=state_key,
                )
            target, target_provision = await self._target_workspace(context)
            resolved_source_key, target_resource_key = _reject_same_or_indeterminate_target(
                workspace,
                target,
            )
            if factory_source_claimed:
                if resolved_source_key != source_resource_key:
                    raise RuntimeError(
                        "SyncBinding source resource_key changed during target factory resolution."
                    )
                _promote_sync_factory_resources(
                    target,
                    source_resource_key=resolved_source_key,
                    target_resource_key=target_resource_key,
                    generation=state_key,
                )
            else:
                source_resource_key = resolved_source_key
                # Fixed targets have both identities before admission, so reserve the complete pair
                # atomically without a provisional phase.
                _reserve_sync_resources(
                    workspace,
                    target,
                    source_resource_key=source_resource_key,
                    target_resource_key=target_resource_key,
                    generation=state_key,
                )
            if source_resource_key is None:
                raise AssertionError("SyncBinding source resource ownership was not established.")
            if target_provision is not None:
                await _await_sync_mutation(
                    lambda: _run_sync_target_provisioner(target_provision),
                    operation="SyncBinding target provisioning",
                )
            staging_transfer = await self.staging_capacity._acquire_transfer()
            try:
                source_paths = await _list_workspace_paths(
                    workspace,
                    self.pattern,
                    limit=self.max_files,
                    role="source",
                )
                source_observation: _SyncSourceObservation | None = None
                source_revisions: tuple[_SyncSourceRevision, ...] = ()
                if self.source_conflict_policy == "require_revision":
                    source_observation = await _capture_sync_source_revisions(
                        workspace,
                        source_paths,
                        max_file_bytes=cast("int", self.max_file_bytes),
                        max_total_bytes=cast("int", self.max_total_bytes),
                        preserve_git_modes=self.preserve_git_modes,
                        staging_capacity=self.staging_capacity,
                    )
                    source_revisions = source_observation.revisions
                _require_sync_copy_staging_capacity(
                    source=workspace,
                    target=target,
                    paths=source_paths,
                    max_file_bytes=self.max_file_bytes,
                    max_total_bytes=self.max_total_bytes,
                    max_archive_bytes=self.max_archive_bytes,
                    staging_capacity=self.staging_capacity,
                    requires_source_observation=source_observation is not None,
                )
                cleaned_paths: tuple[str, ...] = ()
                if self.clean_target == "always":
                    cleaned_paths = await _clear_workspace(target, max_files=self.max_files)
                    target_baseline_paths: tuple[str, ...] = ()
                else:
                    target_baseline_paths = await _list_workspace_paths(
                        target,
                        self.pattern,
                        limit=self.max_files,
                        role="target",
                    )
                copied_bytes = await _copy_paths(
                    source=workspace,
                    target=target,
                    paths=source_paths,
                    max_file_bytes=self.max_file_bytes,
                    max_total_bytes=self.max_total_bytes,
                    max_archive_bytes=self.max_archive_bytes,
                    preserve_git_modes=self.preserve_git_modes,
                    staging_capacity=self.staging_capacity,
                    source_observation=source_observation,
                    archive_policy=self._archive_policy(
                        source_paths,
                        source=workspace,
                        target=target,
                    ),
                )
                if source_observation is not None:
                    await _verify_sync_source_revisions(
                        workspace,
                        source_revisions,
                        max_file_bytes=cast("int", self.max_file_bytes),
                        max_total_bytes=cast("int", self.max_total_bytes),
                        preserve_git_modes=self.preserve_git_modes,
                        staging_capacity=self.staging_capacity,
                    )
            finally:
                staging_transfer.release()
            bind_metadata = {
                **request_metadata,
                "sync_binding": {
                    "source_workspace_id": workspace.id,
                    "target_workspace_id": target.id,
                    "pattern": self.pattern,
                    "max_files": self.max_files,
                    "max_file_bytes": self.max_file_bytes,
                    "max_total_bytes": self.max_total_bytes,
                    "max_archive_bytes": self.max_archive_bytes,
                    "staging_max_concurrency": self.staging_capacity.max_concurrency,
                    "staging_max_bytes": self.staging_capacity.max_staged_bytes,
                    "reuse_sealed_archives": self.staging_capacity.reuse_sealed_archives,
                    "clean_target": self.clean_target,
                    "sync_back": self.sync_back,
                    "delete_missing": self.delete_missing,
                    "source_conflict_policy": self.source_conflict_policy,
                    "preserve_git_modes": self.preserve_git_modes,
                    "copied_files": len(source_paths),
                    "copied_bytes": copied_bytes,
                    "cleaned_target_files": len(cleaned_paths),
                },
            }
            bound = BoundWorkspace(
                workspace=target,
                source_workspace=workspace,
                runner=runner,
                path=self.path,
                metadata=bind_metadata,
                snapshot=WorkspaceSnapshot(
                    snapshot_id=f"sync-bind:{session_id}",
                    workspace_id=workspace.id,
                    source="sync",
                    metadata={
                        "target_workspace_id": target.id,
                        "copied_files": len(source_paths),
                        "copied_bytes": copied_bytes,
                    },
                ),
                state_key=state_key,
            )
            self._record_sync_state(
                state_key,
                _SyncBindingState(
                    source_paths=source_paths,
                    target_baseline_paths=target_baseline_paths,
                    source_revisions=source_revisions,
                    source_resource_key=source_resource_key,
                    target_id=target.id,
                    target_resource_key=target_resource_key,
                ),
            )
            return bound
        except BaseException as bind_error:
            # A failed bind must not leak either a provisional source claim or a promoted pair (the
            # state that would release it was never stored). Success keeps both resources until the
            # bind's state is dropped.
            if source_resource_key is not None:
                if target_resource_key is None:
                    _release_sync_resource(source_resource_key, generation=state_key)
                elif retain_failed_ownership and _sync_resources_are_owned_by(
                    source_resource_key,
                    target_resource_key,
                    generation=state_key,
                ):
                    ownership = _BindingFailureOwnership(
                        source_resource_key=source_resource_key,
                        target_resource_key=target_resource_key,
                        generation=state_key,
                    )
                    if lifecycle_authority is None:
                        ownership.release()
                        raise RuntimeError(
                            "SyncBinding failed without its lifecycle ownership authority."
                        ) from bind_error
                    lifecycle_authority.failure_ownership = ownership
                else:
                    _release_sync_resources(
                        source_resource_key,
                        target_resource_key,
                        generation=state_key,
                    )
            raise

    def _transfer_bound_failure_ownership(
        self,
        bound: BoundWorkspace,
        bind_error: BaseException,
    ) -> _BindingFailureOwnership | None:
        """Detach a completed base bind that an overriding public bind later rejected."""

        state_key = bound.state_key
        if state_key is None:
            return None
        with self._state_lock:
            state = self._states.get(state_key)
            if state is None:
                return None
            _validate_sync_generation_ownership(bound, state)
            if state.phase != "active":
                raise RuntimeError(
                    "SyncBinding override failed while its completed bind was finalizing."
                ) from bind_error
            ownership = _BindingFailureOwnership(
                source_resource_key=state.source_resource_key,
                target_resource_key=state.target_resource_key,
                generation=state_key,
            )
            del self._states[state_key]
        return ownership

    def _validate_lifecycle_bound_result(
        self,
        *,
        returned: BoundWorkspace,
        authoritative: BoundWorkspace,
    ) -> None:
        """Require an override to return the exact admitted bind generation."""

        if type(returned) is not BoundWorkspace:
            raise TypeError("SyncBinding override must return a BoundWorkspace.")
        state_key = authoritative.state_key
        if state_key is None or returned.state_key != state_key:
            raise ValueError("SyncBinding override returned a different admitted bind generation.")
        with self._state_lock:
            state = self._states.get(state_key)
            if state is None:
                raise ValueError(
                    "SyncBinding override returned a bind generation that is no longer active."
                )
            _validate_sync_generation_ownership(authoritative, state)
            _validate_sync_generation_ownership(returned, state)

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        finalize_metadata = _validate_finalize_request(
            bound,
            outcome=outcome,
            metadata=metadata,
        )
        _reject_reserved_sync_finalize_metadata(finalize_metadata)
        _validate_sync_binding_metadata(bound)
        if not _should_sync_back(self.sync_back, outcome):
            state_key, state = self._begin_sync_finalize(bound)
            self._complete_sync_finalize(
                state_key,
                source_paths=state.source_paths,
                target_baseline_paths=state.target_baseline_paths,
                source_revisions=state.source_revisions,
            )
            return None
        if bound.source_workspace is None:
            raise ValueError("SyncBinding finalize requires a source workspace.")
        if bound.workspace is None:
            raise ValueError("SyncBinding finalize requires a bound workspace.")
        source_workspace = bound.source_workspace
        state_key, state = self._begin_sync_finalize(bound)
        staging_transfer: _CapacityLease | None = None
        try:
            staging_transfer = await self.staging_capacity._acquire_transfer()
            target_paths = await _list_workspace_paths(
                bound.workspace,
                self.pattern,
                limit=self.max_files,
                role="target",
            )
            copy_back_paths = _sync_back_paths(
                source_paths=state.source_paths,
                target_baseline_paths=state.target_baseline_paths,
                target_paths=target_paths,
            )
            deleted_paths: tuple[str, ...] = ()
            if self.delete_missing:
                deleted_paths = tuple(sorted(set(state.source_paths) - set(target_paths)))
            if self.source_conflict_policy == "require_revision":
                copied_bytes, source_revisions = await self._copy_back_with_revisions(
                    state_key=state_key,
                    source=source_workspace,
                    target=bound.workspace,
                    copy_back_paths=copy_back_paths,
                    deleted_paths=deleted_paths,
                    revisions=state.source_revisions,
                    staging_capacity=self.staging_capacity,
                    allow_idempotent_replay=state.recovered,
                )
            else:
                copied_bytes = await _copy_paths(
                    source=bound.workspace,
                    target=source_workspace,
                    paths=copy_back_paths,
                    max_file_bytes=self.max_file_bytes,
                    max_total_bytes=self.max_total_bytes,
                    max_archive_bytes=self.max_archive_bytes,
                    preserve_git_modes=self.preserve_git_modes,
                    staging_capacity=self.staging_capacity,
                    source_observation=None,
                    archive_policy=None,
                )
                for path in deleted_paths:
                    await _await_sync_mutation(
                        lambda path=path: source_workspace.delete(path),
                        operation=f"SyncBinding source delete for {path!r}",
                    )
                source_revisions = state.source_revisions
            synced_source_paths = tuple(
                sorted((set(state.source_paths) - set(deleted_paths)).union(copy_back_paths))
            )
            final_snapshot = WorkspaceSnapshot(
                snapshot_id=_final_sync_snapshot_id(bound, outcome),
                workspace_id=bound.source_workspace.id,
                source="sync",
                metadata={
                    **finalize_metadata,
                    "target_workspace_id": bound.workspace.id,
                    "outcome": outcome,
                    "source_conflict_policy": self.source_conflict_policy,
                    "sync_back": self.sync_back,
                    "delete_missing": self.delete_missing,
                    "copied_files": len(copy_back_paths),
                    "copied_bytes": copied_bytes,
                    "deleted_files": len(deleted_paths),
                },
            )
        except BaseException:
            self._restore_sync_state(state_key)
            raise
        finally:
            if staging_transfer is not None:
                staging_transfer.release()
        self._complete_sync_finalize(
            state_key,
            source_paths=synced_source_paths,
            target_baseline_paths=target_paths,
            source_revisions=source_revisions,
        )
        return final_snapshot

    async def _copy_back_with_revisions(
        self,
        *,
        state_key: str,
        source: Workspace,
        target: Workspace,
        copy_back_paths: tuple[str, ...],
        deleted_paths: tuple[str, ...],
        revisions: tuple[_SyncSourceRevision, ...],
        staging_capacity: SyncBindingStagingCapacity,
        allow_idempotent_replay: bool,
    ) -> tuple[int, tuple[_SyncSourceRevision, ...]]:
        max_file_bytes = cast("int", self.max_file_bytes)
        max_total_bytes = cast("int", self.max_total_bytes)
        archive: _SealedTarArchive | None = None
        archive_reader: BinaryIO | None = None
        tar: tarfile.TarFile | None = None
        try:
            members: dict[str, tarfile.TarInfo] = {}
            copied_bytes = 0
            if copy_back_paths:
                archive_reservation = _sync_archive_reservation_bound(
                    copy_back_paths,
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    max_archive_bytes=self.max_archive_bytes,
                )
                if archive_reservation is None:
                    raise AssertionError("Revision-aware SyncBinding requires finite staging.")
                staging_capacity._record_archive_build()
                archive = await _pack_workspace_tar(
                    source=target,
                    source_stream=target.bounded_tar_stream_reader(),
                    paths=copy_back_paths,
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    max_archive_bytes=self.max_archive_bytes,
                    archive_reservation=archive_reservation,
                    staging_capacity=staging_capacity,
                    minimum_transient_bytes=max_file_bytes,
                    preserve_git_modes=self.preserve_git_modes,
                )
                copied_bytes = archive.logical_bytes
                archive_reader = archive.open_reader()
                tar = tarfile.open(  # noqa: SIM115 - closed in the lifecycle finally below
                    fileobj=archive_reader,
                    mode="r",
                )
                members = {member.name: member for member in tar}
            revision_map = {item.path: item for item in revisions}
            if not allow_idempotent_replay:
                await _verify_sync_source_revisions(
                    source,
                    revisions,
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    preserve_git_modes=self.preserve_git_modes,
                    staging_capacity=None if archive is not None else staging_capacity,
                )
            applied: list[str] = []
            operations = tuple(sorted(set(copy_back_paths).union(deleted_paths)))
            for path in operations:
                try:

                    async def mutate_and_record(path: str = path) -> None:
                        if path in members:
                            if archive is None or tar is None:
                                raise AssertionError("SyncBinding staged archive is unavailable.")
                            member = members[path]
                            content = await _await_sync_mutation(
                                lambda tar=tar, member=member: asyncio.to_thread(
                                    _read_sync_tar_member,
                                    tar,
                                    member,
                                ),
                                operation=(
                                    f"SyncBinding revision archive read for {member.name!r}"
                                ),
                            )
                            expected = revision_map.get(path)
                            git_mode = (
                                _tar_member_git_mode(member) if self.preserve_git_modes else None
                            )
                            try:
                                if self.preserve_git_modes:
                                    if not isinstance(source, WorkspaceGitModeMutator):
                                        raise RuntimeError(
                                            "SyncBinding Git-mode copy-back requires a mode-aware "
                                            "source."
                                        )
                                    if git_mode is None:
                                        raise AssertionError(
                                            "SyncBinding validated Git mode authority is "
                                            "unavailable."
                                        )
                                    if expected is None:
                                        result = await source.create_bytes_with_git_mode(
                                            path,
                                            content,
                                            git_mode=git_mode,
                                        )
                                    else:
                                        if expected.git_mode is None:
                                            raise RuntimeError(
                                                "SyncBinding source omitted Git mode authority: "
                                                f"{path}"
                                            )
                                        result = await source.replace_bytes_with_git_mode(
                                            path,
                                            content,
                                            expected_revision=expected.revision,
                                            expected_git_mode=expected.git_mode,
                                            git_mode=git_mode,
                                        )
                                elif expected is None:
                                    result = await source.create_bytes(path, content)
                                else:
                                    result = await source.replace_bytes(
                                        path,
                                        content,
                                        expected_revision=expected.revision,
                                    )
                            except Exception:
                                if not allow_idempotent_replay:
                                    raise
                                current = await _read_sync_file_result(
                                    source,
                                    path,
                                    max_file_bytes=max_file_bytes,
                                    max_total_bytes=max_total_bytes,
                                    copied_bytes=0,
                                )
                                if (
                                    current.content != content
                                    or current.revision is None
                                    or (self.preserve_git_modes and current.git_mode != git_mode)
                                ):
                                    raise
                                result = WorkspaceMutationResult(
                                    operation="replace" if expected is not None else "create",
                                    before_revision=(
                                        current.revision if expected is not None else None
                                    ),
                                    after_revision=current.revision,
                                    before_sha256=current.sha256 if expected is not None else None,
                                    after_sha256=current.sha256,
                                    before_bytes=(
                                        len(current.content) if expected is not None else None
                                    ),
                                    after_bytes=len(current.content),
                                )
                            if result.after_revision is None:
                                raise RuntimeError(
                                    "SyncBinding conditional write returned no resulting revision."
                                )
                            revision_map[path] = _SyncSourceRevision(
                                path=path,
                                revision=result.after_revision,
                                git_mode=git_mode,
                            )
                        else:
                            expected = revision_map[path]
                            try:
                                await source.delete_if_revision(
                                    path,
                                    expected_revision=expected.revision,
                                )
                            except FileNotFoundError:
                                if not allow_idempotent_replay:
                                    raise
                            del revision_map[path]
                        applied.append(path)
                        self._update_conflict_state(
                            state_key,
                            source_paths=tuple(sorted(revision_map)),
                            source_revisions=tuple(
                                sorted(revision_map.values(), key=lambda item: item.path)
                            ),
                        )

                    await _await_sync_mutation(
                        mutate_and_record,
                        operation=f"SyncBinding conditional source mutation for {path!r}",
                    )
                except Exception as exc:
                    raise SyncBindingSourceConflictError(
                        path,
                        applied_paths=tuple(applied),
                    ) from exc
            return copied_bytes, tuple(sorted(revision_map.values(), key=lambda item: item.path))
        finally:
            if tar is not None:
                tar.close()
            if archive_reader is not None:
                archive_reader.close()
            if archive is not None:
                archive.close()

    async def _target_workspace(
        self,
        context: SyncBindingContext,
    ) -> tuple[Workspace, SyncTargetWorkspaceProvisioner | None]:
        if self.target_workspace is not None:
            return self.target_workspace, None
        if self.target_workspace_plan_factory is None:
            raise ValueError(
                "SyncBinding requires target_workspace or target_workspace_plan_factory."
            )
        result = self.target_workspace_plan_factory(context)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, SyncTargetWorkspacePlan):
            return result.workspace, result.provision
        raise TypeError(
            "SyncBinding target_workspace_plan_factory must return SyncTargetWorkspacePlan."
        )

    def abandon(self, bound: BoundWorkspace) -> bool:
        """Drop in-process bind state for a bind whose finalize will never run.

        Lifecycle owners that skip ``finalize`` (crash recovery, cancelled
        sessions) should call this so per-bind state and resource ownership do
        not leak.
        """

        if type(bound) is not BoundWorkspace:
            raise TypeError("SyncBinding abandon requires a BoundWorkspace.")
        self._discard_sync_state(bound)
        return True

    def _defer_finalize_release(self, bound: BoundWorkspace) -> None:
        if type(bound) is not BoundWorkspace:
            raise TypeError("SyncBinding deferred release requires a BoundWorkspace.")
        state_key = bound.state_key
        if state_key is None:
            raise ValueError("SyncBinding deferred release requires in-process bind state.")
        with self._state_lock:
            state = self._states.get(state_key)
            if state is None:
                raise ValueError("SyncBinding deferred release requires in-process bind state.")
            _validate_sync_generation_ownership(bound, state)
            if state.phase != "active":
                raise RuntimeError("SyncBinding release cannot be deferred during finalization.")
            self._states[state_key] = replace(state, defer_finalize_release=True)

    def _requires_mutation_quiescence(self, bound: BoundWorkspace) -> bool:
        if type(bound) is not BoundWorkspace:
            raise TypeError("SyncBinding quiescence query requires a BoundWorkspace.")
        state_key = bound.state_key
        if state_key is None:
            return False
        with self._state_lock:
            state = self._states.get(state_key)
            if state is None:
                return False
            _validate_sync_generation_ownership(bound, state)
            return True

    def _completion_requires_successful_finalization(self, bound: BoundWorkspace) -> bool:
        if type(bound) is not BoundWorkspace:
            raise TypeError("SyncBinding completion query requires a BoundWorkspace.")
        return _should_sync_back(self.sync_back, "completed")

    def _completion_finalization_recovery_state(
        self,
        bound: BoundWorkspace,
    ) -> dict[str, Any] | None:
        if type(bound) is not BoundWorkspace:
            raise TypeError("SyncBinding recovery query requires a BoundWorkspace.")
        state_key = bound.state_key
        if state_key is None:
            raise ValueError("SyncBinding recovery requires in-process bind state.")
        with self._state_lock:
            state = self._states.get(state_key)
            if state is None:
                raise ValueError("SyncBinding recovery requires an active bind generation.")
            _validate_sync_generation_ownership(bound, state)
            if state.phase != "active":
                raise RuntimeError(
                    "SyncBinding recovery state cannot be captured while finalizing."
                )
            source_paths = list(state.source_paths)
            target_baseline_paths = list(state.target_baseline_paths)
            source_revisions = [
                {
                    "path": item.path,
                    "revision": item.revision,
                    "git_mode": item.git_mode,
                }
                for item in state.source_revisions
            ]
        if bound.source_workspace is None or bound.workspace is None:
            raise ValueError("SyncBinding recovery requires source and target workspaces.")
        snapshot = (
            None
            if bound.snapshot is None
            else {
                "snapshot_id": bound.snapshot.snapshot_id,
                "workspace_id": bound.snapshot.workspace_id,
                "version": bound.snapshot.version,
                "source": bound.snapshot.source,
                "metadata": copy_json_value(bound.snapshot.metadata, "snapshot metadata"),
            }
        )
        return copy_durable_json_object(
            {
                "version": 1,
                "kind": "sync_binding",
                "generation": state_key,
                "source_workspace_id": bound.source_workspace.id,
                "target_workspace_id": bound.workspace.id,
                "source_paths": source_paths,
                "target_baseline_paths": target_baseline_paths,
                "source_revisions": source_revisions,
                "path": bound.path,
                "metadata": copy_json_value(bound.metadata, "binding metadata"),
                "snapshot": snapshot,
            },
            "SyncBinding completion finalization recovery state",
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
        if workspace is None:
            raise ValueError("SyncBinding recovery requires a source workspace.")
        state = copy_durable_json_object(recovery_state, "SyncBinding recovery state")
        if state.get("version") != 1 or state.get("kind") != "sync_binding":
            raise ValueError("SyncBinding recovery state has an unsupported format.")
        raw_generation = state.get("generation")
        raw_source_workspace_id = state.get("source_workspace_id")
        raw_target_workspace_id = state.get("target_workspace_id")
        if type(raw_generation) is not str:
            raise ValueError("SyncBinding recovery generation must be a string.")
        if type(raw_source_workspace_id) is not str:
            raise ValueError("SyncBinding recovery source workspace id must be a string.")
        if type(raw_target_workspace_id) is not str:
            raise ValueError("SyncBinding recovery target workspace id must be a string.")
        generation = require_durable_clean_nonblank(
            raw_generation,
            "SyncBinding recovery generation",
        )
        source_workspace_id = require_durable_clean_nonblank(
            raw_source_workspace_id,
            "SyncBinding recovery source workspace id",
        )
        target_workspace_id = require_durable_clean_nonblank(
            raw_target_workspace_id,
            "SyncBinding recovery target workspace id",
        )
        if workspace.id != source_workspace_id:
            raise RuntimeError("SyncBinding recovery resolved a different source workspace.")
        raw_metadata = state.get("metadata")
        if type(raw_metadata) is not dict:
            raise ValueError("SyncBinding recovery metadata must be an object.")
        bound_metadata = copy_durable_json_object(raw_metadata, "SyncBinding recovery metadata")
        durable_config = bound_metadata.get("sync_binding")
        if type(durable_config) is not dict:
            raise ValueError("SyncBinding recovery metadata lost its binding policy.")
        expected_config = {
            "source_workspace_id": workspace.id,
            "target_workspace_id": target_workspace_id,
            "pattern": self.pattern,
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_archive_bytes": self.max_archive_bytes,
            "clean_target": self.clean_target,
            "sync_back": self.sync_back,
            "delete_missing": self.delete_missing,
            "source_conflict_policy": self.source_conflict_policy,
            "preserve_git_modes": self.preserve_git_modes,
        }
        if any(durable_config.get(key) != value for key, value in expected_config.items()):
            raise RuntimeError("SyncBinding recovery policy changed since the original bind.")
        request_metadata = {
            key: copy_json_value(value, f"SyncBinding recovery metadata {key!r}")
            for key, value in bound_metadata.items()
            if key != "sync_binding"
        }
        context = SyncBindingContext(
            source_workspace=workspace,
            runner=runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=request_metadata,
        )
        target, _provision = await self._target_workspace(context)
        if target.id != target_workspace_id:
            raise RuntimeError("SyncBinding recovery resolved a different target workspace.")
        source_resource_key, target_resource_key = _reject_same_or_indeterminate_target(
            workspace,
            target,
        )

        def paths(field_name: str) -> tuple[str, ...]:
            raw = state.get(field_name)
            if type(raw) is not list or len(raw) > self.max_files:
                raise ValueError(f"SyncBinding recovery {field_name} must be a bounded list.")
            parsed = tuple(
                require_durable_clean_nonblank(
                    item,
                    f"SyncBinding recovery {field_name} path",
                )
                for item in raw
            )
            if any(not _safe_git_observation_path(item) for item in parsed):
                raise ValueError(f"SyncBinding recovery {field_name} contains an unsafe path.")
            if tuple(sorted(set(parsed))) != parsed:
                raise ValueError(f"SyncBinding recovery {field_name} must be sorted and unique.")
            return parsed

        source_paths = paths("source_paths")
        target_baseline_paths = paths("target_baseline_paths")
        raw_revisions = state.get("source_revisions")
        if type(raw_revisions) is not list or len(raw_revisions) > self.max_files:
            raise ValueError("SyncBinding recovery source_revisions must be a bounded list.")
        revisions: list[_SyncSourceRevision] = []
        for raw_revision in raw_revisions:
            if type(raw_revision) is not dict:
                raise ValueError("SyncBinding recovery source revision must be an object.")
            path = require_durable_clean_nonblank(
                raw_revision.get("path"),
                "SyncBinding recovery source path",
            )
            if not _safe_git_observation_path(path):
                raise ValueError("SyncBinding recovery source revision path is unsafe.")
            revision = require_durable_clean_nonblank(
                raw_revision.get("revision"),
                "SyncBinding recovery source revision",
            )
            git_mode = raw_revision.get("git_mode")
            if git_mode not in {None, "100644", "100755"}:
                raise ValueError("SyncBinding recovery source Git mode is invalid.")
            revisions.append(
                _SyncSourceRevision(
                    path=path,
                    revision=revision,
                    git_mode=cast("WorkspaceGitMode | None", git_mode),
                )
            )
        source_revisions = tuple(revisions)
        if tuple(sorted({item.path for item in source_revisions})) != tuple(
            item.path for item in source_revisions
        ):
            raise ValueError("SyncBinding recovery source revisions must be sorted and unique.")
        if (
            self.source_conflict_policy == "require_revision"
            and tuple(item.path for item in source_revisions) != source_paths
        ):
            raise ValueError("SyncBinding recovery lost source revision authority.")
        raw_path = state.get("path")
        if raw_path is not None:
            raw_path = require_clean_nonblank(raw_path, "SyncBinding recovery path")
        raw_snapshot = state.get("snapshot")
        snapshot = None
        if raw_snapshot is not None:
            if type(raw_snapshot) is not dict:
                raise ValueError("SyncBinding recovery snapshot must be an object or null.")
            snapshot = WorkspaceSnapshot(
                snapshot_id=raw_snapshot.get("snapshot_id"),
                workspace_id=raw_snapshot.get("workspace_id"),
                version=raw_snapshot.get("version"),
                source=raw_snapshot.get("source"),
                metadata=raw_snapshot.get("metadata", {}),
            )
        bound = BoundWorkspace(
            workspace=target,
            source_workspace=workspace,
            runner=runner,
            path=raw_path,
            metadata=bound_metadata,
            snapshot=snapshot,
            state_key=generation,
        )
        restored = _SyncBindingState(
            source_paths=source_paths,
            target_baseline_paths=target_baseline_paths,
            source_revisions=source_revisions,
            source_resource_key=source_resource_key,
            target_id=target.id,
            target_resource_key=target_resource_key,
            recovered=True,
        )
        with self._state_lock:
            existing = self._states.get(generation)
            if existing is not None:
                _validate_sync_generation_ownership(bound, existing)
                if existing.phase != "active":
                    raise RuntimeError(
                        "SyncBinding recovery generation is already being finalized."
                    )
                return bound
        _recover_sync_resources(
            workspace,
            target,
            source_resource_key=source_resource_key,
            target_resource_key=target_resource_key,
            generation=generation,
        )
        try:
            self._record_sync_state(generation, restored)
        except BaseException:
            _release_sync_resources(
                source_resource_key,
                target_resource_key,
                generation=generation,
            )
            raise
        return bound

    def _record_sync_state(self, state_key: str, state: _SyncBindingState) -> None:
        with self._state_lock:
            if state_key in self._states:
                raise RuntimeError("SyncBinding generated a duplicate state key.")
            if not _sync_resources_are_owned_by(
                state.source_resource_key,
                state.target_resource_key,
                generation=state_key,
            ):
                raise RuntimeError("SyncBinding lost resource ownership during bind.")
            self._states[state_key] = state

    def _remove_state_locked(self, state_key: str) -> None:
        """Pop one state and release both reservations held by its exact generation."""
        state = self._states.pop(state_key, None)
        if state is not None:
            _release_sync_resources(
                state.source_resource_key,
                state.target_resource_key,
                generation=state_key,
            )

    def _begin_sync_finalize(self, bound: BoundWorkspace) -> tuple[str, _SyncBindingState]:
        state_key = bound.state_key
        if state_key is not None:
            with self._state_lock:
                state = self._states.get(state_key)
                if state is not None:
                    _validate_sync_generation_ownership(bound, state)
                    if state.phase == "finalizing":
                        raise RuntimeError("SyncBinding state is already being finalized.")
                    finalizing = replace(state, phase="finalizing")
                    self._states[state_key] = finalizing
                    return state_key, finalizing
        raise ValueError(
            "SyncBinding finalize requires in-process bind state. "
            "Runtime-managed completion retries must use incomplete-session recovery."
        )

    def _restore_sync_state(self, state_key: str) -> None:
        with self._state_lock:
            state = self._states.get(state_key)
            if state is not None and state.phase == "finalizing":
                self._states[state_key] = replace(state, phase="active")

    def _update_conflict_state(
        self,
        state_key: str,
        *,
        source_paths: tuple[str, ...],
        source_revisions: tuple[_SyncSourceRevision, ...],
    ) -> None:
        with self._state_lock:
            state = self._states.get(state_key)
            if state is None or state.phase != "finalizing":
                raise RuntimeError("SyncBinding copy-back lost its ownership state.")
            self._states[state_key] = replace(
                state,
                source_paths=source_paths,
                source_revisions=source_revisions,
            )

    def _complete_sync_finalize(
        self,
        state_key: str,
        *,
        source_paths: tuple[str, ...],
        target_baseline_paths: tuple[str, ...],
        source_revisions: tuple[_SyncSourceRevision, ...],
    ) -> None:
        with self._state_lock:
            state = self._states.get(state_key)
            if state is None or state.phase != "finalizing":
                raise RuntimeError("SyncBinding finalization lost its ownership state.")
            if state.defer_finalize_release:
                # A composite binding keeps ownership until its remaining
                # cleanup succeeds. Advance both path baselines only after this
                # pass completed so a retry can reverse files created or
                # deleted before the next readable snapshot.
                self._states[state_key] = replace(
                    state,
                    source_paths=source_paths,
                    target_baseline_paths=target_baseline_paths,
                    source_revisions=source_revisions,
                    phase="active",
                )
            else:
                self._remove_state_locked(state_key)

    def _discard_sync_state(self, bound: BoundWorkspace) -> None:
        state_key = bound.state_key
        if state_key is None:
            return
        with self._state_lock:
            state = self._states.get(state_key)
            if state is None:
                return
            _validate_sync_generation_ownership(bound, state)
            if state.phase == "finalizing":
                raise RuntimeError("SyncBinding state cannot be abandoned during finalization.")
            self._remove_state_locked(state_key)


def copy_bound_workspace(bound: BoundWorkspace) -> BoundWorkspace:
    """Return a defensive copy of binding result metadata."""

    if type(bound) is not BoundWorkspace:
        raise TypeError("Bound workspace copies require a BoundWorkspace.")
    return BoundWorkspace(
        workspace=bound.workspace,
        source_workspace=bound.source_workspace,
        runner=bound.runner,
        path=bound.path,
        metadata=copy_json_value(bound.metadata, "metadata"),
        snapshot=copy_workspace_snapshot(bound.snapshot),
        state_key=bound.state_key,
    )


def copy_workspace_snapshot(snapshot: WorkspaceSnapshot | None) -> WorkspaceSnapshot | None:
    """Return a defensive copy of a workspace snapshot."""

    if snapshot is None:
        return None
    if type(snapshot) is not WorkspaceSnapshot:
        raise TypeError("Workspace snapshot copies require a WorkspaceSnapshot or None.")
    return WorkspaceSnapshot(
        snapshot_id=snapshot.snapshot_id,
        workspace_id=snapshot.workspace_id,
        version=snapshot.version,
        source=snapshot.source,
        metadata=copy_json_value(snapshot.metadata, "metadata"),
    )


def _validate_bind_request(
    workspace: Workspace | None,
    runner: Runner | None,
    *,
    session_id: str,
    agent_name: str | None,
    environment_name: str | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if workspace is not None and not isinstance(workspace, Workspace):
        raise TypeError("WorkspaceBinding workspace must be a Workspace or None.")
    if runner is not None and not isinstance(runner, Runner):
        raise TypeError("WorkspaceBinding runner must be a Runner or None.")
    require_clean_nonblank(session_id, "session_id")
    if agent_name is not None:
        require_clean_nonblank(agent_name, "agent_name")
    if environment_name is not None:
        require_clean_nonblank(environment_name, "environment_name")
    if metadata is None:
        return {}
    if type(metadata) is not dict:
        raise TypeError("WorkspaceBinding metadata must be a dict or None.")
    return copy_json_value(metadata, "metadata")


def _validate_finalize_request(
    bound: BoundWorkspace,
    *,
    outcome: str | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if type(bound) is not BoundWorkspace:
        raise TypeError("WorkspaceBinding finalize requires a BoundWorkspace.")
    if outcome is not None:
        require_clean_nonblank(outcome, "outcome")
    if metadata is None:
        return {}
    if type(metadata) is not dict:
        raise TypeError("WorkspaceBinding finalize metadata must be a dict or None.")
    return copy_json_value(metadata, "metadata")


def _reject_same_or_indeterminate_target(
    source: Workspace,
    target: Workspace,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Refuse a SyncBinding whose target is, or might be, the same resource as the source.

    Fails closed: when either workspace cannot report a stable ``resource_key`` the identity is
    indeterminate, so the bind is refused rather than risk ``_clear_workspace`` wiping the source.
    """
    if source is target or source.id == target.id:
        raise ValueError(SYNC_DISTINCT_WORKSPACES_ERROR)
    source_key = _validated_workspace_resource_key(source)
    target_key = _validated_workspace_resource_key(target)
    if source_key == target_key:
        raise ValueError(SYNC_DISTINCT_WORKSPACES_ERROR)
    return source_key, target_key


def _validate_sync_generation_ownership(
    bound: BoundWorkspace,
    state: _SyncBindingState,
) -> None:
    source = bound.source_workspace
    if source is None or _validated_workspace_resource_key(source) != state.source_resource_key:
        raise ValueError(
            "SyncBinding bound source workspace does not match the original bind generation."
        )
    target = bound.workspace
    if target is None or _validated_workspace_resource_key(target) != state.target_resource_key:
        raise ValueError(
            "SyncBinding bound target workspace does not match the original bind generation."
        )


def _validated_workspace_resource_key(workspace: Workspace) -> tuple[object, ...]:
    resource_key = workspace.resource_key
    if resource_key is None:
        workspace_type = type(workspace).__name__
        raise ValueError(
            "SyncBinding cannot confirm the source and target are different workspaces: "
            f"{workspace_type} does not define resource_key. Override Workspace.resource_key on "
            f"{workspace_type} to return a stable identity token."
        )
    if type(resource_key) is not tuple or not resource_key:
        raise TypeError("Workspace resource_key must be a non-empty tuple or None.")
    try:
        hash(resource_key)
    except TypeError as exc:
        raise TypeError("Workspace resource_key must be hashable.") from exc
    return resource_key


def _validate_positive_int(value: int, field_name: str, *, owner: str = "SyncBinding") -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{owner} {field_name} must be greater than zero.")
    return value


def _validate_optional_positive_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, field_name)


def _validate_optional_timeout(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, field_name, owner="GitRepositoryBinding")


def _validate_clean_policy(value: object) -> SyncTargetCleanPolicy:
    if value not in {"always", "never"}:
        raise ValueError("SyncBinding clean_target must be 'always' or 'never'.")
    return cast("SyncTargetCleanPolicy", value)


def _validate_sync_back_policy(value: object) -> SyncBackPolicy:
    if value not in {"always", "on_success", "never"}:
        raise ValueError("SyncBinding sync_back must be 'always', 'on_success', or 'never'.")
    return cast("SyncBackPolicy", value)


def _validate_source_conflict_policy(value: object) -> SyncSourceConflictPolicy:
    if value not in {"overwrite", "require_revision"}:
        raise ValueError(
            "SyncBinding source_conflict_policy must be 'overwrite' or 'require_revision'."
        )
    return cast("SyncSourceConflictPolicy", value)


async def _capture_sync_source_revisions(
    workspace: Workspace,
    paths: tuple[str, ...],
    *,
    max_file_bytes: int,
    max_total_bytes: int,
    preserve_git_modes: bool,
    staging_capacity: SyncBindingStagingCapacity | None = None,
    retained_staged_bytes: int = 0,
) -> _SyncSourceObservation:
    revisions: list[_SyncSourceRevision] = []
    manifest: list[tuple[str, str, int, WorkspaceGitMode | None]] = []
    observed_bytes = 0
    for path in paths:
        read_reservation = workspace.bounded_read_limit(max_file_bytes)
        read_lease: _CapacityLease | None = None
        if staging_capacity is not None:
            required = retained_staged_bytes + read_reservation
            if required > staging_capacity.max_staged_bytes:
                raise SyncBindingStagingCapacityError(
                    "SyncBinding retained archive plus source observation exceeds staged byte "
                    f"capacity: required={required}, "
                    f"capacity={staging_capacity.max_staged_bytes}."
                )
            read_lease = await staging_capacity._reserve_bytes(read_reservation)
        try:
            result = await _await_sync_mutation(
                lambda path=path, read_reservation=read_reservation: workspace.read_bytes(
                    path,
                    max_bytes=read_reservation,
                ),
                operation=f"SyncBinding source observation read for {path!r}",
            )
            if type(result) is not WorkspaceReadResult:
                raise TypeError("SyncBinding source read returned an invalid result.")
            content_sha256 = hashlib.sha256(result.content).hexdigest()
            truncated = result.truncated
            total_bytes = result.total_bytes
            revision = result.revision
            git_mode = result.git_mode
            del result
        finally:
            if read_lease is not None:
                read_lease.release()
        if truncated:
            raise RuntimeError(
                f"SyncBinding source file exceeds max_file_bytes={max_file_bytes}: {path}"
            )
        observed_bytes += total_bytes
        if observed_bytes > max_total_bytes:
            raise RuntimeError(
                f"SyncBinding source files exceed max_total_bytes={max_total_bytes}."
            )
        if revision is None:
            raise RuntimeError(
                f"SyncBinding revision-aware copy-back requires source revision support: {path}"
            )
        if preserve_git_modes and git_mode is None:
            raise RuntimeError(
                f"SyncBinding Git-mode copy-back requires source mode support: {path}"
            )
        observed_revision = _SyncSourceRevision(
            path=path,
            revision=revision,
            git_mode=git_mode if preserve_git_modes else None,
        )
        revisions.append(observed_revision)
        manifest.append((path, content_sha256, total_bytes, observed_revision.git_mode))
    encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _SyncSourceObservation(
        revisions=tuple(revisions),
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        logical_bytes=observed_bytes,
    )


async def _verify_sync_source_revisions(
    workspace: Workspace,
    revisions: tuple[_SyncSourceRevision, ...],
    *,
    max_file_bytes: int,
    max_total_bytes: int,
    preserve_git_modes: bool,
    staging_capacity: SyncBindingStagingCapacity | None = None,
    retained_staged_bytes: int = 0,
) -> None:
    try:
        current = await _capture_sync_source_revisions(
            workspace,
            tuple(item.path for item in revisions),
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            preserve_git_modes=preserve_git_modes,
            staging_capacity=staging_capacity,
            retained_staged_bytes=retained_staged_bytes,
        )
    except Exception as exc:
        path = revisions[0].path if revisions else "source-workspace"
        raise SyncBindingSourceConflictError(path) from exc
    expected = {item.path: item for item in revisions}
    for observed in current.revisions:
        if expected[observed.path] != observed:
            raise SyncBindingSourceConflictError(observed.path)


async def _list_workspace_paths(
    workspace: Workspace,
    pattern: str,
    *,
    limit: int,
    role: str,
) -> tuple[str, ...]:
    result = await workspace.list(pattern, limit=limit)
    if result.truncated:
        if result.total_count is not None and result.total_count > limit:
            raise RuntimeError(
                f"SyncBinding {role} workspace file list exceeded max_files={limit}."
            )
        raise RuntimeError(
            f"SyncBinding {role} workspace file list is incomplete within the backend's "
            "traversal or transfer bounds."
        )
    return tuple(result.paths)


async def _clear_workspace(workspace: Workspace, *, max_files: int) -> tuple[str, ...]:
    paths = await _list_workspace_paths(workspace, "**/*", limit=max_files, role="target")
    for path in paths:
        await _await_sync_mutation(
            lambda path=path: workspace.delete(path),
            operation=f"SyncBinding target delete for {path!r}",
        )
    return paths


async def _copy_paths(
    *,
    source: Workspace,
    target: Workspace,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
    preserve_git_modes: bool = False,
    staging_capacity: SyncBindingStagingCapacity,
    source_observation: _SyncSourceObservation | None,
    archive_policy: _SyncArchivePolicy | None,
) -> int:
    """Copy files through one capacity-owned, private, immutable tar spool."""

    if not paths:
        return 0
    source_stream = source.bounded_tar_stream_reader()
    target_stream = target.tar_stream_writer()
    source_supports_bulk = source_stream is not None or isinstance(source, BoundedTarReader)
    target_supports_bulk = target_stream is not None or isinstance(target, TarWriter)
    if (
        preserve_git_modes
        and not target_supports_bulk
        and not isinstance(target, WorkspaceGitModeMutator)
    ):
        raise RuntimeError("SyncBinding Git-mode transfer requires a mode-aware target.")
    requires_staging = any(
        limit is not None for limit in (max_file_bytes, max_total_bytes, max_archive_bytes)
    )
    if not source_supports_bulk and not target_supports_bulk and not requires_staging:
        return await _copy_paths_per_file(
            source=source,
            target=target,
            paths=paths,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            preserve_git_modes=preserve_git_modes,
            staging_capacity=staging_capacity,
        )
    archive_reservation = _sync_archive_reservation_bound(
        paths,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_archive_bytes=max_archive_bytes,
    )
    if archive_reservation is None:
        return await _copy_paths_per_file(
            source=source,
            target=target,
            paths=paths,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            preserve_git_modes=preserve_git_modes,
            staging_capacity=staging_capacity,
        )
    minimum_transient_bytes = _sync_transfer_transient_reservation_bound(
        target=target,
        target_stream=target_stream,
        paths=paths,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_archive_bytes=max_archive_bytes,
        archive_reservation=archive_reservation,
        requires_source_observation=source_observation is not None,
    )

    async def build_archive() -> _SealedTarArchive:
        archive = await _pack_workspace_tar(
            source=source,
            source_stream=source_stream,
            paths=paths,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            max_archive_bytes=max_archive_bytes,
            archive_reservation=archive_reservation,
            staging_capacity=staging_capacity,
            minimum_transient_bytes=minimum_transient_bytes,
            preserve_git_modes=preserve_git_modes,
        )
        if source_observation is not None:
            try:
                await _verify_sync_source_revisions(
                    source,
                    source_observation.revisions,
                    max_file_bytes=cast("int", max_file_bytes),
                    max_total_bytes=cast("int", max_total_bytes),
                    preserve_git_modes=preserve_git_modes,
                    staging_capacity=None,
                )
            except BaseException:
                archive.close()
                raise
        return archive

    if source_observation is not None and archive_policy is not None:
        cache_key = (
            source_observation.revisions,
            source_observation.content_sha256,
            source_observation.logical_bytes,
            archive_policy,
        )
    else:
        cache_key = object()
    reference = await staging_capacity._acquire_archive(cache_key, build_archive)
    archive = reference.archive
    try:
        if source_observation is not None and not reference.is_builder:
            await _verify_sync_source_revisions(
                source,
                source_observation.revisions,
                max_file_bytes=cast("int", max_file_bytes),
                max_total_bytes=cast("int", max_total_bytes),
                preserve_git_modes=preserve_git_modes,
                staging_capacity=staging_capacity,
                retained_staged_bytes=archive.archive_bytes,
            )
        if target_stream is not None:
            reader = archive.open_reader()
            try:
                await _await_sync_mutation(
                    lambda: target_stream.write_tar_stream(
                        reader,
                        archive_bytes=archive.archive_bytes,
                    ),
                    operation="SyncBinding target tar stream write",
                )
            finally:
                reader.close()
        elif isinstance(target, TarWriter):
            duplicate_lease = (
                None
                if reference.is_builder
                else await _reserve_archive_duplicate(staging_capacity, archive)
            )
            reader = archive.open_reader()
            try:
                tar_data = await _await_sync_mutation(
                    lambda: asyncio.to_thread(reader.read),
                    operation="SyncBinding compatibility archive read",
                )
                await _await_sync_mutation(
                    lambda: target.write_tar_bytes(tar_data),
                    operation="SyncBinding target tar write",
                )
            finally:
                reader.close()
                if duplicate_lease is not None:
                    duplicate_lease.release()
        else:
            await _extract_tar_to_workspace(
                target,
                archive,
                staging_capacity=staging_capacity,
                transient_reserved=reference.is_builder,
                preserve_git_modes=preserve_git_modes,
            )
        return archive.logical_bytes
    finally:
        reference.release()


async def _copy_paths_per_file(
    *,
    source: Workspace,
    target: Workspace,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    preserve_git_modes: bool = False,
    staging_capacity: SyncBindingStagingCapacity,
) -> int:
    copied_bytes = 0
    for path in paths:
        policy_read_limit, _, _ = _copy_read_limit(
            source,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            copied_bytes=copied_bytes,
            max_staged_bytes=None,
            staged_limit_label=None,
        )
        capacity_read_limit = _bounded_workspace_read_limit(
            source,
            staging_capacity.max_staged_bytes,
        )
        read_limit = (
            capacity_read_limit
            if policy_read_limit is None
            else min(policy_read_limit, capacity_read_limit)
        )
        read_lease = await staging_capacity._reserve_bytes(read_limit)
        result: WorkspaceReadResult | None = None
        try:
            result = await _await_sync_mutation(
                lambda path=path, read_limit=read_limit: source.read_bytes(
                    path,
                    max_bytes=read_limit,
                ),
                operation=f"SyncBinding source file read for {path!r}",
            )
            if type(result) is not WorkspaceReadResult:
                raise TypeError("SyncBinding source read returned an invalid result.")
            if result.truncated:
                _raise_sync_file_read_limit(
                    path,
                    result.total_bytes,
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    copied_bytes=copied_bytes,
                    max_retained_bytes=staging_capacity.max_staged_bytes,
                )
            if len(result.content) > staging_capacity.max_staged_bytes:
                raise RuntimeError(
                    "SyncBinding files exceed aggregate staging capacity="
                    f"{staging_capacity.max_staged_bytes}."
                )
            _validate_sync_total_bytes(
                copied_bytes + len(result.content),
                max_total_bytes=max_total_bytes,
            )
            if max_file_bytes is not None and len(result.content) > max_file_bytes:
                raise RuntimeError(
                    f"SyncBinding file exceeds max_file_bytes={max_file_bytes}: {path}"
                )
            if preserve_git_modes:
                if not isinstance(target, WorkspaceGitModeMutator):
                    raise RuntimeError(
                        "SyncBinding Git-mode transfer requires a mode-aware target."
                    )
                git_mode = _require_sync_git_mode(result, path)
                await _await_sync_mutation(
                    lambda path=path, result=result, git_mode=git_mode: (
                        target.write_bytes_with_git_mode(
                            path,
                            result.content,
                            git_mode=git_mode,
                        )
                    ),
                    operation=f"SyncBinding target Git-mode write for {path!r}",
                )
            else:
                await _await_sync_mutation(
                    lambda path=path, content=result.content: target.write_bytes(path, content),
                    operation=f"SyncBinding target write for {path!r}",
                )
            copied_bytes += len(result.content)
        finally:
            result = None
            read_lease.release()
    return copied_bytes


def _raise_sync_file_read_limit(
    path: str,
    total_bytes: int,
    *,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    copied_bytes: int,
    max_retained_bytes: int,
) -> NoReturn:
    if total_bytes > max_retained_bytes:
        raise RuntimeError(
            f"SyncBinding files exceed aggregate staging capacity={max_retained_bytes}."
        )
    if max_file_bytes is not None and total_bytes > max_file_bytes:
        raise RuntimeError(f"SyncBinding file exceeds max_file_bytes={max_file_bytes}: {path}")
    if max_total_bytes is not None and copied_bytes + total_bytes > max_total_bytes:
        raise RuntimeError(f"SyncBinding files exceed max_total_bytes={max_total_bytes}.")
    raise RuntimeError(f"SyncBinding file exceeds the workspace read limit: {path}")


async def _pack_workspace_tar(
    *,
    source: Workspace,
    source_stream: BoundedTarStreamReader | None,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
    archive_reservation: int,
    staging_capacity: SyncBindingStagingCapacity,
    minimum_transient_bytes: int = 0,
    preserve_git_modes: bool = False,
) -> _SealedTarArchive:
    archive_overhead_bytes = tar_archive_size_bound(0, paths)
    if max_archive_bytes is not None and archive_overhead_bytes > max_archive_bytes:
        raise RuntimeError(f"SyncBinding tar exceeds max_archive_bytes={max_archive_bytes}.")
    builder_transient_bytes = _sync_archive_builder_transient_reservation_bound(
        source=source,
        source_stream=source_stream,
        paths=paths,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_archive_bytes=max_archive_bytes,
        archive_reservation=archive_reservation,
    )
    working_reservation = archive_reservation + max(
        builder_transient_bytes,
        minimum_transient_bytes,
    )
    # Reserve the builder's peak working set in one admission. Reserving the
    # archive first and its transient read buffer later can deadlock when several
    # independent builders collectively fill the shared capacity.
    archive_lease = await staging_capacity._reserve_bytes(working_reservation)
    spool: BinaryIO | None = None
    try:
        spool = tempfile.TemporaryFile(  # noqa: SIM115 - ownership moves to sealed archive
            mode="w+b",
            prefix="cayu-sync-tar-",
        )
        active_spool: BinaryIO = spool
        if source_stream is not None:
            bounded_destination = _BoundedSyncArchiveWriter(
                active_spool,
                max_bytes=archive_reservation,
            )
            result = await _await_sync_mutation(
                lambda: source_stream.read_tar_stream(
                    paths,
                    cast("BinaryIO", bounded_destination),
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    max_archive_bytes=max_archive_bytes,
                ),
                operation="SyncBinding source tar stream read",
            )
            archive_bytes = result.archive_bytes
            if archive_bytes != bounded_destination.bytes_written:
                raise RuntimeError(
                    "SyncBinding streamed tar byte accounting did not match emitted bytes."
                )
        elif isinstance(source, BoundedTarReader):
            tar_data = await _await_sync_mutation(
                lambda: source.read_tar_bytes(
                    paths,
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    max_archive_bytes=archive_reservation,
                ),
                operation="SyncBinding compatibility source tar read",
            )
            if type(tar_data) is not bytes:
                raise TypeError("SyncBinding compatibility tar reader must return bytes.")
            archive_bytes = len(tar_data)
            if archive_bytes > archive_reservation:
                raise RuntimeError(
                    "SyncBinding compatibility tar exceeds its admitted archive reservation: "
                    f"required={archive_bytes}, capacity={archive_reservation}."
                )
            await _await_sync_mutation(
                lambda: asyncio.to_thread(active_spool.write, tar_data),
                operation="SyncBinding compatibility archive spool write",
            )
            del tar_data
        else:
            copied_bytes = 0
            staged_logical_limit = (
                None if max_archive_bytes is None else max_archive_bytes - archive_overhead_bytes
            )
            with tarfile.open(fileobj=active_spool, mode="w|") as tar:
                for path in paths:
                    result = await _read_sync_file_result(
                        source,
                        path,
                        max_file_bytes=max_file_bytes,
                        max_total_bytes=max_total_bytes,
                        copied_bytes=copied_bytes,
                        max_staged_bytes=staged_logical_limit,
                        staged_limit_label=(
                            None
                            if max_archive_bytes is None
                            else f"max_archive_bytes={max_archive_bytes}"
                        ),
                    )
                    copied_bytes += len(result.content)
                    info = tarfile.TarInfo(name=path)
                    info.size = len(result.content)
                    if preserve_git_modes:
                        git_mode = _require_sync_git_mode(result, path)
                        info.mode = _git_mode_tar_bits(git_mode)
                        info.uname = _SYNC_GIT_MODE_TAR_OWNER
                    await _await_sync_mutation(
                        lambda tar=tar, info=info, result=result: asyncio.to_thread(
                            tar.addfile,
                            info,
                            io.BytesIO(result.content),
                        ),
                        operation=f"SyncBinding private archive write for {path!r}",
                    )
                    del result
            active_spool.seek(0, io.SEEK_END)
            archive_bytes = active_spool.tell()
        active_spool.seek(0, io.SEEK_END)
        if active_spool.tell() != archive_bytes:
            raise RuntimeError("SyncBinding streamed tar byte accounting did not match its spool.")
        copied_bytes = await _await_sync_mutation(
            lambda: asyncio.to_thread(
                _validate_sync_tar_stream,
                active_spool,
                archive_bytes=archive_bytes,
                paths=paths,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                max_archive_bytes=max_archive_bytes,
                preserve_git_modes=preserve_git_modes,
            ),
            operation="SyncBinding private archive validation",
        )
        sealed = _SealedTarArchive(
            active_spool,
            archive_bytes=archive_bytes,
            logical_bytes=copied_bytes,
            capacity_lease=archive_lease,
        )
        spool = None
        return sealed
    except BaseException:
        try:
            if spool is not None:
                try:
                    spool.close()
                finally:
                    archive_lease.record_archive_cleanup()
        finally:
            archive_lease.release()
        raise


async def _extract_tar_to_workspace(
    target: Workspace,
    archive: _SealedTarArchive,
    *,
    staging_capacity: SyncBindingStagingCapacity,
    transient_reserved: bool,
    preserve_git_modes: bool = False,
) -> None:
    if preserve_git_modes and not isinstance(target, WorkspaceGitModeMutator):
        raise RuntimeError("SyncBinding Git-mode transfer requires a mode-aware target.")
    reader = archive.open_reader()
    try:
        with tarfile.open(fileobj=reader, mode="r") as tar:
            for member in tar:
                if (
                    not transient_reserved
                    and archive.archive_bytes + member.size > staging_capacity.max_staged_bytes
                ):
                    raise SyncBindingStagingCapacityError(
                        "SyncBinding archive plus extracted file exceeds staged byte capacity: "
                        f"required={archive.archive_bytes + member.size}, "
                        f"capacity={staging_capacity.max_staged_bytes}."
                    )
                content_lease = (
                    None
                    if transient_reserved
                    else await staging_capacity._reserve_bytes(member.size)
                )
                try:
                    content = await _await_sync_mutation(
                        lambda tar=tar, member=member: asyncio.to_thread(
                            _read_sync_tar_member,
                            tar,
                            member,
                        ),
                        operation=f"SyncBinding private archive read for {member.name!r}",
                    )
                    try:
                        if preserve_git_modes:
                            git_mode = _tar_member_git_mode(member)
                            mode_target = cast("WorkspaceGitModeMutator", target)
                            await _await_sync_mutation(
                                lambda name=member.name, content=content, git_mode=git_mode, mode_target=mode_target: (
                                    mode_target.write_bytes_with_git_mode(
                                        name,
                                        content,
                                        git_mode=git_mode,
                                    )
                                ),
                                operation=(
                                    f"SyncBinding target Git-mode write for {member.name!r}"
                                ),
                            )
                        else:
                            await _await_sync_mutation(
                                lambda name=member.name, content=content: target.write_bytes(
                                    name, content
                                ),
                                operation=f"SyncBinding target write for {member.name!r}",
                            )
                    finally:
                        # The next member read may allocate before assigning the
                        # next loop value. Drop this buffer while its byte lease
                        # still accounts for it, including after a failed write.
                        del content
                finally:
                    if content_lease is not None:
                        content_lease.release()
    finally:
        reader.close()


def _read_sync_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = tar.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"SyncBinding tar member could not be read: {member.name}")
    try:
        return extracted.read()
    finally:
        extracted.close()


async def _reserve_archive_duplicate(
    staging_capacity: SyncBindingStagingCapacity,
    archive: _SealedTarArchive,
) -> _CapacityLease:
    required = archive.archive_bytes * 2
    if required > staging_capacity.max_staged_bytes:
        raise SyncBindingStagingCapacityError(
            "SyncBinding compatibility tar copy exceeds staged byte capacity: "
            f"required={required}, capacity={staging_capacity.max_staged_bytes}."
        )
    return await staging_capacity._reserve_bytes(archive.archive_bytes)


def _sync_archive_reservation_bound(
    paths: tuple[str, ...],
    *,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
) -> int | None:
    logical_bounds: list[int] = []
    if max_file_bytes is not None:
        logical_bounds.append(max_file_bytes * len(paths))
    if max_total_bytes is not None:
        logical_bounds.append(max_total_bytes)
    raw_bounds = [tar_archive_size_bound(min(logical_bounds), paths)] if logical_bounds else []
    if max_archive_bytes is not None:
        raw_bounds.append(max_archive_bytes)
    return min(raw_bounds) if raw_bounds else None


def _sync_file_reservation_bound(
    *,
    copied_bytes: int,
    archive_overhead_bytes: int,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
) -> int:
    bounds: list[int] = []
    if max_file_bytes is not None:
        bounds.append(max_file_bytes)
    if max_total_bytes is not None:
        bounds.append(max(0, max_total_bytes - copied_bytes))
    if max_archive_bytes is not None:
        bounds.append(max(0, max_archive_bytes - archive_overhead_bytes - copied_bytes))
    if not bounds:
        raise AssertionError("A staged SyncBinding transfer requires a finite read bound.")
    return min(bounds)


def _sync_transfer_transient_reservation_bound(
    *,
    target: Workspace,
    target_stream: TarStreamWriter | None,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
    archive_reservation: int,
    requires_source_observation: bool,
) -> int:
    transient_bounds = [0]
    if requires_source_observation:
        if max_file_bytes is None:
            raise AssertionError("Revision-aware SyncBinding requires a finite file bound.")
        transient_bounds.append(max_file_bytes)
    if target_stream is None:
        if isinstance(target, TarWriter):
            transient_bounds.append(archive_reservation)
        else:
            transient_bounds.append(
                max(
                    1,
                    _sync_file_reservation_bound(
                        copied_bytes=0,
                        archive_overhead_bytes=tar_archive_size_bound(0, paths),
                        max_file_bytes=max_file_bytes,
                        max_total_bytes=max_total_bytes,
                        max_archive_bytes=max_archive_bytes,
                    ),
                )
            )
    return max(transient_bounds)


def _sync_archive_builder_transient_reservation_bound(
    *,
    source: Workspace,
    source_stream: BoundedTarStreamReader | None,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
    archive_reservation: int,
) -> int:
    if source_stream is not None:
        return 0
    if isinstance(source, BoundedTarReader):
        return archive_reservation
    return max(
        1,
        _sync_file_reservation_bound(
            copied_bytes=0,
            archive_overhead_bytes=tar_archive_size_bound(0, paths),
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            max_archive_bytes=max_archive_bytes,
        ),
    )


def _require_sync_copy_staging_capacity(
    *,
    source: Workspace,
    target: Workspace,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
    staging_capacity: SyncBindingStagingCapacity,
    requires_source_observation: bool,
) -> None:
    if not paths:
        return
    source_stream = source.bounded_tar_stream_reader()
    target_stream = target.tar_stream_writer()
    source_supports_bulk = source_stream is not None or isinstance(source, BoundedTarReader)
    target_supports_bulk = target_stream is not None or isinstance(target, TarWriter)
    requires_staging = any(
        limit is not None for limit in (max_file_bytes, max_total_bytes, max_archive_bytes)
    )
    if not source_supports_bulk and not target_supports_bulk and not requires_staging:
        return
    archive_reservation = _sync_archive_reservation_bound(
        paths,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_archive_bytes=max_archive_bytes,
    )
    if archive_reservation is None:
        return
    consumer_transient_bytes = _sync_transfer_transient_reservation_bound(
        target=target,
        target_stream=target_stream,
        paths=paths,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_archive_bytes=max_archive_bytes,
        archive_reservation=archive_reservation,
        requires_source_observation=requires_source_observation,
    )
    builder_transient_bytes = _sync_archive_builder_transient_reservation_bound(
        source=source,
        source_stream=source_stream,
        paths=paths,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_archive_bytes=max_archive_bytes,
        archive_reservation=archive_reservation,
    )
    required = archive_reservation + max(
        consumer_transient_bytes,
        builder_transient_bytes,
    )
    if required > staging_capacity.max_staged_bytes:
        raise SyncBindingStagingCapacityError(
            "SyncBinding transfer peak exceeds staged byte capacity: "
            f"required={required}, capacity={staging_capacity.max_staged_bytes}."
        )


def _validate_sync_tar_stream(
    source: BinaryIO,
    *,
    archive_bytes: int,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
    preserve_git_modes: bool = False,
) -> int:
    if max_archive_bytes is not None and archive_bytes > max_archive_bytes:
        raise RuntimeError(f"SyncBinding tar exceeds max_archive_bytes={max_archive_bytes}.")
    copied_bytes = 0
    member_names: list[str] = []
    source.seek(0)
    try:
        with tarfile.open(fileobj=source, mode="r") as tar:
            for member in tar:
                if not member.isreg():
                    raise RuntimeError(
                        f"SyncBinding tar member must be a regular file: {member.name}"
                    )
                if preserve_git_modes:
                    _tar_member_git_mode(member)
                if max_file_bytes is not None and member.size > max_file_bytes:
                    raise RuntimeError(
                        f"SyncBinding file exceeds max_file_bytes={max_file_bytes}: {member.name}"
                    )
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"SyncBinding tar member could not be read: {member.name}")
                try:
                    while extracted.read(1 << 16):
                        pass
                finally:
                    extracted.close()
                member_names.append(member.name)
                copied_bytes += member.size
                _validate_sync_total_bytes(copied_bytes, max_total_bytes=max_total_bytes)
    except tarfile.TarError as exc:
        raise RuntimeError("SyncBinding bulk transfer returned an invalid tar archive.") from exc
    if sorted(member_names) != sorted(paths):
        raise RuntimeError("SyncBinding bulk transfer paths do not match the requested files.")
    return copied_bytes


async def _await_sync_mutation(
    operation_factory: Callable[[], Awaitable[_MutationResultT]],
    *,
    operation: str,
) -> _MutationResultT:
    """Keep a dispatched workspace operation fenced until it is quiescent.

    Workspace implementations can delegate filesystem or SDK work to a thread.
    Cancelling the await does not prove that work stopped. Run each operation in
    a shielded child task and defer propagation of caller cancellation or fatal
    signals until the child has a terminal outcome, so SyncBinding cannot release
    staging capacity or restore ownership while old work can still consume a
    staged object or affect a new owner.
    """

    # Deliver a cancellation already pending at entry before dispatching work.
    # A historical, already-handled cancellation request does not raise here.
    await asyncio.sleep(0)
    current_task = asyncio.current_task()
    observed_cancellation_requests = 0 if current_task is None else current_task.cancelling()

    async def run_operation() -> _MutationResultT:
        return await operation_factory()

    mutation_task = asyncio.create_task(run_operation())
    caller_signal: BaseException | None = None

    while not mutation_task.done():
        try:
            await asyncio.shield(mutation_task)
        except asyncio.CancelledError as cancellation:
            cancellation_requests = 0 if current_task is None else current_task.cancelling()
            if cancellation_requests > observed_cancellation_requests:
                observed_cancellation_requests = cancellation_requests
                if caller_signal is None:
                    caller_signal = cancellation
                else:
                    caller_signal.add_note(
                        f"Additional caller cancellation arrived while draining {operation}."
                    )
                continue
            if mutation_task.done():
                break
            raise
        except BaseException as signal:
            if mutation_task.done():
                break
            if caller_signal is None:
                caller_signal = signal
            else:
                caller_signal = BaseExceptionGroup(
                    f"{operation} received multiple caller control signals.",
                    [caller_signal, signal],
                )

    try:
        result = mutation_task.result()
    except asyncio.CancelledError as child_cancellation:
        mutation_error: BaseException = unexpected_child_cancellation_error(
            child_cancellation,
            operation=operation,
        )
    except BaseException as child_error:
        mutation_error = child_error
    else:
        if caller_signal is not None:
            raise caller_signal
        return result

    if caller_signal is not None:
        raise BaseExceptionGroup(
            f"{operation} failed after a caller control signal.",
            [caller_signal, mutation_error],
        ) from mutation_error
    raise mutation_error


async def _run_sync_target_provisioner(
    provision: SyncTargetWorkspaceProvisioner,
) -> None:
    """Run one admitted target setup callback and validate its result."""

    result = provision()
    if inspect.isawaitable(result):
        result = await result
    if result is not None:
        raise TypeError("SyncTargetWorkspacePlan provision must return None.")


def _validate_sync_tar(
    tar_data: bytes,
    paths: tuple[str, ...],
    *,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    max_archive_bytes: int | None,
    preserve_git_modes: bool = False,
) -> int:
    if type(tar_data) is not bytes:
        raise TypeError("SyncBinding bulk transfer must produce tar bytes.")
    _validate_sync_archive_bytes(tar_data, max_archive_bytes=max_archive_bytes)
    copied_bytes = 0
    member_names: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as archive:
            for member in archive.getmembers():
                if not member.isreg():
                    raise RuntimeError(
                        f"SyncBinding tar member must be a regular file: {member.name}"
                    )
                if preserve_git_modes:
                    _tar_member_git_mode(member)
                if max_file_bytes is not None and member.size > max_file_bytes:
                    raise RuntimeError(
                        f"SyncBinding file exceeds max_file_bytes={max_file_bytes}: {member.name}"
                    )
                member_names.append(member.name)
                copied_bytes += member.size
                _validate_sync_total_bytes(
                    copied_bytes,
                    max_total_bytes=max_total_bytes,
                )
    except tarfile.TarError as exc:
        raise RuntimeError("SyncBinding bulk transfer returned an invalid tar archive.") from exc
    if sorted(member_names) != sorted(paths):
        raise RuntimeError("SyncBinding bulk transfer paths do not match the requested files.")
    return copied_bytes


async def _read_sync_file(
    source: Workspace,
    path: str,
    *,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    copied_bytes: int,
    max_staged_bytes: int | None = None,
    staged_limit_label: str | None = None,
) -> bytes:
    result = await _read_sync_file_result(
        source,
        path,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        copied_bytes=copied_bytes,
        max_staged_bytes=max_staged_bytes,
        staged_limit_label=staged_limit_label,
    )
    return result.content


async def _read_sync_file_result(
    source: Workspace,
    path: str,
    *,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    copied_bytes: int,
    max_staged_bytes: int | None = None,
    staged_limit_label: str | None = None,
) -> WorkspaceReadResult:
    read_limit, limit_label, active_aggregate_limit = _copy_read_limit(
        source,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        copied_bytes=copied_bytes,
        max_staged_bytes=max_staged_bytes,
        staged_limit_label=staged_limit_label,
    )
    result = await _await_sync_mutation(
        lambda: source.read_bytes(path, max_bytes=read_limit),
        operation=f"SyncBinding source file read for {path!r}",
    )
    if type(result) is not WorkspaceReadResult:
        raise TypeError("SyncBinding source read returned an invalid result.")
    if result.truncated:
        if active_aggregate_limit is not None:
            aggregate_bytes, aggregate_label = active_aggregate_limit
            _validate_sync_aggregate_bytes(
                copied_bytes + result.total_bytes,
                max_bytes=aggregate_bytes,
                limit_label=aggregate_label,
            )
        raise RuntimeError(f"SyncBinding file exceeds {limit_label}: {path}")
    _validate_sync_total_bytes(
        copied_bytes + len(result.content),
        max_total_bytes=max_total_bytes,
    )
    if max_staged_bytes is not None:
        if staged_limit_label is None:
            raise RuntimeError("SyncBinding staged byte limit is missing its label.")
        _validate_sync_aggregate_bytes(
            copied_bytes + len(result.content),
            max_bytes=max_staged_bytes,
            limit_label=staged_limit_label,
        )
    return result


def _require_sync_git_mode(result: WorkspaceReadResult, path: str) -> WorkspaceGitMode:
    if result.git_mode is None:
        raise RuntimeError(f"SyncBinding source omitted Git mode authority: {path}")
    return result.git_mode


def _git_mode_tar_bits(git_mode: WorkspaceGitMode) -> int:
    return 0o755 if git_mode == "100755" else 0o644


def _tar_member_git_mode(member: tarfile.TarInfo) -> WorkspaceGitMode:
    if member.uname != _SYNC_GIT_MODE_TAR_OWNER or member.mode not in {0o644, 0o755}:
        raise RuntimeError(
            f"SyncBinding tar member omitted valid Git mode authority: {member.name}"
        )
    return "100755" if member.mode == 0o755 else "100644"


def _copy_read_limit(
    source: Workspace,
    *,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    copied_bytes: int,
    max_staged_bytes: int | None,
    staged_limit_label: str | None,
) -> tuple[int | None, str, tuple[int, str] | None]:
    read_limit = max_file_bytes
    limit_label = (
        f"max_file_bytes={max_file_bytes}"
        if max_file_bytes is not None
        else "the workspace read limit"
    )
    aggregate_limits: list[tuple[int, str]] = []
    if max_total_bytes is not None:
        aggregate_limits.append((max_total_bytes, f"max_total_bytes={max_total_bytes}"))
    if max_staged_bytes is not None:
        if staged_limit_label is None:
            raise RuntimeError("SyncBinding staged byte limit is missing its label.")
        aggregate_limits.append((max_staged_bytes, staged_limit_label))
    active_aggregate_limit: tuple[int, str] | None = None
    if aggregate_limits:
        aggregate_bytes, aggregate_label = min(aggregate_limits, key=lambda item: item[0])
        remaining_bytes = aggregate_bytes - copied_bytes
        if remaining_bytes < 0:
            _validate_sync_aggregate_bytes(
                copied_bytes,
                max_bytes=aggregate_bytes,
                limit_label=aggregate_label,
            )
        # Workspace reads require a positive max_bytes. Probe one byte when the
        # aggregate is exactly exhausted so trailing empty files remain valid.
        total_read_ceiling = max(1, remaining_bytes)
        if max_file_bytes is None:
            read_limit = _bounded_workspace_read_limit(source, total_read_ceiling)
            if read_limit == total_read_ceiling:
                limit_label = aggregate_label
                active_aggregate_limit = (aggregate_bytes, aggregate_label)
        elif remaining_bytes == 0 or total_read_ceiling < max_file_bytes:
            read_limit = total_read_ceiling
            limit_label = aggregate_label
            active_aggregate_limit = (aggregate_bytes, aggregate_label)
    return read_limit, limit_label, active_aggregate_limit


def _validate_sync_total_bytes(
    copied_bytes: int,
    *,
    max_total_bytes: int | None,
) -> None:
    if max_total_bytes is not None and copied_bytes > max_total_bytes:
        raise RuntimeError(f"SyncBinding files exceed max_total_bytes={max_total_bytes}.")


def _validate_sync_aggregate_bytes(
    copied_bytes: int,
    *,
    max_bytes: int,
    limit_label: str,
) -> None:
    if copied_bytes > max_bytes:
        raise RuntimeError(f"SyncBinding files exceed {limit_label}.")


def _validate_sync_archive_bytes(
    tar_data: bytes,
    *,
    max_archive_bytes: int | None,
) -> None:
    if max_archive_bytes is not None and len(tar_data) > max_archive_bytes:
        raise RuntimeError(f"SyncBinding tar exceeds max_archive_bytes={max_archive_bytes}.")


def _bounded_workspace_read_limit(source: Workspace, max_bytes: int) -> int:
    value = source.bounded_read_limit(max_bytes)
    if type(value) is not int or value <= 0 or value > max_bytes:
        raise RuntimeError(
            f"{type(source).__name__}.bounded_read_limit() must return a positive integer "
            f"no greater than max_bytes={max_bytes}."
        )
    return value


def _validate_sync_binding_metadata(bound: BoundWorkspace) -> dict[str, Any]:
    value = bound.metadata.get("sync_binding")
    if type(value) is not dict:
        raise ValueError("SyncBinding bound metadata is missing sync_binding state.")
    return copy_json_value(value, "sync_binding")


def _reject_reserved_sync_finalize_metadata(metadata: dict[str, Any]) -> None:
    reserved_keys = sorted(SYNC_FINAL_METADATA_KEYS.intersection(metadata))
    if reserved_keys:
        names = ", ".join(repr(key) for key in reserved_keys)
        raise ValueError(f"SyncBinding finalize metadata key is reserved: {names}.")


def _should_sync_back(policy: SyncBackPolicy, outcome: str | None) -> bool:
    if policy == "never":
        return False
    if policy == "always":
        return True
    return outcome == "completed"


def _sync_back_paths(
    *,
    source_paths: tuple[str, ...],
    target_baseline_paths: tuple[str, ...],
    target_paths: tuple[str, ...],
) -> tuple[str, ...]:
    source_set = set(source_paths)
    target_baseline_set = set(target_baseline_paths)
    return tuple(
        path for path in target_paths if path in source_set or path not in target_baseline_set
    )


def _final_sync_snapshot_id(bound: BoundWorkspace, outcome: str | None) -> str:
    source_id = bound.source_workspace.id if bound.source_workspace is not None else "unknown"
    suffix = outcome or "unknown"
    return f"sync-final:{source_id}:{suffix}"


class _GitWorkspaceExecutor:
    def __init__(
        self,
        *,
        runner: Runner,
        cwd: str | None,
        git_executable: str,
        timeout_s: int | None,
        output_limit_bytes: int,
    ) -> None:
        self.runner = runner
        self.cwd = cwd
        self.git_executable = git_executable
        self.timeout_s = timeout_s
        self.output_limit_bytes = output_limit_bytes

    async def run(self, *args: str) -> None:
        result = await self._exec(*args)
        if result.exit_code != 0:
            _raise_git_error(args, result)

    async def stdout(self, *args: str) -> str:
        result = await self._exec(*args)
        if result.exit_code != 0:
            _raise_git_error(args, result)
        if result.stdout_truncated:
            raise RuntimeError(f"Git command output exceeded limit: {_git_command_label(args)}")
        return result.stdout.strip()

    async def is_work_tree(self) -> bool:
        result = await self._exec("rev-parse", "--is-inside-work-tree")
        return result.exit_code == 0 and result.stdout.strip() == "true"

    async def is_dirty(self) -> bool:
        result = await self._exec("status", "--porcelain")
        if result.exit_code != 0:
            _raise_git_error(("status", "--porcelain"), result)
        if result.stdout_truncated:
            raise RuntimeError("Git status output exceeded limit.")
        return bool(result.stdout.strip())

    async def ref_exists(self, ref: str) -> bool:
        result = await self._exec("rev-parse", "--verify", "--quiet", ref)
        return result.exit_code == 0

    async def capture(self, *args: str):
        """Return bounded raw Git output for typed observation parsing."""

        cwd = self.runner.resolve_cwd(self.cwd)
        windows_workspace = bool(ntpath.splitdrive(cwd)[0])
        null_device = "NUL" if windows_workspace else "/dev/null"
        file_mode_config = () if windows_workspace else ("-c", "core.fileMode=true")
        return await self.runner.exec(
            ExecCommand.process(
                self.git_executable,
                "--no-pager",
                "-c",
                "core.fsmonitor=false",
                *file_mode_config,
                "-c",
                f"core.hooksPath={null_device}",
                *args,
            ),
            cwd=self.cwd,
            env={
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_CEILING_DIRECTORIES": cwd,
                "GIT_CONFIG_GLOBAL": null_device,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            },
            env_remove=_GIT_OBSERVATION_ENV_REMOVE,
            timeout_s=self.timeout_s,
            output_limit_bytes=self.output_limit_bytes,
        )

    async def _exec(self, *args: str):
        return await self.runner.exec(
            ExecCommand.process(self.git_executable, *args),
            cwd=self.cwd,
            timeout_s=self.timeout_s,
            output_limit_bytes=self.output_limit_bytes,
        )


async def _git_snapshot_identity(
    executor: _GitWorkspaceExecutor,
) -> tuple[str | None, str]:
    """Return Git snapshot identity without requiring an initial commit."""

    head_result = await executor.capture("rev-parse", "--verify", "--quiet", "HEAD")
    if head_result.stdout_truncated or head_result.stderr_truncated:
        raise RuntimeError("Git HEAD identity exceeded the configured output limit.")
    if head_result.exit_code == 0:
        commit = head_result.stdout.strip()
        if not _valid_git_object_id(commit):
            raise RuntimeError("Git HEAD identity is malformed.")
    elif (
        head_result.exit_code == 1
        and not head_result.stdout.strip()
        and not head_result.stderr.strip()
    ):
        commit = None
    else:
        raise RuntimeError("Git HEAD identity could not be read.")

    branch_result = await executor.capture("symbolic-ref", "--quiet", "--short", "HEAD")
    if branch_result.stdout_truncated or branch_result.stderr_truncated:
        raise RuntimeError("Git branch identity exceeded the configured output limit.")
    if branch_result.exit_code == 0:
        try:
            branch = require_durable_clean_nonblank(branch_result.stdout.strip(), "branch")
            if len(branch.encode("utf-8")) > 4096:
                raise ValueError("Git branch identity is too large.")
        except (UnicodeEncodeError, ValueError):
            raise RuntimeError("Git branch identity is malformed.") from None
    elif (
        commit is not None
        and branch_result.exit_code == 1
        and not branch_result.stdout.strip()
        and not branch_result.stderr.strip()
    ):
        branch = "HEAD"
    else:
        raise RuntimeError("Git branch identity could not be read.")
    return commit, branch


def _git_snapshot_version_label(commit: str | None) -> str:
    return "unborn" if commit is None else commit[:12]


async def _observe_git_workspace_revision(
    *,
    workspace: Workspace,
    executor: _GitWorkspaceExecutor,
    identity: WorkspaceIdentity,
    limits: WorkspaceRevisionObservationLimits,
) -> WorkspaceRevisionObservation:
    head_result = await executor.capture("rev-parse", "--verify", "--quiet", "HEAD")
    if head_result.stdout_truncated or head_result.stderr_truncated:
        return _git_observation_limit(identity, "git_head_output_truncated")
    if head_result.exit_code == 0:
        head_revision = head_result.stdout.strip()
    elif (
        head_result.exit_code == 1
        and not head_result.stdout.strip()
        and not head_result.stderr.strip()
    ):
        head_revision = None
    else:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="git_head_observation_failed",
        )
    if head_revision is not None and not _valid_git_object_id(head_revision):
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="git_head_invalid",
        )

    branch_result = await executor.capture("symbolic-ref", "--quiet", "--short", "HEAD")
    if branch_result.stdout_truncated or branch_result.stderr_truncated:
        return _git_observation_limit(identity, "git_branch_output_truncated")
    if branch_result.exit_code == 0:
        branch = branch_result.stdout.strip()
    elif (
        branch_result.exit_code == 1
        and not branch_result.stdout.strip()
        and not branch_result.stderr.strip()
    ):
        branch = None
    else:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            head_revision=head_revision,
            detail_code="git_branch_observation_failed",
        )
    if branch is not None and len(branch.encode("utf-8")) > 4096:
        return _git_observation_limit(
            identity,
            "git_branch_output_truncated",
            head_revision=head_revision,
        )

    status_result = await executor.capture(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
        "--renames",
    )
    if status_result.exit_code != 0:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            head_revision=head_revision,
            branch=branch,
            detail_code="git_status_failed",
        )
    if status_result.stdout_truncated or status_result.stderr_truncated:
        return _git_observation_limit(
            identity,
            "git_status_output_truncated",
            head_revision=head_revision,
            branch=branch,
        )
    try:
        records = _parse_git_status_records(status_result.stdout)
    except ValueError:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            head_revision=head_revision,
            branch=branch,
            detail_code="git_status_invalid",
        )

    index_result = await executor.capture("ls-files", "--stage", "-v", "-z")
    if index_result.stdout_truncated or index_result.stderr_truncated:
        return _git_observation_limit(
            identity,
            "git_index_output_truncated",
            head_revision=head_revision,
            branch=branch,
        )
    if index_result.exit_code != 0:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.INCOMPLETE,
            head_revision=head_revision,
            branch=branch,
            detail_code="git_index_observation_failed",
        )
    try:
        index_entries = _parse_git_index_entries(index_result.stdout)
    except ValueError:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.INCOMPLETE,
            head_revision=head_revision,
            branch=branch,
            detail_code="git_index_observation_invalid",
        )

    if any(tag == "S" or tag.islower() for _, _, tag in index_entries.values()):
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.INCOMPLETE,
            head_revision=head_revision,
            branch=branch,
            total_paths=len(index_entries),
            detail_code="git_index_visibility_flags_unsupported",
        )

    status_by_path: dict[str, tuple[str, str, str | None]] = {}
    for staged, working_tree, path, renamed_from in records:
        if path in status_by_path:
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.FAILED,
                head_revision=head_revision,
                branch=branch,
                detail_code="git_status_duplicate_path",
            )
        status_by_path[path] = (staged, working_tree, renamed_from)
    observed_paths = sorted(index_entries.keys() | status_by_path.keys())
    if len(observed_paths) > limits.max_paths:
        return _git_observation_limit(
            identity,
            "path_count_limit_exceeded",
            head_revision=head_revision,
            branch=branch,
            total_paths=len(observed_paths),
        )

    paths: list[WorkspacePathRevision] = []
    observed_file_bytes = 0
    for path in observed_paths:
        status_record = status_by_path.get(path)
        staged, working_tree, renamed_from = status_record or (" ", " ", None)
        if not _safe_git_observation_path(path) or (
            renamed_from is not None and not _safe_git_observation_path(renamed_from)
        ):
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.FAILED,
                head_revision=head_revision,
                branch=branch,
                total_paths=len(observed_paths),
                detail_code="unsafe_git_path",
            )
        path_values = (path,) if renamed_from is None else (path, renamed_from)
        if any(len(value.encode("utf-8")) > limits.max_path_bytes for value in path_values):
            return _git_observation_limit(
                identity,
                "path_byte_limit_exceeded",
                head_revision=head_revision,
                branch=branch,
                total_paths=len(observed_paths),
            )

        mode, index_object_id, _index_tag = index_entries.get(path, (None, None, None))
        kind = _git_mode_kind(mode)
        staged_code = None if staged in {" ", "?", "!"} else staged
        working_tree_code = None if working_tree in {" ", "?", "!"} else working_tree
        untracked = staged in {"?", "!"} and working_tree == staged
        ignored = staged == "!" and working_tree == "!"
        content_digest: str | None = None
        present: bool | None = None
        should_read_worktree = status_record is not None and (
            untracked
            or not (working_tree_code == "D" or (staged_code == "D" and working_tree_code is None))
        )
        if kind in {"symlink", "submodule"} and (working_tree_code is not None or untracked):
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.INCOMPLETE,
                head_revision=head_revision,
                branch=branch,
                path_scope="complete",
                total_paths=len(observed_paths),
                detail_code="git_worktree_special_path_unsupported",
            )
        if should_read_worktree and kind not in {"symlink", "submodule"}:
            remaining_file_bytes = limits.max_total_file_bytes - observed_file_bytes
            try:
                read = await workspace.read_bytes(
                    path,
                    # Preserve exact-limit success for trailing empty files;
                    # a one-byte probe still rejects any additional content.
                    max_bytes=min(limits.max_file_bytes, max(1, remaining_file_bytes)),
                )
            except (FileNotFoundError, IsADirectoryError, PermissionError, ValueError):
                return WorkspaceRevisionObservation(
                    identity=identity,
                    status=WorkspaceRevisionObservationStatus.INCOMPLETE,
                    head_revision=head_revision,
                    branch=branch,
                    total_paths=len(observed_paths),
                    detail_code="git_worktree_path_unreadable",
                )
            if read.truncated:
                return _git_observation_limit(
                    identity,
                    (
                        "total_file_byte_limit_exceeded"
                        if remaining_file_bytes < limits.max_file_bytes
                        else "file_byte_limit_exceeded"
                    ),
                    head_revision=head_revision,
                    branch=branch,
                    total_paths=len(observed_paths),
                )
            observed_file_bytes += read.total_bytes
            if observed_file_bytes > limits.max_total_file_bytes:
                return _git_observation_limit(
                    identity,
                    "total_file_byte_limit_exceeded",
                    head_revision=head_revision,
                    branch=branch,
                    total_paths=len(observed_paths),
                )
            content_digest = hashlib.sha256(read.content).hexdigest()
            present = True
        elif status_record is not None and not should_read_worktree:
            present = False
        elif kind in {"symlink", "submodule"}:
            present = staged_code != "D"
        else:
            present = True
        paths.append(
            WorkspacePathRevision(
                path=path,
                staged=staged_code,
                working_tree=working_tree_code,
                untracked=untracked,
                ignored=ignored,
                kind=kind,
                content_sha256=content_digest,
                index_object_id=index_object_id,
                index_mode=mode,
                renamed_from=renamed_from,
                present=present,
                tracked=not untracked,
            )
        )

    paths.sort(key=lambda item: item.path)
    manifest = {
        "head_revision": head_revision,
        "branch": branch,
        "path_scope": "complete",
        "paths": [path.model_dump(mode="json") for path in paths],
    }
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > limits.max_manifest_bytes:
        return _git_observation_limit(
            identity,
            "manifest_byte_limit_exceeded",
            head_revision=head_revision,
            branch=branch,
            total_paths=len(paths),
        )
    return WorkspaceRevisionObservation(
        identity=identity,
        status=WorkspaceRevisionObservationStatus.SUPPORTED,
        revision="sha256:" + hashlib.sha256(encoded).hexdigest(),
        head_revision=head_revision,
        branch=branch,
        path_scope="complete",
        paths=tuple(paths),
        total_paths=len(paths),
    )


def _parse_git_status_records(
    output: str,
) -> list[tuple[str, str, str, str | None]]:
    if "\ufffd" in output:
        raise ValueError("Git status path is not portable UTF-8.")
    tokens = output.split("\x00")
    if tokens and tokens[-1] == "":
        tokens.pop()
    records: list[tuple[str, str, str, str | None]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if len(token) < 4 or token[2] != " ":
            raise ValueError("Git status record is malformed.")
        staged, working_tree, path = token[0], token[1], token[3:]
        if staged not in " MADRCU?!" or working_tree not in " MADRCU?!":
            raise ValueError("Git status record has an unknown state code.")
        renamed_from: str | None = None
        if staged in {"R", "C"} or working_tree in {"R", "C"}:
            index += 1
            if index >= len(tokens):
                raise ValueError("Git rename record is incomplete.")
            renamed_from = tokens[index]
        records.append((staged, working_tree, path, renamed_from))
        index += 1
    return records


def _parse_git_index_entries(output: str) -> dict[str, tuple[str, str, str]]:
    if "\ufffd" in output:
        raise ValueError("Git index path is not portable UTF-8.")
    entries: dict[str, tuple[str, str, str]] = {}
    tokens = output.split("\x00")
    if tokens and tokens[-1] == "":
        tokens.pop()
    for token in tokens:
        header, separator, path = token.partition("\t")
        fields = header.split()
        if not separator or not path or len(fields) != 4:
            raise ValueError("Git index entry is malformed.")
        tag, mode, object_id, stage = fields
        if len(tag) != 1 or not tag.isascii() or not tag.isalpha():
            raise ValueError("Git index entry tag is malformed.")
        if len(mode) != 6 or any(char not in "01234567" for char in mode):
            raise ValueError("Git index entry mode is malformed.")
        if not _valid_git_object_id(object_id):
            raise ValueError("Git index entry object id is malformed.")
        if stage != "0" or path in entries:
            raise ValueError("Git index contains an unsupported unmerged entry.")
        entries[path] = (mode, object_id, tag)
    return entries


def _valid_git_object_id(value: str) -> bool:
    return len(value) in {40, 64} and all(char in "0123456789abcdefABCDEF" for char in value)


def _git_mode_kind(mode: str | None) -> Literal["file", "symlink", "submodule", "unknown"]:
    if mode == "120000":
        return "symlink"
    if mode == "160000":
        return "submodule"
    if mode is not None:
        return "file"
    return "unknown"


def _safe_git_observation_path(path: str) -> bool:
    if type(path) is not str or not path or "\x00" in path:
        return False
    candidate = PurePosixPath(path)
    return not candidate.is_absolute() and ".." not in candidate.parts and str(candidate) == path


def _git_observation_limit(
    identity: WorkspaceIdentity,
    detail_code: str,
    *,
    head_revision: str | None = None,
    branch: str | None = None,
    total_paths: int = 0,
) -> WorkspaceRevisionObservation:
    return WorkspaceRevisionObservation(
        identity=identity,
        status=WorkspaceRevisionObservationStatus.TRUNCATED,
        head_revision=head_revision,
        branch=branch,
        path_scope="changed",
        total_paths=total_paths,
        detail_code=detail_code,
    )


def _git_executor_for_workspace(
    workspace: Workspace,
    *,
    git_executable: str,
    timeout_s: int | None,
    output_limit_bytes: int,
) -> _GitWorkspaceExecutor:
    if isinstance(workspace, LocalWorkspace):
        return _GitWorkspaceExecutor(
            runner=LocalRunner(workspace.root),
            cwd=None,
            git_executable=git_executable,
            timeout_s=timeout_s,
            output_limit_bytes=output_limit_bytes,
        )
    if isinstance(workspace, RunnerWorkspace):
        return _GitWorkspaceExecutor(
            runner=workspace._control_plane_runner(),
            cwd=workspace.cwd,
            git_executable=git_executable,
            timeout_s=timeout_s,
            output_limit_bytes=output_limit_bytes,
        )
    raise TypeError(
        "GitRepositoryBinding requires a LocalWorkspace or RunnerWorkspace. "
        "For E2B, Microsandbox, or Docker runners, wrap the runner with RunnerWorkspace."
    )


async def _require_empty_workspace_for_git_clone(
    workspace: Workspace,
    *,
    timeout_s: int | None,
    output_limit_bytes: int,
) -> None:
    nonempty = False
    if isinstance(workspace, LocalWorkspace):
        nonempty = any(workspace.root.iterdir())
    elif isinstance(workspace, RunnerWorkspace):
        result = await workspace._control_plane_runner().exec(
            ExecCommand.process(
                workspace.python_executable,
                "-c",
                (
                    "import os, sys\n"
                    "with os.scandir('.') as entries:\n"
                    "    sys.exit(10 if any(entries) else 0)\n"
                ),
            ),
            cwd=workspace.cwd,
            timeout_s=timeout_s,
            output_limit_bytes=output_limit_bytes,
        )
        if result.exit_code == 10:
            nonempty = True
        elif result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"GitRepositoryBinding could not inspect workspace: {detail}")
    else:
        existing = await workspace.list("**/*", limit=1)
        nonempty = bool(existing.paths)
    if nonempty:
        raise ValueError(
            "GitRepositoryBinding can only clone into an empty workspace or an existing Git work tree."
        )


async def _reset_workspace_after_failed_clone(
    workspace: Workspace,
    *,
    timeout_s: int | None,
    output_limit_bytes: int,
) -> None:
    """Best-effort removal of a failed clone's partial artifacts, returning the workspace to the
    empty state verified before the clone so a later bind can retry.

    Only ``LocalWorkspace`` and ``RunnerWorkspace`` reach here (``_git_executor_for_workspace``
    rejects other types). Cleanup failures are swallowed so the original clone error is what
    propagates from the caller's ``raise``; a ``CancelledError`` mid-cleanup still propagates.
    """

    try:
        if isinstance(workspace, LocalWorkspace):
            # Off the event loop: removing a large partial clone with shutil.rmtree is blocking.
            await asyncio.to_thread(_remove_local_workspace_contents, workspace.root)
        elif isinstance(workspace, RunnerWorkspace):
            await workspace._control_plane_runner().exec(
                ExecCommand.process(
                    workspace.python_executable,
                    "-c",
                    (
                        "import os, shutil\n"
                        # Materialize before removing: deleting entries while iterating the live
                        # directory can skip siblings.
                        "for entry in list(os.scandir('.')):\n"
                        "    if entry.is_dir(follow_symlinks=False):\n"
                        "        shutil.rmtree(entry.path, ignore_errors=True)\n"
                        "    else:\n"
                        "        os.remove(entry.path)\n"
                    ),
                ),
                cwd=workspace.cwd,
                timeout_s=timeout_s,
                output_limit_bytes=output_limit_bytes,
            )
    except Exception:
        # Swallow cleanup errors so the original clone error surfaces, but let a CancelledError
        # (or other BaseException) propagate rather than dropping a requested cancellation.
        pass


def _remove_local_workspace_contents(root: Path) -> None:
    # Materialize before removing: iterdir() is lazy, and deleting entries while iterating the live
    # directory skips siblings.
    for entry in list(root.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def _validate_git_repo_url(value: str) -> str:
    repo_url = _validate_git_value(value, "repo_url")
    parsed = urlsplit(repo_url)
    if parsed.scheme in {"http", "https"} and (parsed.username or parsed.password):
        raise ValueError(
            "GitRepositoryBinding repo_url must not contain embedded credentials because "
            "the URL is stored in durable binding metadata."
        )
    return repo_url


def _validate_git_value(value: str, field_name: str) -> str:
    checked = require_clean_nonblank(value, field_name)
    if checked.startswith("-"):
        raise ValueError(f"GitRepositoryBinding {field_name} must not start with '-'.")
    return checked


def _raise_git_error(args: tuple[str, ...], result) -> None:
    detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
    raise RuntimeError(
        f"Git command failed with exit code {result.exit_code}: {_git_command_label(args)}: {detail}"
    )


def _git_command_label(args: tuple[str, ...]) -> str:
    return "git " + " ".join(args)


def _reject_reserved_metadata(metadata: dict[str, Any], key: str) -> None:
    if key in metadata:
        raise ValueError(f"GitRepositoryBinding metadata key {key!r} is reserved.")


def _runtime_owned_workspace_observer_name(binding: object) -> str | None:
    """Return an observer name only for exact built-in binding implementations.

    Extension subclasses deliberately do not inherit this authority: equality
    with a built-in class name is not evidence that the runtime produced it.
    Private runtime wrappers are likewise treated as extension-shaped unless
    their observation identity is projected through a separate owned boundary.
    """

    if type(binding) not in {
        NativeBinding,
        DeterministicWorkspaceBinding,
        NoWorkspaceBinding,
        GitRepositoryBinding,
        SyncBinding,
    }:
        return None
    return type(binding).__name__
