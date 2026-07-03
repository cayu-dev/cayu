"""Workspace binding contracts for bridging storage and compute."""

from __future__ import annotations

import inspect
import io
import tarfile
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import urlsplit
from uuid import uuid4

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.runners import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, ExecCommand, LocalRunner, Runner
from cayu.workspaces import LocalWorkspace, RunnerWorkspace, Workspace


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
        object.__setattr__(self, "metadata", copy_json_value(self.metadata, "metadata"))


@dataclass(frozen=True)
class _SyncBindingState:
    session_id: str
    created_at: float
    source_paths: tuple[str, ...]
    target_baseline_paths: tuple[str, ...]


DEFAULT_SYNC_STATE_TTL_S = 24 * 60 * 60.0


SYNC_FINAL_METADATA_KEYS = frozenset(
    {
        "target_workspace_id",
        "outcome",
        "copied_files",
        "copied_bytes",
        "deleted_files",
    }
)


SyncTargetWorkspaceFactory = Callable[
    [SyncBindingContext],
    Workspace | Awaitable[Workspace],
]
SyncTargetCleanPolicy = Literal["always", "never"]
SyncBackPolicy = Literal["always", "on_success", "never"]

GIT_REPOSITORY_METADATA_KEY = "git_repository"


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
        object.__setattr__(self, "metadata", copy_json_value(self.metadata, "metadata"))


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
        object.__setattr__(self, "metadata", copy_json_value(self.metadata, "metadata"))
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
        require_clean: bool = True,
        verify_remote_url: bool = True,
        timeout_s: int | None = 120,
        output_limit_bytes: int = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
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
        self.require_clean = require_clean
        self.verify_remote_url = verify_remote_url
        self.timeout_s = _validate_optional_timeout(timeout_s, "timeout_s")
        self.output_limit_bytes = _validate_positive_int(
            output_limit_bytes,
            "output_limit_bytes",
            owner="GitRepositoryBinding",
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
            await executor.run("clone", self.repo_url, ".")

        if self.ref is not None:
            await self._checkout_configured_ref(executor)
        commit = await executor.stdout("rev-parse", "HEAD")
        branch = await executor.stdout("rev-parse", "--abbrev-ref", "HEAD")
        dirty = await executor.is_dirty()
        git_metadata = {
            "repo_url": self.repo_url,
            "remote_name": self.remote_name,
            "ref": self.ref,
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "fetch": self.fetch,
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
                snapshot_id=f"git-bind:{session_id}:{commit[:12]}",
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
        commit = await executor.stdout("rev-parse", "HEAD")
        branch = await executor.stdout("rev-parse", "--abbrev-ref", "HEAD")
        dirty = await executor.is_dirty()
        git_metadata = {
            **copy_json_value(bind_metadata, GIT_REPOSITORY_METADATA_KEY),
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "outcome": outcome,
        }
        return WorkspaceSnapshot(
            snapshot_id=f"git-final:{bound.workspace.id}:{commit[:12]}:{outcome or 'unknown'}",
            workspace_id=bound.workspace.id,
            version=commit,
            source="git",
            metadata={
                **finalize_metadata,
                GIT_REPOSITORY_METADATA_KEY: git_metadata,
            },
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

    async def _checkout_configured_ref(self, executor: _GitWorkspaceExecutor) -> None:
        if self.ref is None:
            return
        await executor.run("checkout", self.ref)
        fetched_ref = f"refs/remotes/{self.remote_name}/{self.ref}"
        if self.fetch and await executor.ref_exists(fetched_ref):
            await executor.run("merge", "--ff-only", fetched_ref)


class SyncBinding(WorkspaceBinding):
    """Copy a durable workspace into a bound workspace and sync changes back.

    ``workspace`` passed to ``bind`` is the durable source. ``target_workspace``
    or ``target_workspace_factory`` identifies the workspace visible to tools
    during the run, typically a sandbox filesystem wrapper. The target workspace
    should be dedicated to this binding because the default clean policy deletes
    files in the target before copying source files in.

    File copies use one bulk tar transfer per direction when a workspace
    exposes ``read_tar_bytes``/``write_tar_bytes`` (RunnerWorkspace does), and
    fall back to per-file copies otherwise. Per-bind state is keyed by session:
    rebinding a session replaces its leaked state, ``abandon`` drops state for
    a bind whose finalize will never run, and states older than
    ``state_ttl_s`` are pruned on the next bind.
    """

    def __init__(
        self,
        *,
        target_workspace: Workspace | None = None,
        target_workspace_factory: SyncTargetWorkspaceFactory | None = None,
        path: str | None = None,
        pattern: str = "**/*",
        max_files: int = 10_000,
        max_file_bytes: int | None = None,
        clean_target: SyncTargetCleanPolicy = "always",
        sync_back: SyncBackPolicy = "always",
        delete_missing: bool = True,
        state_ttl_s: float | None = DEFAULT_SYNC_STATE_TTL_S,
    ) -> None:
        if target_workspace is not None and not isinstance(target_workspace, Workspace):
            raise TypeError("SyncBinding target_workspace must be a Workspace or None.")
        if target_workspace_factory is not None and not callable(target_workspace_factory):
            raise TypeError("SyncBinding target_workspace_factory must be callable or None.")
        if target_workspace is not None and target_workspace_factory is not None:
            raise ValueError(
                "SyncBinding accepts either target_workspace or target_workspace_factory, not both."
            )
        if path is not None:
            require_clean_nonblank(path, "path")
        self.target_workspace = target_workspace
        self.target_workspace_factory = target_workspace_factory
        self.path = path
        self.pattern = require_clean_nonblank(pattern, "pattern")
        self.max_files = _validate_positive_int(max_files, "max_files")
        self.max_file_bytes = _validate_optional_positive_int(max_file_bytes, "max_file_bytes")
        self.clean_target = _validate_clean_policy(clean_target)
        self.sync_back = _validate_sync_back_policy(sync_back)
        if type(delete_missing) is not bool:
            raise TypeError("SyncBinding delete_missing must be a bool.")
        self.delete_missing = delete_missing
        self.state_ttl_s = _validate_optional_positive_number(state_ttl_s, "state_ttl_s")
        self._states: dict[str, _SyncBindingState] = {}

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
            raise ValueError("SyncBinding requires a source workspace.")
        if "sync_binding" in request_metadata:
            raise ValueError("SyncBinding metadata key 'sync_binding' is reserved.")
        self._prune_sync_states(session_id=session_id)
        context = SyncBindingContext(
            source_workspace=workspace,
            runner=runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=request_metadata,
        )
        target = await self._target_workspace(context)
        if _same_workspace_resource(workspace, target):
            raise ValueError("SyncBinding source and target workspaces must be different.")
        source_paths = await _list_workspace_paths(
            workspace,
            self.pattern,
            limit=self.max_files,
            role="source",
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
        )
        bind_metadata = {
            **request_metadata,
            "sync_binding": {
                "source_workspace_id": workspace.id,
                "target_workspace_id": target.id,
                "pattern": self.pattern,
                "max_files": self.max_files,
                "max_file_bytes": self.max_file_bytes,
                "clean_target": self.clean_target,
                "sync_back": self.sync_back,
                "delete_missing": self.delete_missing,
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
            state_key=uuid4().hex,
        )
        if bound.state_key is None:
            raise RuntimeError("SyncBinding bound workspace missing state key.")
        self._states[bound.state_key] = _SyncBindingState(
            session_id=session_id,
            created_at=time.monotonic(),
            source_paths=source_paths,
            target_baseline_paths=target_baseline_paths,
        )
        return bound

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
            self._discard_sync_state(bound)
            return None
        if bound.source_workspace is None:
            raise ValueError("SyncBinding finalize requires a source workspace.")
        if bound.workspace is None:
            raise ValueError("SyncBinding finalize requires a bound workspace.")
        state = self._get_sync_state(bound)
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
        copied_bytes = await _copy_paths(
            source=bound.workspace,
            target=bound.source_workspace,
            paths=copy_back_paths,
            max_file_bytes=self.max_file_bytes,
        )
        deleted_paths: tuple[str, ...] = ()
        if self.delete_missing:
            deleted_paths = tuple(sorted(set(state.source_paths) - set(target_paths)))
            for path in deleted_paths:
                await bound.source_workspace.delete(path)
        self._discard_sync_state(bound)
        return WorkspaceSnapshot(
            snapshot_id=_final_sync_snapshot_id(bound, outcome),
            workspace_id=bound.source_workspace.id,
            source="sync",
            metadata={
                **finalize_metadata,
                "target_workspace_id": bound.workspace.id,
                "outcome": outcome,
                "copied_files": len(copy_back_paths),
                "copied_bytes": copied_bytes,
                "deleted_files": len(deleted_paths),
            },
        )

    async def _target_workspace(
        self,
        context: SyncBindingContext,
    ) -> Workspace:
        if self.target_workspace is not None:
            return self.target_workspace
        if self.target_workspace_factory is None:
            raise ValueError("SyncBinding requires target_workspace or target_workspace_factory.")
        result = self.target_workspace_factory(context)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Workspace):
            raise TypeError("SyncBinding target workspace factory must return a Workspace.")
        return result

    def abandon(self, bound: BoundWorkspace) -> None:
        """Drop in-process bind state for a bind whose finalize will never run.

        Lifecycle owners that skip ``finalize`` (crash recovery, cancelled
        sessions) should call this so per-bind state does not leak until the
        TTL prune catches it.
        """

        if type(bound) is not BoundWorkspace:
            raise TypeError("SyncBinding abandon requires a BoundWorkspace.")
        self._discard_sync_state(bound)

    def _prune_sync_states(self, *, session_id: str) -> None:
        now = time.monotonic()
        stale_keys = [
            key
            for key, state in self._states.items()
            if state.session_id == session_id
            or (self.state_ttl_s is not None and now - state.created_at > self.state_ttl_s)
        ]
        for key in stale_keys:
            del self._states[key]

    def _get_sync_state(self, bound: BoundWorkspace) -> _SyncBindingState:
        if bound.state_key is not None:
            state = self._states.get(bound.state_key)
            if state is not None:
                return state
        raise ValueError(
            "SyncBinding finalize requires in-process bind state. "
            "Use a custom WorkspaceBinding when sync finalization must survive process restart."
        )

    def _discard_sync_state(self, bound: BoundWorkspace) -> None:
        if bound.state_key is not None:
            self._states.pop(bound.state_key, None)


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


def _same_workspace_resource(source: Workspace, target: Workspace) -> bool:
    if target is source or target.id == source.id:
        return True

    source_root = getattr(source, "root", None)
    target_root = getattr(target, "root", None)
    if source_root is not None and target_root is not None:
        source_runner = getattr(source, "runner", None)
        target_runner = getattr(target, "runner", None)
        return _runner_resource_key(source_runner) == _runner_resource_key(target_runner) and str(
            source_root
        ) == str(target_root)

    if (
        hasattr(source, "runner")
        and hasattr(target, "runner")
        and hasattr(source, "cwd")
        and hasattr(target, "cwd")
    ):
        source_cwd = getattr(source, "cwd", None)
        target_cwd = getattr(target, "cwd", None)
        source_runner = getattr(source, "runner", None)
        target_runner = getattr(target, "runner", None)
        return _runner_resource_key(source_runner) == _runner_resource_key(target_runner) and (
            source_cwd or "."
        ) == (target_cwd or ".")

    return False


def _runner_resource_key(runner: Any) -> tuple[Any, ...]:
    if runner is None:
        return (None,)
    for attr in ("sandbox_id", "name", "container_name", "sandbox_name", "root"):
        value = getattr(runner, attr, None)
        if value is not None:
            return (type(runner), attr, str(value))
    return (type(runner), "object", id(runner))


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


def _validate_optional_positive_number(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise TypeError(f"SyncBinding {field_name} must be a number or None.")
    if value <= 0:
        raise ValueError(f"SyncBinding {field_name} must be greater than zero.")
    return float(value)


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


async def _list_workspace_paths(
    workspace: Workspace,
    pattern: str,
    *,
    limit: int,
    role: str,
) -> tuple[str, ...]:
    result = await workspace.list(pattern, limit=limit)
    if result.truncated:
        raise RuntimeError(f"SyncBinding {role} workspace file list exceeded max_files={limit}.")
    return tuple(result.paths)


async def _clear_workspace(workspace: Workspace, *, max_files: int) -> tuple[str, ...]:
    paths = await _list_workspace_paths(workspace, "**/*", limit=max_files, role="target")
    for path in paths:
        await workspace.delete(path)
    return paths


async def _copy_paths(
    *,
    source: Workspace,
    target: Workspace,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
) -> int:
    """Copy files between workspaces, preferring one bulk tar transfer.

    When either side exposes the optional ``read_tar_bytes``/``write_tar_bytes``
    capability (RunnerWorkspace does), the whole file set moves as a single
    tar so a runner-backed workspace costs O(1) execs instead of one exec per
    file. Workspaces without the capability fall back to per-file copies.
    """

    if not paths:
        return 0
    read_tar = getattr(source, "read_tar_bytes", None)
    write_tar = getattr(target, "write_tar_bytes", None)
    if not callable(read_tar) and not callable(write_tar):
        return await _copy_paths_per_file(
            source=source,
            target=target,
            paths=paths,
            max_file_bytes=max_file_bytes,
        )
    if callable(read_tar):
        tar_data = await read_tar(paths, max_file_bytes=max_file_bytes)
    else:
        tar_data = await _pack_workspace_tar(source, paths, max_file_bytes=max_file_bytes)
    copied_bytes = _validate_sync_tar(tar_data, paths, max_file_bytes=max_file_bytes)
    if callable(write_tar):
        await write_tar(tar_data)
    else:
        await _extract_tar_to_workspace(target, tar_data)
    return copied_bytes


async def _copy_paths_per_file(
    *,
    source: Workspace,
    target: Workspace,
    paths: tuple[str, ...],
    max_file_bytes: int | None,
) -> int:
    copied_bytes = 0
    for path in paths:
        result = await source.read_bytes(path, max_bytes=max_file_bytes)
        if result.truncated:
            raise RuntimeError(
                f"SyncBinding file exceeds {_copy_limit_label(max_file_bytes)}: {path}"
            )
        await target.write_bytes(path, result.content)
        copied_bytes += len(result.content)
    return copied_bytes


async def _pack_workspace_tar(
    source: Workspace,
    paths: tuple[str, ...],
    *,
    max_file_bytes: int | None,
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in paths:
            result = await source.read_bytes(path, max_bytes=max_file_bytes)
            if result.truncated:
                raise RuntimeError(
                    f"SyncBinding file exceeds {_copy_limit_label(max_file_bytes)}: {path}"
                )
            info = tarfile.TarInfo(name=path)
            info.size = len(result.content)
            archive.addfile(info, io.BytesIO(result.content))
    return buffer.getvalue()


async def _extract_tar_to_workspace(target: Workspace, tar_data: bytes) -> None:
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as archive:
        for member in archive.getmembers():
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"SyncBinding tar member could not be read: {member.name}")
            await target.write_bytes(member.name, extracted.read())


def _validate_sync_tar(
    tar_data: bytes,
    paths: tuple[str, ...],
    *,
    max_file_bytes: int | None,
) -> int:
    if type(tar_data) is not bytes:
        raise TypeError("SyncBinding bulk transfer must produce tar bytes.")
    copied_bytes = 0
    member_names: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as archive:
            for member in archive.getmembers():
                if not member.isreg():
                    raise RuntimeError(
                        f"SyncBinding tar member must be a regular file: {member.name}"
                    )
                if max_file_bytes is not None and member.size > max_file_bytes:
                    raise RuntimeError(
                        f"SyncBinding file exceeds max_file_bytes={max_file_bytes}: {member.name}"
                    )
                member_names.append(member.name)
                copied_bytes += member.size
    except tarfile.TarError as exc:
        raise RuntimeError("SyncBinding bulk transfer returned an invalid tar archive.") from exc
    if sorted(member_names) != sorted(paths):
        raise RuntimeError("SyncBinding bulk transfer paths do not match the requested files.")
    return copied_bytes


def _copy_limit_label(max_file_bytes: int | None) -> str:
    if max_file_bytes is None:
        return "the workspace read limit"
    return f"max_file_bytes={max_file_bytes}"


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

    async def _exec(self, *args: str):
        return await self.runner.exec(
            ExecCommand.process(self.git_executable, *args),
            cwd=self.cwd,
            timeout_s=self.timeout_s,
            output_limit_bytes=self.output_limit_bytes,
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
            runner=workspace.runner,
            cwd=workspace.cwd,
            git_executable=git_executable,
            timeout_s=timeout_s,
            output_limit_bytes=output_limit_bytes,
        )
    raise TypeError(
        "GitRepositoryBinding requires a LocalWorkspace or RunnerWorkspace. "
        "For E2B, Microsandbox, Docker, or SBX runners, wrap the runner with RunnerWorkspace."
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
        result = await workspace.runner.exec(
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
