"""Workspace binding contracts for bridging storage and compute."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from uuid import uuid4

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.runners import Runner
from cayu.workspaces import Workspace


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
    source_paths: tuple[str, ...]
    target_baseline_paths: tuple[str, ...]


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


class SyncBinding(WorkspaceBinding):
    """Copy a durable workspace into a bound workspace and sync changes back.

    ``workspace`` passed to ``bind`` is the durable source. ``target_workspace``
    or ``target_workspace_factory`` identifies the workspace visible to tools
    during the run, typically a sandbox filesystem wrapper. The target workspace
    should be dedicated to this binding because the default clean policy deletes
    files in the target before copying source files in.
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


def _validate_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"SyncBinding {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"SyncBinding {field_name} must be greater than zero.")
    return value


def _validate_optional_positive_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, field_name)


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
    copied_bytes = 0
    for path in paths:
        result = await source.read_bytes(path, max_bytes=max_file_bytes)
        if result.truncated:
            limit = (
                "the workspace read limit"
                if max_file_bytes is None
                else f"max_file_bytes={max_file_bytes}"
            )
            raise RuntimeError(f"SyncBinding file exceeds {limit}: {path}")
        await target.write_bytes(path, result.content)
        copied_bytes += len(result.content)
    return copied_bytes


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
