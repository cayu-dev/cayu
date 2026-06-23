"""Workspace binding contracts for bridging storage and compute."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.runners import Runner
from cayu.workspaces import Workspace


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
    when the binding backend can provide one.
    """

    workspace: Workspace | None = None
    runner: Runner | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot: WorkspaceSnapshot | None = None

    def __post_init__(self) -> None:
        if self.workspace is not None and not isinstance(self.workspace, Workspace):
            raise TypeError("BoundWorkspace workspace must be a Workspace or None.")
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


def copy_bound_workspace(bound: BoundWorkspace) -> BoundWorkspace:
    """Return a defensive copy of binding result metadata."""

    if type(bound) is not BoundWorkspace:
        raise TypeError("Bound workspace copies require a BoundWorkspace.")
    return BoundWorkspace(
        workspace=bound.workspace,
        runner=bound.runner,
        path=bound.path,
        metadata=copy_json_value(bound.metadata, "metadata"),
        snapshot=copy_workspace_snapshot(bound.snapshot),
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
