"""Workspace contracts."""

from cayu.workspaces.base import Workspace, WorkspaceListResult, WorkspaceReadResult
from cayu.workspaces.local import LocalWorkspace

__all__ = [
    "LocalWorkspace",
    "Workspace",
    "WorkspaceListResult",
    "WorkspaceReadResult",
]
