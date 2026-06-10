"""Workspace contracts."""

from cayu.workspaces.base import Workspace, WorkspaceListResult, WorkspaceReadResult
from cayu.workspaces.local import LocalWorkspace
from cayu.workspaces.microsandbox import (
    DEFAULT_MICROSANDBOX_WORKSPACE_LIST_LIMIT,
    DEFAULT_MICROSANDBOX_WORKSPACE_READ_LIMIT_BYTES,
    MicrosandboxWorkspace,
)
from cayu.workspaces.runner import (
    DEFAULT_RUNNER_WORKSPACE_LIST_LIMIT,
    DEFAULT_RUNNER_WORKSPACE_READ_LIMIT_BYTES,
    RunnerWorkspace,
)

__all__ = [
    "DEFAULT_MICROSANDBOX_WORKSPACE_LIST_LIMIT",
    "DEFAULT_MICROSANDBOX_WORKSPACE_READ_LIMIT_BYTES",
    "DEFAULT_RUNNER_WORKSPACE_LIST_LIMIT",
    "DEFAULT_RUNNER_WORKSPACE_READ_LIMIT_BYTES",
    "LocalWorkspace",
    "MicrosandboxWorkspace",
    "RunnerWorkspace",
    "Workspace",
    "WorkspaceListResult",
    "WorkspaceReadResult",
]
