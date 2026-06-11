"""Workspace contracts."""

from cayu.workspaces.base import Workspace, WorkspaceListResult, WorkspaceReadResult
from cayu.workspaces.e2b import (
    DEFAULT_E2B_WORKSPACE_LIST_DEPTH,
    DEFAULT_E2B_WORKSPACE_LIST_LIMIT,
    DEFAULT_E2B_WORKSPACE_READ_LIMIT_BYTES,
    E2BWorkspace,
)
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
    "DEFAULT_E2B_WORKSPACE_LIST_DEPTH",
    "DEFAULT_E2B_WORKSPACE_LIST_LIMIT",
    "DEFAULT_E2B_WORKSPACE_READ_LIMIT_BYTES",
    "DEFAULT_MICROSANDBOX_WORKSPACE_LIST_LIMIT",
    "DEFAULT_MICROSANDBOX_WORKSPACE_READ_LIMIT_BYTES",
    "DEFAULT_RUNNER_WORKSPACE_LIST_LIMIT",
    "DEFAULT_RUNNER_WORKSPACE_READ_LIMIT_BYTES",
    "E2BWorkspace",
    "LocalWorkspace",
    "MicrosandboxWorkspace",
    "RunnerWorkspace",
    "Workspace",
    "WorkspaceListResult",
    "WorkspaceReadResult",
]
