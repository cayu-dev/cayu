"""Workspace contracts."""

from cayu.workspaces.base import Workspace
from cayu.workspaces.local import LocalWorkspace

__all__ = ["LocalWorkspace", "Workspace"]
