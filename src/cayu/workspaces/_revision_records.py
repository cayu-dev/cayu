"""Workspace path records and their shared event-schema fields."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator

from cayu._validation import require_durable_clean_nonblank


class WorkspacePathRevision(BaseModel):
    """Content-free Git/workspace state for one bounded relative path."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    path: str
    staged: str | None = None
    working_tree: str | None = None
    untracked: StrictBool = False
    ignored: StrictBool = False
    present: StrictBool | None = None
    tracked: StrictBool | None = None
    kind: Literal["file", "symlink", "submodule", "unknown"] = "unknown"
    content_sha256: str | None = None
    index_object_id: str | None = None
    index_mode: str | None = None
    worktree_mode: str | None = None
    renamed_from: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = require_durable_clean_nonblank(value, "path")
        if not _safe_relative_path(path):
            raise ValueError("Workspace revision path must be relative and traversal-free.")
        if str(PurePosixPath(path)) != path:
            raise ValueError("Workspace revision path must use canonical POSIX spelling.")
        return path

    @field_validator(
        "staged",
        "working_tree",
        "content_sha256",
        "index_object_id",
        "index_mode",
        "worktree_mode",
        "renamed_from",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        text = require_durable_clean_nonblank(value, info.field_name)
        if info.field_name == "renamed_from" and not _safe_relative_path(text):
            raise ValueError("Workspace revision rename source must be traversal-free.")
        if info.field_name == "renamed_from" and str(PurePosixPath(text)) != text:
            raise ValueError("Workspace revision rename source must use canonical POSIX spelling.")
        if info.field_name in {"index_mode", "worktree_mode"} and (
            len(text) != 6 or any(char not in "01234567" for char in text)
        ):
            raise ValueError(
                f"Workspace revision {info.field_name.replace('_', ' ')} must be a "
                "six-digit octal mode."
            )
        return text


_WORKSPACE_PATH_REVISION_FIELDS = frozenset(WorkspacePathRevision.model_fields)


_WORKSPACE_PATH_REVISION_AUTHORITY_FIELDS = frozenset(
    {
        "content_sha256",
        "index_mode",
        "worktree_mode",
        "index_object_id",
        "kind",
        "staged",
        "working_tree",
    }
)


class WorkspacePathRevisionDelta(BaseModel):
    """One content-free path change between two observations."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    path: str
    change: Literal["added", "modified", "deleted", "renamed"]
    renamed_from: str | None = None

    @field_validator("path", "renamed_from")
    @classmethod
    def validate_path_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        path = require_durable_clean_nonblank(value, info.field_name)
        if not _safe_relative_path(path):
            raise ValueError("Workspace revision delta path must be traversal-free.")
        return path


_WORKSPACE_PATH_REVISION_DELTA_FIELDS = frozenset(WorkspacePathRevisionDelta.model_fields)


_WORKSPACE_PATH_REVISION_DELTA_AUTHORITY_FIELDS = frozenset({"change"})


def _safe_relative_path(path: str) -> bool:
    if type(path) is not str or not path or "\x00" in path:
        return False
    candidate = PurePosixPath(path)
    return not candidate.is_absolute() and ".." not in candidate.parts and str(candidate) == path
