"""Which registered workspace source, at which commit, a session's source workspace was.

A coding environment factory given ``workspace_repository`` records, as each
session binds, the commit its source workspace (a Git checkout of that
registered source) was at, and whether the checkout had uncommitted changes.
``cayu eval add-session`` turns that into the case's workspace fixture.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, field_validator

from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.local import LocalRunner
from cayu.workspaces.base import Workspace
from cayu.workspaces.local import LocalWorkspace
from cayu.workspaces.sources import _require_commit_id, _source_name

logger = logging.getLogger(__name__)

#: The bound-metadata key a coding binding records its source's origin under.
WORKSPACE_ORIGIN_METADATA_KEY = "workspace_origin"
# Each command reads a commit id, a path or whether anything changed.
_GIT_TIMEOUT_SECONDS = 60
_GIT_OUTPUT_LIMIT_BYTES = 4_096
_GLOB_SPECIAL = re.compile(r"([*?\[\]\\])")


class WorkspaceOrigin(BaseModel):
    """A registered source's commit a source workspace was a checkout of when a session bound.

    ``clean`` is true only when Git confirmed that the checkout had no
    uncommitted change outside the paths the environment excludes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: StrictStr
    revision: StrictStr
    clean: StrictBool

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        return _source_name(value)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return _require_commit_id(value)


async def _git(runner: LocalRunner, *arguments: str) -> ExecResult:
    return await runner.exec(
        ExecCommand.process("git", "-c", "core.fsmonitor=false", *arguments),
        env={"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"},
        timeout_s=_GIT_TIMEOUT_SECONDS,
        output_limit_bytes=_GIT_OUTPUT_LIMIT_BYTES,
    )


def _excluded_directory_pathspecs(names: tuple[str, ...]) -> list[str]:
    # The environment copies none of these in, so a change under them changes nothing it ran on.
    escaped = (_GLOB_SPECIAL.sub(r"\\\1", name) for name in names)
    return [f":(exclude,glob,icase)**/{name}/**" for name in escaped]


async def observe_workspace_origin(
    repository: str, workspace: LocalWorkspace
) -> WorkspaceOrigin | None:
    """The commit ``workspace`` is checked out at, or ``None`` when Git can't name one.

    ``None`` when the workspace's root isn't the top of a Git checkout, its
    HEAD has no commit, or Git fails. The checkout's index and files are only
    read.
    """

    root = Path(workspace.root).resolve()
    try:
        runner = LocalRunner(root)
        top = await _git(runner, "rev-parse", "--show-toplevel")
        if top.exit_code != 0 or top.timed_out or Path(top.stdout.strip()).resolve() != root:
            return None
        head = await _git(runner, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
        if head.exit_code != 0 or head.timed_out:
            return None
        status = await _git(
            runner,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=normal",
            "--",
            ".",
            *_excluded_directory_pathspecs(workspace.excluded_directory_names),
        )
        clean = (
            status.exit_code == 0
            and not status.timed_out
            and not status.stdout_truncated
            and not status.stdout
        )
        return WorkspaceOrigin(repository=repository, revision=head.stdout.strip(), clean=clean)
    except Exception:
        # The origin is evidence for later curation; reading it never fails the session.
        logger.warning(
            "Reading the Git revision of workspace source %r failed.", repository, exc_info=True
        )
        return None


async def metadata_with_workspace_origin(
    repository: str | None,
    workspace: Workspace | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """``metadata`` with ``workspace``'s origin added, when ``repository`` names its source."""

    if repository is None or not isinstance(workspace, LocalWorkspace):
        return metadata
    if metadata is not None and WORKSPACE_ORIGIN_METADATA_KEY in metadata:
        raise ValueError(f"Binding metadata key {WORKSPACE_ORIGIN_METADATA_KEY!r} is reserved.")
    origin = await observe_workspace_origin(repository, workspace)
    if origin is None:
        return metadata
    return {**(metadata or {}), WORKSPACE_ORIGIN_METADATA_KEY: origin.model_dump(mode="json")}


__all__ = [
    "WORKSPACE_ORIGIN_METADATA_KEY",
    "WorkspaceOrigin",
    "metadata_with_workspace_origin",
    "observe_workspace_origin",
]
