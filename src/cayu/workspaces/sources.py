"""Repositories an application registers so a workspace can be prepared at one revision.

An eval case names a source and a full commit id in its workspace fixture; the
eval runner asks the source to check that revision out into a fresh directory
for every trial. :class:`GitWorkspaceSource` checks revisions out of a local Git
repository (a working checkout or a bare mirror the application keeps current).
An application whose repository lives behind a remote with its own credentials
subclasses :class:`WorkspaceSource` and fetches before it checks out.
"""

from __future__ import annotations

import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from cayu._validation import require_durable_clean_nonblank
from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.local import LocalRunner

#: How long one Git command of a checkout may run, in seconds, unless the source says otherwise.
DEFAULT_GIT_WORKSPACE_SOURCE_TIMEOUT_SECONDS = 300
# Every command here prints at most a commit id; the rest is error text.
_GIT_OUTPUT_LIMIT_BYTES = 4_096
_SOURCE_NAME_MAX_CHARS = 128
_SOURCE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z", re.ASCII)
_COMMIT_ID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)


def _source_name(value: object) -> str:
    if type(value) is not str or len(value) > _SOURCE_NAME_MAX_CHARS:
        raise ValueError(
            f"A workspace source name is a string of at most {_SOURCE_NAME_MAX_CHARS} characters."
        )
    name = require_durable_clean_nonblank(value, "name")
    if _SOURCE_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            "A workspace source name starts with a lowercase ASCII letter and contains only "
            "lowercase ASCII letters, digits, '.', '_', or '-'."
        )
    return name


class WorkspaceSource(ABC):
    """A named repository that can check one revision out into an empty directory."""

    def __init__(self, name: str) -> None:
        self.name = _source_name(name)

    @abstractmethod
    async def checkout(self, revision: str, destination: Path) -> None:
        """Write the tree of ``revision`` (a full commit id) into the empty ``destination``.

        Raise when the revision is unknown or the tree can't be written; the
        caller removes ``destination`` either way.
        """


class GitWorkspaceSource(WorkspaceSource):
    """Check revisions out of a local Git repository without changing it.

    ``repository`` is a working checkout or a bare repository that already
    holds the commits cases name; fetching new ones is the application's job.
    A checkout reads the revision's tree into a private index and writes it to
    the destination with the repository's own attributes and filters, as
    ``git checkout`` would. The repository's HEAD, index and working tree are
    never touched, and the destination gets no ``.git`` of its own. Each Git
    command may run for ``timeout_s`` seconds.
    """

    def __init__(
        self,
        name: str,
        repository: str | Path,
        *,
        timeout_s: int = DEFAULT_GIT_WORKSPACE_SOURCE_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(name)
        if not isinstance(repository, str | Path):
            raise TypeError("repository must be a path.")
        root = Path(repository).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Git workspace source {self.name!r} has no directory {root}.")
        if type(timeout_s) is not int or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive integer.")
        self.repository = root
        self.timeout_s = timeout_s

    async def checkout(self, revision: str, destination: Path) -> None:
        if type(revision) is not str or _COMMIT_ID_PATTERN.fullmatch(revision) is None:
            raise ValueError("revision must be a full lowercase commit id.")
        target = Path(destination)
        if not target.is_dir() or any(target.iterdir()):
            raise ValueError("A checkout destination must be an existing empty directory.")
        runner = LocalRunner(self.repository)
        found = await self._git(
            runner, "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"
        )
        # --quiet --verify exits 1 without a message for a commit it doesn't have.
        if found.exit_code == 1 and not found.stderr.strip():
            raise LookupError(
                f"Git workspace source {self.name!r} has no commit {revision}; "
                "fetch it into the repository first."
            )
        self._require_success("rev-parse", found)
        if found.stdout.strip() != revision:
            # A tag object peels to the commit it points at; the fixture must name that commit.
            raise LookupError(
                f"{revision} in Git workspace source {self.name!r} is not a commit; "
                f"name the commit {found.stdout.strip()} instead."
            )
        with tempfile.TemporaryDirectory(prefix="cayu-workspace-index-") as index_directory:
            index = {"GIT_INDEX_FILE": str(Path(index_directory) / "index")}
            self._require_success(
                "read-tree", await self._git(runner, "read-tree", revision, env=index)
            )
            self._require_success(
                "checkout-index",
                await self._git(
                    runner, "checkout-index", "--all", work_tree=target.resolve(), env=index
                ),
            )

    async def _git(
        self,
        runner: LocalRunner,
        command: str,
        *arguments: str,
        work_tree: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        options = () if work_tree is None else (f"--work-tree={work_tree}",)
        result = await runner.exec(
            ExecCommand.process("git", *options, command, *arguments),
            env={"GIT_TERMINAL_PROMPT": "0", **(env or {})},
            timeout_s=self.timeout_s,
            output_limit_bytes=_GIT_OUTPUT_LIMIT_BYTES,
        )
        if result.timed_out:
            raise TimeoutError(
                f"git {command} in workspace source {self.name!r} took longer than "
                f"{self.timeout_s} seconds."
            )
        return result

    def _require_success(self, command: str, result: ExecResult) -> None:
        if result.exit_code != 0:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            raise RuntimeError(f"git {command} in workspace source {self.name!r} failed: {detail}")


__all__ = [
    "DEFAULT_GIT_WORKSPACE_SOURCE_TIMEOUT_SECONDS",
    "GitWorkspaceSource",
    "WorkspaceSource",
]
