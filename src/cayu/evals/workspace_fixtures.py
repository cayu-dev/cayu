"""Workspace fixtures: the repository revision and files each trial of an eval case starts from."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.evals._structural_paths import _validate_portable_structural_workspace_path

#: Most files one fixture writes over its revision.
EVAL_WORKSPACE_FIXTURE_MAX_FILES = 256
#: Most UTF-8 bytes of file content one fixture writes.
EVAL_WORKSPACE_FIXTURE_MAX_BYTES = 1 << 20
#: Longest fixture file path, in characters.
EVAL_WORKSPACE_FIXTURE_MAX_PATH_CHARS = 1_024
_REPOSITORY_MAX_CHARS = 128

_REPOSITORY_PATTERN = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z", re.ASCII)
# A full commit id: SHA-1 or SHA-256 object format. Abbreviations and ref names
# are refused, so the same revision always names the same tree.
_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_FILES_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_GIT_DIRECTORY_NAME = ".git"


def _repository_name(value: str) -> str:
    if type(value) is not str or len(value) > _REPOSITORY_MAX_CHARS:
        raise ValueError(
            f"repository must be a string of at most {_REPOSITORY_MAX_CHARS} characters."
        )
    value = require_durable_clean_nonblank(value, "repository")
    if _REPOSITORY_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "repository must start with a lowercase ASCII letter and contain only "
            "lowercase ASCII letters, digits, '.', '_', or '-'."
        )
    return value


def _revision(value: str) -> str:
    if type(value) is not str or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "revision must be a full lowercase commit id (40 or 64 hexadecimal characters)."
        )
    return value


def _fixture_path(value: object) -> str:
    if type(value) is not str or len(value) > EVAL_WORKSPACE_FIXTURE_MAX_PATH_CHARS:
        raise ValueError(
            "Workspace fixture paths must be strings of at most "
            f"{EVAL_WORKSPACE_FIXTURE_MAX_PATH_CHARS} characters."
        )
    path = _validate_portable_structural_workspace_path(
        require_durable_clean_nonblank(value, "workspace fixture path")
    )
    if any(component.casefold() == _GIT_DIRECTORY_NAME for component in path.split("/")):
        raise ValueError("Workspace fixture paths cannot be inside a .git directory.")
    return path


class WorkspaceFixtureIdentityV1(BaseModel):
    """What a trial's workspace was prepared from: the revision and a digest of the written files.

    Two fixtures with the same identity prepare the same workspace, since the
    revision is a full commit id and ``files_digest`` covers every written path
    and its content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    repository: StrictStr
    revision: StrictStr
    files_digest: StrictStr

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        return _repository_name(value)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return _revision(value)

    @field_validator("files_digest")
    @classmethod
    def validate_files_digest(cls, value: str) -> str:
        if _FILES_DIGEST_PATTERN.fullmatch(value) is None:
            raise ValueError("files_digest must be a lowercase sha256 content revision.")
        return value


class WorkspaceFixtureSpec(BaseModel):
    """A repository at one commit, with files written over it, that each trial starts from.

    ``repository`` names a :class:`~cayu.workspaces.WorkspaceSource` the
    application registered with ``CayuApp.register_workspace_source``.
    ``revision`` is a full commit id. ``files`` maps canonical relative POSIX
    paths to UTF-8 text written over the checkout, replacing a file the revision
    has at that path. Every trial gets its own copy, so no trial sees another
    trial's writes.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        revalidate_instances="always",
    )

    repository: StrictStr
    revision: StrictStr
    files: dict[str, str] = Field(default_factory=dict)

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        return _repository_name(value)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return _revision(value)

    @field_validator("files", mode="before")
    @classmethod
    def validate_files(cls, value: object) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError("files must be an object mapping paths to text.")
        if len(value) > EVAL_WORKSPACE_FIXTURE_MAX_FILES:
            raise ValueError(
                f"A workspace fixture writes at most {EVAL_WORKSPACE_FIXTURE_MAX_FILES} files."
            )
        files: dict[str, str] = {}
        total_bytes = 0
        for raw_path, content in value.items():
            path = _fixture_path(raw_path)
            if type(content) is not str:
                raise ValueError(f"Workspace fixture file {path!r} must be text.")
            total_bytes += len(require_durable_text(content, f"files[{path!r}]").encode("utf-8"))
            if total_bytes > EVAL_WORKSPACE_FIXTURE_MAX_BYTES:
                raise ValueError(
                    "Workspace fixture files exceed "
                    f"{EVAL_WORKSPACE_FIXTURE_MAX_BYTES} UTF-8 bytes in total."
                )
            files[path] = content
        folded: dict[str, str] = {}
        for path in files:
            key = path.casefold()
            if key in folded:
                raise ValueError(
                    f"Workspace fixture paths {folded[key]!r} and {path!r} differ only in case."
                )
            folded[key] = path
        for path in files:
            parent = path.rpartition("/")[0]
            while parent:
                if parent.casefold() in folded:
                    raise ValueError(
                        f"Workspace fixture path {path!r} is inside the file {parent!r}."
                    )
                parent = parent.rpartition("/")[0]
        return dict(sorted(files.items()))

    def files_digest(self) -> str:
        """The ``sha256:`` digest of every written path and its content."""

        material = {
            path: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for path, content in self.files.items()
        }
        digest = hashlib.sha256(
            canonical_durable_json_bytes(material, "workspace fixture files")
        ).hexdigest()
        return f"sha256:{digest}"

    def identity(self) -> WorkspaceFixtureIdentityV1:
        """The identity a trial prepared from this fixture records."""

        return WorkspaceFixtureIdentityV1(
            repository=self.repository,
            revision=self.revision,
            files_digest=self.files_digest(),
        )


__all__ = [
    "EVAL_WORKSPACE_FIXTURE_MAX_BYTES",
    "EVAL_WORKSPACE_FIXTURE_MAX_FILES",
    "EVAL_WORKSPACE_FIXTURE_MAX_PATH_CHARS",
    "WorkspaceFixtureIdentityV1",
    "WorkspaceFixtureSpec",
]
