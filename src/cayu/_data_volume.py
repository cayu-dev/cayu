"""The wire contract of a Cayu server's read-only data-volume routes.

A host that gives the application a persistent data volume names its directory with
``CAYU_DATA_VOLUME``; Cayu Cloud sets it to ``/data`` on the Agent's own processes. A Cayu
server then lets its authenticated full-access callers read that directory, and only that
directory, through four bounded routes under its API path, which
`cayu cloud data ls|cat|get|stat --application NAME PATH` calls. Nothing here writes.

Dependency-light: the CLI validates the server's answers with these models without the
server extra.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DATA_VOLUME_ENV = "CAYU_DATA_VOLUME"
# What a download of the volume's root is named: `data.tar.gz`, members under `data/`.
DATA_VOLUME_ROOT_NAME = "data"

# Routes under the server's API path; each takes the path relative to the volume as `path`.
DATA_VOLUME_ENTRY_PATH = "/data-volume/entry"
DATA_VOLUME_ENTRIES_PATH = "/data-volume/entries"
DATA_VOLUME_CONTENT_PATH = "/data-volume/content"
DATA_VOLUME_ARCHIVE_PATH = "/data-volume/archive"

# Paths: relative, `/`-separated, at most this many characters, each name at most this many
# UTF-8 bytes (the usual file-system limit).
MAX_PATH_CHARS = 1024
MAX_NAME_BYTES = 255
# Listing: names a page holds, and how many entries of one directory the server reads.
DEFAULT_LIST_LIMIT = 200
MAX_LIST_LIMIT = 1000
MAX_DIRECTORY_SCAN = 10_000
# Bytes of a file one content read returns.
MAX_READ_BYTES = 4 * 1024 * 1024
# A directory archive: entries, file bytes read, compressed bytes returned, and depth.
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_INPUT_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_DEPTH = 32
# The skipped entries an archive names; it counts the rest.
MAX_ARCHIVE_SKIPPED_NAMED = 100
# How long one listing or archive may spend reading the volume.
DATA_VOLUME_DEADLINE_SECONDS = 20.0

DataVolumeEntryType = Literal["file", "directory", "symlink", "other"]
DataVolumeSkipReason = Literal["symlink", "special_file", "unreadable_name"]
DataVolumeRefusalCode = Literal[
    "data_volume_requires_authentication",
    "data_volume_not_configured",
    "data_volume_unavailable",
    "path_refused",
    "path_outside_volume",
    "path_not_found",
    "path_unreadable",
    "not_a_file",
    "not_a_directory",
    "changed_while_read",
    "archive_too_large",
    "archive_busy",
]

_BASE64_READ_CHARS = 4 * -(-MAX_READ_BYTES // 3)
_BASE64_ARCHIVE_CHARS = 4 * -(-MAX_ARCHIVE_BYTES // 3)
# A listed entry's path is the directory's path and one name. An archive member is the
# directory's own name, one name per level below it and its own name, which a skipped name
# that isn't UTF-8 writes with each byte escaped (four characters).
_MAX_ENTRY_PATH_CHARS = MAX_PATH_CHARS + 1 + MAX_NAME_BYTES
_MAX_MEMBER_CHARS = (MAX_ARCHIVE_DEPTH + 1) * (MAX_NAME_BYTES + 1) + 4 * MAX_NAME_BYTES


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataVolumeEntry(_Model):
    """One entry of the volume; ``path`` is relative to the volume, ``""`` for its root."""

    path: str = Field(max_length=_MAX_ENTRY_PATH_CHARS)
    name: str = Field(max_length=MAX_NAME_BYTES)
    type: DataVolumeEntryType
    size: int | None = Field(default=None, ge=0, description="Bytes, for a file.")
    modified_at: datetime


class DataVolumeListing(_Model):
    """A page of one directory's entries, in name order."""

    path: str = Field(max_length=MAX_PATH_CHARS)
    entries: list[DataVolumeEntry] = Field(max_length=MAX_LIST_LIMIT)
    next_after: str | None = Field(
        default=None,
        max_length=MAX_NAME_BYTES,
        description="When more entries follow: the `after` that reads the next page.",
    )
    complete: bool = Field(
        description=(
            f"False when the directory holds more than {MAX_DIRECTORY_SCAN} entries; the "
            "pages then hold only the entries the server read."
        )
    )
    omitted: int = Field(ge=0, description="Entries left out because their names aren't UTF-8.")


class DataVolumeContent(_Model):
    """Up to ``MAX_READ_BYTES`` of a file from ``offset``, base64-encoded."""

    path: str = Field(max_length=MAX_PATH_CHARS)
    size: int = Field(ge=0, description="The file's size when it was read.")
    modified_at: datetime
    offset: int = Field(ge=0)
    content_base64: str = Field(max_length=_BASE64_READ_CHARS)


class DataVolumeSkipped(_Model):
    """An entry an archive left out, named as its member would have been."""

    member: str = Field(max_length=_MAX_MEMBER_CHARS)
    reason: DataVolumeSkipReason


class DataVolumeArchive(_Model):
    """A directory as a gzip-compressed tar archive, base64-encoded.

    Members are named under the directory's own name. Symbolic links and special files are
    left out and named in ``skipped`` (the first ``MAX_ARCHIVE_SKIPPED_NAMED``).
    """

    path: str = Field(max_length=MAX_PATH_CHARS)
    name: str = Field(
        max_length=MAX_NAME_BYTES + len(".tar.gz"), description="A file name for the archive."
    )
    files: int = Field(ge=0)
    directories: int = Field(ge=0)
    bytes: int = Field(ge=0, description="File bytes archived, before compression.")
    skipped: list[DataVolumeSkipped] = Field(max_length=MAX_ARCHIVE_SKIPPED_NAMED)
    skipped_count: int = Field(ge=0)
    content_base64: str = Field(max_length=_BASE64_ARCHIVE_CHARS)


class DataVolumeRefusalDetail(_Model):
    code: DataVolumeRefusalCode
    message: str


class DataVolumeRefusalResponse(_Model):
    detail: DataVolumeRefusalDetail


def data_volume_path_parts(path: str) -> tuple[str, ...]:
    """The names of ``path`` from the volume's root, or ``ValueError`` saying why it's refused.

    ``""`` and ``"."`` name the root, and one trailing ``/`` is ignored. Absolute paths, ``.``
    and ``..`` names, empty names (``a//b``), control characters and over-long paths or
    names are refused. This is only the path's shape: the server resolves it on the volume
    itself, so a symbolic link can't lead out either.
    """

    if len(path) > MAX_PATH_CHARS:
        raise ValueError(f"A path must be at most {MAX_PATH_CHARS} characters.")
    if any(unicodedata.category(character) == "Cc" for character in path):
        raise ValueError("A path must not contain control characters.")
    if path.startswith("/"):
        raise ValueError(
            "A path is relative to the data volume, such as runs/RUN/run.json; absolute "
            "paths are refused."
        )
    if path in {"", "."}:
        return ()
    names = tuple(path.removesuffix("/").split("/"))
    for name in names:
        if name in {"", ".", ".."}:
            raise ValueError(
                "A path must not contain empty, `.` or `..` names; name the file from the "
                "data volume's root."
            )
        try:
            encoded = name.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("A path must be UTF-8 text.") from None
        if len(encoded) > MAX_NAME_BYTES:
            raise ValueError(f"Each name in a path must be at most {MAX_NAME_BYTES} bytes.")
    return names
