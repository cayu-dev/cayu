"""Read-only routes over the host's data volume, for authenticated full-access callers.

`cayu cloud data ls|cat|get|stat --application NAME PATH` reaches them on a Cloud Agent, with
the Agent's Developer dashboard login. They serve the directory ``CAYU_DATA_VOLUME`` names and
nothing outside it (`_data_volume_reader`), and only to a caller the server's auth dependency
admitted: a server without one refuses them, and under ``ControlPlaneRoles`` only full-access
roles may use them. Each read is bounded (`cayu._data_volume`) and writes nothing to the volume.

Every request a route answers, refusals included, writes one JSON audit line to standard
error, ``{"event": "cayu.data_volume.read", ...}``, naming the operation, the requested path,
the outcome, the caller's subject and how much was returned, never what a file holds. Cayu
Cloud keeps a web process's standard error with the Agent's logs.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, cast

from fastapi import HTTPException, Query

from cayu._data_volume import (
    DATA_VOLUME_ARCHIVE_PATH,
    DATA_VOLUME_CONTENT_PATH,
    DATA_VOLUME_DEADLINE_SECONDS,
    DATA_VOLUME_ENTRIES_PATH,
    DATA_VOLUME_ENTRY_PATH,
    DATA_VOLUME_ENV,
    DEFAULT_LIST_LIMIT,
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_INPUT_BYTES,
    MAX_DIRECTORY_SCAN,
    MAX_LIST_LIMIT,
    MAX_NAME_BYTES,
    MAX_PATH_CHARS,
    MAX_READ_BYTES,
    DataVolumeArchive,
    DataVolumeContent,
    DataVolumeEntry,
    DataVolumeListing,
    DataVolumeRefusalResponse,
    data_volume_path_parts,
)
from cayu.server._data_volume_reader import (
    DataVolume,
    DataVolumeRefusal,
    data_volume_from_environment,
)
from cayu.server.auth import AuthContext

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

AUDIT_EVENT = "cayu.data_volume.read"
# How long a caller refused because another archive is being built waits.
ARCHIVE_RETRY_AFTER_SECONDS = 5
# The largest offset a content read accepts: JavaScript's largest exact integer.
_MAX_OFFSET = 2**53 - 1
_MIB = 1024 * 1024

_Answer = TypeVar(
    "_Answer", DataVolumeEntry, DataVolumeListing, DataVolumeContent, DataVolumeArchive
)

_PATH_DESCRIPTION = (
    "The path relative to the data volume, such as `runs/RUN/run.json`; empty for its root. "
    "Absolute paths and `.`, `..` or empty names are refused."
)
_COMMON_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": DataVolumeRefusalResponse, "description": "The path isn't in the data volume."},
    409: {
        "model": DataVolumeRefusalResponse,
        "description": (
            "The server serves no data volume (it has no authentication, or "
            f"`{DATA_VOLUME_ENV}` isn't set or names no directory), or the path is the wrong "
            "kind of entry, unreadable, or changed while it was read."
        ),
    },
    422: {
        "model": DataVolumeRefusalResponse,
        "description": (
            "The path is refused: absolute, with `.`, `..` or empty names, or leading outside "
            "the data volume through a symbolic link. A parameter outside its bounds answers "
            "422 with a plain `detail`."
        ),
    },
}


def register_data_volume_routes(router: APIRouter, *, optional_auth_context: Depends) -> None:
    """Add ``GET {api}/data-volume/entry|entries|content|archive`` over the host's volume.

    ``optional_auth_context`` is the API's auth dependency in the form that yields the
    caller (``None`` on a server without authentication, which these routes refuse).
    ``CAYU_DATA_VOLUME`` is read once, here.
    """

    volume = data_volume_from_environment(os.environ)
    archives = threading.Lock()
    auth_default = cast("AuthContext | None", optional_auth_context)

    def answer(
        operation: str,
        path: str,
        auth_context: AuthContext | None,
        read: Callable[[DataVolume, tuple[str, ...]], _Answer],
    ) -> _Answer:
        subject = None if auth_context is None else auth_context.subject
        try:
            result = read(_served(volume, auth_context), _parts(path))
        except DataVolumeRefusal as refusal:
            _audit(operation, path, outcome=refusal.code, subject=subject)
            headers = (
                {"Retry-After": str(ARCHIVE_RETRY_AFTER_SECONDS)}
                if refusal.code == "archive_busy"
                else None
            )
            raise HTTPException(
                status_code=refusal.status_code,
                detail={"code": refusal.code, "message": refusal.message},
                headers=headers,
            ) from None
        except Exception:
            _audit(operation, path, outcome="error", subject=subject)
            raise
        _audit(operation, path, outcome="ok", subject=subject, **_amounts(result))
        return result

    @router.get(
        DATA_VOLUME_ENTRY_PATH,
        response_model=DataVolumeEntry,
        responses=_COMMON_RESPONSES,
        description=(
            "Describe one entry of the host's data volume (`CAYU_DATA_VOLUME`): its type, "
            "size and modification time, after following symbolic links that stay within the "
            "volume. For authenticated full-access callers; nothing is written."
        ),
    )
    def read_data_volume_entry(
        path: Annotated[str, Query(max_length=MAX_PATH_CHARS, description=_PATH_DESCRIPTION)] = "",
        auth_context: AuthContext | None = auth_default,
    ) -> DataVolumeEntry:
        return answer("entry", path, auth_context, lambda served, parts: served.entry(parts))

    @router.get(
        DATA_VOLUME_ENTRIES_PATH,
        response_model=DataVolumeListing,
        responses=_COMMON_RESPONSES,
        description=(
            "List a directory of the host's data volume in name order, a page of at most "
            f"{MAX_LIST_LIMIT} entries after the name `after`. Each entry is described without "
            "following it, so a symbolic link is listed as one. The server reads at most "
            f"{MAX_DIRECTORY_SCAN} entries of one directory and spends at most "
            f"{DATA_VOLUME_DEADLINE_SECONDS:g} seconds on a page; `next_after` is set when "
            "more follow. For authenticated full-access callers; nothing is written."
        ),
    )
    def list_data_volume_entries(
        path: Annotated[str, Query(max_length=MAX_PATH_CHARS, description=_PATH_DESCRIPTION)] = "",
        limit: Annotated[int, Query(ge=1, le=MAX_LIST_LIMIT)] = DEFAULT_LIST_LIMIT,
        after: Annotated[str | None, Query(min_length=1, max_length=MAX_NAME_BYTES)] = None,
        auth_context: AuthContext | None = auth_default,
    ) -> DataVolumeListing:
        return answer(
            "entries",
            path,
            auth_context,
            lambda served, parts: served.entries(parts, limit=limit, after=after),
        )

    @router.get(
        DATA_VOLUME_CONTENT_PATH,
        response_model=DataVolumeContent,
        responses=_COMMON_RESPONSES,
        description=(
            f"Read up to {MAX_READ_BYTES // _MIB} MiB of a regular file of the host's data "
            "volume from `offset`, base64-encoded, with the file's size and modification "
            "time so a caller reading it in parts can tell it changed. For authenticated "
            "full-access callers; nothing is written."
        ),
    )
    def read_data_volume_content(
        path: Annotated[str, Query(max_length=MAX_PATH_CHARS, description=_PATH_DESCRIPTION)],
        offset: Annotated[int, Query(ge=0, le=_MAX_OFFSET)] = 0,
        length: Annotated[int, Query(ge=1, le=MAX_READ_BYTES)] = MAX_READ_BYTES,
        auth_context: AuthContext | None = auth_default,
    ) -> DataVolumeContent:
        return answer(
            "content",
            path,
            auth_context,
            lambda served, parts: served.content(parts, offset=offset, length=length),
        )

    def archive(served: DataVolume, parts: tuple[str, ...]) -> DataVolumeArchive:
        # An archive holds up to its compressed bound in memory, so one is built at a time.
        if not archives.acquire(blocking=False):
            raise DataVolumeRefusal(
                429, "archive_busy", "Another archive is being built; try again shortly."
            )
        try:
            return served.archive(parts)
        finally:
            archives.release()

    @router.get(
        DATA_VOLUME_ARCHIVE_PATH,
        response_model=DataVolumeArchive,
        responses={
            **_COMMON_RESPONSES,
            413: {
                "model": DataVolumeRefusalResponse,
                "description": (
                    f"The directory holds more than {MAX_ARCHIVE_ENTRIES} entries or "
                    f"{MAX_ARCHIVE_INPUT_BYTES // _MIB} MiB of files, compresses to more than "
                    f"{MAX_ARCHIVE_BYTES // _MIB} MiB, nests too deep, or takes longer than "
                    f"{DATA_VOLUME_DEADLINE_SECONDS:g} seconds to read."
                ),
            },
            429: {
                "model": DataVolumeRefusalResponse,
                "description": (
                    "Another archive is being built in this process. Retry after the "
                    "`Retry-After` header."
                ),
            },
        },
        description=(
            "Archive a directory of the host's data volume as a gzip-compressed tar, "
            "base64-encoded, its members named under the directory's own name. Symbolic "
            "links and special files are left out and named in `skipped`. Bounded in "
            "entries, bytes read, compressed size, depth and time; one archive is built at a "
            "time per process. For authenticated full-access callers; nothing is written."
        ),
    )
    def read_data_volume_archive(
        path: Annotated[str, Query(max_length=MAX_PATH_CHARS, description=_PATH_DESCRIPTION)] = "",
        auth_context: AuthContext | None = auth_default,
    ) -> DataVolumeArchive:
        return answer("archive", path, auth_context, archive)


def _served(volume: DataVolume | None, auth_context: AuthContext | None) -> DataVolume:
    if auth_context is None:
        raise DataVolumeRefusal(
            409,
            "data_volume_requires_authentication",
            "This server has no authentication, so it doesn't serve its data volume.",
        )
    if volume is None:
        raise DataVolumeRefusal(
            409,
            "data_volume_not_configured",
            f"This server has no data volume: its host doesn't set {DATA_VOLUME_ENV}. On Cayu "
            "Cloud, deploy the Agent again so Cloud sets it.",
        )
    return volume


def _parts(path: str) -> tuple[str, ...]:
    try:
        return data_volume_path_parts(path)
    except ValueError as exc:
        raise DataVolumeRefusal(422, "path_refused", str(exc)) from None


def _amounts(result: Any) -> dict[str, int]:
    """How much a read returned, for its audit line."""

    if isinstance(result, DataVolumeListing):
        return {"entries": len(result.entries)}
    if isinstance(result, DataVolumeContent):
        return {"bytes": _decoded_length(result.content_base64), "offset": result.offset}
    if isinstance(result, DataVolumeArchive):
        return {
            "bytes": _decoded_length(result.content_base64),
            "files": result.files,
        }
    return {}


def _decoded_length(encoded: str) -> int:
    return len(encoded) // 4 * 3 - encoded[-2:].count("=")


def _audit(operation: str, path: str, *, outcome: str, subject: str | None, **amounts: int) -> None:
    record = {
        "at": datetime.now(UTC).isoformat(),
        "event": AUDIT_EVENT,
        "operation": operation,
        "outcome": outcome,
        "path": path,
        "subject": subject,
        **amounts,
    }
    # ASCII escapes keep a requested path's control characters from forging a log line.
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)
