"""Large file uploads and streamed downloads for an application's own routes.

Cayu's ``/api`` artifact routes carry a file as base64 JSON, at most 4 MiB. An application
that takes bigger files (a workbook, a ZIP) reads a multipart upload with
:func:`read_upload` and serves a stored file back with :func:`artifact_download`, which
reads it from the artifact store a range at a time::

    from fastapi import APIRouter, Request

    from cayu.artifacts import ArtifactScope
    from cayu.server.app_files import artifact_download, read_upload

    router = APIRouter(prefix="/app")


    @router.post("/files", status_code=201)
    async def upload(request: Request) -> dict:
        upload = await read_upload(request, max_bytes=50 * 1024 * 1024)
        stored = await files.put_bytes(
            upload.content,
            filename=upload.filename,
            content_type=upload.content_type,
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="files",
        )
        return {"id": stored.id}


    @router.get("/files/{artifact_id}")
    async def download(artifact_id: str):
        return await artifact_download(files, artifact_id)

``files`` is a raw ``ArtifactStore`` the application guards itself. The download takes any
handle with ``read_range``, so it has the same access checks as the rest of the app: with
the person's scoped handle
(``(await app.access(subject)).artifacts(store, environment_name=...)``) every range it
reads is checked, and the upload is stored with
``handle.put_bytes(upload.content, labels=..., filename=..., content_type=...)``.

The upload is counted as it arrives, so a body without ``Content-Length`` (a chunked
upload, or one a proxy re-chunked) is refused as soon as it passes the limit, not after it
was read whole. ``read_upload`` needs ``python-multipart``, which the ``server`` extra
installs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol
from urllib.parse import quote

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from cayu.artifacts.base import ArtifactReadResult
from cayu.server.routes import _artifact_content_disposition_kind

__all__ = [
    "DOWNLOAD_CHUNK_BYTES",
    "UPLOAD_ENVELOPE_BYTES",
    "ArtifactRangeReader",
    "UploadedFile",
    "artifact_download",
    "read_upload",
]

# What a multipart body may carry beyond its file: boundaries, part headers, small fields.
UPLOAD_ENVELOPE_BYTES = 64 * 1024
# How much of a stored file a download reads at a time; the scoped handle allows 4 MiB.
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """The file of a multipart upload, and the form's other (text) fields."""

    filename: str
    content_type: str | None
    content: bytes
    fields: Mapping[str, str] = field(default_factory=dict)


class _BodyTooLarge(Exception):
    pass


def _declared_length(request: Request) -> int | None:
    try:
        return int(request.headers["content-length"])
    except (KeyError, ValueError):
        return None


def _clean_filename(name: str | None) -> str:
    # A browser sends the bare name; some clients send a path. Keep the last part only.
    return PurePosixPath((name or "").replace("\\", "/")).name.strip()


async def read_upload(
    request: Request,
    *,
    max_bytes: int,
    file_field: str = "file",
    max_fields: int = 20,
    too_large: str | None = None,
) -> UploadedFile:
    """Read a ``multipart/form-data`` request's one file, at most ``max_bytes`` of it.

    A body over ``max_bytes`` plus :data:`UPLOAD_ENVELOPE_BYTES` is refused with 413
    (``too_large``) as soon as it passes the limit, whether or not it declared its length;
    a file over ``max_bytes`` is refused with 413 too. A request that isn't multipart, or
    has no ``file_field`` file or no file name, gets 422. The form's text fields come back
    in ``fields``. The file is spooled to a temporary file while it arrives and returned as
    bytes, which is what ``ArtifactStore.put_bytes`` takes.
    """

    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("read_upload needs a positive max_bytes.")
    message = too_large or f"The file is larger than {max_bytes} bytes."
    limit = max_bytes + UPLOAD_ENVELOPE_BYTES
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise HTTPException(422, "Send the file as multipart/form-data.")
    declared = _declared_length(request)
    if declared is not None and declared > limit:
        raise HTTPException(413, message)

    received = 0

    async def receive() -> Any:
        nonlocal received
        event = await request.receive()
        if event["type"] == "http.request":
            received += len(event.get("body", b""))
            if received > limit:
                raise _BodyTooLarge()
        return event

    counted = Request(request.scope, receive)
    try:
        async with counted.form(
            max_files=1, max_fields=max_fields, max_part_size=UPLOAD_ENVELOPE_BYTES
        ) as form:
            upload = form.get(file_field)
            if upload is None or isinstance(upload, str):
                raise HTTPException(422, f"Send the file in the form field {file_field!r}.")
            filename = _clean_filename(upload.filename)
            if not filename:
                raise HTTPException(422, "The uploaded file has no name.")
            await upload.seek(0)
            content = await upload.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise HTTPException(413, message)
            fields = {
                name: value
                for name, value in form.multi_items()
                if name != file_field and isinstance(value, str)
            }
            return UploadedFile(
                filename=filename,
                content_type=upload.content_type or None,
                content=content,
                fields=fields,
            )
    except _BodyTooLarge:
        raise HTTPException(413, message) from None


class ArtifactRangeReader(Protocol):
    """What :func:`artifact_download` reads from: an ``ArtifactStore``, or a scoped handle."""

    async def read_range(
        self, artifact_id: str, *, offset: int, max_bytes: int
    ) -> ArtifactReadResult: ...


def _content_disposition(disposition: str, filename: str) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode().replace('"', "").replace("\\", "")
    value = f'{disposition}; filename="{ascii_name or "download"}"'
    if ascii_name != filename:
        value += f"; filename*=UTF-8''{quote(filename, safe='')}"
    return value


async def artifact_download(
    reader: ArtifactRangeReader,
    artifact_id: str,
    *,
    media_type: str | None = None,
    filename: str | None = None,
    disposition: Literal["attachment", "inline"] = "attachment",
    headers: Mapping[str, str] | None = None,
    chunk_bytes: int = DOWNLOAD_CHUNK_BYTES,
) -> StreamingResponse:
    """A stored file, streamed from ``reader`` one range at a time, never held whole.

    The first range is read before the response starts, so a missing file
    (``FileNotFoundError``) or a refused one (the scoped handle's access error) raises
    before any byte is sent; answer it as the route answers other errors
    (``cayu.server.app_errors``). The response has the file's ``Content-Length``, its
    stored content type unless ``media_type`` says otherwise (``application/octet-stream``
    when neither does), ``Content-Disposition`` with its stored name unless ``filename``
    says otherwise, ``X-Content-Type-Options: nosniff`` and ``Cache-Control: private,
    no-store``. ``disposition="inline"`` applies only to the content types Cayu's own
    artifact routes show inline (plain text, Markdown, CSV, JSON, PDF and raster images);
    anything else, such as HTML or SVG an uploader named, is sent as an attachment, so an
    upload can't run script on the app's origin. A store without range reads
    raises ``NotImplementedError``.
    """

    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= _MAX_CHUNK_BYTES:
        raise ValueError(f"chunk_bytes must be between 1 and {_MAX_CHUNK_BYTES}.")
    first = await reader.read_range(artifact_id, offset=0, max_bytes=chunk_bytes)
    total = first.total_bytes

    async def body() -> AsyncIterator[bytes]:
        result = first
        offset = 0
        while True:
            if result.total_bytes != total or result.offset != offset:
                raise RuntimeError("The stored file changed while it was downloaded.")
            if result.content:
                yield result.content
            offset += len(result.content)
            if offset >= total:
                return
            if not result.content:
                raise RuntimeError("The artifact store returned an empty range before the end.")
            result = await reader.read_range(artifact_id, offset=offset, max_bytes=chunk_bytes)

    metadata = first.metadata
    served_type = media_type or metadata.content_type or "application/octet-stream"
    response_headers = {
        **dict(headers or {}),
        "Content-Length": str(total),
        "Content-Disposition": _content_disposition(
            _artifact_content_disposition_kind(served_type, disposition),
            filename or metadata.filename or artifact_id,
        ),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-store",
    }
    return StreamingResponse(body(), media_type=served_type, headers=response_headers)
