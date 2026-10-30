"""`cayu cloud data ls|cat|get|stat --application NAME PATH`: read an Agent's /data files.

The Agent's own Cayu server serves its data volume (``CAYU_DATA_VOLUME``, which Cayu Cloud
sets to ``/data``) to full-access callers (`cayu.server._data_volume_routes`). The CLI reaches
it as `cayu task --application` does: it signs in to Cloud, asks for the Agent's Cayu address
and Developer dashboard login, and reads the Agent's API with that login. PATH is relative to
the volume; the server refuses absolute paths, ``..`` and symbolic links that lead out of it.

Everything is read-only and bounded: the server bounds each answer, `cat` prints a file of
at most ``MAX_CAT_BYTES`` and `get` downloads a file of at most ``MAX_GET_BYTES`` in parts, or
a directory as one tar.gz archive. There is no `rm` or `put`: changing the volume stays with
the Agent's own code and the web app.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from cayu._data_volume import (
    DATA_VOLUME_ARCHIVE_PATH,
    DATA_VOLUME_CONTENT_PATH,
    DATA_VOLUME_DEADLINE_SECONDS,
    DATA_VOLUME_ENTRIES_PATH,
    DATA_VOLUME_ENTRY_PATH,
    DATA_VOLUME_ROOT_NAME,
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    DataVolumeArchive,
    DataVolumeContent,
    DataVolumeEntry,
    DataVolumeListing,
)
from cayu.cli._cloud_releases import add_output_argument, render_failure
from cayu.cli._cloud_settings import _cell, _table
from cayu.cli._output import terminal_safe

if TYPE_CHECKING:
    from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
    from cayu.cli._cloud_api import CloudApiClient

MAX_CAT_BYTES = 16 * 1024 * 1024
MAX_GET_BYTES = 256 * 1024 * 1024
# The archive's read can spend the server's whole deadline before its first byte.
_ARCHIVE_TIMEOUT_SECONDS = DATA_VOLUME_DEADLINE_SECONDS + 15
_MIB = 1024 * 1024
_COPY_BYTES = _MIB
_SKIPPED_SHOWN = 10

_Model = TypeVar("_Model", bound=BaseModel)


def add_data_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    data = commands.add_parser(
        "data",
        help="Read the files in an Agent's /data.",
        description=(
            "Read the files the Agent keeps in its /data volume, through the Agent's own Cayu "
            "server. Read-only: there is no rm or put. Needs a release built with this Cayu."
        ),
    )
    actions = data.add_subparsers(dest="data_command", required=True)
    listing = actions.add_parser(
        "ls",
        description=(
            f"List a directory of /data in name order, {DEFAULT_LIST_LIMIT} entries a page "
            "by default. A symbolic link is listed as one, not followed."
        ),
    )
    listing.add_argument(
        "path", nargs="?", default="", help="Relative to /data; its root when omitted."
    )
    listing.add_argument(
        "--limit",
        type=_list_limit,
        default=DEFAULT_LIST_LIMIT,
        help=f"Entries per page, at most {MAX_LIST_LIMIT} ({DEFAULT_LIST_LIMIT}).",
    )
    listing.add_argument("--after", help="Go on after this name, from a page's next_after.")
    add_output_argument(listing, render_listing)
    entry = actions.add_parser(
        "stat", description="Show an entry's type, size and modification time."
    )
    entry.add_argument("path", help="Relative to /data.")
    add_output_argument(entry, render_entry)
    cat = actions.add_parser(
        "cat",
        description=(
            f"Print a file of at most {MAX_CAT_BYTES // _MIB} MiB. Its bytes go to standard "
            "output as they are; a terminal gets UTF-8 text only, with control characters "
            "shown as escapes."
        ),
    )
    cat.add_argument("path", help="Relative to /data.")
    cat.set_defaults(_cloud_render=_printed, _cloud_render_failure=render_failure)
    download = actions.add_parser(
        "get",
        description=(
            f"Download a file of at most {MAX_GET_BYTES // _MIB} MiB, or a directory as one "
            "tar.gz archive (symbolic links and special files left out)."
        ),
    )
    download.add_argument("path", help="Relative to /data; its root when empty.")
    download.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Where to write it: the path's own name (NAME.tar.gz for a directory) by default.",
    )
    download.add_argument("--force", action="store_true", help="Replace an existing file.")
    add_output_argument(download, render_download)
    for parser in (listing, entry, cat, download):
        parser.add_argument("--application", required=True)


def data_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    """Run one `cayu cloud data` command on the Agent ``application_id``."""

    from cayu.cli._cloud_agent_api import CayuApiError, cloud_agent_api

    action = arguments.data_command
    try:
        api = cloud_agent_api(
            client,
            application_id,
            timeout_seconds=max(arguments.timeout_seconds, _ARCHIVE_TIMEOUT_SECONDS),
        )
    except CayuApiError as exc:
        raise _command_error(exc) from None
    try:
        # A route every Cayu serves finds the Agent's API first, so an Agent whose Cayu
        # predates these routes is told apart from one that serves no Cayu API.
        api.get("/contract", resource="server contract")
        if action == "ls":
            return {"operation": "data.ls", "result": _listing(api, arguments)}
        if action == "stat":
            return {
                "operation": "data.stat",
                "result": _entry(api, arguments.path).model_dump(mode="json"),
            }
        if action == "cat":
            return {"operation": "data.cat", "result": _cat(api, arguments.path)}
        return {"operation": "data.get", "result": _download(api, arguments)}
    except CayuApiError as exc:
        raise _command_error(exc, api=api) from None
    finally:
        api.close()


def _listing(api: CayuApi, arguments: argparse.Namespace) -> dict[str, Any]:
    query = {"path": arguments.path, "limit": str(arguments.limit)}
    if arguments.after is not None:
        query["after"] = arguments.after
    answer = api.get(DATA_VOLUME_ENTRIES_PATH, query=query, resource="data volume listing")
    return _validated(DataVolumeListing, answer, api).model_dump(mode="json")


def _entry(api: CayuApi, path: str) -> DataVolumeEntry:
    answer = api.get(DATA_VOLUME_ENTRY_PATH, query={"path": path}, resource="data volume entry")
    return _validated(DataVolumeEntry, answer, api)


def _cat(api: CayuApi, path: str) -> dict[str, Any]:
    content = b"".join(_file_parts(api, path, limit=MAX_CAT_BYTES, command="cat"))
    stream = sys.stdout
    if stream.isatty():
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise _not_text(path) from None
        stream.write(terminal_safe(text, keep_lines=True))
        stream.flush()
    else:
        stream.flush()
        stream.buffer.write(content)
        stream.buffer.flush()
    return {"path": path, "bytes": len(content)}


def _download(api: CayuApi, arguments: argparse.Namespace) -> dict[str, Any]:
    entry = _entry(api, arguments.path)
    # The local name comes from the requested path, never from what the Agent answers.
    name = PurePosixPath(arguments.path).name or DATA_VOLUME_ROOT_NAME
    if entry.type == "directory":
        output = arguments.output or Path(f"{name}.tar.gz")
        _refuse_existing(output, force=arguments.force)
        answer = api.get(
            DATA_VOLUME_ARCHIVE_PATH,
            query={"path": arguments.path},
            resource="data volume archive",
        )
        archive = _validated(DataVolumeArchive, answer, api)
        payload = _decoded(archive.content_base64, api)
        _write(output, iter((payload,)), force=arguments.force)
        return {
            "path": archive.path,
            "type": "directory",
            "output": str(output),
            "bytes": len(payload),
            "files": archive.files,
            "directories": archive.directories,
            "skipped": [item.model_dump(mode="json") for item in archive.skipped],
            "skipped_count": archive.skipped_count,
        }
    if entry.type != "file":
        raise _error("not_a_file", f"`{arguments.path}` is not a file or a directory.")
    output = arguments.output or Path(name)
    _refuse_existing(output, force=arguments.force)
    size = _write(
        output,
        _file_parts(api, arguments.path, limit=MAX_GET_BYTES, command="get"),
        force=arguments.force,
    )
    return {"path": entry.path, "type": "file", "output": str(output), "bytes": size}


def _file_parts(api: CayuApi, path: str, *, limit: int, command: str) -> Iterator[bytes]:
    """A file's bytes in the parts the server reads, refused when it passes ``limit`` or
    changes between them."""

    first: DataVolumeContent | None = None
    offset = 0
    while first is None or offset < first.size:
        answer = api.get(
            DATA_VOLUME_CONTENT_PATH,
            query={"path": path, "offset": str(offset)},
            resource="data volume file",
        )
        part = _validated(DataVolumeContent, answer, api)
        if first is None:
            first = part
            if part.size > limit:
                raise _error(
                    "too_large",
                    f"`{path}` is {part.size} bytes; `{command}` reads at most "
                    f"{limit // _MIB} MiB."
                    + (" Download it with `cayu cloud data get`." if command == "cat" else ""),
                )
        content = _decoded(part.content_base64, api)
        if (
            part.offset != offset
            or (part.size, part.modified_at) != (first.size, first.modified_at)
            or (not content and offset < first.size)
        ):
            raise _error("changed_while_read", f"`{path}` changed while it was read; try again.")
        offset += len(content)
        yield content


def _write(output: Path, parts: Iterator[bytes], *, force: bool) -> int:
    """Write ``parts`` to a new file beside ``output``, then put it there.

    With ``force`` it replaces whatever ``output`` is then. Without, it puts it there only if
    nothing took the name while the download ran (`_publish_new`), so nothing is ever replaced
    that the earlier check didn't see.
    """

    try:
        descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=".cayu-data-")
    except OSError as exc:
        raise _unwritable(output, exc) from None
    written = 0
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for part in parts:
                stream.write(part)
                written += len(part)
        if force:
            os.replace(temporary, output)
        else:
            _publish_new(temporary, output)
    except FileExistsError:
        raise _exists(output) from None
    except OSError as exc:
        raise _unwritable(output, exc) from None
    finally:
        Path(temporary).unlink(missing_ok=True)
    return written


def _publish_new(temporary: str, output: Path) -> None:
    """Put the finished ``temporary`` at ``output`` only if the name is free, or raise
    ``FileExistsError``.

    A hard link does that in one step. A file system without hard links (exFAT, many network
    and shared-folder mounts) refuses the link; there the name is created exclusively and the
    bytes are copied in, and a copy that fails removes what it created.
    """

    try:
        os.link(temporary, output)
        return
    except FileExistsError:
        raise
    except OSError:
        pass
    # Opening with "x" creates the name or raises FileExistsError, leaving another file alone.
    with open(temporary, "rb") as source, open(output, "xb") as target:
        try:
            shutil.copyfileobj(source, target, _COPY_BYTES)
        except BaseException:
            output.unlink(missing_ok=True)
            raise


def _unwritable(output: Path, error: OSError) -> Exception:
    return _error(
        "output_unwritable", f"Could not write {output}: {error.strerror or type(error).__name__}."
    )


def _refuse_existing(output: Path, *, force: bool) -> None:
    if output.is_dir():
        raise _error("output_exists", f"{output} is a directory; name a file with --output.")
    if output.exists() and not force:
        raise _exists(output)


def _exists(output: Path) -> Exception:
    return _error(
        "output_exists", f"{output} exists; pass --force to replace it, or choose --output."
    )


def _validated(model: type[_Model], answer: object, api: CayuApi) -> _Model:
    try:
        return model.model_validate(answer)
    except ValidationError:
        raise _error(
            "invalid_response",
            f"{api.description} answered with data this CLI can't read; a newer Cayu CLI may "
            "read it.",
        ) from None


def _decoded(encoded: str, api: CayuApi) -> bytes:
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise _error(
            "invalid_response", f"{api.description} answered with content that isn't base64."
        ) from None


def _command_error(exc: CayuApiError, *, api: CayuApi | None = None) -> Exception:
    from cayu.cli.cloud import CloudCommandError

    category = exc.category
    if category == "unsupported" and api is not None:
        return CloudCommandError(
            "unsupported",
            f"{api.description} runs a Cayu that can't read its data volume. Deploy a release "
            "built with this CLI's Cayu version or newer, then try again.",
        )
    return CloudCommandError(category, str(exc))


def _error(category: str, message: str) -> Exception:
    from cayu.cli.cloud import CloudCommandError

    return CloudCommandError(category, message)


def _not_text(path: str) -> Exception:
    return _error(
        "not_text",
        f"`{path}` isn't UTF-8 text, so it isn't printed to a terminal. Redirect the output, "
        "or download it with `cayu cloud data get`.",
    )


def _list_limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if not 1 <= number <= MAX_LIST_LIMIT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_LIST_LIMIT}")
    return number


def _printed(_: Mapping[str, Any]) -> str:
    """`cat` wrote the file itself; there is nothing more to print."""

    return ""


def render_listing(output: Mapping[str, Any]) -> str:
    listing = output["result"]
    rows = [
        (
            entry["type"],
            "-" if entry["size"] is None else str(entry["size"]),
            _cell(entry["modified_at"]),
            _cell(entry["name"], limit=1024) + ("/" if entry["type"] == "directory" else ""),
        )
        for entry in listing["entries"]
    ]
    lines = _table(("TYPE", "SIZE", "MODIFIED", "NAME"), rows) if rows else ["(empty)"]
    if listing["omitted"]:
        lines.append(f"{listing['omitted']} entries with names that aren't UTF-8 are left out.")
    if not listing["complete"]:
        lines.append(
            "The directory holds more entries than the Agent reads at once; these pages hold "
            "only the ones it read."
        )
    if listing["next_after"] is not None:
        lines.append(
            "More entries follow; go on with --after "
            + _cell(listing["next_after"], limit=1024)
            + "."
        )
    return "\n".join(lines)


def render_entry(output: Mapping[str, Any]) -> str:
    entry = output["result"]
    size = "" if entry["size"] is None else f", {entry['size']} bytes"
    return (
        f"{_cell(entry['path'], limit=2048) or '/'}: {entry['type']}{size}, modified "
        f"{_cell(entry['modified_at'])}"
    )


def render_download(output: Mapping[str, Any]) -> str:
    result = output["result"]
    lines = [f"Wrote {_cell(result['output'], limit=4096)} ({result['bytes']} bytes)."]
    if result["type"] == "directory":
        lines.append(f"{result['files']} files in {result['directories']} directories.")
        if result["skipped_count"]:
            named = ", ".join(
                f"{_cell(item['member'], limit=1024)} ({item['reason']})"
                for item in result["skipped"][:_SKIPPED_SHOWN]
            )
            more = result["skipped_count"] - min(len(result["skipped"]), _SKIPPED_SHOWN)
            lines.append(
                f"Left out {result['skipped_count']}: {named}"
                + (f" and {more} more." if more else ".")
            )
    return "\n".join(lines)
