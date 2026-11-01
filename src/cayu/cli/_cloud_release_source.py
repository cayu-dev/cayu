"""`cayu cloud deployment source RELEASE --output DIR [--archive]`: a Release's uploaded source.

Cloud keeps the source bundle `cayu cloud deploy` uploaded for a Release and serves it at
`GET /v1/applications/{id}/deployments/{release}/source`: the gzip-compressed tar exactly as
uploaded, with its SHA-256 in `X-Cayu-Content-Digest`. The CLI streams it to a temporary file,
checks the digest and refuses a mismatch, then either extracts it into DIR or, with
`--archive`, puts the verified archive at the `--output` path.

Extraction never trusts the archive. Every member is checked before anything is written:
names stay inside DIR (no absolute paths, no `..`, no duplicates, nothing written through a
link), only files, directories and links are allowed, and every link has to resolve inside
DIR, following the archive's other links. Names are compared as a case-insensitive file system
(macOS, Windows) sees them, so two that differ only in case are duplicates there too. At most
`MAX_EXTRACTED_MEMBERS` members and `MAX_EXTRACTED_BYTES` of files are extracted, counted from
the headers as they are read, after the tar is decompressed under a bound and its extension
headers (long names, PAX records) are checked against `_MAX_EXTENSION_BYTES`. Python's `data`
extraction filter then applies its own checks; a Python without it (3.11 before 3.11.4) only
keeps the archive (`--archive`). Everything is downloaded and staged in hidden `.cayu-source-`
entries inside DIR, so a DIR that is a mount point or under a read-only parent works, and is
published only once all of it is written, relative to directory descriptors that never follow
a link below DIR (so Windows only keeps the archive): every name is created exclusively
(`_publish`), so nothing another process put in DIR meanwhile is ever replaced. If publication
fails, already-published entries remain for inspection: deleting by name could remove another
process's replacement. An existing empty DIR stays the same directory. The bundle
keeps the project under `source/`; extraction puts that directory's contents in DIR.

A Release built from a connected repository has no uploaded bundle (`source_not_uploaded`):
its source is that repository at the recorded commit.
"""

from __future__ import annotations

import contextlib
import errno
import gzip
import os
import re
import shlex
import shutil
import stat
import tarfile
import tempfile
import unicodedata
import zlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import quote, urlsplit

from cayu.cli._cloud_api import CloudApiClient, CloudApiError, DownloadedFile
from cayu.cli._cloud_data import _publish_new
from cayu.cli._cloud_project import SOURCE_BUNDLE_MAX_BYTES
from cayu.cli._cloud_settings import _cell

_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")
_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
_REPOSITORY_LIMIT = 512
_BUNDLE_ROOT = "source"
# Hops a link may take through the archive's other links before it counts as a loop.
_LINK_HOPS = 40
#: What one extraction writes at most: Cloud admits no larger unpacked source.
MAX_EXTRACTED_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_MEMBERS = 100_000
_MIB = 1024 * 1024
# What a path the archive doesn't list is when it holds another member: a directory.
_DIRECTORY = tarfile.TarInfo()
_DIRECTORY.type = tarfile.DIRTYPE


class SourceDownloadError(CloudApiError):
    """A source download the CLI refused or couldn't finish, with Cloud's code if it had one.

    ``details`` are the error's other fields: the repository and revision of a Release built
    from a connected repository.
    """

    def __init__(
        self, category: str, message: str, *, details: dict[str, str] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(category, message, **kwargs)
        self.details = details or {}


def download_release_source(
    client: CloudApiClient,
    *,
    application_id: str,
    deployment_id: str,
    output: Path,
    archive: bool,
) -> dict[str, Any]:
    """Download the Release's source bundle, verify it, and extract it or keep the archive."""

    # Absolute, so DIR's parent is never DIR itself (`--output .`).
    output = Path(os.path.abspath(output.expanduser()))
    if not archive and not hasattr(tarfile, "data_filter"):
        raise SourceDownloadError(
            "source_extraction_unsupported",
            "This Python can't extract an archive safely (it needs 3.11.4 or newer); "
            "pass --archive to keep the archive instead.",
        )
    if not archive and not _DIRECTORY_RELATIVE:
        raise SourceDownloadError(
            "source_extraction_unsupported",
            "This platform can't publish extracted files relative to the output directory "
            "(Windows); pass --archive to keep the archive instead.",
        )
    if archive:
        _refuse_existing_file(output)
    else:
        _refuse_nonempty_directory(output)
    # The archive sits beside the file it becomes; for extraction it is downloaded into DIR
    # itself, which is created now when missing and removed again if nothing lands in it.
    created = not archive and not output.is_dir()
    try:
        if archive:
            output.parent.mkdir(parents=True, exist_ok=True)
        else:
            output.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=output.parent if archive else output, prefix=_TEMPORARY_PREFIX
        )
    except OSError as exc:
        raise _unwritable(output, exc) from None
    succeeded = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            downloaded = _download(client, application_id, deployment_id, stream)
        expected = _expected_digest(downloaded.headers)
        if downloaded.sha256 != expected:
            raise SourceDownloadError(
                "source_digest_mismatch",
                f"The downloaded source of {deployment_id} doesn't match the digest Cayu Cloud "
                f"sent (sha256:{expected}); nothing was written. Run the command again.",
            )
        if archive:
            # The archive is kept as Cloud stores it; only extraction needs its members safe.
            counts = _archive_counts(temporary, output.parent, deployment_id)
            try:
                _publish_new(temporary, output)
            except FileExistsError:
                raise _exists(output) from None
            except OSError as exc:
                raise _unwritable(output, exc) from None
        else:
            counts = _extract(temporary, output, deployment_id)
        succeeded = True
    finally:
        Path(temporary).unlink(missing_ok=True)
        if created and not succeeded:
            with contextlib.suppress(OSError):
                output.rmdir()
    return {
        "application_id": application_id,
        "deployment_id": deployment_id,
        "output": str(output),
        "format": "archive" if archive else "directory",
        "digest": f"sha256:{downloaded.sha256}",
        "bytes": downloaded.size,
        **counts,
    }


def _download(
    client: CloudApiClient, application_id: str, deployment_id: str, stream: BinaryIO
) -> DownloadedFile:
    path = (
        f"/v1/applications/{quote(application_id, safe='')}/deployments/"
        f"{quote(deployment_id, safe='')}/source"
    )
    try:
        return client.download(path, stream, max_bytes=SOURCE_BUNDLE_MAX_BYTES)
    except CloudApiError as exc:
        raise _refusal(exc, deployment_id) from None


def _refusal(exc: CloudApiError, deployment_id: str) -> CloudApiError:
    """Cloud's refusal, said in terms of what the caller can do next."""

    if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
        return SourceDownloadError(
            "source_download_unsupported",
            "This Cayu Cloud can't download a release's source yet.",
            status_code=exc.status_code,
        )
    if exc.category == "api_response_too_large":
        return SourceDownloadError(
            "source_bundle_invalid",
            f"Cayu Cloud sent more than {SOURCE_BUNDLE_MAX_BYTES // _MIB} MiB for the source of "
            f"{deployment_id}, more than a source bundle can be; nothing was written.",
        )
    if exc.code == "source_not_uploaded":
        body = exc.body or {}
        repository = _repository(body.get("repository"))
        revision = _shown(body.get("revision"), _REVISION)
        if repository is not None and revision is not None:
            message = (
                f"Release {deployment_id} was built from {repository} at {revision}, so Cayu "
                "Cloud keeps no uploaded source for it. Fetch that commit from the repository "
                f"instead, for example `git clone {shlex.quote(repository)}` and then `git "
                f"checkout {shlex.quote(revision)}`."
            )
        else:
            message = (
                f"Release {deployment_id} was built from a connected repository, so Cayu Cloud "
                "keeps no uploaded source for it. Fetch the commit `cayu cloud deployment "
                "status` names from that repository."
            )
        return SourceDownloadError(
            exc.category,
            message,
            status_code=exc.status_code,
            code=exc.code,
            body=exc.body,
            details={
                key: value
                for key, value in (("repository", repository), ("revision", revision))
                if value is not None
            },
        )
    return exc


def _repository(value: object) -> str | None:
    """A repository Cloud names, shown only without control characters or a credential."""

    if not isinstance(value, str) or not value or len(value) > _REPOSITORY_LIMIT:
        return None
    if any(unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"} for character in value):
        return None
    if " " in value or "@" in urlsplit(value).netloc:
        return None
    return value


def _shown(value: object, pattern: re.Pattern[str]) -> str | None:
    return value if isinstance(value, str) and pattern.fullmatch(value) is not None else None


def _expected_digest(headers: Mapping[str, str]) -> str:
    match = _DIGEST.fullmatch(headers.get("x-cayu-content-digest", "").strip())
    if match is None:
        raise SourceDownloadError(
            "api_response_invalid",
            "Cayu Cloud sent the source without a valid X-Cayu-Content-Digest, so it can't be "
            "verified; nothing was written.",
        )
    return match.group(1)


def _archive_counts(
    archive_path: str, directory: Path, deployment_id: str
) -> dict[str, int | None]:
    """What a kept archive holds, read under the same bounds as an extraction. A bundle past
    them is still kept for inspection (that's what `--archive` is for), and its counts are
    then unknown (None); one that isn't a tar.gz at all is refused."""

    tar_path = None
    try:
        tar_path = _decompressed(archive_path, directory, deployment_id)
        _refuse_large_metadata(tar_path, deployment_id)
        with tarfile.open(tar_path, mode="r:") as bundle:
            return dict(_counts(_bounded_members(bundle, deployment_id)))
    except SourceDownloadError as exc:
        if exc.category == "source_bundle_invalid":
            raise
        return {"files": None, "links": None, "file_bytes": None}
    except (OSError, EOFError, tarfile.TarError):
        raise _invalid(deployment_id) from None
    finally:
        if tar_path is not None:
            Path(tar_path).unlink(missing_ok=True)


def _bounded_members(bundle: tarfile.TarFile, deployment_id: str) -> list[tarfile.TarInfo]:
    """The bundle's members, read header by header and refused as soon as there are more of
    them, or more file bytes, than one extraction writes."""

    members: list[tarfile.TarInfo] = []
    size = 0
    while (member := bundle.next()) is not None:
        members.append(member)
        size += member.size if member.isreg() else 0
        if len(members) > MAX_EXTRACTED_MEMBERS or size > MAX_EXTRACTED_BYTES:
            raise _too_large(deployment_id)
    return members


def _folded(name: tuple[str, ...]) -> tuple[str, ...]:
    """``name`` as a case-insensitive, normalization-insensitive file system compares it."""

    return tuple(
        unicodedata.normalize("NFC", unicodedata.normalize("NFC", part).casefold()) for part in name
    )


def _invalid(deployment_id: str) -> SourceDownloadError:
    return SourceDownloadError(
        "source_bundle_invalid",
        f"The source of {deployment_id} isn't a readable .tar.gz archive; nothing was written.",
    )


def _checked(members: list[tarfile.TarInfo], deployment_id: str) -> list[tarfile.TarInfo]:
    """The members to extract, checked and renamed to their places in the output directory."""

    names = [_normalized(member, deployment_id) for member in members]
    # The upload keeps the project under source/; its contents are what lands in DIR.
    strip = bool(names) and all(
        name == (_BUNDLE_ROOT,) or (len(name) > 1 and name[0] == _BUNDLE_ROOT) for name in names
    )
    kept: list[tarfile.TarInfo] = []
    seen: dict[tuple[str, ...], tarfile.TarInfo] = {}
    links: dict[tuple[str, ...], str] = {}
    for member, name in zip(members, names, strict=True):
        if strip:
            if name == (_BUNDLE_ROOT,):
                if not member.isdir():
                    raise _unsafe(deployment_id, member.name, "is not a directory")
                continue
            name = name[1:]
        if not name:
            if member.isdir():
                continue
            raise _unsafe(deployment_id, member.name, "has no name")
        key = _folded(name)
        if key in seen:
            raise _unsafe(
                deployment_id,
                member.name,
                "appears more than once (names that differ only in case are the same file on "
                "macOS and Windows)",
            )
        if member.issym():
            links[key] = member.linkname
        elif member.islnk():
            target = _link_member(member, strip, deployment_id)
            original = seen.get(_folded(target))
            if original is None or not original.isreg():
                raise _unsafe(deployment_id, member.name, "links to a file not before it")
            member.linkname = "/".join(target)
        elif not (member.isreg() or member.isdir()):
            raise _unsafe(deployment_id, member.name, "is a device, fifo or other special file")
        member.name = "/".join(name)
        seen[key] = member
        kept.append(member)
    for key, member in seen.items():
        # Only directories hold other members, whichever of them the archive lists first.
        if any(not seen.get(key[:depth], _DIRECTORY).isdir() for depth in range(1, len(key))):
            raise _unsafe(deployment_id, member.name, "would be written through a link or file")
    for key, target in links.items():
        if target.startswith("/") or _resolved(key[:-1], target, links) is None:
            raise _unsafe(deployment_id, seen[key].name, "links outside the output directory")
    return kept


def _normalized(member: tarfile.TarInfo, deployment_id: str) -> tuple[str, ...]:
    name = member.name
    if (
        name.startswith(("/", "\\"))
        or re.match(r"[A-Za-z]:", name) is not None
        or "\x00" in name
        or "\\" in name
    ):
        raise _unsafe(deployment_id, name, "has an absolute or unsupported path")
    parts = tuple(part for part in name.split("/") if part not in {"", "."})
    if ".." in parts:
        raise _unsafe(deployment_id, name, "leads out of the output directory")
    return parts


def _link_member(member: tarfile.TarInfo, strip: bool, deployment_id: str) -> tuple[str, ...]:
    target = tarfile.TarInfo(member.linkname)
    parts = _normalized(target, deployment_id)
    if strip:
        if not parts or parts[0] != _BUNDLE_ROOT:
            raise _unsafe(deployment_id, member.name, "links outside the output directory")
        parts = parts[1:]
    return parts


def _resolved(
    directory: tuple[str, ...], target: str, links: Mapping[tuple[str, ...], str]
) -> tuple[str, ...] | None:
    """Where the link ``target`` in ``directory`` leads within the archive, or None outside.

    It follows the archive's own links the way the file system will once they are written,
    so a chain of links that each stay inside but together climb out is caught too. Names are
    compared folded (`_folded`), as ``links`` is keyed, so `UP` follows a link named `up`.
    """

    hops = 0
    pending = list(target.split("/"))
    resolved: list[str] = list(directory)
    while pending:
        part = pending.pop(0)
        if part in {"", "."}:
            continue
        if part == "..":
            if not resolved:
                return None
            resolved.pop()
            continue
        candidate = _folded((*resolved, part))
        followed = links.get(candidate)
        if followed is None:
            resolved.append(part)
            continue
        hops += 1
        if hops > _LINK_HOPS or followed.startswith("/"):
            return None
        pending = followed.split("/") + pending
    return tuple(resolved)


def _counts(members: list[tarfile.TarInfo]) -> dict[str, int]:
    files = [member for member in members if member.isreg()]
    return {
        "files": len(files),
        "links": sum(1 for member in members if member.issym() or member.islnk()),
        "file_bytes": sum(member.size for member in files),
    }


def _extract(archive_path: str, output: Path, deployment_id: str) -> dict[str, int]:
    """Extract the checked bundle into ``output``, an empty directory that exists by now.

    Everything is staged inside ``output`` itself (hidden `.cayu-source-` entries), so a DIR
    that is a mount point or whose parent isn't writable works, and only then published.
    """

    tar_path = None
    staging = None
    try:
        tar_path = _decompressed(archive_path, output, deployment_id)
        _refuse_large_metadata(tar_path, deployment_id)
        with tarfile.open(tar_path, mode="r:") as bundle:
            try:
                members = _checked(_bounded_members(bundle, deployment_id), deployment_id)
            except (OSError, EOFError, tarfile.TarError):
                raise _invalid(deployment_id) from None
            staging = Path(tempfile.mkdtemp(dir=output, prefix=_TEMPORARY_PREFIX))
            try:
                # A Python without the filter never gets here (`download_release_source`).
                bundle.extractall(staging, members=members, filter="data")
            except tarfile.TarError:
                # Python's `data` filter refused a member the checks above let through.
                raise SourceDownloadError(
                    "source_bundle_unsafe",
                    f"The source of {deployment_id} has a member that can't be extracted "
                    "safely; nothing was written. `--archive` keeps the archive as it is.",
                ) from None
        _publish(staging, output)
    except (EOFError, tarfile.ReadError, tarfile.CompressionError):
        raise _invalid(deployment_id) from None
    except FileExistsError:
        raise SourceDownloadError(
            "output_exists",
            f"Something else wrote into {output} while the source was extracted. "
            "Partially published source files may remain; inspect the directory and retry "
            "with a new or empty --output directory.",
        ) from None
    except OSError as exc:
        raise SourceDownloadError(
            "output_unwritable",
            f"Could not write {output}: {exc.strerror or type(exc).__name__}. "
            "Partially published source files may remain; inspect the directory and retry "
            "with a new or empty --output directory.",
        ) from None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if tar_path is not None:
            Path(tar_path).unlink(missing_ok=True)
    return _counts(members)


#: Cloud's own bound on one tar extension header (a long name, PAX records).
_MAX_EXTENSION_BYTES = 64 * 1024
#: Headers and padding on top of the file bytes: at most two 512-byte blocks per member,
#: plus each member's extension headers.
_MAX_TAR_BYTES = MAX_EXTRACTED_BYTES + MAX_EXTRACTED_MEMBERS * 2 * 512 + 64 * _MIB
_TEMPORARY_PREFIX = ".cayu-source-"


def _decompressed(archive_path: str, output: Path, deployment_id: str) -> str:
    """The bundle's tar, decompressed beside the staging and refused past `_MAX_TAR_BYTES`, so
    neither tarfile's metadata reads nor the disk see more than that."""

    descriptor, tar_path = tempfile.mkstemp(dir=output, prefix=_TEMPORARY_PREFIX)
    try:
        with os.fdopen(descriptor, "wb") as target, gzip.open(archive_path, "rb") as source:
            written = 0
            while chunk := source.read(_MIB):
                written += len(chunk)
                if written > _MAX_TAR_BYTES:
                    raise _too_large(deployment_id)
                target.write(chunk)
    except (OSError, EOFError, zlib.error) as exc:
        Path(tar_path).unlink(missing_ok=True)
        if isinstance(exc, gzip.BadGzipFile | EOFError | zlib.error):
            raise _invalid(deployment_id) from None
        raise
    except BaseException:
        Path(tar_path).unlink(missing_ok=True)
        raise
    return tar_path


def _refuse_large_metadata(tar_path: str, deployment_id: str) -> None:
    """Refuse a tar whose extension headers (GNU long names, PAX records) are larger than
    `_MAX_EXTENSION_BYTES`, read from the raw headers before tarfile reads any of them."""

    with open(tar_path, "rb") as stream:
        while len(header := stream.read(512)) == 512 and header.strip(b"\0"):
            size = _header_size(header[124:136], deployment_id)
            if header[156:157] in {b"L", b"K", b"x", b"g"} and size > _MAX_EXTENSION_BYTES:
                raise SourceDownloadError(
                    "source_bundle_unsafe",
                    f"The source of {deployment_id} has tar metadata larger than "
                    f"{_MAX_EXTENSION_BYTES // 1024} KiB, so it isn't extracted and nothing was "
                    "written. `--archive` keeps the archive as it is.",
                )
            stream.seek((size + 511) // 512 * 512, os.SEEK_CUR)


def _header_size(field: bytes, deployment_id: str) -> int:
    """A tar header's size field: octal digits, or GNU base-256 when its first byte is 0x80."""

    if field[:1] == b"\x80":
        return int.from_bytes(field[1:], "big")
    if field[:1] == b"\xff":
        # A negative base-256 size is never a valid member.
        raise _invalid(deployment_id)
    digits = field.split(b"\0", 1)[0].strip(b" ")
    # Octal digits only: `int` would also take a sign, and a negative size would seek back.
    if digits.strip(b"01234567"):
        raise _invalid(deployment_id)
    return int(digits, 8) if digits else 0


def _too_large(deployment_id: str) -> SourceDownloadError:
    return SourceDownloadError(
        "source_bundle_too_large",
        f"The source of {deployment_id} unpacks to more than {MAX_EXTRACTED_MEMBERS} "
        f"members or {MAX_EXTRACTED_BYTES // _MIB} MiB, so it isn't extracted and "
        "nothing was written. `--archive` keeps the archive as it is.",
    )


#: Publication works only relative to directory descriptors, so a path below DIR is never
#: followed (Windows has none of these).
_DIRECTORY_FUNCTIONS = (
    os.open,
    os.mkdir,
    os.symlink,
    os.link,
    os.readlink,
    os.stat,
    os.unlink,
    os.rmdir,
)
_DIRECTORY_RELATIVE = (
    all(function in os.supports_dir_fd for function in _DIRECTORY_FUNCTIONS)
    and os.stat in os.supports_follow_symlinks
)
_OPEN_DIRECTORY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _publish(staging: Path, output: Path) -> None:
    """Copy what ``staging`` holds into ``output`` without ever replacing anything there.

    Everything works relative to open directory descriptors (`_DIRECTORY_RELATIVE`), never a
    path below DIR, and each directory is opened with `O_NOFOLLOW`, so a directory another user
    swaps for a link mid-publication is never followed out of DIR. Every name is created
    exclusively: a directory with `mkdir`, a link with `symlink`, a file with a hard link or an
    `O_EXCL` copy. Each fails if the name exists, so a file another process put in DIR meanwhile
    is never touched. On failure, published entries remain. There is no portable atomic
    unlink-if-inode-matches, so a name-based rollback could delete concurrent replacements.
    """

    output_fd = os.open(output, _OPEN_DIRECTORY)
    try:
        staging_fd = os.open(staging.name, _OPEN_DIRECTORY, dir_fd=output_fd)
        try:
            if any(
                name != staging.name and not name.startswith(_TEMPORARY_PREFIX)
                for name in os.listdir(output_fd)
            ):
                raise FileExistsError(errno.ENOTEMPTY, os.strerror(errno.ENOTEMPTY), str(output))
            _publish_tree(staging_fd, output_fd)
        finally:
            os.close(staging_fd)
    finally:
        os.close(output_fd)


def _publish_tree(source_fd: int, destination_fd: int) -> None:
    """Publish one directory without replacing or rolling back destination entries."""

    for name in sorted(os.listdir(source_fd)):
        found = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        mode = stat.S_IMODE(found.st_mode)
        if stat.S_ISLNK(found.st_mode):
            os.symlink(os.readlink(name, dir_fd=source_fd), name, dir_fd=destination_fd)
        elif stat.S_ISDIR(found.st_mode):
            os.mkdir(name, 0o700, dir_fd=destination_fd)
            child_fd = os.open(name, _OPEN_DIRECTORY, dir_fd=destination_fd)
            try:
                if os.listdir(child_fd):
                    # Not the directory just made: someone else's took its name.
                    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), name)
                source_child = os.open(name, _OPEN_DIRECTORY, dir_fd=source_fd)
                try:
                    _publish_tree(source_child, child_fd)
                finally:
                    os.close(source_child)
                os.fchmod(child_fd, mode)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(found.st_mode):
            _publish_file(name, source_fd, destination_fd, mode, found)


def _publish_file(
    name: str, source_fd: int, destination_fd: int, mode: int, found: os.stat_result
) -> os.stat_result:
    """A hard link where the file system has them, which never replaces a name; else a copy
    into a file created with `O_EXCL`. Returns what was created."""

    try:
        os.link(name, name, src_dir_fd=source_fd, dst_dir_fd=destination_fd)
        return found
    except FileExistsError:
        raise
    except OSError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    target = os.open(name, flags, 0o600, dir_fd=destination_fd)
    try:
        made = os.fstat(target)
        source = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=source_fd)
        with open(source, "rb") as reader, open(target, "wb", closefd=False) as writer:
            shutil.copyfileobj(reader, writer, _MIB)
        os.fchmod(target, mode)
    finally:
        os.close(target)
    return made


def _refuse_existing_file(output: Path) -> None:
    if output.is_dir():
        raise SourceDownloadError(
            "output_exists",
            f"{output} is a directory; name a new file with --output, or leave out --archive to "
            "extract into a directory.",
        )
    if output.exists() or output.is_symlink():
        raise _exists(output)


def _refuse_nonempty_directory(output: Path) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise SourceDownloadError(
            "output_exists",
            f"{output} exists and isn't a directory; name a new or empty directory with "
            "--output, or pass --archive to write the archive itself.",
        )
    if output.is_dir() and any(output.iterdir()):
        raise SourceDownloadError(
            "output_exists",
            f"{output} isn't empty; name a new or empty directory with --output.",
        )


def _exists(output: Path) -> SourceDownloadError:
    return SourceDownloadError("output_exists", f"{output} exists; name a new file with --output.")


def _unwritable(output: Path, error: OSError) -> SourceDownloadError:
    return SourceDownloadError(
        "output_unwritable",
        f"Could not write {output}: {error.strerror or type(error).__name__}.",
    )


def _unsafe(deployment_id: str, member: str, reason: str) -> SourceDownloadError:
    return SourceDownloadError(
        "source_bundle_unsafe",
        f"The source of {deployment_id} has a member that {reason} "
        f"({_cell(PurePosixPath(member).as_posix(), limit=256)}), so it isn't extracted and "
        "nothing was written. `--archive` keeps the archive as it is.",
    )


def render_source(output: Mapping[str, Any]) -> str:
    result = output["result"]
    path = _cell(result["output"], limit=4096)
    if result["files"] is None:
        # A kept archive too large to count within the extraction bounds.
        return (
            f"Wrote {path} ({result['bytes']} bytes; too large to count its members).\n"
            f"Verified digest {result['digest']}."
        )
    files = f"{result['files']} file{'' if result['files'] == 1 else 's'}"
    if result["links"]:
        files += f" and {result['links']} link{'' if result['links'] == 1 else 's'}"
    if result["format"] == "archive":
        lines = [f"Wrote {path} ({result['bytes']} bytes, {files})."]
    else:
        lines = [
            f"Extracted {files} ({result['file_bytes']} bytes) from release "
            f"{_cell(result['deployment_id'])} into {path}."
        ]
    lines.append(f"Verified digest {result['digest']}.")
    return "\n".join(lines)
