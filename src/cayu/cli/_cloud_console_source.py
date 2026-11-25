"""A Console project's source on disk, for `cayu cloud console pull` and `push`.

A pull writes a revision's files into a directory with `CHECKOUT_METADATA`, which names the
project, the revision the work started from and that revision's source digest. A push reads the
directory's source as a Console workspace save reads it (the tree's own `.gitignore` rules, less
the directories no save records and what is never source), packs it as every Console revision is
packed, a deterministic gzip tar under `source/` whose digest is its SHA-256, and checks it fits
what a Console workspace can save. Cloud and every client that writes this record read the same
format, so a checkout one makes another can push.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import unicodedata
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_project import SOURCE_BUNDLE_MAX_BYTES

#: Where a pull records what it pulled; under `.cayu`, so no push ever records it.
CHECKOUT_METADATA = Path(".cayu") / "console-pull.json"
#: Directories, at any depth, that no workspace save records.
WORKSPACE_EXCLUDED_DIRECTORIES = frozenset(
    {".git", ".venv", ".cayu", "data", "__pycache__", ".pytest_cache"}
)
#: Directories, at any depth, a push leaves out: those, and the caches and environments local
#: tools make, as `cayu cloud deploy` leaves them out.
LOCAL_ONLY_DIRECTORIES = WORKSPACE_EXCLUDED_DIRECTORIES | {
    ".hg",
    ".mypy_cache",
    ".ruff_cache",
    ".svn",
    "node_modules",
    "venv",
}
#: Files a push leaves out: local secrets and clutter. `.env.*` templates stay source.
LOCAL_ONLY_FILES = frozenset({".DS_Store", ".env"})
ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})
#: Path components no revision made outside a Console turn may hold, in any letter case: git's
#: own repository, which runs commands its configuration names, and the pull record's directory.
PROTECTED_NAMES = frozenset({".git", ".cayu"})
#: The most files and bytes a source archive may hold when it is read back.
MAX_SOURCE_FILES = 10_000
MAX_UNPACKED_SOURCE_BYTES = 512 * 1024 * 1024
#: What a Console workspace can save: at most this many paths, in a snapshot archive of at most
#: this many bytes.
MAX_CONSOLE_SOURCE_PATHS = 4096
MAX_SNAPSHOT_BYTES = 16 * 2**20
#: The longest path a Console revision holds, in UTF-8 bytes.
MAX_SOURCE_PATH_BYTES = 1024
_ROOT = "source"
_COMPILED_SUFFIXES = (".pyc", ".pyo")
_GIT_TIMEOUT_SECONDS = 60
# A Windows short (8.3) name, such as `GIT~1`, which Windows resolves to a long name.
_SHORT_NAME = re.compile(r"~[0-9]")
# Unicode's default-ignorable code points outside the format (Cf) category, which some
# filesystems (HFS+) drop when they look a name up, so `.g\u200cit` there is `.git`.
_IGNORABLE_RANGES = (
    (0x034F, 0x034F),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFFA0, 0xFFA0),
    (0xE0000, 0xE0FFF),
)
# How `git archive --format=tar` lays out a workspace snapshot: 512-byte blocks in 10 KiB
# records, a ustar header with a 100-byte name and a 155-byte prefix, and a pax header for a
# path that fits neither.
_TAR_BLOCK = 512
_TAR_RECORD = 20 * _TAR_BLOCK
_TAR_NAME_BYTES = 100
_TAR_PREFIX_BYTES = 155
# The global pax record a snapshot of a commit carries: its 40-character ID as `comment`.
_COMMIT_COMMENT_BYTES = 40
# What a source archive may decompress to: its content, and a few blocks of headers per entry.
_MAX_TAR_BYTES = MAX_UNPACKED_SOURCE_BYTES + (MAX_SOURCE_FILES + 1) * 8 * _TAR_BLOCK
_INFLATE_CHUNK_BYTES = 1024 * 1024


class ConsoleError(CloudApiError):
    """A refused Console command, with what more it reports (`details`)."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(category, message, status_code=status_code, code=code)
        self.details: dict[str, object] = dict(details or {})


@dataclass(frozen=True)
class ConsoleCheckout:
    """The project and revision a directory's source was pulled from, or last pushed as."""

    project_id: str
    revision: int
    source_digest: str

    def write(self, root: Path) -> None:
        path = root / CHECKOUT_METADATA
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "project_id": self.project_id,
            "revision": self.revision,
            "source_digest": self.source_digest,
        }
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, root: Path) -> ConsoleCheckout:
        path = root / CHECKOUT_METADATA
        try:
            payload: Any = json.loads(path.read_text())
        except FileNotFoundError:
            raise ConsoleError(
                "console_checkout_missing",
                f"{root} holds no {CHECKOUT_METADATA.as_posix()}; pull the project into it "
                "first with `cayu cloud console pull PROJECT_ID --dir DIR`.",
            ) from None
        except (OSError, ValueError) as exc:
            raise ConsoleError(
                "console_checkout_invalid", f"Could not read {path}: {exc}"
            ) from None
        if not isinstance(payload, dict):
            payload = {}
        project_id = payload.get("project_id")
        revision = payload.get("revision")
        digest = payload.get("source_digest")
        if (
            not isinstance(project_id, str)
            or not project_id
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 1
            or not isinstance(digest, str)
            or not digest.startswith("sha256:")
        ):
            raise ConsoleError(
                "console_checkout_invalid",
                f"{path} is not a Console pull record; pull the project again.",
            )
        return cls(project_id=project_id, revision=revision, source_digest=digest)


def unportable_name(part: str) -> bool:
    """Whether a path component names something else on some system a pull writes to:
    Windows reads `C:` as a drive, drops a trailing dot or space and resolves `GIT~1`."""

    return ":" in part or part != part.rstrip(". ") or _SHORT_NAME.search(part) is not None


def _ignorable(character: str) -> bool:
    point = ord(character)
    return unicodedata.category(character) == "Cf" or any(
        low <= point <= high for low, high in _IGNORABLE_RANGES
    )


def _folded(part: str) -> str:
    """``part`` as a case-insensitive filesystem that ignores ignorable characters reads it:
    HFS+ drops format characters (Unicode category Cf, such as U+200C) and other
    default-ignorable code points when it looks a name up, so `.g\u200cit` there is `.git`.
    Such characters are ordinary in names otherwise (an emoji's U+FE0F, a Persian U+200C)."""

    return "".join(character for character in part if not _ignorable(character)).casefold()


def pushed(path: str) -> bool:
    """Whether a push records `path`, a file of the tree, once `.gitignore` kept it."""

    *directories, name = path.split("/")
    if LOCAL_ONLY_DIRECTORIES.intersection(directories):
        return False
    if name in LOCAL_ONLY_FILES or name.endswith(_COMPILED_SUFFIXES):
        return False
    return not (name.startswith(".env.") and name not in ENV_TEMPLATES)


def pulled(path: str) -> bool:
    """Whether a pull writes `path`, a file of a revision.

    It never writes under a directory a push leaves out, in any letter case (on a
    case-insensitive filesystem `.VENV` is the directory's own `.venv`), nor any path naming
    `.git` or `.cayu` in any letter case or with ignorable characters (`_folded`): a revision's `.git/config` would run its commands in the next git command,
    and `.cayu` holds the pull record. A file a push leaves out for its own name, such as
    `.env.local`, is source a revision may hold, and is written.
    """

    *directories, _ = parts = path.split("/")
    if LOCAL_ONLY_DIRECTORIES.intersection(_folded(part) for part in directories):
        return False
    return not any(_folded(part) in PROTECTED_NAMES for part in parts)


def source_paths(root: Path) -> list[str]:
    """The paths of `root`'s files a push records, sorted.

    git lists the tree against a throwaway repository, so `.gitignore` files apply as a
    workspace's `git add -A` reads them, while neither an enclosing repository's index nor the
    user's global ignore rules change what is recorded.
    """

    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    with tempfile.TemporaryDirectory(prefix="cayu-console-push-") as scratch:
        git = ["git", "-c", f"core.excludesFile={os.devnull}", "-c", "core.hooksPath=/dev/null"]
        try:
            subprocess.run(
                [*git, "init", "-q", scratch],
                check=True,
                capture_output=True,
                env=environment,
                stdin=subprocess.DEVNULL,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
            listed = subprocess.run(
                [
                    *git,
                    f"--git-dir={Path(scratch) / '.git'}",
                    "--work-tree=.",
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ],
                check=True,
                capture_output=True,
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                timeout=_GIT_TIMEOUT_SECONDS,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ConsoleError(
                "source_git_failed",
                f"git could not list the source in {root}, which push needs to apply its "
                f".gitignore rules: {type(exc).__name__}",
            ) from None
    paths = [os.fsdecode(item) for item in listed.split(b"\0") if item]
    # A repository inside what a push leaves out, such as an editable package in .venv, is
    # never source.
    nested = next((path for path in paths if path.endswith("/") and pushed(path)), None)
    if nested is not None:
        raise ConsoleError(
            "source_invalid",
            f"{nested} in {root} is a git repository of its own; a revision can't hold one.",
        )
    return sorted(path for path in paths if pushed(path))


def source_tree(root: Path) -> dict[str, bytes]:
    """The files of `root` a push records, with their content."""

    files: dict[str, bytes] = {}
    for path in source_paths(root):
        target = root / path
        try:
            mode = target.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ConsoleError(
                    "source_invalid", f"{path} is a symlink, and a revision can't hold one."
                )
            if not stat.S_ISREG(mode):
                raise ConsoleError(
                    "source_invalid", f"{path} is not a regular file, and a revision can't hold it."
                )
            files[path] = target.read_bytes()
        except OSError as exc:
            raise ConsoleError(
                "source_unavailable", f"Could not read {path}: {type(exc).__name__}"
            ) from None
    return files


def write_tree(
    root: Path, files: Mapping[str, bytes], *, project_id: str, force: bool
) -> tuple[list[str], list[str]]:
    """Make `root`'s source exactly `files`, a pull of Console project `project_id`; return the
    paths of `files` a pull never writes (see `pulled`), and those it kept as they were.

    A missing `root` is created. A non-empty one is refused unless `force` and it holds a pull
    of the same project, whose source is then replaced: the files a push would record that
    `files` doesn't hold are removed, and what a push leaves out (`.git`, `.venv`, `.env`, ...)
    stays, even where `files` holds it (a local `.env.local` is never overwritten). Every path
    is checked before anything is written or removed, and the pull record is removed first and
    written by the caller last, so an interrupted pull leaves no record.
    """

    if root.exists() and not root.is_dir():
        raise ConsoleError("console_directory_invalid", f"{root} is not a directory.")
    replacing = root.exists() and any(root.iterdir())
    if replacing:
        if not force:
            raise ConsoleError(
                "console_directory_not_empty",
                f"{root} is not empty; pull into a new directory, or pass --force to replace "
                "the source of an earlier pull of this project there.",
            )
        try:
            recorded_project: str | None = ConsoleCheckout.read(root).project_id
        except CloudApiError:
            recorded_project = None
        if recorded_project != project_id:
            raise ConsoleError(
                "console_directory_not_empty",
                f"{root} is not empty and holds no pull of Console project {project_id}, so "
                "--force won't replace it; pull into a new directory.",
            )
    repository = bare_repository(root, files)
    if repository is not None:
        where = "the directory's root" if repository == "" else f"{repository!a}"
        raise ConsoleError(
            "console_source_unsafe",
            f"The revision would make {where} a bare git repository (a HEAD file beside objects/ "
            "and refs/, or beside a commondir file), whose config git reads and whose commands "
            "it runs in the pulled tree; nothing was written. Look at the revision with "
            "`cayu cloud console source`.",
        )
    written = {path: content for path, content in files.items() if pulled(path)}
    kept: list[str] = []
    stale: list[str] = []
    if replacing:
        # What a push records, as push lists it (the tree's .gitignore applied), before anything
        # changes; so never anything under .git or .cayu, and listing can refuse.
        recorded = set(source_paths(root))
        # Compared ignoring case, as a case-insensitive filesystem finds them: there a
        # revision's README.md is the checkout's Readme.md, a source file, not a local one.
        folded_recorded = {path.casefold() for path in recorded}
        # Anything already here that a push leaves out, such as `.env.local` or a file
        # .gitignore lists, is local: it is kept as it is, never overwritten.
        kept = sorted(
            path
            for path in written
            if path.casefold() not in folded_recorded
            and os.path.lexists(root / path)
            and (not (root / path).is_dir() or (root / path).is_symlink())
        )
        for path in kept:
            del written[path]
        stale = sorted(recorded - written.keys())
    for path in sorted(written):
        require_writable(root, path, removed=frozenset(stale))
    if replacing:
        (root / CHECKOUT_METADATA).unlink()
        for path in stale:
            (root / path).unlink()
    root.mkdir(parents=True, exist_ok=True)
    for path, content in sorted(written.items()):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.write_bytes(content)
    return sorted(set(files) - set(written) - set(kept)), kept


def bare_repository(root: Path, files: Mapping[str, bytes]) -> str | None:
    """The directory writing `files` into `root` would make a bare git repository, `""` for the
    root, or None.

    git takes any directory holding a `HEAD` file, an `objects/` directory and a `refs/`
    directory as a repository, and reads its `config`, which can name commands (a pager, an
    SSH command, aliases) that a git command run there executes. With a `commondir` file beside
    `HEAD` it takes `objects`, `refs` and `config` from the directory that file names instead.
    So no directory may get a `HEAD` file beside `objects/` and `refs/`, or beside a
    `commondir` file, counting what it already holds on disk. Names are compared as a
    case-insensitive filesystem that ignores ignorable characters reads them, as `pulled`
    compares them; a lone `HEAD` file is fine.
    """

    # Directories are keyed by their folded names, as HFS+ reads them, so a revision can't
    # split one directory across spellings (`vendor/HEAD` and `vend\u200cor/objects/...`).
    found: dict[tuple[str, ...], set[str]] = {}
    shown: dict[tuple[str, ...], str] = {}
    for path in files:
        raw = path.split("/")
        parts = [_folded(part) for part in raw]
        for depth, name in enumerate(parts):
            last = depth == len(parts) - 1
            if (last and name in {"head", "commondir"}) or (
                not last and name in {"objects", "refs"}
            ):
                key = tuple(parts[:depth])
                found.setdefault(key, set()).add(name)
                shown.setdefault(key, "/".join(raw[:depth]))
    for key, names in sorted(found.items()):
        for directory in _folded_directories(root, key):
            for entry in directory.iterdir():
                name = _folded(entry.name)
                is_directory = entry.is_dir() and not entry.is_symlink()
                if (name in {"head", "commondir"} and not is_directory) or (
                    name in {"objects", "refs"} and is_directory
                ):
                    names.add(name)
        if names >= {"head", "objects", "refs"} or names >= {"head", "commondir"}:
            return shown[key]
    return None


def _folded_directories(root: Path, key: tuple[str, ...]) -> list[Path]:
    """The directories under `root`, never through a symlink, whose path reads as `key` once
    each name is folded."""

    directories = [root] if root.is_dir() and not root.is_symlink() else []
    for name in key:
        directories = [
            entry
            for directory in directories
            for entry in directory.iterdir()
            if entry.is_dir() and not entry.is_symlink() and _folded(entry.name) == name
        ]
    return directories


def require_writable(root: Path, path: str, *, removed: frozenset[str] = frozenset()) -> None:
    """Refuse `path` when writing it would leave `root`, follow a symlink, write under a file
    (other than one of `removed`, which the pull removes first) or replace a directory; every
    component must be one plain name on every system."""

    if any(
        PurePath(part).anchor or PurePath(part).parts != (part,) or unportable_name(part)
        for part in path.split("/")
    ):
        raise ConsoleError(
            "console_directory_invalid",
            f"{path!a} can't be written: it names a drive, a path separator, a name ending in a "
            "dot or a space, or a short name such as GIT~1.",
        )
    current = root
    for part in path.split("/")[:-1]:
        current = current / part
        if current.relative_to(root).as_posix() in removed and not current.is_symlink():
            break
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ConsoleError(
                "console_directory_invalid",
                f"{path} can't be written: {current.relative_to(root).as_posix()} is a "
                "symlink or a file in this directory; move it away and pull again.",
            )
    target = root / path
    if target.is_dir() and not target.is_symlink():
        raise ConsoleError(
            "console_directory_invalid",
            f"{path} can't be written: it is a directory here; move it away and pull again.",
        )


def source_archive(files: Mapping[str, bytes]) -> bytes:
    """The deterministic, `source/`-rooted gzip tar every Console revision is stored as."""

    entries = sorted(validated_source(files).items())
    long = next((path for path, _ in entries if len(path.encode()) > MAX_SOURCE_PATH_BYTES), None)
    if long is not None:
        raise ConsoleError(
            "source_invalid",
            f"it holds a path of {len(long.encode())} bytes ({long[:80]}...), and a Console "
            f"revision's paths are at most {MAX_SOURCE_PATH_BYTES} bytes.",
        )
    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as bundle,
    ):
        for path, content in entries:
            info = tarfile.TarInfo(f"{_ROOT}/{path}")
            info.size = len(content)
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(content))
    archive = output.getvalue()
    if len(archive) > SOURCE_BUNDLE_MAX_BYTES:
        raise ConsoleError(
            "source_invalid",
            f"The source is larger than the {SOURCE_BUNDLE_MAX_BYTES // 2**20} MiB compressed "
            "source bundle limit.",
        )
    return archive


def source_digest(archive: bytes) -> str:
    return "sha256:" + hashlib.sha256(archive).hexdigest()


def source_files(archive: bytes) -> dict[str, bytes]:
    """The files of a Console source archive, refused once it holds more than
    `MAX_SOURCE_FILES` entries or unpacks past its bounds."""

    try:
        tar = _inflated(archive)
        files: dict[str, bytes] = {}
        unpacked = 0
        with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as bundle:
            for count, member in enumerate(bundle, start=1):
                if not member.isfile():
                    raise ValueError(f"it holds a non-file entry: {member.name!r}")
                if count > MAX_SOURCE_FILES:
                    raise ValueError(f"it holds more than {MAX_SOURCE_FILES} files")
                unpacked += member.size
                if unpacked > MAX_UNPACKED_SOURCE_BYTES:
                    raise ValueError(
                        f"it unpacks to more than {MAX_UNPACKED_SOURCE_BYTES // 2**20} MiB"
                    )
                top, _, path = member.name.partition("/")
                if top != _ROOT:
                    raise ValueError(f"it holds an entry outside {_ROOT}/")
                if path in files:
                    raise ValueError(f"it holds {path} twice")
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"{path} can't be read")
                files[path] = extracted.read()
        return validated_source(files)
    except ConsoleError as exc:
        raise ConsoleError(
            "api_response_invalid", f"Cayu Cloud returned an unusable source archive: {exc}"
        ) from None
    except (OSError, EOFError, ValueError, zlib.error, tarfile.TarError) as exc:
        raise ConsoleError(
            "api_response_invalid", f"Cayu Cloud returned an unreadable source archive: {exc}"
        ) from None


def _inflated(archive: bytes) -> bytes:
    """`archive`'s tar stream, decompressed no further than a source archive can reach."""

    inflate = zlib.decompressobj(wbits=31)
    output = bytearray()
    pending = archive
    while pending:
        output += inflate.decompress(pending, _INFLATE_CHUNK_BYTES)
        if len(output) > _MAX_TAR_BYTES:
            raise ValueError("it decompresses past what a source archive holds")
        pending = inflate.unconsumed_tail
        if inflate.eof:
            break
    if not inflate.eof:
        raise ValueError("it ends before its gzip stream does")
    return bytes(output)


def validated_source(files: Mapping[str, bytes]) -> dict[str, bytes]:
    if not files:
        raise ConsoleError("source_invalid", "A Console source holds at least one file.")
    if len(files) > MAX_SOURCE_FILES:
        raise ConsoleError(
            "source_invalid", f"A Console source holds at most {MAX_SOURCE_FILES} files."
        )
    for path in files:
        if (
            not path
            or path.startswith(("/", "\\"))
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ConsoleError("source_invalid", f"The source has an invalid path: {path!r}")
    if sum(len(content) for content in files.values()) > MAX_UNPACKED_SOURCE_BYTES:
        raise ConsoleError(
            "source_invalid",
            f"The source unpacks to more than {MAX_UNPACKED_SOURCE_BYTES // 2**20} MiB.",
        )
    for path in files:
        parts = path.split("/")
        for depth in range(1, len(parts)):
            parent = "/".join(parts[:depth])
            if parent in files:
                raise ConsoleError(
                    "source_invalid",
                    f"The source has {parent} as both a file and a directory of {path}.",
                )
    return dict(files)


def console_source_problem(files: Mapping[str, bytes]) -> str | None:
    """Why `files` can't be a Console revision made outside a turn, or None.

    It may name no `PROTECTED_NAMES` component nor a name another system reads differently,
    and must fit what a Console workspace can save: at most `MAX_CONSOLE_SOURCE_PATHS` files,
    in a committed snapshot archive of at most `MAX_SNAPSHOT_BYTES`.
    """

    protected = next(
        (
            path
            for path in sorted(files)
            if any(part.casefold() in PROTECTED_NAMES for part in path.split("/"))
        ),
        None,
    )
    if protected is not None:
        return f"it holds {protected}, and no revision holds a .git or .cayu path."
    unportable = next(
        (path for path in sorted(files) if any(map(unportable_name, path.split("/")))), None
    )
    if unportable is not None:
        return (
            f"it holds {unportable}, and no revision path holds a ':', a name ending in a dot "
            "or a space, or a short name such as GIT~1."
        )
    if len(files) > MAX_CONSOLE_SOURCE_PATHS:
        return (
            f"it holds {len(files)} files, and a Console revision holds at most "
            f"{MAX_CONSOLE_SOURCE_PATHS}."
        )
    size = snapshot_size(files)
    if size > MAX_SNAPSHOT_BYTES:
        return (
            f"it is {size / 2**20:.1f} MiB as an uncompressed archive, and a Console workspace "
            f"saves at most {MAX_SNAPSHOT_BYTES // 2**20} MiB."
        )
    return None


def snapshot_size(files: Mapping[str, bytes]) -> int:
    """The exact size of the `git archive --format=tar` of a commit of `files`: per directory
    and per file, a pax header when the path fits neither the name nor a prefix split, a
    header, and the content, in 512-byte blocks, after the commit's global header; then at
    least two zero blocks, up to a whole 10 KiB record."""

    directories = {
        "/".join(path.split("/")[:depth]) + "/"
        for path in files
        for depth in range(1, path.count("/") + 1)
    }
    size = _pax_header_size("comment", _COMMIT_COMMENT_BYTES)
    for path in (*directories, *files):
        name = len(path.encode())
        if name > _TAR_NAME_BYTES and not _fits_prefix_split(path.encode()):
            size += _pax_header_size("path", name)
        size += _TAR_BLOCK + _blocks(len(files.get(path, b"")))
    tail = _TAR_RECORD - size % _TAR_RECORD
    return size + tail + (_TAR_RECORD if tail < 2 * _TAR_BLOCK else 0)


def _fits_prefix_split(path: bytes) -> bool:
    """Whether git stores `path` in a ustar header's prefix and name, split at a slash."""

    end = len(path) - 1 if len(path) > 1 and path.endswith(b"/") else len(path)
    split = min(end, _TAR_PREFIX_BYTES)
    while True:
        split -= 1
        if split <= 0 or path[split : split + 1] == b"/":
            break
    return split > 0 and len(path) - split - 1 <= _TAR_NAME_BYTES


def _pax_header_size(keyword: str, value_bytes: int) -> int:
    """A pax extended header of one record, `"<length> <keyword>=<value>\\n"`."""

    length = 1 + 1 + len(keyword) + 1 + value_bytes + 1
    digits = 1
    while length // 10**digits:
        length += 1
        digits += 1
    return _TAR_BLOCK + _blocks(length)


def _blocks(size: int) -> int:
    return -(-size // _TAR_BLOCK) * _TAR_BLOCK
