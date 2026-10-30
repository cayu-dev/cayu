"""Repositories an application registers so a workspace can be prepared at one revision.

An eval case names a source and a full commit id in its workspace fixture; the
eval runner asks the source to check that revision out into a fresh directory
for every trial. :class:`GitWorkspaceSource` checks revisions out of a local Git
repository (a working checkout or a bare mirror the application keeps current).
:class:`GitRemoteWorkspaceSource` keeps its own bare mirror of a Git remote and
fetches a commit the mirror lacks, with credentials the application resolves
from its vault.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import stat
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from cayu._filesystem_lock import cooperative_path_lock
from cayu._validation import require_durable_clean_nonblank
from cayu.delivery import _git_process
from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.local import LocalRunner
from cayu.vaults.base import SecretEnv, SecretRef, SecretResolver, validate_secret_resolver

#: How long one Git command of a checkout may run, in seconds, unless the source says otherwise.
DEFAULT_GIT_WORKSPACE_SOURCE_TIMEOUT_SECONDS = 300
#: How long one fetch from a Git remote may run, in seconds, unless the source says otherwise.
DEFAULT_GIT_REMOTE_FETCH_TIMEOUT_SECONDS = 600
#: The most bytes one clone or fetch from a Git remote may write, unless the source says
#: otherwise.
DEFAULT_GIT_REMOTE_MAX_FETCH_BYTES = 4 * 1024 * 1024 * 1024
#: The most files one clone or fetch from a Git remote may write. Git's smart protocols write
#: a pack and its index; the dumb HTTP protocol can write an object per file.
MAX_GIT_REMOTE_FETCH_FILES = 100_000
# Every command here prints at most a commit id; the rest is error text.
_GIT_OUTPUT_LIMIT_BYTES = 4_096
_SOURCE_NAME_MAX_CHARS = 128
_SOURCE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z", re.ASCII)
_COMMIT_ID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_REMOTE_URL_MAX_CHARS = 4_096
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# The resource-limit wrapper refuses a smaller file-size limit.
_MIN_FETCH_BYTES = 1_024
_FILE_TOO_LARGE = "File too large"
_FETCH_LOCK_DIRECTORY = "cayu-workspace-source-locks"
_FETCH_LOCK_POLL_SECONDS = 0.1
# How often a running fetch's writes are measured against its budget.
_FETCH_POLL_SECONDS = 0.2
# A ref list names every branch and tag of the remote; a large repository has many.
_REF_LIST_LIMIT_BYTES = 64 * 1024 * 1024
_LOOSE_OBJECT_DIRECTORY = re.compile(r"[0-9a-f]{2}\Z", re.ASCII)
# Where a commit fetched by id is kept, so the mirror holds it like any branch.
_REVISION_REF_PREFIX = "refs/cayu/revisions/"
_USERNAME_ENV = "CAYU_GIT_USERNAME"
_PASSWORD_ENV = "CAYU_GIT_PASSWORD"
_ASKPASS_SCRIPT = (
    "#!/bin/sh\n"
    'case "$1" in\n'
    f"  *[Uu]sername*) printf '%s\\n' \"${_USERNAME_ENV}\" ;;\n"
    f"  *[Pp]assword*) printf '%s\\n' \"${_PASSWORD_ENV}\" ;;\n"
    "  *) exit 1 ;;\n"
    "esac\n"
)
# Git behavior a remote or the host's configuration could otherwise switch on for
# a fetch: hooks, credential helpers, redirects, other transports, unpacking a
# fetch into loose objects (which the file-size limit wouldn't bound), and garbage
# collection that could drop a commit a case still names.
_FETCH_CONFIGURATION = (
    "core.hooksPath=/dev/null",
    "core.fsmonitor=false",
    "credential.helper=",
    "http.followRedirects=false",
    "protocol.allow=never",
    "protocol.https.allow=always",
    "protocol.http.allow=always",
    "protocol.file.allow=always",
    "fetch.unpackLimit=0",
    "transfer.unpackLimit=0",
    "gc.auto=0",
    "maintenance.auto=false",
)


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


def _require_commit_id(revision: object) -> str:
    if type(revision) is not str or _COMMIT_ID_PATTERN.fullmatch(revision) is None:
        raise ValueError("revision must be a full lowercase commit id.")
    return revision


def _require_empty_destination(destination: Path) -> Path:
    target = Path(destination)
    if not target.is_dir() or any(target.iterdir()):
        raise ValueError("A checkout destination must be an existing empty directory.")
    return target


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


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
    holds the commits cases name; fetching new ones is the application's job,
    or use :class:`GitRemoteWorkspaceSource`, which fetches for itself.
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
        self.repository = root
        self.timeout_s = _positive_int(timeout_s, "timeout_s")

    async def checkout(self, revision: str, destination: Path) -> None:
        _require_commit_id(revision)
        target = _require_empty_destination(destination)
        runner = LocalRunner(self.repository)
        found = await self._resolve_commit(runner, revision)
        if found is None:
            raise LookupError(
                f"Git workspace source {self.name!r} has no commit {revision}; "
                "fetch it into the repository first."
            )
        if found != revision:
            # A tag object peels to the commit it points at; the fixture must name that commit.
            raise LookupError(
                f"{revision} in Git workspace source {self.name!r} is not a commit; "
                f"name the commit {found} instead."
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

    async def _has_object(self, revision: str) -> bool:
        """Whether the repository holds ``revision``, as a commit or an object peeling to one."""

        return await self._resolve_commit(LocalRunner(self.repository), revision) is not None

    async def _resolve_commit(self, runner: LocalRunner, revision: str) -> str | None:
        """The commit ``revision`` peels to, or ``None`` when the repository lacks it."""

        found = await self._git(
            runner, "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"
        )
        # --quiet --verify exits 1 without a message for a commit it doesn't have.
        if found.exit_code == 1 and not found.stderr.strip():
            return None
        self._require_success("rev-parse", found)
        return found.stdout.strip()

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


@dataclass(frozen=True, kw_only=True)
class GitRemoteCredentials:
    """The HTTPS username and password (or token) a Git remote source fetches with.

    Both are references the ``resolver`` (a ``Vault`` or ``CredentialProxy``)
    resolves for each fetch. The values reach Git only through the fetch's own
    environment, read by an askpass helper: never a URL, an argument, the
    mirror's configuration or a log, and they are scrubbed from Git's output
    before it can reach an error.
    """

    username: SecretRef
    password: SecretRef
    resolver: SecretResolver

    def __post_init__(self) -> None:
        if type(self.username) is not SecretRef or type(self.password) is not SecretRef:
            raise TypeError("Git remote credentials require SecretRef values.")
        validate_secret_resolver(self.resolver, "resolver")


def _remote_url(value: object, *, has_credentials: bool) -> str:
    if type(value) is not str or len(value) > _REMOTE_URL_MAX_CHARS:
        raise ValueError(f"url must be a string of at most {_REMOTE_URL_MAX_CHARS} characters.")
    url = require_durable_clean_nonblank(value, "url")
    # Git reads "<helper>::<address>" as a remote helper; past "://" a "::" is an IPv6 host.
    if url.startswith("-") or "::" in url.partition("://")[0]:
        raise ValueError("A Git remote URL can't start with '-' or name a remote helper.")
    if "://" not in url:
        if not Path(url).is_absolute():
            raise ValueError(
                "A Git remote is an https:// URL, or a file:// URL or absolute path on this "
                "machine."
            )
        if has_credentials:
            raise ValueError("Credentials are only sent to an https:// remote.")
        return url
    parts = urlsplit(url)
    if parts.scheme not in {"https", "http", "file"}:
        raise ValueError(
            "A Git remote is an https:// URL (http:// only on a loopback host), or a file:// "
            "URL or absolute path on this machine."
        )
    if parts.username is not None or parts.password is not None:
        raise ValueError(
            "A Git remote URL can't carry credentials; pass them as credentials, which the "
            "source resolves from the vault for each fetch."
        )
    if parts.query or parts.fragment:
        raise ValueError("A Git remote URL can't have a query or a fragment.")
    if parts.scheme == "https" and parts.hostname:
        return url
    if parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS:
        return url
    if parts.scheme == "file" and not parts.netloc and parts.path.startswith("/"):
        if has_credentials:
            raise ValueError("Credentials are only sent to an https:// remote.")
        return url
    raise ValueError(
        "A Git remote is an https:// URL (http:// only on a loopback host), or a file:// URL "
        "or absolute path on this machine."
    )


class GitRemoteWorkspaceSource(WorkspaceSource):
    """Check revisions out of a Git remote, through a bare mirror this source keeps.

    ``url`` is the remote: an ``https://`` URL (``http://`` only on a loopback
    host), or a ``file://`` URL or absolute path on this machine. It can't
    carry credentials, a query or a fragment; a remote that needs them gets
    ``credentials``, references the application's vault resolves for each
    fetch. ``mirror`` is a directory this source owns: on first use it becomes
    a bare clone of the remote. A checkout of a commit the mirror lacks first
    fetches every branch and tag, then, if the commit is on none of them, the
    commit itself by id, which most hosts allow. ``refresh`` fetches the
    branches and tags ahead of a run. Checkouts then work as
    :class:`GitWorkspaceSource`'s do, from the mirror.

    Every clone or fetch writes into a new repository beside the mirror, which
    borrows the mirror's objects, so only what is new is fetched. It runs for
    at most ``fetch_timeout_s`` seconds and may write at most
    ``max_fetch_bytes`` bytes in at most ``MAX_GIT_REMOTE_FETCH_FILES`` files,
    whichever protocol the remote speaks; past either limit it is stopped and
    removed, and the mirror is as it was. Only a fetch that finished within
    them moves into the mirror: its objects as packs, each visible only once
    whole, then its refs. Fetches into one mirror
    wait for each other, in this process and others. Hooks, the host's Git
    configuration and credential helpers, redirects and transports other than
    the URL's are off for every fetch, and automatic garbage collection is off,
    so a commit fetched once stays in the mirror. Each local Git command,
    checkouts included, may run for ``timeout_s`` seconds.
    """

    def __init__(
        self,
        name: str,
        url: str,
        mirror: str | Path,
        *,
        credentials: GitRemoteCredentials | None = None,
        fetch_timeout_s: int = DEFAULT_GIT_REMOTE_FETCH_TIMEOUT_SECONDS,
        max_fetch_bytes: int = DEFAULT_GIT_REMOTE_MAX_FETCH_BYTES,
        timeout_s: int = DEFAULT_GIT_WORKSPACE_SOURCE_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(name)
        if os.name != "posix":
            raise NotImplementedError(
                "A Git remote workspace source needs a POSIX host, which can limit what a "
                "fetch writes."
            )
        if credentials is not None and type(credentials) is not GitRemoteCredentials:
            raise TypeError("credentials must be GitRemoteCredentials or None.")
        self.url = _remote_url(url, has_credentials=credentials is not None)
        if not isinstance(mirror, str | Path):
            raise TypeError("mirror must be a path.")
        self.mirror = Path(mirror).expanduser().resolve()
        self.credentials = credentials
        self.fetch_timeout_s = _positive_int(fetch_timeout_s, "fetch_timeout_s")
        self.max_fetch_bytes = _positive_int(max_fetch_bytes, "max_fetch_bytes")
        if self.max_fetch_bytes < _MIN_FETCH_BYTES:
            raise ValueError(f"max_fetch_bytes must be at least {_MIN_FETCH_BYTES}.")
        self.timeout_s = _positive_int(timeout_s, "timeout_s")

    async def checkout(self, revision: str, destination: Path) -> None:
        _require_commit_id(revision)
        _require_empty_destination(destination)
        local = self._mirror_source()
        if local is None or not await local._has_object(revision):
            async with self._fetch_lock():
                local = await self._ready_mirror()
                if not await local._has_object(revision):
                    await self._fetch_into_mirror(*self._branch_and_tag_refspecs())
                if not await local._has_object(revision):
                    await self._fetch_revision(local, revision)
        await local.checkout(revision, destination)

    async def refresh(self) -> None:
        """Bring every branch and tag of the remote into the mirror, cloning it first if needed."""

        async with self._fetch_lock():
            cloned = self._mirror_source() is None
            await self._ready_mirror()
            if not cloned:
                await self._fetch_into_mirror(*self._branch_and_tag_refspecs())

    def _branch_and_tag_refspecs(self) -> tuple[str, ...]:
        return ("--no-tags", self.url, "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*")

    def _mirror_source(self) -> GitWorkspaceSource | None:
        """The mirror as a local source, once it is a bare repository."""

        if not (self.mirror / "HEAD").is_file() or not (self.mirror / "objects").is_dir():
            return None
        return GitWorkspaceSource(self.name, self.mirror, timeout_s=self.timeout_s)

    def _staging_directory(self, purpose: str) -> Path:
        # Beside the mirror, on its filesystem, so what it fetched moves in by renaming.
        self.mirror.parent.mkdir(parents=True, exist_ok=True)
        return Path(
            tempfile.mkdtemp(prefix=f".{self.mirror.name}.{purpose}-", dir=self.mirror.parent)
        )

    async def _ready_mirror(self) -> GitWorkspaceSource:
        """The mirror, cloned from the remote if this source hasn't made it yet."""

        local = self._mirror_source()
        if local is not None:
            if await self._git(self.mirror, "rev-parse", "--is-bare-repository") != "true":
                raise ValueError(
                    f"The mirror of Git workspace source {self.name!r} is not a bare repository."
                )
            return local
        if self.mirror.exists() and (not self.mirror.is_dir() or any(self.mirror.iterdir())):
            raise ValueError(
                f"The mirror of Git workspace source {self.name!r} must be an empty directory "
                "or a bare repository."
            )
        # Cloned beside the mirror and renamed into place whole, so a checkout that
        # looks at the mirror without the fetch lock never sees half a clone.
        staging = self._staging_directory("clone")
        try:
            await self._transfer(
                staging,
                "clone",
                "--bare",
                "--quiet",
                # A path remote is copied or hard-linked otherwise, past the limits.
                "--no-local",
                "--template=",
                "--",
                self.url,
                str(staging),
            )
            staging.rename(self.mirror)
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, staging, True)
            raise
        local = self._mirror_source()
        if local is None:
            raise RuntimeError(f"Cloning Git workspace source {self.name!r} made no mirror.")
        return local

    async def _fetch_into_mirror(self, *arguments: str) -> None:
        """Fetch into a quarantine that borrows the mirror's objects, then adopt what it got.

        The quarantine lists the mirror as its alternate, so Git negotiates
        with the mirror's refs and fetches only what the mirror lacks. Nothing
        reaches the mirror unless the fetch finished within its limits.
        """

        staging = self._staging_directory("fetch")
        try:
            object_format = await self._git(self.mirror, "rev-parse", "--show-object-format")
            await self._git(
                staging,
                "init",
                "--bare",
                "--quiet",
                "--template=",
                f"--object-format={object_format}",
                ".",
            )
            (staging / "objects" / "info").mkdir(parents=True, exist_ok=True)
            (staging / "objects" / "info" / "alternates").write_text(
                f"{self.mirror / 'objects'}\n", encoding="utf-8"
            )
            await self._transfer(staging, "fetch", *arguments)
            if await asyncio.to_thread(_has_loose_objects, staging / "objects"):
                # Loose objects would reach the mirror one at a time, a commit perhaps
                # before its trees; a pack becomes visible whole, when its index does.
                await self._git(
                    staging, "-c", "repack.writeBitmaps=false", "repack", "-a", "-d", "-l", "-q"
                )
            refs = await self._git(
                staging,
                "for-each-ref",
                "--format=%(objectname) %(refname)",
                output_limit_bytes=_REF_LIST_LIMIT_BYTES,
            )
            await _finish_in_thread(self._adopt_packs, staging / "objects" / "pack")
            updates = "".join(
                f"update {name} {object_id}\n"
                for object_id, _, name in (line.partition(" ") for line in refs.splitlines())
            )
            if updates:
                await self._git(self.mirror, "update-ref", "--stdin", stdin=updates)
        finally:
            await asyncio.to_thread(shutil.rmtree, staging, True)

    def _adopt_packs(self, packs: Path) -> None:
        """Move a finished fetch's packs into the mirror's object store."""

        if not packs.is_dir():
            return
        # Git finds a pack through its index, so each index arrives after its pack.
        for file in sorted(packs.iterdir(), key=lambda path: path.suffix == ".idx"):
            target = self.mirror / "objects" / "pack" / file.name
            if not target.exists():
                os.replace(file, target)

    async def _fetch_revision(self, local: GitWorkspaceSource, revision: str) -> None:
        try:
            await self._fetch_into_mirror(
                "--no-tags", self.url, f"+{revision}:{_REVISION_REF_PREFIX}{revision}"
            )
        except RuntimeError as error:
            raise LookupError(
                f"Git workspace source {self.name!r} has no commit {revision} on any branch or "
                f"tag of its remote, and fetching it by id failed: {error}"
            ) from None
        if not await local._has_object(revision):
            raise LookupError(
                f"Git workspace source {self.name!r}'s remote has no commit {revision}."
            )

    async def _git(
        self,
        repository: Path,
        *arguments: str,
        stdin: str | None = None,
        output_limit_bytes: int = _GIT_OUTPUT_LIMIT_BYTES,
    ) -> str:
        """Run a local Git command in ``repository`` and return its output."""

        result = await LocalRunner(repository).exec(
            ExecCommand.process(
                "git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *arguments
            ),
            env={
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "LC_ALL": "C",
            },
            timeout_s=self.timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        if result.timed_out:
            raise TimeoutError(
                f"git {arguments[0]} in workspace source {self.name!r} took longer than "
                f"{self.timeout_s} seconds."
            )
        if result.exit_code != 0 or result.stdout_truncated:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            raise RuntimeError(
                f"git {arguments[0]} in workspace source {self.name!r} failed: {detail}"
            )
        return result.stdout.strip()

    async def _transfer(self, repository: Path, command: str, *arguments: str) -> None:
        """Run ``git clone`` or ``git fetch`` into ``repository``, bounded in time and size."""

        git = shutil.which("git")
        if git is None:
            raise FileNotFoundError("Git workspace sources need git on PATH.")
        options = [item for setting in _FETCH_CONFIGURATION for item in ("-c", setting)]
        with tempfile.TemporaryDirectory(prefix="cayu-git-fetch-") as helper_directory:
            environment = {
                # An empty home: no ~/.gitconfig, ~/.config/git or ~/.netrc of the host's
                # can add configuration or credentials to the fetch.
                "HOME": helper_directory,
                "XDG_CONFIG_HOME": helper_directory,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "LC_ALL": "C",
            }
            secret_env: tuple[SecretEnv, ...] = ()
            resolver: SecretResolver | None = None
            if self.credentials is not None:
                helper = Path(helper_directory) / "askpass"
                helper.write_text(_ASKPASS_SCRIPT, encoding="utf-8")
                helper.chmod(stat.S_IRWXU)
                environment["GIT_ASKPASS"] = str(helper)
                secret_env = (
                    SecretEnv(name=_USERNAME_ENV, ref=self.credentials.username),
                    SecretEnv(name=_PASSWORD_ENV, ref=self.credentials.password),
                )
                resolver = self.credentials.resolver
            runner = LocalRunner(
                repository,
                inherit_env=False,
                secret_env=secret_env,
                secret_resolver=resolver,
            )
            transfer = asyncio.ensure_future(
                runner.exec(
                    ExecCommand.process(
                        sys.executable,
                        "-I",
                        str(Path(_git_process.__file__)),
                        # No one file, a pack above all, may outgrow the whole budget.
                        str(self.max_fetch_bytes),
                        git,
                        *options,
                        command,
                        *arguments,
                    ),
                    env=environment,
                    timeout_s=self.fetch_timeout_s,
                    output_limit_bytes=_GIT_OUTPUT_LIMIT_BYTES,
                )
            )
            try:
                # Git's dumb HTTP protocol writes an object per file, which the file-size
                # limit doesn't add up, so what the fetch has written is measured as it runs.
                while not (await asyncio.wait({transfer}, timeout=_FETCH_POLL_SECONDS))[0]:
                    await self._require_within_budget(command, repository)
                result = transfer.result()
            finally:
                if not transfer.done():
                    transfer.cancel()
                    await asyncio.gather(transfer, return_exceptions=True)
        if result.timed_out:
            raise TimeoutError(
                f"git {command} for workspace source {self.name!r} took longer than "
                f"{self.fetch_timeout_s} seconds."
            )
        # The file-size limit makes Git's write fail with EFBIG: "File too large" in the C locale.
        if result.exit_code != 0 and _FILE_TOO_LARGE in result.stderr:
            raise self._over_budget(command)
        if result.exit_code != 0:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            raise RuntimeError(f"git {command} for workspace source {self.name!r} failed: {detail}")
        await self._require_within_budget(command, repository)

    async def _require_within_budget(self, command: str, repository: Path) -> None:
        written, files = await asyncio.to_thread(_directory_usage, repository / "objects")
        if written > self.max_fetch_bytes or files > MAX_GIT_REMOTE_FETCH_FILES:
            raise self._over_budget(command)

    def _over_budget(self, command: str) -> RuntimeError:
        return RuntimeError(
            f"git {command} for workspace source {self.name!r} would write more than "
            f"max_fetch_bytes ({self.max_fetch_bytes}) bytes or "
            f"{MAX_GIT_REMOTE_FETCH_FILES} files."
        )

    @asynccontextmanager
    async def _fetch_lock(self) -> AsyncIterator[None]:
        """Hold the mirror's fetch lock, which other processes' sources of it share.

        The holder's own fetches are bounded and a dead holder's lock is released
        with its process, so the wait ends; the caller's deadline bounds it too.
        """

        self.mirror.parent.mkdir(parents=True, exist_ok=True)
        while True:
            lock = cooperative_path_lock(
                self.mirror.parent,
                self.mirror.name,
                lock_directory_name=_FETCH_LOCK_DIRECTORY,
                blocking=False,
            )
            try:
                lock.__enter__()
            except BlockingIOError:
                await asyncio.sleep(_FETCH_LOCK_POLL_SECONDS)
                continue
            break
        try:
            yield
        finally:
            lock.__exit__(None, None, None)


def _has_loose_objects(objects: Path) -> bool:
    return any(
        _LOOSE_OBJECT_DIRECTORY.fullmatch(directory.name) and any(directory.iterdir())
        for directory in objects.iterdir()
        if directory.is_dir()
    )


async def _finish_in_thread(function: Callable[[Path], None], argument: Path) -> None:
    """Run ``function`` in a worker thread and wait for it to finish, cancelled or not.

    A thread can't be stopped, so returning while it still worked would let
    what follows, removing the directory it reads, run beside it. A
    cancellation waits for the thread, then goes on.
    """

    worker = asyncio.ensure_future(asyncio.to_thread(function, argument))
    cancelled = False
    while True:
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            if worker.done():
                raise
            cancelled = True
            continue
        break
    if cancelled:
        raise asyncio.CancelledError


def _directory_usage(root: Path) -> tuple[int, int]:
    """The bytes and the number of regular files under ``root``."""

    written = files = 0
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                metadata = os.lstat(os.path.join(directory, name))
            except FileNotFoundError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                written += metadata.st_size
                files += 1
    return written, files


__all__ = [
    "DEFAULT_GIT_REMOTE_FETCH_TIMEOUT_SECONDS",
    "DEFAULT_GIT_REMOTE_MAX_FETCH_BYTES",
    "DEFAULT_GIT_WORKSPACE_SOURCE_TIMEOUT_SECONDS",
    "MAX_GIT_REMOTE_FETCH_FILES",
    "GitRemoteCredentials",
    "GitRemoteWorkspaceSource",
    "GitWorkspaceSource",
    "WorkspaceSource",
]
