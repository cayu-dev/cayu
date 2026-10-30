"""A Git remote workspace source: a bare mirror it keeps, fetched on demand, bounded and private.

Every remote here is local: a bare repository reached over ``file://`` or as a path, or served
over HTTP on the loopback interface by a thread of the test, with Basic authentication.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import logging
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cayu.vaults import SecretRef, StaticVault
from cayu.workspaces import GitRemoteCredentials, GitRemoteWorkspaceSource
from cayu.workspaces import sources as sources_module

_USERNAME = "fixture-bot"
_TOKEN = "ghp-local-fixture-token-5f2b9c"


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Eval", "-c", "user.email=eval@example.test", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(work: Path, content: str) -> str:
    (work / "calc.py").write_text(content)
    _git(work, "add", "-A")
    _git(work, "commit", "--quiet", "-m", content[:20])
    return _git(work, "rev-parse", "HEAD")


def _remote(tmp_path: Path) -> tuple[Path, Path, str]:
    """A working repository, a bare remote of it, and the remote's first commit."""

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "--quiet", "--initial-branch=main")
    first = _commit(work, "def add(a, b):\n    return a - b\n")
    remote = tmp_path / "remote.git"
    _git(tmp_path, "clone", "--quiet", "--bare", str(work), str(remote))
    return work, remote, first


def _push(work: Path, remote: Path, refspec: str) -> None:
    _git(work, "push", "--quiet", str(remote), refspec)


def _checkout(source: GitRemoteWorkspaceSource, revision: str, destination: Path) -> str:
    destination.mkdir()
    asyncio.run(source.checkout(revision, destination))
    return (destination / "calc.py").read_text()


def _refs(repository: Path) -> str:
    return _git(repository, "for-each-ref", "--format=%(refname) %(objectname)")


def test_a_remote_source_mirrors_the_remote_and_fetches_a_commit_it_lacks(
    tmp_path: Path,
) -> None:
    work, remote, first = _remote(tmp_path)
    mirror = tmp_path / "mirrors" / "payments"
    source = GitRemoteWorkspaceSource("payments", f"file://{remote}", mirror)

    assert "a - b" in _checkout(source, first, tmp_path / "first")
    assert _git(mirror, "rev-parse", "--is-bare-repository") == "true"

    # A commit pushed to a branch since the mirror was made comes with the next fetch.
    second = _commit(work, "def add(a, b):\n    return a + b\n")
    _push(work, remote, "main")
    assert "a + b" in _checkout(source, second, tmp_path / "second")

    # A commit on no branch or tag is fetched by its id and kept under a ref of its own.
    third = _commit(work, "def add(a, b):\n    return b + a\n")
    _push(work, remote, f"{third}:refs/pull/7/head")
    remote_refs = _refs(remote)
    assert "b + a" in _checkout(source, third, tmp_path / "third")
    assert f"refs/cayu/revisions/{third} {third}" in _refs(mirror)
    # The remote is only read, and a checkout holds no repository of its own.
    assert _refs(remote) == remote_refs
    assert not (tmp_path / "third" / ".git").exists()


def test_a_remote_source_refuses_a_commit_its_remote_lacks(tmp_path: Path) -> None:
    _, remote, first = _remote(tmp_path)
    mirror = tmp_path / "mirror"
    source = GitRemoteWorkspaceSource("payments", str(remote), mirror)
    asyncio.run(source.refresh())

    (tmp_path / "missing").mkdir()
    with pytest.raises(LookupError, match=f"no commit {'f' * 40}"):
        asyncio.run(source.checkout("f" * 40, tmp_path / "missing"))
    # The mirror is as it was and still checks out what it holds.
    assert "a - b" in _checkout(source, first, tmp_path / "first")


def test_refresh_brings_the_remote_branches_and_tags_into_the_mirror(tmp_path: Path) -> None:
    work, remote, first = _remote(tmp_path)
    mirror = tmp_path / "mirror"
    source = GitRemoteWorkspaceSource("payments", str(remote), mirror)
    asyncio.run(source.refresh())
    assert _git(mirror, "rev-parse", "refs/heads/main") == first

    second = _commit(work, "def add(a, b):\n    return a + b\n")
    _git(work, "tag", "v2")
    _push(work, remote, "main")
    _push(work, remote, "refs/tags/v2")
    asyncio.run(source.refresh())
    assert _git(mirror, "rev-parse", "refs/heads/main") == second
    assert _git(mirror, "rev-parse", "refs/tags/v2") == second


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://bot:token@git.example.test/payments.git", "can't carry credentials"),
        ("https://token@git.example.test/payments.git", "can't carry credentials"),
        ("https://git.example.test/payments.git?private_token=x", "query or a fragment"),
        ("http://git.example.test/payments.git", "loopback"),
        ("ssh://git@git.example.test/payments.git", "https://"),
        ("git@git.example.test:payments.git", "https://"),
        ("checkouts/payments", "https://"),
        ("-upayments", "start with '-'"),
        ("ext::sh -c touch% /tmp/pwned", "remote helper"),
    ],
)
def test_a_remote_url_carries_no_credentials_or_other_transport(
    tmp_path: Path, url: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GitRemoteWorkspaceSource("payments", url, tmp_path / "mirror")


def test_an_ipv6_remote_is_no_remote_helper(tmp_path: Path) -> None:
    for url in ("https://[2001:db8::1]/payments.git", "http://[::1]:8080/payments.git"):
        assert GitRemoteWorkspaceSource("payments", url, tmp_path / "mirror").url == url
    with pytest.raises(ValueError, match="remote helper"):
        GitRemoteWorkspaceSource("payments", "ext::https://[::1]/payments.git", tmp_path / "m")


def test_credentials_are_sent_only_to_an_https_remote(tmp_path: Path) -> None:
    for url in (f"file://{tmp_path}/remote.git", f"{tmp_path}/remote.git"):
        with pytest.raises(ValueError, match="only sent to an https:// remote"):
            GitRemoteWorkspaceSource(
                "payments", url, tmp_path / "mirror", credentials=_credentials(_TOKEN)
            )


def _credentials(token: str) -> GitRemoteCredentials:
    return GitRemoteCredentials(
        username=SecretRef(name="git-username"),
        password=SecretRef(name="git-token"),
        resolver=StaticVault({"git-username": _USERNAME, "git-token": token}),
    )


class _GitHttpHandler(SimpleHTTPRequestHandler):
    """Git's dumb HTTP protocol over the served directory, behind Basic authentication."""

    expected = "Basic " + base64.b64encode(f"{_USERNAME}:{_TOKEN}".encode()).decode()
    stall_s = 0.0
    seen: list[tuple[str, str | None]]

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        self.seen.append((self.path, self.headers.get("Authorization")))
        if self.stall_s:
            time.sleep(self.stall_s)
        if self.headers.get("Authorization") != self.expected:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="git"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


@contextmanager
def _served(directory: Path, *, stall_s: float = 0.0) -> Iterator[tuple[str, list]]:
    seen: list[tuple[str, str | None]] = []
    handler = type("Handler", (_GitHttpHandler,), {"stall_s": stall_s, "seen": seen})
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(handler, directory=str(directory))
    )
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def _files_mentioning(root: Path, text: str) -> list[Path]:
    return [
        path for path in root.rglob("*") if path.is_file() and text.encode() in path.read_bytes()
    ]


def test_credentials_reach_git_only_through_the_fetch_environment(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    work, remote, first = _remote(tmp_path)
    second = _commit(work, "def add(a, b):\n    return a + b\n")
    _push(work, remote, "main")
    _git(remote, "update-server-info")
    mirror = tmp_path / "mirror"
    with _served(tmp_path) as (base, seen):
        url = f"{base}/remote.git"
        source = GitRemoteWorkspaceSource("payments", url, mirror, credentials=_credentials(_TOKEN))
        assert "a + b" in _checkout(source, second, tmp_path / "second")
        assert "a - b" in _checkout(source, first, tmp_path / "first")
        assert any(header == _GitHttpHandler.expected for _, header in seen)
        assert source.url == url

        refused = GitRemoteWorkspaceSource(
            "payments",
            url,
            tmp_path / "refused",
            credentials=_credentials("revoked-fixture-token-11aa"),
        )
        (tmp_path / "denied").mkdir()
        with pytest.raises(RuntimeError, match="Authentication failed") as denied:
            asyncio.run(refused.checkout(second, tmp_path / "denied"))
    message = str(denied.value)
    for secret in (_TOKEN, "revoked-fixture-token-11aa", _USERNAME):
        assert secret not in message
        assert secret not in caplog.text
        # Nothing the source keeps, the mirror's configuration included, holds a credential.
        assert _files_mentioning(mirror, secret) == []
    assert not (tmp_path / "refused").exists()


def test_a_fetch_writes_no_more_than_its_size_limit(tmp_path: Path) -> None:
    work, remote, first = _remote(tmp_path)
    mirror = tmp_path / "mirror"
    source = GitRemoteWorkspaceSource("payments", str(remote), mirror, max_fetch_bytes=64_000)
    assert "a - b" in _checkout(source, first, tmp_path / "first")

    (work / "blob.bin").write_bytes(os.urandom(256_000))
    large = _commit(work, "def add(a, b):\n    return a + b\n")
    _push(work, remote, "main")
    (tmp_path / "large").mkdir()
    with pytest.raises(RuntimeError, match=r"more than max_fetch_bytes \(64000\)"):
        asyncio.run(source.checkout(large, tmp_path / "large"))
    # The cut-off pack is gone and the mirror still serves what it held.
    leftovers = [path.name for path in (mirror / "objects" / "pack").iterdir()]
    assert not [name for name in leftovers if name.startswith("tmp_") or name.endswith(".temp")]
    assert "a - b" in _checkout(source, first, tmp_path / "again")

    # A first clone is bounded too, and leaves no mirror behind.
    fresh = GitRemoteWorkspaceSource(
        "payments", f"file://{remote}", tmp_path / "fresh", max_fetch_bytes=64_000
    )
    (tmp_path / "cloned").mkdir()
    with pytest.raises(RuntimeError, match="max_fetch_bytes"):
        asyncio.run(fresh.checkout(large, tmp_path / "cloned"))
    assert not (tmp_path / "fresh").exists()


def _loose_remote(tmp_path: Path, work: Path, objects: int) -> str:
    """Push ``objects`` incompressible 4 KB files, which the remote keeps as loose objects."""

    for index in range(objects):
        (work / f"data-{index}.bin").write_bytes(os.urandom(4_096))
    revision = _commit(work, f"def add(a, b):\n    return a + b + {objects}\n")
    _push(work, tmp_path / "loose.git", "main")
    _git(tmp_path / "loose.git", "update-server-info")
    return revision


def test_a_dumb_http_fetch_is_bounded_by_what_it_writes_in_all(tmp_path: Path) -> None:
    work, _, first = _remote(tmp_path)
    _git(tmp_path, "init", "--quiet", "--bare", "loose.git")
    _push(work, tmp_path / "loose.git", "main")
    _git(tmp_path / "loose.git", "update-server-info")
    mirror = tmp_path / "mirror"
    with _served(tmp_path) as (base, _):
        source = GitRemoteWorkspaceSource(
            "payments",
            f"{base}/loose.git",
            mirror,
            credentials=_credentials(_TOKEN),
            max_fetch_bytes=64_000,
        )
        assert "a - b" in _checkout(source, first, tmp_path / "first")
        refs, objects = _refs(mirror), _git(mirror, "count-objects", "-v")
        # Forty 4 KB objects: each under the limit, together past it.
        large = _loose_remote(tmp_path, work, 40)
        (tmp_path / "large").mkdir()
        with pytest.raises(RuntimeError, match=r"more than max_fetch_bytes \(64000\)"):
            asyncio.run(source.checkout(large, tmp_path / "large"))
        # Nothing of the stopped fetch reached the mirror, and nothing of it is left beside it.
        assert (_refs(mirror), _git(mirror, "count-objects", "-v")) == (refs, objects)
        assert sorted(path.name for path in tmp_path.iterdir() if path.name.startswith(".")) == []

        # A first clone of the same remote is bounded the same way.
        fresh = GitRemoteWorkspaceSource(
            "payments",
            f"{base}/loose.git",
            tmp_path / "fresh",
            credentials=_credentials(_TOKEN),
            max_fetch_bytes=64_000,
        )
        (tmp_path / "cloned").mkdir()
        with pytest.raises(RuntimeError, match="max_fetch_bytes"):
            asyncio.run(fresh.checkout(large, tmp_path / "cloned"))
        assert not (tmp_path / "fresh").exists()


def test_a_fetch_cancelled_while_its_objects_move_in_leaves_no_partial_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, _, first = _remote(tmp_path)
    _git(tmp_path, "init", "--quiet", "--bare", "loose.git")
    _push(work, tmp_path / "loose.git", "main")
    _git(tmp_path / "loose.git", "update-server-info")
    mirror = tmp_path / "mirror"
    replace = os.replace
    moving = threading.Event()

    def slow_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        # The deadline strikes once the commit, or the pack holding it, is in the mirror.
        if ".mirror.fetch-" not in str(source):
            replace(source, target)
            return
        time.sleep(0.05)
        replace(source, target)
        name = Path(source).name
        if name.endswith(".pack") or Path(source).parent.name + name == large:
            moving.set()

    with _served(tmp_path) as (base, _):
        source = GitRemoteWorkspaceSource(
            "payments", f"{base}/loose.git", mirror, credentials=_credentials(_TOKEN)
        )
        assert "a - b" in _checkout(source, first, tmp_path / "first")
        large = _loose_remote(tmp_path, work, 20)
        monkeypatch.setattr(sources_module.os, "replace", slow_replace)

        async def cancel_while_moving() -> None:
            (tmp_path / "cancelled").mkdir()
            checkout = asyncio.ensure_future(source.checkout(large, tmp_path / "cancelled"))
            while not moving.is_set() and not checkout.done():
                await asyncio.sleep(0.01)
            checkout.cancel()
            with pytest.raises(asyncio.CancelledError):
                await checkout

        asyncio.run(cancel_while_moving())
        assert moving.is_set()
        monkeypatch.setattr(sources_module.os, "replace", replace)
        # Whatever reached the mirror is whole: the commit checks out, fetched again or not.
        assert "+ 20" in _checkout(source, large, tmp_path / "after")
    assert sorted(path.name for path in tmp_path.iterdir() if path.name.startswith(".")) == []


def test_a_later_fetch_brings_only_what_the_mirror_lacks(tmp_path: Path) -> None:
    work, remote, _ = _remote(tmp_path)
    (work / "blob.bin").write_bytes(os.urandom(256_000))
    large = _commit(work, "def add(a, b):\n    return a + b\n")
    _push(work, remote, "main")
    mirror = tmp_path / "mirror"
    assert "a + b" in _checkout(
        GitRemoteWorkspaceSource("payments", f"file://{remote}", mirror),
        large,
        tmp_path / "large",
    )
    small = _commit(work, "def add(a, b):\n    return b + a\n")
    _push(work, remote, "main")
    # Far below the repository's size: the fetch negotiates with the mirror's own commits.
    bounded = GitRemoteWorkspaceSource(
        "payments", f"file://{remote}", mirror, max_fetch_bytes=16_000
    )
    assert "b + a" in _checkout(bounded, small, tmp_path / "small")
    assert _git(mirror, "rev-parse", "refs/heads/main") == small


def test_a_fetch_runs_no_longer_than_its_timeout(tmp_path: Path) -> None:
    _, remote, first = _remote(tmp_path)
    _git(remote, "update-server-info")
    with _served(tmp_path, stall_s=10.0) as (base, _):
        source = GitRemoteWorkspaceSource(
            "payments",
            f"{base}/remote.git",
            tmp_path / "mirror",
            credentials=_credentials(_TOKEN),
            fetch_timeout_s=1,
        )
        (tmp_path / "slow").mkdir()
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="longer than 1 seconds"):
            asyncio.run(source.checkout(first, tmp_path / "slow"))
        assert time.monotonic() - started < 8
    assert not (tmp_path / "mirror").exists()


def test_concurrent_checkouts_of_a_missing_commit_share_one_mirror(tmp_path: Path) -> None:
    work, remote, first = _remote(tmp_path)
    second = _commit(work, "def add(a, b):\n    return a + b\n")
    _push(work, remote, "main")
    source = GitRemoteWorkspaceSource("payments", f"file://{remote}", tmp_path / "mirror")
    # Another process's source of the same mirror shares its fetch lock.
    other = GitRemoteWorkspaceSource("payments", f"file://{remote}", tmp_path / "mirror")
    destinations = [tmp_path / f"trial-{index}" for index in range(6)]
    for destination in destinations:
        destination.mkdir()

    async def check_out_all() -> None:
        await asyncio.gather(
            *(
                (source if index % 2 else other).checkout(
                    second if index % 3 else first, destination
                )
                for index, destination in enumerate(destinations)
            )
        )

    asyncio.run(check_out_all())
    for index, destination in enumerate(destinations):
        expected = "a + b" if index % 3 else "a - b"
        assert expected in (destination / "calc.py").read_text()
