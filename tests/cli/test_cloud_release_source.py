"""`cayu cloud deployment source`: download a Release's uploaded source, verify it, and extract
it safely or keep the archive."""

from __future__ import annotations

import errno
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, _cloud_release_source, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_APP = "/v1/applications/my-agent"
_SOURCE = ("GET", f"{_APP}/deployments/dep_live/source")
_AGENTS = {"items": [{"id": "my-agent", "name": "My Agent", "current_deployment_id": "dep_live"}]}
_RELEASE = {"id": "dep_live", "application_id": "my-agent", "status": "promoted"}

# (name, type, content or link target, mode)
_Member = tuple[str, bytes, bytes | str, int]


def _bundle(*members: _Member) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, kind, content, mode in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.mode = mode
            if kind == tarfile.REGTYPE:
                assert isinstance(content, bytes)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            else:
                if isinstance(content, str):
                    info.linkname = content
                archive.addfile(info)
    return buffer.getvalue()


def _file(name: str, content: bytes = b"x", mode: int = 0o644) -> _Member:
    return (name, tarfile.REGTYPE, content, mode)


def _symlink(name: str, target: str) -> _Member:
    return (name, tarfile.SYMTYPE, target, 0o777)


# A bundle as `cayu cloud deploy` uploads it: the project under source/.
_PROJECT = _bundle(
    _file("source/pyproject.toml", b"[project]\nname = 'my-agent'\n"),
    _file("source/uv.lock", b"version = 1\n"),
    _file("source/bin/run", b"#!/bin/sh\n", mode=0o755),
    _file("source/src/my_agent/__init__.py", b""),
    _symlink("source/AGENTS.md", "CLAUDE.md"),
    _file("source/CLAUDE.md", b"# Notes\n"),
)


def _served(body: bytes, *, digest: str | None = None) -> httpx.Response:
    headers = {
        "Content-Type": "application/gzip",
        "Content-Disposition": 'attachment; filename="my-agent-dep_live.tar.gz"',
    }
    headers["X-Cayu-Content-Digest"] = (
        digest if digest is not None else "sha256:" + hashlib.sha256(body).hexdigest()
    )
    return httpx.Response(200, headers=headers, content=body)


def _refused(status: int, code: str, message: str, **extra: str) -> httpx.Response:
    return httpx.Response(status, json={"detail": {"code": code, "message": message, **extra}})


class _Cloud:
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = {
            ("GET", "/v1/applications?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("GET", "/v1/deployments/dep_live"): httpx.Response(200, json=_RELEASE),
            **routes,
        }
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.raw_path.decode())
        self.requests.append(" ".join(key))
        return self.routes.get(key) or httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], httpx.Response]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], httpx.Response]) -> _Cloud:
        fake = _Cloud(routes)
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
        )
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )
        return fake

    return install


def _error(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)["error"]


def _leftovers(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.startswith(".cayu-"))


def test_extracts_the_uploaded_project_into_a_new_directory(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({_SOURCE: _served(_PROJECT)})
    output = tmp_path / "checkout" / "dep_live"

    assert main(["cloud", "deployment", "source", "dep_live", "--output", str(output)]) == 0

    out = capsys.readouterr().out
    assert f"Extracted 5 files and 1 link (58 bytes) from release dep_live into {output}." in out
    assert f"Verified digest sha256:{hashlib.sha256(_PROJECT).hexdigest()}." in out
    # The project lands in DIR itself, not under the bundle's source/.
    assert (output / "pyproject.toml").read_bytes() == b"[project]\nname = 'my-agent'\n"
    assert (output / "src" / "my_agent" / "__init__.py").exists()
    assert os.readlink(output / "AGENTS.md") == "CLAUDE.md"
    assert (output / "AGENTS.md").read_text() == "# Notes\n"
    assert os.access(output / "bin" / "run", os.X_OK)
    assert _leftovers(output.parent) == []
    # The Agent came from the release id alone.
    assert "GET /v1/applications?lifecycle=all" not in fake.requests


def test_json_reports_what_was_written_and_takes_an_empty_directory(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_SOURCE: _served(_PROJECT)})
    output = tmp_path / "empty"
    output.mkdir()

    arguments = ["cloud", "deployment", "source", "dep_live", "--application", "my-agent"]
    assert main([*arguments, "--output", str(output), "--json"]) == 0

    answer = json.loads(capsys.readouterr().out)
    assert answer["operation"] == "deployment.source"
    assert answer["result"] == {
        "application_id": "my-agent",
        "deployment_id": "dep_live",
        "output": str(output),
        "format": "directory",
        "digest": "sha256:" + hashlib.sha256(_PROJECT).hexdigest(),
        "bytes": len(_PROJECT),
        "files": 5,
        "links": 1,
        "file_bytes": 58,
    }
    assert (output / "uv.lock").exists()


def test_archive_writes_the_verified_bundle_as_it_is(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_SOURCE: _served(_PROJECT)})
    output = tmp_path / "dep_live.tar.gz"
    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--archive"]

    assert main(arguments) == 0
    assert output.read_bytes() == _PROJECT
    assert f"Wrote {output} ({len(_PROJECT)} bytes, 5 files and 1 link)." in (
        capsys.readouterr().out
    )
    assert _leftovers(tmp_path) == []

    # An existing file is never replaced.
    assert main([*arguments, "--json"]) == 2
    assert _error(capsys)["category"] == "output_exists"
    assert output.read_bytes() == _PROJECT


def test_a_non_empty_directory_is_refused_before_downloading(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({_SOURCE: _served(_PROJECT)})
    (tmp_path / "kept.txt").write_text("mine")

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(tmp_path), "--json"]
    assert main(arguments) == 2
    error = _error(capsys)
    assert error["category"] == "output_exists"
    assert "isn't empty" in str(error["message"])
    assert " ".join(_SOURCE) not in fake.requests
    assert [path.name for path in tmp_path.iterdir()] == ["kept.txt"]


@pytest.mark.parametrize("digest", ["sha256:" + "0" * 64, None])
def test_a_digest_mismatch_or_a_missing_digest_writes_nothing(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    digest: str | None,
) -> None:
    response = _served(_PROJECT, digest=digest or "")
    cloud({_SOURCE: response})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    error = _error(capsys)
    expected = "source_digest_mismatch" if digest else "api_response_invalid"
    assert error["category"] == expected
    assert not output.exists()
    assert _leftovers(tmp_path) == []
    assert main([*arguments, "--archive"]) == 2
    assert _error(capsys)["category"] == expected
    assert not output.exists()


@pytest.mark.parametrize(
    ("members", "reason"),
    [
        ((_file("/etc/cron.d/agent"),), "absolute"),
        ((_file("source/../../outside"),), "leads out"),
        ((_symlink("source/escape", "../../.."),), "links outside"),
        ((_symlink("source/etc", "/etc"),), "links outside"),
        # Each link stays inside on its own; together they climb out.
        ((_symlink("source/up", "."), _symlink("source/escape", "up/..")), "links outside"),
        ((_symlink("source/loop", "loop/x"),), "links outside"),
        ((_symlink("source/lib", "pkg"), _file("source/lib/__init__.py")), "written through"),
        ((_file("source/a"), _file("source/a")), "more than once"),
        ((("source/pipe", tarfile.FIFOTYPE, b"", 0o644),), "special file"),
        ((("source/tty", tarfile.CHRTYPE, b"", 0o644),), "special file"),
        ((("source/hard", tarfile.LNKTYPE, "/etc/passwd", 0o644),), "absolute"),
        ((("source/hard", tarfile.LNKTYPE, "source/missing", 0o644),), "not before it"),
    ],
)
def test_an_unsafe_member_is_refused_and_nothing_is_extracted(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    members: tuple[_Member, ...],
    reason: str,
) -> None:
    bundle = _bundle(_file("source/pyproject.toml"), *members)
    cloud({_SOURCE: _served(bundle)})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output)]
    assert main([*arguments, "--json"]) == 2
    error = _error(capsys)
    assert error["category"] == "source_bundle_unsafe"
    assert reason in str(error["message"])
    assert not output.exists()
    assert _leftovers(tmp_path) == []

    # The archive itself is still available, as Cloud stores it.
    archive = tmp_path / "bundle.tar.gz"
    assert main([*arguments[:-1], str(archive), "--archive", "--json"]) == 0
    assert archive.read_bytes() == bundle


@pytest.mark.parametrize(
    ("members", "reason"),
    [
        # `UP` is the link `up` on a case-insensitive file system, so `esc` climbs out of DIR
        # and `ESC/evil` would be written beside it.
        (
            (
                _symlink("source/a/b/up", "../.."),
                _symlink("source/a/b/esc", "UP/.."),
                _file("source/a/b/ESC/evil"),
            ),
            "would be written through a link or file",
        ),
        ((_symlink("source/a/up", ".."), _symlink("source/a/esc", "UP/..")), "links outside"),
        ((_file("source/Foo.py"), _file("source/foo.py")), "differ only in case"),
    ],
)
def test_names_are_compared_as_a_case_insensitive_file_system_sees_them(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    members: tuple[_Member, ...],
    reason: str,
) -> None:
    cloud({_SOURCE: _served(_bundle(_file("source/pyproject.toml"), *members))})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    error = _error(capsys)
    assert error["category"] == "source_bundle_unsafe"
    assert reason in str(error["message"])
    # Nothing was written: no DIR, and nothing beside it either.
    assert list(tmp_path.iterdir()) == []


def test_dot_extracts_into_an_empty_current_directory_and_keeps_it(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cloud({_SOURCE: _served(_PROJECT)})
    work = tmp_path / "dep"
    work.mkdir()
    work.chmod(0o750)
    before = work.stat()
    monkeypatch.chdir(work)

    assert main(["cloud", "deployment", "source", "dep_live", "--output", "."]) == 0

    assert f"into {work}." in capsys.readouterr().out
    after = work.stat()
    # The same directory, with its own permissions: a shell already in it sees the files.
    assert (after.st_ino, after.st_mode) == (before.st_ino, before.st_mode)
    assert sorted(os.listdir(".")) == [
        "AGENTS.md",
        "CLAUDE.md",
        "bin",
        "pyproject.toml",
        "src",
        "uv.lock",
    ]
    assert _leftovers(tmp_path) == []


def test_a_file_written_into_dir_meanwhile_is_never_replaced(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cloud({_SOURCE: _served(_PROJECT)})
    output = tmp_path / "dep"
    output.mkdir()
    extract = tarfile.TarFile.extractall

    def extract_then_race(self: tarfile.TarFile, *args: object, **kwargs: object) -> None:
        extract(self, *args, **kwargs)  # type: ignore[arg-type]
        # Another process writes into DIR after its emptiness was checked.
        (output / "uv.lock").write_text("theirs")

    monkeypatch.setattr(tarfile.TarFile, "extractall", extract_then_race)

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "output_exists"
    # Their file stands, and nothing of the release is left beside it.
    assert sorted(os.listdir(output)) == ["uv.lock"]
    assert (output / "uv.lock").read_text() == "theirs"
    assert _leftovers(tmp_path) == []


def _staged(output: Path, *entries: tuple[str, str]) -> Path:
    """A staging directory inside ``output``, as extraction leaves it, holding ``entries``."""

    staging = output / ".cayu-source-staging"
    staging.mkdir(parents=True)
    for name, content in entries:
        (staging / name).parent.mkdir(parents=True, exist_ok=True)
        (staging / name).write_text(content)
    return staging


def test_publishing_never_replaces_what_is_already_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    staging = _staged(output, ("kept.txt", "release"), ("taken/inner", "release"))
    published = _cloud_release_source._publish_file

    def race(name: str, *args: Any) -> os.stat_result:
        # Another process takes a name after the emptiness check, mid-publication.
        if name == "inner":
            raise FileExistsError(errno.EEXIST, "taken")
        return published(name, *args)

    monkeypatch.setattr(_cloud_release_source, "_publish_file", race)
    with pytest.raises(FileExistsError):
        _cloud_release_source._publish(staging, output)
    # Published entries remain; rolling them back by name could delete replacements.
    assert sorted(os.listdir(output)) == [staging.name, "kept.txt", "taken"]
    assert (output / "kept.txt").read_text() == "release"


def test_a_finished_subdirectory_is_preserved_when_a_later_entry_collides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    staging = _staged(output, ("a/b/x.py", "release"), ("a/y.py", "release"), ("z.txt", "release"))
    published = _cloud_release_source._publish_file

    def race(name: str, *args: Any) -> os.stat_result:
        if name == "z.txt":
            # After a/ is fully published, another process takes z.txt.
            (output / "z.txt").write_text("theirs")
        return published(name, *args)

    monkeypatch.setattr(_cloud_release_source, "_publish_file", race)
    with pytest.raises(FileExistsError):
        _cloud_release_source._publish(staging, output)
    # Retain the partial tree, including nested files, and their colliding file.
    assert sorted(os.listdir(output)) == [staging.name, "a", "z.txt"]
    assert (output / "a/b/x.py").read_text() == "release"
    assert (output / "a/y.py").read_text() == "release"
    assert (output / "z.txt").read_text() == "theirs"


def test_failed_publication_never_removes_concurrent_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    staging = _staged(output, ("a.txt", "release"), ("z.txt", "release"))
    published = _cloud_release_source._publish_file

    def race(name: str, *args: Any) -> os.stat_result:
        if name == "z.txt":
            # After a.txt was published: another process replaces it and takes z.txt.
            (tmp_path / "theirs").write_text("theirs")
            os.replace(tmp_path / "theirs", output / "a.txt")
            (output / "z.txt").write_text("theirs too")
        return published(name, *args)

    monkeypatch.setattr(_cloud_release_source, "_publish_file", race)
    with pytest.raises(FileExistsError):
        _cloud_release_source._publish(staging, output)
    assert (output / "a.txt").read_text() == "theirs"
    assert (output / "z.txt").read_text() == "theirs too"


def test_a_directory_swapped_for_a_link_mid_publication_is_never_followed(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _bundle(_file("source/pyproject.toml"), _file("source/sub/evil.txt", b"release"))
    cloud({_SOURCE: _served(bundle)})
    outside = tmp_path / "private"
    outside.mkdir(mode=0o700)
    output = tmp_path / "out"
    mkdir = os.mkdir

    def swap(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        mkdir(path, mode, dir_fd=dir_fd)
        if dir_fd is not None and path == "sub":
            # Another local user replaces the new directory with a link out of DIR.
            os.rename("sub", "sub.gone", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.symlink(str(outside), "sub", dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", swap)
    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "output_unwritable"
    # Nothing was written through the link, and the outside directory's mode is untouched.
    assert os.listdir(outside) == []
    assert stat.S_IMODE(outside.stat().st_mode) == 0o700


def test_a_platform_without_directory_relative_calls_only_keeps_the_archive(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cloud({_SOURCE: _served(_PROJECT)})
    monkeypatch.setattr(_cloud_release_source, "_DIRECTORY_RELATIVE", False)
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "source_extraction_unsupported"
    assert not output.exists()


def test_a_directory_under_a_read_only_parent_is_extracted_in_place(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_SOURCE: _served(_PROJECT)})
    parent = tmp_path / "ro"
    output = parent / "out"
    output.mkdir(parents=True)
    parent.chmod(0o555)
    try:
        arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output)]
        assert main(arguments) == 0, capsys.readouterr().out
        assert (output / "pyproject.toml").exists()
        assert _leftovers(output) == []
    finally:
        parent.chmod(0o755)


def test_oversized_tar_metadata_is_refused_before_tarfile_reads_it(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_SOURCE: _served(_long_name_bundle())})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    error = _error(capsys)
    assert error["category"] == "source_bundle_unsafe"
    assert "tar metadata larger than 64 KiB" in str(error["message"])
    assert not output.exists()


@pytest.mark.parametrize("limit", ["MAX_EXTRACTED_BYTES", "MAX_EXTRACTED_MEMBERS"])
def test_a_bundle_that_unpacks_too_large_is_refused_as_its_headers_are_read(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    limit: str,
) -> None:
    monkeypatch.setattr(_cloud_release_source, limit, 3)
    cloud({_SOURCE: _served(_PROJECT)})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "source_bundle_too_large"
    assert not output.exists()
    assert _leftovers(tmp_path) == []


def test_a_python_without_the_data_filter_only_keeps_the_archive(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = cloud({_SOURCE: _served(_PROJECT)})
    monkeypatch.delattr(tarfile, "data_filter")
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "source_extraction_unsupported"
    assert " ".join(_SOURCE) not in fake.requests
    archive = tmp_path / "bundle.tar.gz"
    assert main([*arguments[:-2], str(archive), "--archive", "--json"]) == 0
    assert archive.read_bytes() == _PROJECT


def test_links_within_the_directory_and_a_hard_link_extract(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _bundle(
        _file("source/docs/guide.md", b"guide"),
        _symlink("source/pkg/README.md", "../docs/guide.md"),
        ("source/copy.md", tarfile.LNKTYPE, "source/docs/guide.md", 0o644),
    )
    cloud({_SOURCE: _served(bundle)})
    output = tmp_path / "out"

    assert main(["cloud", "deployment", "source", "dep_live", "--output", str(output)]) == 0
    assert (output / "pkg" / "README.md").read_text() == "guide"
    assert (output / "copy.md").read_text() == "guide"
    assert "Extracted 1 file and 2 links" in capsys.readouterr().out


def test_a_bundle_without_the_source_directory_extracts_as_it_is(
    cloud: Callable[..., _Cloud], tmp_path: Path
) -> None:
    cloud({_SOURCE: _served(_bundle(_file("app.py"), _file("source/inner.py")))})
    output = tmp_path / "out"

    assert main(["cloud", "deployment", "source", "dep_live", "--output", str(output)]) == 0
    assert sorted(path.name for path in output.iterdir()) == ["app.py", "source"]


def _long_name_bundle() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.GNU_FORMAT) as archive:
        archive.addfile(tarfile.TarInfo("source/" + "n" * (128 * 1024)), io.BytesIO(b""))
    return buffer.getvalue()


def test_archive_mode_keeps_a_bundle_with_oversized_metadata_without_counting_it(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _long_name_bundle()
    cloud({_SOURCE: _served(bundle)})
    archive = tmp_path / "bundle.tar.gz"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(archive)]
    assert main([*arguments, "--archive", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert (result["files"], result["links"], result["file_bytes"]) == (None, None, None)
    assert archive.read_bytes() == bundle
    assert _leftovers(tmp_path) == []
    archive.unlink()
    assert main([*arguments, "--archive"]) == 0
    assert "too large to count its members" in capsys.readouterr().out


def test_a_signed_size_in_a_tar_header_is_refused_rather_than_read_backwards(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    header = bytearray(tarfile.TarInfo("source/x").tobuf(format=tarfile.USTAR_FORMAT))
    header[124:136] = b"-0000001000\0"
    # The checksum is left as the original's; the size check runs first.
    bundle = gzip.compress(bytes(header) + b"\0" * 1024)
    cloud({_SOURCE: _served(bundle)})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "source_bundle_invalid"
    assert not output.exists()


def test_an_unreadable_archive_is_refused(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_SOURCE: _served(b"not a tar.gz")})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "source_bundle_invalid"
    assert not output.exists()
    assert _leftovers(tmp_path) == []


def test_a_release_built_from_a_repository_names_the_commit_to_fetch(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    revision = "3f2a9c1d0b7e4a5f6c8d9e0f1a2b3c4d5e6f7a8b"
    cloud(
        {
            _SOURCE: _refused(
                409,
                "source_not_uploaded",
                "This Release was built from a connected repository.",
                repository="https://github.com/acme/my-agent",
                revision=revision,
            )
        }
    )
    output = tmp_path / "out"
    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output)]

    assert main(arguments) == 2
    message = capsys.readouterr().err
    assert message.startswith("Error: Release dep_live was built from https://github.com/acme")
    assert f"`git checkout {revision}`" in message

    assert main([*arguments, "--json"]) == 2
    error = _error(capsys)
    assert error["code"] == "source_not_uploaded"
    assert error["repository"] == "https://github.com/acme/my-agent"
    assert error["revision"] == revision
    assert not output.exists()
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (404, "release_not_found"),
        (410, "source_bundle_expired"),
        (503, "source_bundles_unavailable"),
        (403, "developer_required"),
        (403, "agent_not_yours"),
        (403, "key_scope_required"),
    ],
)
def test_cloud_refusals_carry_their_code_and_message(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
    code: str,
) -> None:
    cloud({_SOURCE: _refused(status, code, "Cloud says why.")})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    error = _error(capsys)
    assert error["code"] == code
    assert error["message"] == f"Cayu Cloud API returned HTTP {status}: Cloud says why."
    assert not output.exists()
    assert _leftovers(tmp_path) == []


def test_a_cloud_without_the_route_says_it_cannot_download_source_yet(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({})
    output = tmp_path / "out"

    assert main(["cloud", "deployment", "source", "dep_live", "--output", str(output)]) == 2
    assert capsys.readouterr().err.strip() == (
        "Error: This Cayu Cloud can't download a release's source yet."
    )


def test_a_release_name_that_is_not_an_id_needs_the_agent(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({})

    arguments = ["cloud", "deployment", "source", "latest", "--output", str(tmp_path / "out")]
    assert main([*arguments, "--json"]) == 2
    assert "--application" in str(_error(capsys)["message"])


def test_an_oversized_answer_is_refused(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import _cloud_release_source

    monkeypatch.setattr(_cloud_release_source, "SOURCE_BUNDLE_MAX_BYTES", 16)
    cloud({_SOURCE: _served(_PROJECT)})
    output = tmp_path / "out"

    arguments = ["cloud", "deployment", "source", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "source_bundle_invalid"
    assert not output.exists()
    assert _leftovers(tmp_path) == []


def test_a_rejected_login_is_refreshed_once_for_the_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        if request.headers["Authorization"] == "Bearer stale":
            return httpx.Response(401, json={"detail": "Unauthorized"})
        return _served(_PROJECT)

    real_client = httpx.Client
    monkeypatch.setattr(
        _cloud_api.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(answer), **kwargs),
    )
    client = CloudApiClient(
        api_url="https://cloud.example",
        api_key="stale",
        rejected_api_key_provider=lambda _rejected: "fresh",
    )
    sink = io.BytesIO()

    downloaded = client.download(_SOURCE[1], sink, max_bytes=len(_PROJECT))

    assert seen == ["Bearer stale", "Bearer fresh"]
    assert sink.getvalue() == _PROJECT
    assert downloaded.sha256 == hashlib.sha256(_PROJECT).hexdigest()
    assert downloaded.headers["x-cayu-content-digest"] == f"sha256:{downloaded.sha256}"
