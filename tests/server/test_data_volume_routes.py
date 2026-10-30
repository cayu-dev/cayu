"""The data-volume routes read the host's volume, only it, and only for full-access callers."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import tarfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu._data_volume import (
    DATA_VOLUME_ENV,
    MAX_ARCHIVE_DEPTH,
    MAX_LIST_LIMIT,
    MAX_READ_BYTES,
)
from cayu.server import AuthenticatedAccess, BasicAuth, ServerConfig, create_server
from cayu.server import _data_volume_reader as reader
from cayu.server._data_volume_reader import DataVolume, DataVolumeRefusal
from cayu.server.auth import AuthContext
from cayu.server.proxy_identity import ControlPlaneRoles

_API = "/api/data-volume"
_LOGIN = ("operator", "operator-password")
_RUN = b'{"run": "RUN", "steps": ["clone", "reproduce"], "error": "boom"}\n'
_SECRET = "OUTSIDE-THE-VOLUME-SECRET"


@pytest.fixture
def volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`data/` holds a run; `outside/` beside it holds what no read may return."""

    root = tmp_path / "data"
    (root / "scout" / "runs" / "RUN").mkdir(parents=True)
    (root / "scout" / "runs" / "RUN" / "run.json").write_bytes(_RUN)
    (root / "scout" / "notes.txt").write_text("notes\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(_SECRET)
    monkeypatch.setenv(DATA_VOLUME_ENV, str(root))
    return root


@pytest.fixture
def client(volume: Path) -> Iterator[TestClient]:
    access = AuthenticatedAccess(dependency=BasicAuth(username=_LOGIN[0], password=_LOGIN[1]))
    app = CayuApp(enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig(access=access))) as served:
        yield served


def _get(client: TestClient, route: str, **params: Any) -> Any:
    return client.get(f"{_API}/{route}", params=params, auth=_LOGIN)


def _audit(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().err.splitlines() if line]


def _refusal(response: Any) -> str:
    return response.json()["detail"]["code"]


def test_a_file_its_entry_and_its_directory_are_read_and_audited(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    content = _get(client, "content", path="scout/runs/RUN/run.json")
    entry = _get(client, "entry", path="scout/runs/RUN/run.json")
    listing = _get(client, "entries", path="scout/")

    assert content.status_code == 200, content.text
    assert content.headers["cache-control"] == "private, no-store"
    body = content.json()
    assert base64.b64decode(body["content_base64"]) == _RUN
    assert (body["path"], body["size"], body["offset"]) == ("scout/runs/RUN/run.json", len(_RUN), 0)
    assert entry.json() | {"modified_at": None} == {
        "path": "scout/runs/RUN/run.json",
        "name": "run.json",
        "type": "file",
        "size": len(_RUN),
        "modified_at": None,
    }
    assert [(item["name"], item["type"], item["size"]) for item in listing.json()["entries"]] == [
        ("notes.txt", "file", 6),
        ("runs", "directory", None),
    ]
    assert listing.json()["path"] == "scout"
    # One content-free line per read, naming who read what and how much.
    lines = _audit(capsys)
    assert [(line["operation"], line["outcome"], line["path"]) for line in lines] == [
        ("content", "ok", "scout/runs/RUN/run.json"),
        ("entry", "ok", "scout/runs/RUN/run.json"),
        ("entries", "ok", "scout/"),
    ]
    assert {line["event"] for line in lines} == {"cayu.data_volume.read"}
    assert {line["subject"] for line in lines} == {"operator"}
    assert (lines[0]["bytes"], lines[0]["offset"], lines[2]["entries"]) == (len(_RUN), 0, 2)
    assert "boom" not in json.dumps(lines)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../outside/secret.txt",
        "scout/../../outside/secret.txt",
        "scout/..",
        "./scout",
        "scout//runs",
        "scout/./runs",
        "scout/\x1b[2Jruns",
        "x" * 256,
    ],
)
def test_paths_outside_the_volume_are_refused_before_any_read(
    client: TestClient, capsys: pytest.CaptureFixture[str], path: str
) -> None:
    capsys.readouterr()
    for route in ("entry", "entries", "content", "archive"):
        response = _get(client, route, path=path)
        assert response.status_code == 422, (route, response.text)
        assert _refusal(response) == "path_refused"
        assert _SECRET not in response.text

    # The refusal is audited too, on one line whatever the path holds.
    lines = _audit(capsys)
    assert [(line["outcome"], line["path"]) for line in lines] == [("path_refused", path)] * 4


def test_an_over_long_path_is_refused_without_echoing_it(client: TestClient) -> None:
    response = _get(client, "entry", path="a/" * 600)

    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            "Invalid control-plane request: query parameter path must be at most 1024 characters."
        )
    }


def _escapes(volume: Path) -> list[str]:
    """Symbolic links that lead out of the volume, each in its own way."""

    outside = volume.parent / "outside"
    os.symlink("../outside/secret.txt", volume / "relative")
    os.symlink(str(outside / "secret.txt"), volume / "absolute")
    os.symlink("../outside", volume / "directory")
    os.symlink("../..", volume / "scout" / "up")
    os.symlink("hop", volume / "chain")
    os.symlink("../outside/secret.txt", volume / "hop")
    os.symlink("/etc", volume / "system")
    return [
        "relative",
        "absolute",
        "directory/secret.txt",
        "scout/up/outside/secret.txt",
        "chain",
        "system/passwd",
    ]


def test_symlinks_that_lead_out_of_the_volume_are_refused(client: TestClient, volume: Path) -> None:
    for path in _escapes(volume):
        for route in ("entry", "content"):
            response = _get(client, route, path=path)
            assert response.status_code == 422, (route, path, response.text)
            assert _refusal(response) == "path_outside_volume"
            assert _SECRET not in response.text
    for path in ("directory", "scout/up", "system"):
        for route in ("entries", "archive"):
            response = _get(client, route, path=path)
            assert response.status_code == 422, (route, path, response.text)
            assert _refusal(response) == "path_outside_volume"


def test_an_archive_leaves_every_symlink_out(client: TestClient, volume: Path) -> None:
    _escapes(volume)

    response = _get(client, "archive", path="")

    assert response.status_code == 200, response.text
    body = response.json()
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(body["content_base64"]))) as archive:
        assert archive.getnames() == [
            "data",
            "data/scout",
            "data/scout/notes.txt",
            "data/scout/runs",
            "data/scout/runs/RUN",
            "data/scout/runs/RUN/run.json",
        ]
        contents = b""
        for member in archive.getmembers():
            extracted = archive.extractfile(member)
            if extracted is not None:
                contents += extracted.read()
    assert _SECRET.encode() not in contents
    assert body["skipped_count"] == 7
    assert {item["reason"] for item in body["skipped"]} == {"symlink"}
    assert (body["name"], body["files"], body["directories"]) == ("data.tar.gz", 2, 4)


def test_an_archive_names_a_name_that_isnt_utf8_and_keeps_the_rest(
    volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # How Linux gives a name with the byte 0xff; APFS refuses to create one.
    names = reader._names

    def with_undecodable(fd: int) -> Iterator[str]:
        yield from names(fd)
        if "notes.txt" in os.listdir(fd):
            yield "bad-\udcff"

    monkeypatch.setattr(reader, "_names", with_undecodable)

    archive = DataVolume(str(volume)).archive(("scout",))

    assert [item.model_dump() for item in archive.skipped] == [
        {"member": "scout/bad-\\xff", "reason": "unreadable_name"}
    ]
    assert (archive.files, archive.directories) == (2, 3)
    json.dumps(archive.model_dump(mode="json")).encode("utf-8")


def test_a_real_name_that_isnt_utf8_is_skipped(client: TestClient, volume: Path) -> None:
    try:
        os.close(os.open(os.fsencode(volume / "scout") + b"/bad-\xff", os.O_CREAT | os.O_WRONLY))
    except OSError:
        pytest.skip("This file system only takes UTF-8 names.")

    archive = _get(client, "archive", path="scout")
    listing = _get(client, "entries", path="scout")

    assert archive.status_code == 200, archive.text
    assert archive.json()["skipped"] == [{"member": "scout/bad-\\xff", "reason": "unreadable_name"}]
    assert listing.json()["omitted"] == 1


def test_a_platform_that_cant_confine_reads_serves_no_volume(
    volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As on Windows: no O_DIRECTORY or O_NOFOLLOW. The reader still imports, so the server
    # does, and refuses every read.
    monkeypatch.delattr(os, "O_DIRECTORY")
    monkeypatch.delattr(os, "O_NOFOLLOW")
    spec = importlib.util.spec_from_file_location("_unconfined_reader", reader.__file__)
    assert spec is not None and spec.loader is not None
    unconfined = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, unconfined)
    spec.loader.exec_module(unconfined)

    with pytest.raises(unconfined.DataVolumeRefusal) as refused:
        unconfined.DataVolume(str(volume)).entry(("scout",))

    assert unconfined.CONFINED_READS_SUPPORTED is False
    assert (refused.value.status_code, refused.value.code) == (409, "data_volume_unavailable")
    assert reader.CONFINED_READS_SUPPORTED is True


def test_an_archive_names_a_skipped_entry_at_its_deepest_level(volume: Path) -> None:
    # The longest member an archive can name: the directory, a 255-byte name at each of its
    # levels, and a link at the bottom. Built through descriptors, past any path length limit.
    top = "t" * 255
    level = os.open(volume / "scout", os.O_RDONLY | os.O_DIRECTORY)
    names = [top, *["d" * 255] * MAX_ARCHIVE_DEPTH]
    try:
        for name in names:
            os.mkdir(name, dir_fd=level)
            below = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=level)
            os.close(level)
            level = below
        os.symlink("elsewhere", "l" * 255, dir_fd=level)
    finally:
        os.close(level)

    archive = DataVolume(str(volume)).archive(("scout", top))

    (skipped,) = archive.skipped
    assert skipped.member == "/".join([*names, "l" * 255])
    assert archive.directories == MAX_ARCHIVE_DEPTH + 1


def test_symlinks_within_the_volume_are_followed(client: TestClient, volume: Path) -> None:
    os.symlink("scout/runs/RUN", volume / "latest")
    os.symlink(str(volume / "scout" / "runs"), volume / "runs")
    os.symlink("../RUN", volume / "scout" / "runs" / "RUN" / "self")

    for path in ("latest/run.json", "runs/RUN/run.json", "latest/self/self/run.json"):
        response = _get(client, "content", path=path)
        assert response.status_code == 200, (path, response.text)
        assert base64.b64decode(response.json()["content_base64"]) == _RUN
    assert _get(client, "entry", path="latest").json()["type"] == "directory"
    listed = _get(client, "entries", path="").json()["entries"]
    assert {item["name"]: item["type"] for item in listed}["latest"] == "symlink"


def test_a_link_loop_is_refused(client: TestClient, volume: Path) -> None:
    os.symlink("loop", volume / "loop")

    response = _get(client, "content", path="loop")

    assert response.status_code == 422
    assert _refusal(response) == "path_refused"


def test_a_name_swapped_after_it_was_examined_is_not_followed(volume: Path) -> None:
    name = volume / "scout" / "notes.txt"
    examined = os.lstat(name)
    root = os.open(volume / "scout", os.O_RDONLY | os.O_DIRECTORY)
    try:
        # The examined file stays in place under another name, so no file the swaps create
        # can take over its inode number (Linux reuses a freed one at once).
        name.rename(name.with_name("examined.txt"))
        # Swapped for a link out of the volume: O_NOFOLLOW refuses to open it.
        os.symlink(volume.parent / "outside" / "secret.txt", name)
        with pytest.raises(DataVolumeRefusal) as linked:
            reader._open_at("notes.txt", root, reader._FILE_FLAGS, examined, ("notes.txt",))
        # Swapped for another file: not the one examined.
        name.unlink()
        name.write_text("another file\n")
        with pytest.raises(DataVolumeRefusal) as replaced:
            reader._open_at("notes.txt", root, reader._FILE_FLAGS, examined, ("notes.txt",))
    finally:
        os.close(root)

    assert linked.value.code == replaced.value.code == "changed_while_read"


def test_special_files_are_described_but_never_opened(client: TestClient, volume: Path) -> None:
    os.mkfifo(volume / "scout" / "pipe")

    entry = _get(client, "entry", path="scout/pipe")
    content = _get(client, "content", path="scout/pipe")
    archive = _get(client, "archive", path="scout")

    assert entry.json()["type"] == "other"
    assert content.status_code == 409
    assert _refusal(content) == "not_a_file"
    assert {"member": "scout/pipe", "reason": "special_file"} in archive.json()["skipped"]


def test_reads_name_the_wrong_kind_of_entry_and_a_missing_one(client: TestClient) -> None:
    assert _refusal(_get(client, "content", path="scout")) == "not_a_file"
    assert _refusal(_get(client, "entries", path="scout/notes.txt")) == "not_a_directory"
    assert _refusal(_get(client, "archive", path="scout/notes.txt")) == "not_a_directory"
    missing = _get(client, "entry", path="scout/missing.json")
    assert missing.status_code == 404
    assert _refusal(missing) == "path_not_found"
    assert _refusal(_get(client, "entry", path="scout/notes.txt/inside")) == "path_not_found"


def test_a_listing_pages_in_name_order(client: TestClient, volume: Path) -> None:
    for name in ("e", "a", "d", "b", "c"):
        (volume / "page" / name).mkdir(parents=True)

    pages: list[list[str]] = []
    after: str | None = None
    while True:
        params: dict[str, Any] = {"path": "page", "limit": 2}
        if after is not None:
            params["after"] = after
        page = _get(client, "entries", **params).json()
        pages.append([item["name"] for item in page["entries"]])
        after = page["next_after"]
        if after is None:
            break

    assert pages == [["a", "b"], ["c", "d"], ["e"]]
    for limit in (0, MAX_LIST_LIMIT + 1):
        refused = _get(client, "entries", path="page", limit=limit)
        assert refused.status_code == 422
        assert "query parameter limit must be" in refused.json()["detail"]


def test_a_listing_reads_a_bounded_part_of_a_huge_directory(
    client: TestClient, volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(5):
        (volume / "many" / f"{index}").mkdir(parents=True)
    monkeypatch.setattr(reader, "MAX_DIRECTORY_SCAN", 3)

    page = _get(client, "entries", path="many").json()

    assert len(page["entries"]) == 3
    assert page["complete"] is False


def _ticking(step: float) -> Callable[[], float]:
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


def test_a_listing_stops_at_its_deadline_and_says_where_to_go_on(volume: Path) -> None:
    for name in ("a", "b", "c"):
        (volume / "slow" / name).mkdir(parents=True)
    # Each reading of the clock is 25 seconds later, so the deadline passes after one entry.
    served = DataVolume(str(volume), clock=_ticking(25.0))

    page = served.entries(("slow",), limit=10, after=None)

    assert [entry.name for entry in page.entries] == ["a"]
    assert page.next_after == "a"


def test_content_is_read_in_bounded_parts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reader, "MAX_READ_BYTES", 10)

    parts = [
        base64.b64decode(
            _get(client, "content", path="scout/runs/RUN/run.json", offset=offset).json()[
                "content_base64"
            ]
        )
        for offset in range(0, len(_RUN) + 10, 10)
    ]

    assert all(len(part) <= 10 for part in parts)
    assert b"".join(parts) == _RUN
    assert parts[-1] == b""
    refused = _get(client, "content", path="scout/notes.txt", length=MAX_READ_BYTES + 1)
    assert refused.status_code == 422


@pytest.mark.parametrize(
    ("bound", "value", "reason"),
    [
        ("MAX_ARCHIVE_ENTRIES", 3, "holds more than 3 entries"),
        ("MAX_ARCHIVE_INPUT_BYTES", 3, "MiB of files"),
        ("MAX_ARCHIVE_DEPTH", 1, "nests directories more than 1 deep"),
    ],
)
def test_an_archive_past_a_bound_is_refused(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
    value: int,
    reason: str,
) -> None:
    monkeypatch.setattr(reader, bound, value)

    response = _get(client, "archive", path="scout")

    assert response.status_code == 413
    assert _refusal(response) == "archive_too_large"
    assert reason in response.json()["detail"]["message"]


def test_an_archive_that_compresses_past_its_bound_is_refused(
    volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (volume / "scout" / "noise.bin").write_bytes(os.urandom(64 * 1024))
    monkeypatch.setattr(reader, "MAX_ARCHIVE_BYTES", 16 * 1024)

    with pytest.raises(DataVolumeRefusal) as refused:
        DataVolume(str(volume)).archive(("scout",))

    assert refused.value.status_code == 413
    assert "compresses to more than" in refused.value.message


def test_an_archive_that_takes_too_long_is_refused(volume: Path) -> None:
    with pytest.raises(DataVolumeRefusal) as refused:
        DataVolume(str(volume), clock=_ticking(25.0)).archive(("scout",))

    assert refused.value.code == "archive_too_large"
    assert "took longer to read than one archive may" in refused.value.message


def test_one_archive_is_built_at_a_time(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = threading.Event(), threading.Event()
    build = DataVolume.archive

    def held(self: DataVolume, parts: tuple[str, ...]) -> Any:
        started.set()
        release.wait(10)
        return build(self, parts)

    monkeypatch.setattr(DataVolume, "archive", held)
    first: list[Any] = []
    worker = threading.Thread(target=lambda: first.append(_get(client, "archive", path="scout")))
    worker.start()
    try:
        assert started.wait(10)
        busy = _get(client, "archive", path="scout")
    finally:
        release.set()
        worker.join(10)

    assert busy.status_code == 429
    assert busy.headers["retry-after"] == "5"
    assert _refusal(busy) == "archive_busy"
    assert first[0].status_code == 200


def test_only_authenticated_full_access_callers_may_read(volume: Path) -> None:
    roles = ControlPlaneRoles()
    roles.add_api_path("/api")

    class Auth:
        control_plane_roles = roles

        def __call__(self, request):
            role = request.headers.get("x-test-role")
            if role is None:
                raise HTTPException(401, "Authentication required")
            context = AuthContext(subject=f"{role}-person", claims={"role": role})
            roles.require(request, context)
            return context

    path = {"path": "scout/runs/RUN/run.json"}
    config = ServerConfig(access=AuthenticatedAccess(dependency=Auth()))
    with TestClient(create_server(CayuApp(enable_logging=False), config=config)) as client:
        assert client.get(f"{_API}/content", params=path).status_code == 401
        member = client.get(f"{_API}/content", params=path, headers={"x-test-role": "member"})
        developer = client.get(f"{_API}/content", params=path, headers={"x-test-role": "developer"})
    with TestClient(
        create_server(CayuApp(enable_logging=False), config=ServerConfig.local_development())
    ) as open_server:
        unauthenticated = open_server.get(f"{_API}/content", params=path)

    assert member.status_code == 403
    assert "content_base64" not in member.text
    assert developer.status_code == 200
    assert base64.b64decode(developer.json()["content_base64"]) == _RUN
    # A server without authentication never serves its volume.
    assert unauthenticated.status_code == 409
    assert _refusal(unauthenticated) == "data_volume_requires_authentication"
    assert "content_base64" not in unauthenticated.text


@pytest.mark.parametrize(
    ("configured", "code"),
    [
        (None, "data_volume_not_configured"),
        ("missing", "data_volume_unavailable"),
        ("relative/data", "data_volume_unavailable"),
    ],
)
def test_a_server_without_its_data_volume_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: str | None, code: str
) -> None:
    if configured is None:
        monkeypatch.delenv(DATA_VOLUME_ENV, raising=False)
    elif configured == "missing":
        monkeypatch.setenv(DATA_VOLUME_ENV, str(tmp_path / "missing"))
    else:
        monkeypatch.setenv(DATA_VOLUME_ENV, configured)
    access = AuthenticatedAccess(dependency=BasicAuth(username=_LOGIN[0], password=_LOGIN[1]))
    config = ServerConfig(access=access)
    with TestClient(create_server(CayuApp(enable_logging=False), config=config)) as client:
        response = _get(client, "entries", path="")

    assert response.status_code == 409
    assert _refusal(response) == code
