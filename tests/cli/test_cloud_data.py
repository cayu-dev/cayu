"""`cayu cloud data` reads an Agent's /data through the Agent's own Cayu server."""

from __future__ import annotations

import errno
import json
import os
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu._data_volume import DATA_VOLUME_ENV
from cayu.cli import _cloud_data, main
from cayu.server import AuthenticatedAccess, BasicAuth, ServerConfig, create_server
from cayu.server import _data_volume_reader as reader

_OPERATOR_PASSWORD = "operator-password"
_RUN = b'{"run": "RUN", "steps": ["clone", "reproduce"], "error": "boom"}\n'
_SECRET = "OUTSIDE-THE-VOLUME-SECRET"
_AUDIT_MARK = '"event": "cayu.data_volume.read"'


@pytest.fixture
def volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "agent-data"
    (root / "scout" / "runs" / "RUN").mkdir(parents=True)
    (root / "scout" / "runs" / "RUN" / "run.json").write_bytes(_RUN)
    (root / "scout" / "runs" / "RUN" / "report.md").write_text("# Report\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(_SECRET)
    os.symlink("../../outside/secret.txt", root / "scout" / "escape")
    monkeypatch.setenv(DATA_VOLUME_ENV, str(root))
    monkeypatch.chdir(tmp_path)
    return root


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A fake Cayu Cloud whose Agent `cayu-scout` serves Cayu at `/cayu/`.

    ``state["agent"]`` answers the Agent's requests and ``state["agent_requests"]`` records
    them, with their query strings.
    """

    state: dict[str, Any] = {"agent": None, "agent_requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            target = urlsplit(self.path)
            origin = f"http://{self.headers['Host']}"
            cloud_answers = {
                "/v1/applications": {"items": [{"id": "cayu-scout", "name": "Cayu Scout"}]},
                "/v1/applications/cayu-scout/service": {
                    "status": "running",
                    "cayu_url": f"{origin}/cayu/",
                },
                "/v1/applications/cayu-scout/operator-credentials": {
                    "username": "operator",
                    "password": _OPERATOR_PASSWORD,
                    "env": {"username": "U", "password": "P"},
                },
            }
            if target.path in cloud_answers:
                status, answer = 200, cloud_answers[target.path]
            elif target.path.startswith("/cayu/"):
                under_mount = self.path.removeprefix("/cayu")
                state["agent_requests"].append(under_mount)
                status, answer = state["agent"](self.headers.get("Authorization"), under_mount)
            else:
                status, answer = 404, {"detail": "Not Found"}
            payload = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setenv("CAYU_CLOUD_API_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def served(cloud: dict[str, Any], volume: Path) -> Iterator[dict[str, Any]]:
    """The Agent is a real Cayu server behind the operator login Cloud hands out."""

    access = AuthenticatedAccess(
        dependency=BasicAuth(username="operator", password=_OPERATOR_PASSWORD)
    )
    app = CayuApp(enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig(access=access))) as client:

        def forward(authorization: str | None, target: str) -> tuple[int, Any]:
            headers = {} if authorization is None else {"Authorization": authorization}
            response = client.get(target, headers=headers)
            return response.status_code, response.json()

        cloud["agent"] = forward
        yield cloud


def _data(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str, str]:
    code = main(["cloud", "data", *arguments])
    captured = capsys.readouterr()
    # The Agent's server runs in this process; its audit lines aren't the CLI's output.
    errors = [line for line in captured.err.splitlines() if _AUDIT_MARK not in line]
    return code, captured.out, "\n".join(errors)


def _assert_private_text_absent(*texts: str) -> None:
    for text in texts:
        assert _SECRET not in text
        assert _OPERATOR_PASSWORD not in text
        assert "customer-secret-material" not in text


def test_cat_prints_a_runs_file(served: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _data(capsys, "cat", "--application", "cayu-scout", "scout/runs/RUN/run.json")

    assert code == 0, err
    assert out.encode() == _RUN
    assert served["agent_requests"] == [
        "/api/contract",
        "/api/data-volume/content?path=scout%2Fruns%2FRUN%2Frun.json&offset=0",
    ]


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("../outside/secret.txt", "must not contain empty, `.` or `..` names"),
        ("/etc/passwd", "absolute paths are refused"),
        ("scout/escape", "leads outside the data volume through a symbolic link"),
    ],
)
def test_paths_out_of_the_volume_are_refused(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str], path: str, reason: str
) -> None:
    code, out, err = _data(capsys, "cat", "--application", "cayu-scout", path)

    assert code == 2
    assert out == ""
    assert err.startswith("Error: The Agent cayu-scout rejected the data volume file request: ")
    assert reason in err
    _assert_private_text_absent(out, err)


def test_ls_lists_a_directory_and_pages_on(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _data(capsys, "ls", "--application", "cayu-scout", "scout/runs/RUN")

    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["TYPE", "SIZE", "MODIFIED", "NAME"]
    assert [line.split()[::3] for line in lines[1:]] == [
        ["file", "report.md"],
        ["file", "run.json"],
    ]

    code, out, _ = _data(capsys, "ls", "--application", "cayu-scout", "--limit", "1", "--json")
    envelope = json.loads(out)
    assert code == 0
    assert envelope["ok"] is True
    assert envelope["operation"] == "data.ls"
    assert [entry["name"] for entry in envelope["result"]["entries"]] == ["scout"]
    assert envelope["result"]["next_after"] is None

    code, out, _ = _data(capsys, "ls", "--application", "cayu-scout", "scout", "--limit", "1")
    assert out.splitlines()[-1] == "More entries follow; go on with --after escape."


def test_stat_describes_an_entry(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _data(capsys, "stat", "--application", "cayu-scout", "scout/runs/RUN/run.json")

    assert code == 0
    assert out.startswith(f"scout/runs/RUN/run.json: file, {len(_RUN)} bytes, modified ")


def test_get_downloads_a_file_and_never_replaces_one_silently(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = _data(capsys, "get", "--application", "cayu-scout", "scout/runs/RUN/run.json")

    assert code == 0
    assert out.strip() == f"Wrote run.json ({len(_RUN)} bytes)."
    assert (tmp_path / "run.json").read_bytes() == _RUN

    (tmp_path / "run.json").write_text("kept")
    code, _, err = _data(capsys, "get", "--application", "cayu-scout", "scout/runs/RUN/run.json")
    assert code == 2
    assert "run.json exists; pass --force" in err
    assert (tmp_path / "run.json").read_text() == "kept"

    code, out, _ = _data(
        capsys,
        "get",
        "--application",
        "cayu-scout",
        "scout/runs/RUN/run.json",
        "--force",
        "--json",
    )
    assert code == 0
    assert json.loads(out)["result"] == {
        "bytes": len(_RUN),
        "output": "run.json",
        "path": "scout/runs/RUN/run.json",
        "type": "file",
    }
    assert (tmp_path / "run.json").read_bytes() == _RUN


def test_get_never_replaces_a_file_that_appears_during_the_download(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    forward = served["agent"]

    def meanwhile(authorization: str | None, target: str) -> tuple[int, Any]:
        if "/data-volume/content" in target:
            # Another download, or the person, takes the name after the command checked it.
            (tmp_path / "run.json").write_text("written meanwhile")
        return forward(authorization, target)

    served["agent"] = meanwhile
    code, _, err = _data(capsys, "get", "--application", "cayu-scout", "scout/runs/RUN/run.json")

    assert code == 2
    assert "run.json exists; pass --force" in err
    assert (tmp_path / "run.json").read_text() == "written meanwhile"
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".cayu-data-")]


def _without_hard_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """As on exFAT and many network or shared-folder mounts."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", refuse)


def test_get_works_where_the_file_system_has_no_hard_links(
    served: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_hard_links(monkeypatch)

    code, out, err = _data(capsys, "get", "--application", "cayu-scout", "scout/runs/RUN/run.json")

    assert code == 0, err
    assert out.strip() == f"Wrote run.json ({len(_RUN)} bytes)."
    assert (tmp_path / "run.json").read_bytes() == _RUN
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".cayu-data-")]


def test_without_hard_links_get_still_never_replaces_a_file_that_appears(
    served: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_hard_links(monkeypatch)
    forward = served["agent"]

    def meanwhile(authorization: str | None, target: str) -> tuple[int, Any]:
        if "/data-volume/content" in target:
            (tmp_path / "run.json").write_text("written meanwhile")
        return forward(authorization, target)

    served["agent"] = meanwhile
    code, _, err = _data(capsys, "get", "--application", "cayu-scout", "scout/runs/RUN/run.json")

    assert code == 2
    assert "run.json exists; pass --force" in err
    assert (tmp_path / "run.json").read_text() == "written meanwhile"


def test_get_downloads_a_directory_as_an_archive(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = _data(capsys, "get", "--application", "cayu-scout", "scout/")

    assert code == 0
    assert out.splitlines() == [
        f"Wrote scout.tar.gz ({(tmp_path / 'scout.tar.gz').stat().st_size} bytes).",
        "2 files in 3 directories.",
        "Left out 1: scout/escape (symlink).",
    ]
    with tarfile.open(tmp_path / "scout.tar.gz") as archive:
        assert archive.getnames() == [
            "scout",
            "scout/runs",
            "scout/runs/RUN",
            "scout/runs/RUN/report.md",
            "scout/runs/RUN/run.json",
        ]
        member = archive.extractfile("scout/runs/RUN/run.json")
        assert member is not None and member.read() == _RUN


def test_cat_refuses_a_file_past_its_bound(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_cloud_data, "MAX_CAT_BYTES", 8)

    code, out, err = _data(capsys, "cat", "--application", "cayu-scout", "scout/runs/RUN/run.json")

    assert code == 2
    assert out == ""
    assert "Download it with `cayu cloud data get`." in err


def test_a_file_that_changes_between_parts_is_refused(
    served: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    volume: Path,
) -> None:
    monkeypatch.setattr(reader, "MAX_READ_BYTES", 16)
    forward = served["agent"]

    def growing(authorization: str | None, target: str) -> tuple[int, Any]:
        answer = forward(authorization, target)
        if "offset=0" in target:
            with (volume / "scout" / "runs" / "RUN" / "run.json").open("ab") as run:
                run.write(b'{"step": "late"}\n')
        return answer

    served["agent"] = growing
    code, out, err = _data(capsys, "cat", "--application", "cayu-scout", "scout/runs/RUN/run.json")

    assert code == 2
    assert out == ""
    assert "changed while it was read; try again" in err


def test_a_terminal_gets_text_only_with_controls_escaped(
    served: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    volume: Path,
) -> None:
    (volume / "scout" / "colored.txt").write_bytes(b"red \x1b[31mtext\x1b[0m\n")
    (volume / "scout" / "image.bin").write_bytes(b"\x89PNG\xff\xfe")
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    code, out, _ = _data(capsys, "cat", "--application", "cayu-scout", "scout/colored.txt")
    assert code == 0
    assert out == "red \\x1b[31mtext\\x1b[0m\n"

    code, out, err = _data(capsys, "cat", "--application", "cayu-scout", "scout/image.bin")
    assert code == 2
    assert out == ""
    assert "isn't UTF-8 text, so it isn't printed to a terminal" in err


def test_an_agent_on_an_older_cayu_is_unsupported(
    cloud: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    def older(authorization: str | None, target: str) -> tuple[int, Any]:
        if target == "/api/contract":
            return 200, {"server_contract_version": "84"}
        return 404, {"detail": "Not Found"}

    cloud["agent"] = older
    code, out, _ = _data(capsys, "ls", "--application", "cayu-scout", "--json")

    error = json.loads(out)["error"]
    assert code == 2
    assert error["category"] == "unsupported"
    assert error["message"] == (
        "The Agent cayu-scout runs a Cayu that can't read its data volume. Deploy a release "
        "built with this CLI's Cayu version or newer, then try again."
    )


def test_an_agent_without_a_data_volume_says_how_to_get_one(
    cloud: dict[str, Any], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DATA_VOLUME_ENV, raising=False)
    access = AuthenticatedAccess(
        dependency=BasicAuth(username="operator", password=_OPERATOR_PASSWORD)
    )
    app = CayuApp(enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig(access=access))) as client:

        def forward(authorization: str | None, target: str) -> tuple[int, Any]:
            response = client.get(target, headers={"Authorization": authorization or ""})
            return response.status_code, response.json()

        cloud["agent"] = forward
        code, out, _ = _data(capsys, "stat", "--application", "cayu-scout", "scout", "--json")

    error = json.loads(out)["error"]
    assert code == 2
    assert error["category"] == "request_rejected"
    assert "doesn't set CAYU_DATA_VOLUME" in error["message"]
    assert "deploy the Agent again" in error["message"]


def _tamper_archive(served: dict[str, Any], **fields: str) -> None:
    forward = served["agent"]

    def tampered(authorization: str | None, target: str) -> tuple[int, Any]:
        status, answer = forward(authorization, target)
        if "/data-volume/archive" in target:
            answer = {**answer, **fields}
        return status, answer

    served["agent"] = tampered


def test_the_local_name_comes_from_the_requested_path(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _tamper_archive(served, name="../../escaped.tar.gz")

    code, _, err = _data(capsys, "get", "--application", "cayu-scout", "scout")

    assert code == 0, err
    assert (tmp_path / "scout.tar.gz").is_file()
    assert not (tmp_path.parent.parent / "escaped.tar.gz").exists()


def test_an_unreadable_archive_writes_nothing(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _tamper_archive(served, content_base64="%%%")

    code, _, err = _data(capsys, "get", "--application", "cayu-scout", "scout")

    assert code == 2
    assert "content that isn't base64" in err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["agent-data", "outside"]
