"""Tests for the newer-release notice in ``cayu new``."""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from cayu.cli import _release_check as release_check
from cayu.cli import main
from cayu.cli import scaffold as scaffold_cli

_fetch_latest_release = release_check.fetch_latest_release


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def pypi(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Run `cayu new` as an installed 0.8.1 release against a stubbed PyPI answer."""

    state: dict[str, object] = {"latest": "0.14.0", "fetches": 0}

    def fetch() -> str | None:
        state["fetches"] = int(state["fetches"]) + 1
        latest = state["latest"]
        return latest if isinstance(latest, str) else None

    monkeypatch.delenv(release_check.NO_VERSION_CHECK_VARIABLE, raising=False)
    monkeypatch.setattr(release_check, "fetch_latest_release", fetch)
    monkeypatch.setattr(scaffold_cli, "_cayu_source_checkout", lambda: None)
    monkeypatch.setattr(scaffold_cli, "_installed_cayu_version", lambda: "0.8.1")
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    return state


@pytest.mark.parametrize(
    ("latest", "installed", "newer"),
    [
        ("0.14.0", "0.8.1", True),
        ("0.10.0", "0.9.9", True),
        ("0.14.0", "0.14.0", False),
        ("0.14", "0.14.0", False),
        ("0.14.0", "0.15.0", False),
        ("0.15.0rc1", "0.14.0", False),
        ("0.15.0.dev1", "0.14.0", False),
        ("0.15.0a1", "0.14.0", False),
        ("0.14.0", "0.14.0rc1", True),
        ("0.14.0", "0.14.0.dev3", True),
        ("0.14.0", "0.14.0.post1", False),
        ("0.14.0", "0.14.0+local", False),
        ("0.14.0", "0.15.0.dev0", False),
        ("not-a-version", "0.8.1", False),
        ("0.14.0", "unknown", False),
    ],
)
def test_only_final_releases_count_as_newer(latest: str, installed: str, newer: bool) -> None:
    assert release_check.is_newer_release(latest, installed) is newer


def test_latest_release_asks_pypi_at_most_once_a_day(tmp_path: Path) -> None:
    cache = tmp_path / "cayu" / "release-check.json"
    calls: list[str] = []

    def fetch() -> str | None:
        calls.append("fetch")
        return "0.14.0"

    assert release_check.latest_release(fetch=fetch, cache_path=cache, now=1000.0) == "0.14.0"
    assert release_check.latest_release(fetch=fetch, cache_path=cache, now=1000.0 + 3600) == (
        "0.14.0"
    )
    assert calls == ["fetch"]
    assert json.loads(cache.read_text()) == {"checked_at": 1000.0, "latest": "0.14.0"}

    release_check.latest_release(fetch=fetch, cache_path=cache, now=1000.0 + 24 * 3600)
    assert calls == ["fetch", "fetch"]


def test_failed_check_is_cached_and_a_bad_cache_is_ignored(tmp_path: Path) -> None:
    cache = tmp_path / "release-check.json"
    calls: list[str] = []

    def failing() -> str | None:
        calls.append("fetch")
        return None

    assert release_check.latest_release(fetch=failing, cache_path=cache, now=50.0) is None
    assert release_check.latest_release(fetch=failing, cache_path=cache, now=60.0) is None
    assert calls == ["fetch"]

    cache.write_text("{not json")
    assert release_check.latest_release(fetch=lambda: "0.14.0", cache_path=cache, now=70.0) == (
        "0.14.0"
    )
    # A cache stamped in the future (clock moved back) is stale.
    cache.write_text(json.dumps({"checked_at": 10_000.0, "latest": "0.1.0"}))
    assert release_check.latest_release(fetch=lambda: "0.14.0", cache_path=cache, now=80.0) == (
        "0.14.0"
    )


def test_unwritable_cache_still_answers(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    cache = blocker / "cayu" / "release-check.json"
    assert release_check.latest_release(fetch=lambda: "0.14.0", cache_path=cache, now=1.0) == (
        "0.14.0"
    )


def test_opt_out_variable_skips_the_check(tmp_path: Path) -> None:
    def fetch() -> str | None:
        raise AssertionError("the opted-out check must not fetch")

    assert (
        release_check.newer_release(
            "0.8.1",
            fetch=fetch,
            cache_path=tmp_path / "cache.json",
            environ={release_check.NO_VERSION_CHECK_VARIABLE: "1"},
        )
        is None
    )
    assert release_check.newer_release(
        "0.8.1",
        fetch=lambda: "0.14.0",
        cache_path=tmp_path / "cache.json",
        environ={release_check.NO_VERSION_CHECK_VARIABLE: "0"},
    ) == release_check.NewerRelease(installed="0.8.1", latest="0.14.0")


def test_fetch_reads_pypi_latest_with_a_short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    @contextmanager
    def stream(method: str, url: str, **kwargs: object):
        seen.update(url=url, **kwargs)
        yield httpx.Response(
            200, json={"info": {"version": "0.14.0"}}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "stream", stream)
    assert _fetch_latest_release() == "0.14.0"
    assert seen["url"] == "https://pypi.org/pypi/cayu/json"
    assert seen["timeout"] == 2.0


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.ConnectTimeout("timed out"),
        httpx.Response(503, request=httpx.Request("GET", "https://pypi.org/pypi/cayu/json")),
        httpx.Response(
            200, content=b"<html>", request=httpx.Request("GET", "https://pypi.org/pypi/cayu/json")
        ),
        httpx.Response(
            200, json={"info": {}}, request=httpx.Request("GET", "https://pypi.org/pypi/cayu/json")
        ),
    ],
)
def test_fetch_is_silent_when_pypi_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, outcome: object
) -> None:
    @contextmanager
    def stream(method: str, url: str, **_: object):
        if isinstance(outcome, Exception):
            raise outcome
        yield outcome

    monkeypatch.setattr(httpx, "stream", stream)
    assert _fetch_latest_release() is None


def test_cayu_new_behind_pypi_warns_and_continues_without_a_terminal(
    pypi: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["new", "proj", "--dir", str(tmp_path), "--preset", "agent"]) == 0

    captured = capsys.readouterr()
    assert "Cayu 0.14.0 is available; this is Cayu 0.8.1." in captured.err
    command = f"uvx --from 'cayu==0.14.0' cayu new proj --dir {tmp_path} --preset agent"
    assert f"  {command}\n" in captured.err
    assert "[y/N]" not in captured.err
    assert "0.14.0" not in captured.out
    assert (tmp_path / "proj" / "app.py").is_file()


def test_cayu_new_json_reports_the_newer_release(
    pypi: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["new", "proj", "--dir", str(tmp_path), "--with", "knowledge", "--dry-run", "--json"]
    assert main(argv) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["newer_release"] == {
        "installed": "0.8.1",
        "latest": "0.14.0",
        "command": (
            f"uvx --from 'cayu==0.14.0' cayu new proj --dir {tmp_path} "
            "--with knowledge --dry-run --json"
        ),
    }
    assert "Cayu 0.14.0 is available" in captured.err


@pytest.mark.parametrize("latest", ["0.8.1", "0.8.0", "0.15.0rc1", None])
def test_cayu_new_is_quiet_when_current_or_pypi_is_unavailable(
    pypi: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    latest: str | None,
) -> None:
    pypi["latest"] = latest
    assert main(["new", "proj", "--dir", str(tmp_path), "--dry-run", "--json"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "newer_release" not in json.loads(captured.out)
    assert pypi["fetches"] == 1


def test_cayu_new_opt_out_does_not_ask_pypi(
    pypi: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(release_check.NO_VERSION_CHECK_VARIABLE, "1")
    assert main(["new", "proj", "--dir", str(tmp_path), "--dry-run"]) == 0

    assert capsys.readouterr().err == ""
    assert pypi["fetches"] == 0


def test_cayu_new_from_a_source_checkout_does_not_ask_pypi(
    pypi: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scaffold_cli,
        "_cayu_source_checkout",
        lambda: scaffold_cli._CayuSourceCheckout(root=tmp_path, version="0.8.1"),
    )
    assert main(["new", "proj", "--dir", str(tmp_path), "--dry-run"]) == 0

    assert capsys.readouterr().err == ""
    assert pypi["fetches"] == 0


@pytest.mark.parametrize("answer", ["", "n\n", "\n"])
def test_interactive_cayu_new_stops_by_default(
    pypi: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    stderr = _Terminal()
    monkeypatch.setattr(sys, "stdin", _Terminal(answer))
    monkeypatch.setattr(sys, "stderr", stderr)

    assert main(["new", "proj", "--dir", str(tmp_path)]) == 1

    assert "Continue with 0.8.1 anyway? [y/N] " in stderr.getvalue()
    assert "uvx --from 'cayu==0.14.0' cayu new proj" in stderr.getvalue()
    assert not (tmp_path / "proj").exists()


def test_interactive_cayu_new_continues_when_confirmed(
    pypi: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stderr = _Terminal()
    monkeypatch.setattr(sys, "stdin", _Terminal("y\n"))
    monkeypatch.setattr(sys, "stderr", stderr)

    assert main(["new", "proj", "--dir", str(tmp_path)]) == 0

    assert "Continue with 0.8.1 anyway? [y/N] " in stderr.getvalue()
    assert (tmp_path / "proj" / "app.py").is_file()


def test_terminal_json_and_dry_run_never_prompt(
    pypi: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdin = _Terminal()
    stderr = _Terminal()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stderr", stderr)

    assert main(["new", "proj", "--dir", str(tmp_path), "--json"]) == 0
    assert main(["new", "other", "--dir", str(tmp_path), "--dry-run"]) == 0

    assert "[y/N]" not in stderr.getvalue()
    assert stderr.getvalue().count("Cayu 0.14.0 is available") == 2


@pytest.mark.parametrize("stream_name", ["stdin", "stderr"])
@pytest.mark.parametrize("missing", [True, False])
def test_newer_release_notice_tolerates_unavailable_streams(
    pypi, tmp_path, monkeypatch, stream_name, missing
) -> None:
    stream = None
    if not missing:
        stream = io.StringIO()
        stream.close()
    with monkeypatch.context() as patch:
        patch.setattr(sys, stream_name, stream)
        assert main(["new", "proj", "--dir", str(tmp_path), "--preset", "agent"]) == 0
    assert (tmp_path / "proj" / "app.py").is_file()


def test_interactive_scaffold_with_missing_stdin_reports_terminal_requirement(
    pypi, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(sys, "stdin", None)
    assert main(["new", "proj", "--dir", str(tmp_path), "--interactive"]) == 1
    assert "requires a terminal" in capsys.readouterr().err


def test_upgrade_command_identifies_its_shell(pypi, tmp_path, capsys) -> None:
    assert main(["new", "proj", "--dir", str(tmp_path / "with spaces"), "--dry-run"]) == 0
    assert "run in a POSIX shell:" in capsys.readouterr().err


def test_slow_drip_lookup_returns_within_total_budget(monkeypatch, tmp_path) -> None:
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    stopped = threading.Event()
    connected = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "10000")
            self.end_headers()
            connected.set()
            try:
                # Each read succeeds before the per-read timeout. Only the
                # overall deadline can bound this response.
                while not stopped.wait(0.05):
                    self.wfile.write(b" ")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        release_check, "_PYPI_PROJECT_URL", f"http://127.0.0.1:{server.server_port}/"
    )
    monkeypatch.setattr(release_check, "_FETCH_TIMEOUT_S", 0.4)
    cache = tmp_path / "release-cache.json"
    try:
        start = time.monotonic()
        assert release_check.latest_release(fetch=_fetch_latest_release, cache_path=cache) is None
        elapsed = time.monotonic() - start
        assert connected.is_set()
        assert elapsed < 0.9
        assert json.loads(cache.read_text())["latest"] is None
        # A timed-out lookup is cached; another scaffold does not launch it again.
        assert (
            release_check.latest_release(
                fetch=lambda: pytest.fail("cache missed"), cache_path=cache
            )
            is None
        )
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
