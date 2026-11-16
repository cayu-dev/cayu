"""Tests for the newer-release notice every `cayu` command prints on stderr."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import cayu.cli as cli
from cayu.cli import _release_check as release_check
from cayu.cli import main
from cayu.cli import scaffold as scaffold_cli

DAY = 24 * 60 * 60
NOTICE = "cayu 0.14.0 is available; you're running 0.8.1."


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def pypi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    """An installed 0.8.1 uv tool, a stubbed PyPI answer, a temp cache and a terminal."""

    state: dict[str, object] = {"latest": "0.14.0", "fetches": 0, "now": 1_000_000.0}

    def fetch() -> str | None:
        state["fetches"] = int(state["fetches"]) + 1
        latest = state["latest"]
        return latest if isinstance(latest, str) else None

    cache = tmp_path / "cache" / "release-check.json"
    monkeypatch.delenv(release_check.NO_VERSION_CHECK_VARIABLE, raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(release_check, "fetch_latest_release", fetch)
    monkeypatch.setattr(release_check, "release_cache_path", lambda: cache)
    monkeypatch.setattr(release_check, "installed_release", lambda: "0.8.1")
    monkeypatch.setattr(release_check, "install_method", lambda: "uv-tool")
    monkeypatch.setattr(release_check, "time", SimpleNamespace(time=lambda: float(state["now"])))
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    state["stderr"] = _Terminal()
    state["monkeypatch"] = monkeypatch
    return state


def _run(state: dict[str, object], arguments: list[str]) -> int:
    """Run `cayu` with ``state["stderr"]`` as stderr.

    Pytest's own capture replaces ``sys.stderr`` when the test body starts, so the
    stream is attached here rather than in the fixture.
    """

    monkeypatch = state["monkeypatch"]
    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    monkeypatch.setattr(sys, "stderr", state["stderr"])
    return main(arguments)


def _stderr(state: dict[str, object]) -> str:
    stream = state["stderr"]
    assert isinstance(stream, io.StringIO)
    return stream.getvalue()


def test_notice_follows_the_command_once_per_cache_day(
    pypi: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == f"{NOTICE}\nTo update, run `uv tool upgrade cayu`.\n"
    assert _run(pypi, ["version"]) == 0
    pypi["now"] = float(pypi["now"]) + DAY - 1
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi).count(NOTICE) == 1
    assert pypi["fetches"] == 1

    pypi["now"] = float(pypi["now"]) + 1
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi).count(NOTICE) == 2
    assert pypi["fetches"] == 2
    assert "available" not in capsys.readouterr().out


def test_notice_never_goes_to_stdout(
    pypi: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(pypi, ["version"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("cayu ")
    assert "available" not in out
    assert NOTICE in _stderr(pypi)


def test_notice_comes_after_the_command_output(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _Terminal()
    monkeypatch.setattr(sys, "stdout", shared)
    pypi["stderr"] = shared
    assert _run(pypi, ["version"]) == 0
    output, notice = shared.getvalue().split(NOTICE)
    assert output.startswith("cayu ")
    assert notice == "\nTo update, run `uv tool upgrade cayu`.\n"


@pytest.mark.parametrize(
    "variable", [(release_check.NO_VERSION_CHECK_VARIABLE, "1"), ("CI", "true"), ("CI", "1")]
)
def test_disabled_and_ci_do_not_ask_pypi(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch, variable: tuple[str, str]
) -> None:
    monkeypatch.setenv(*variable)
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == ""
    assert pypi["fetches"] == 0


def test_ci_false_still_checks(pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "false")
    assert _run(pypi, ["version"]) == 0
    assert NOTICE in _stderr(pypi)


def test_stderr_that_is_not_a_terminal_gets_nothing(
    pypi: dict[str, object], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pypi["stderr"] = io.StringIO()
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == ""
    assert "available" not in capsys.readouterr().out
    assert pypi["fetches"] == 0


@pytest.mark.parametrize("latest", [None, "0.8.1", "0.8.0", "0.15.0rc1"])
def test_offline_or_on_the_latest_release_is_silent_and_retried_tomorrow(
    pypi: dict[str, object], latest: str | None
) -> None:
    pypi["latest"] = latest
    assert _run(pypi, ["version"]) == 0
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == ""
    assert pypi["fetches"] == 1

    pypi["latest"] = "0.14.0"
    pypi["now"] = float(pypi["now"]) + DAY
    assert _run(pypi, ["version"]) == 0
    assert NOTICE in _stderr(pypi)


def test_a_failing_lookup_never_breaks_the_command(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> str | None:
        raise RuntimeError("boom")

    monkeypatch.setattr(release_check, "fetch_latest_release", broken)
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["session", "list", "--json"],
        ["session", "events", "--jsonl"],
        ["session", "list", "--format", "json"],
        ["session", "list", "--format=jsonl"],
        ["cloud", "service", "status", "--json"],
        ["check", "--json=true"],
    ],
)
def test_json_output_is_never_followed_by_a_notice(
    pypi: dict[str, object], arguments: list[str]
) -> None:
    stream = pypi["stderr"]
    assert isinstance(stream, io.StringIO)
    assert release_check.print_newer_release_notice(arguments, stream=stream) is False
    assert _stderr(pypi) == ""
    assert pypi["fetches"] == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["session", "list"],
        ["session", "list", "--format", "table"],
        ["agent", "run", "--", "--json"],
        ["knowledge", "add", "notes.json"],
    ],
)
def test_other_output_is_not_mistaken_for_json(arguments: list[str]) -> None:
    assert release_check.requests_json_output(arguments) is False


def test_json_command_through_main_exits_as_before_without_a_notice(
    pypi: dict[str, object],
) -> None:
    with pytest.raises(SystemExit) as exited:
        _run(pypi, ["version", "--json"])
    assert exited.value.code == 2
    assert NOTICE not in _stderr(pypi)
    assert pypi["fetches"] == 0


def test_cayu_new_prints_only_its_own_notice(
    pypi: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scaffold_cli, "_cayu_source_checkout", lambda: None)
    monkeypatch.setattr(scaffold_cli, "_installed_cayu_version", lambda: "0.8.1")
    assert _run(pypi, ["new", "proj", "--dir", str(tmp_path), "--dry-run"]) == 0
    assert _stderr(pypi).count("0.14.0 is available") == 1
    assert f"cayu new: {NOTICE}" in _stderr(pypi)


@pytest.mark.parametrize(
    ("method", "hint"),
    [
        (
            "project",
            "This project's cayu requirement decides its version. To move it to the new "
            "release, allow it in pyproject.toml if the requirement excludes it, then run "
            "`uv lock --upgrade-package cayu` and `uv sync`.",
        ),
        ("uv-tool", "To update, run `uv tool upgrade cayu`."),
        ("uvx", "To run the latest release, use `uvx cayu@latest`."),
        ("unknown", "To update, upgrade the cayu package the way you installed it."),
    ],
)
def test_each_install_method_gets_its_own_hint(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch, method: str, hint: str
) -> None:
    monkeypatch.setattr(release_check, "install_method", lambda: method)
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == f"{NOTICE}\n{hint}\n"


def test_a_source_checkout_does_not_ask_pypi(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release_check, "install_method", lambda: "source")
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == ""
    assert pypi["fetches"] == 0


def test_another_command_holding_the_claim_wins(pypi: dict[str, object], tmp_path: Path) -> None:
    lock = tmp_path / "cache" / "release-notice.json.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == ""
    assert lock.exists()


def test_a_stale_claim_is_cleared_for_the_next_command(
    pypi: dict[str, object], tmp_path: Path
) -> None:
    lock = tmp_path / "cache" / "release-notice.json.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    os.utime(lock, (0, 0))
    pypi["now"] = 10_000.0
    assert _run(pypi, ["version"]) == 0
    assert not lock.exists()
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi).count(NOTICE) == 1


def test_a_claim_is_won_once_per_day(tmp_path: Path) -> None:
    path = tmp_path / "release-notice.json"
    assert release_check.claim_daily_notice(path, now=100.0, latest="0.14.0") is True
    assert release_check.claim_daily_notice(path, now=100.0 + DAY - 1, latest="0.15.0") is False
    assert release_check.claim_daily_notice(path, now=100.0 + DAY, latest="0.15.0") is True
    path.write_text("not json")
    assert release_check.claim_daily_notice(path, now=200.0, latest="0.15.0") is True
    assert not path.with_name("release-notice.json.lock").exists()


def test_an_unwritable_cache_shows_nothing(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("")
    monkeypatch.setattr(release_check, "release_cache_path", lambda: blocker / "release-check.json")
    assert _run(pypi, ["version"]) == 0
    assert _stderr(pypi) == ""


# Exit codes ---------------------------------------------------------------


def test_a_failing_command_keeps_its_exit_code(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_run_version", lambda _args: 3)
    assert _run(pypi, ["version"]) == 3
    assert NOTICE in _stderr(pypi)


@pytest.mark.parametrize("code", [0, 4, "stopped"])
def test_a_command_exiting_through_system_exit_keeps_its_code(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch, code: int | str
) -> None:
    def exits(_args: object) -> int:
        raise SystemExit(code)

    monkeypatch.setattr(cli, "_run_version", exits)
    with pytest.raises(SystemExit) as exited:
        _run(pypi, ["version"])
    assert exited.value.code == code
    assert NOTICE in _stderr(pypi)


def test_a_usage_error_keeps_exit_code_two(pypi: dict[str, object]) -> None:
    with pytest.raises(SystemExit) as exited:
        _run(pypi, ["no-such-command"])
    assert exited.value.code == 2


def test_a_crash_propagates_without_waiting_for_pypi(
    pypi: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def crashes(_args: object) -> int:
        raise RuntimeError("crash")

    monkeypatch.setattr(cli, "_run_version", crashes)
    with pytest.raises(RuntimeError, match="crash"):
        _run(pypi, ["version"])
    assert pypi["fetches"] == 0


# Install method detection --------------------------------------------------


@pytest.fixture
def site_package(tmp_path: Path) -> Path:
    package = tmp_path / "lib" / "python3.12" / "site-packages" / "cayu"
    package.mkdir(parents=True)
    return package


def _environment(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "pyvenv.cfg").write_text("home = /usr/bin\nuv = 0.9.0\n")
    return path


def test_detects_a_uv_tool_install(tmp_path: Path, site_package: Path) -> None:
    prefix = _environment(tmp_path / "share" / "uv" / "tools" / "cayu")
    (prefix / "uv-receipt.toml").write_text('[tool]\nrequirements = [{ name = "cayu" }]\n')
    assert release_check.install_method(prefix=prefix, package_dir=site_package) == "uv-tool"


def test_a_uv_tool_for_another_package_is_not_cayus(tmp_path: Path, site_package: Path) -> None:
    prefix = _environment(tmp_path / "share" / "uv" / "tools" / "other")
    (prefix / "uv-receipt.toml").write_text('[tool]\nrequirements = [{ name = "other" }]\n')
    assert release_check.install_method(prefix=prefix, package_dir=site_package) == "unknown"


def test_detects_uvx(tmp_path: Path, site_package: Path) -> None:
    prefix = _environment(tmp_path / "cache" / "uv" / "archive-v0" / "Xy3kQ9")
    assert release_check.install_method(prefix=prefix, package_dir=site_package) == "uvx"


@pytest.mark.parametrize(
    "pyproject",
    [
        '[project]\nname = "app"\ndependencies = ["cayu[postgres,server]==0.16.1"]\n',
        '[project]\nname = "app"\ndependencies = ["Cayu>=0.16"]\n',
        '[project]\nname = "app"\n[dependency-groups]\ndev = ["cayu==0.16.1"]\n',
        '[project]\nname = "app"\n[project.optional-dependencies]\nagents = ["cayu"]\n',
        "[tool.uv]\ndev-dependencies = [\"cayu ; python_version >= '3.11'\"]\n",
    ],
)
def test_detects_a_uv_project_dependency(
    tmp_path: Path, site_package: Path, pyproject: str
) -> None:
    project = tmp_path / "app"
    prefix = _environment(project / ".venv")
    (project / "pyproject.toml").write_text(pyproject)
    (project / "uv.lock").write_text("")
    assert release_check.install_method(prefix=prefix, package_dir=site_package) == "project"


@pytest.mark.parametrize(
    ("pyproject", "lock"),
    [
        ('[project]\nname = "app"\ndependencies = ["cayu==0.16.1"]\n', False),
        ('[project]\nname = "app"\ndependencies = ["cayu-extras"]\n', True),
        ("not toml [", True),
    ],
)
def test_other_projects_fall_back_to_the_generic_hint(
    tmp_path: Path, site_package: Path, pyproject: str, lock: bool
) -> None:
    project = tmp_path / "app"
    prefix = _environment(project / ".venv")
    (project / "pyproject.toml").write_text(pyproject)
    if lock:
        (project / "uv.lock").write_text("")
    assert release_check.install_method(prefix=prefix, package_dir=site_package) == "unknown"


def test_a_package_outside_site_packages_is_a_source_checkout(tmp_path: Path) -> None:
    package = tmp_path / "src" / "cayu"
    package.mkdir(parents=True)
    prefix = _environment(tmp_path / ".venv")
    assert release_check.install_method(prefix=prefix, package_dir=package) == "source"


def test_a_claim_reads_the_clock_once_it_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A command whose lookup started first but finished last must not record an
    # earlier time than the command that already showed the notice.
    path = tmp_path / "release-notice.json"
    clock = [100.1]
    monkeypatch.setattr(release_check, "time", SimpleNamespace(time=lambda: clock[0]))
    assert release_check.claim_daily_notice(path, latest="0.14.0") is True
    clock[0] = 100.2
    assert release_check.claim_daily_notice(path, latest="0.14.0") is False
