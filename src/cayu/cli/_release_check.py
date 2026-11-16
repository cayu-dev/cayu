"""Best-effort check for a newer Cayu release on PyPI.

The check never blocks or fails the caller: a slow, offline, or malformed answer
reads as "no newer release". PyPI is asked at most once a day; the answer (or
the failure) is cached in the user's cache directory.

:func:`print_newer_release_notice` is the notice every ``cayu`` command prints on
stderr, at most once a day, with an update hint for the way Cayu is installed
(:func:`install_method`, :func:`upgrade_hint`). ``cayu new`` builds its own
notice from :func:`newer_release` and :func:`describe_newer_release`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from cayu.cli._scaffold_test_tools import _package_name

NO_VERSION_CHECK_VARIABLE = "CAYU_NO_VERSION_CHECK"

_PYPI_PROJECT_URL = "https://pypi.org/pypi/cayu/json"
_FETCH_TIMEOUT_S = 2.0
_CACHE_TTL_S = 24 * 60 * 60
_CACHE_FILENAME = "release-check.json"
_NOTICE_FILENAME = "release-notice.json"
# A claim lock older than this was left by a process that died while holding it.
_NOTICE_LOCK_STALE_S = 60.0
_JSON_FORMATS = frozenset({"json", "jsonl"})
_UVX_ENVIRONMENT = re.compile(r"archive-v\d+")

InstallMethod = Literal["project", "uv-tool", "uvx", "source", "unknown"]
_FINAL_RELEASE = re.compile(r"\d+(?:\.\d+)*")
_INSTALLED_RELEASE = re.compile(r"v?(\d+(?:\.\d+)*)(.*)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class NewerRelease:
    installed: str
    latest: str


def version_check_disabled(environ: Mapping[str, str] = os.environ) -> bool:
    return environ.get(NO_VERSION_CHECK_VARIABLE, "").strip() not in {"", "0"}


def fetch_latest_release() -> str | None:
    """Return within the total lookup budget, including response-body reads.

    The daemon worker is never joined: DNS or transport cleanup must not hold
    up scaffolding after the deadline. It stops between chunks once the caller
    leaves, and HTTPX also bounds individual network operations.
    """

    finished = threading.Event()
    abandoned = threading.Event()
    result: list[str] = []

    def fetch() -> None:
        try:
            import httpx

            with httpx.stream(
                "GET",
                _PYPI_PROJECT_URL,
                timeout=_FETCH_TIMEOUT_S,
                headers={"Accept": "application/json"},
            ) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    if abandoned.is_set():
                        return
                    body.extend(chunk)
                if abandoned.is_set():
                    return
                latest = json.loads(body)["info"]["version"]
                if isinstance(latest, str):
                    result.append(latest)
        except Exception:
            pass
        finally:
            finished.set()

    threading.Thread(target=fetch, name="cayu-release-check", daemon=True).start()
    if not finished.wait(_FETCH_TIMEOUT_S):
        abandoned.set()
        return None
    return result[0] if result else None


def release_cache_path() -> Path | None:
    try:
        if sys.platform == "darwin":
            root = Path.home() / "Library" / "Caches"
        elif sys.platform == "win32":
            local = os.environ.get("LOCALAPPDATA")
            root = Path(local) if local else Path.home() / "AppData" / "Local"
        else:
            configured = os.environ.get("XDG_CACHE_HOME")
            root = (
                Path(configured)
                if configured and os.path.isabs(configured)
                else Path.home() / ".cache"
            )
    except RuntimeError:
        return None
    return root / "cayu" / _CACHE_FILENAME


def _release_key(release: str) -> tuple[int, ...]:
    parts = [int(part) for part in release.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def is_newer_release(latest: str, installed: str) -> bool:
    """Whether final release ``latest`` is ahead of ``installed``.

    A pre-release or dev build is never ``latest``. An installed pre-release or
    dev build is behind the final release it leads up to.
    """

    latest = latest.strip()
    match = _INSTALLED_RELEASE.fullmatch(installed.strip())
    if not _FINAL_RELEASE.fullmatch(latest) or match is None:
        return False
    latest_key, installed_key = _release_key(latest), _release_key(match[1])
    if latest_key != installed_key:
        return latest_key > installed_key
    suffix = match[2].lstrip(".-_").lower()
    return bool(suffix) and not suffix.startswith(("post", "+"))


def _read_cache(path: Path, *, now: float) -> tuple[bool, str | None]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False, None
    if not isinstance(document, dict):
        return False, None
    checked_at = document.get("checked_at")
    latest = document.get("latest")
    if not isinstance(checked_at, int | float) or isinstance(checked_at, bool):
        return False, None
    if not 0 <= now - checked_at < _CACHE_TTL_S:
        return False, None
    if latest is not None and not isinstance(latest, str):
        return False, None
    return True, latest


def _write_json(path: Path, document: Mapping[str, object]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError:
        return False
    return True


def _write_cache(path: Path, *, now: float, latest: str | None) -> None:
    _write_json(path, {"checked_at": now, "latest": latest})


def latest_release(
    *,
    fetch: Callable[[], str | None] | None = None,
    cache_path: Path | None = None,
    now: float | None = None,
) -> str | None:
    """Return the latest Cayu release, from a day-old cache or one PyPI request."""

    current = time.time() if now is None else now
    path = release_cache_path() if cache_path is None else cache_path
    if path is not None:
        cached, latest = _read_cache(path, now=current)
        if cached:
            return latest
    latest = fetch_latest_release() if fetch is None else fetch()
    if path is not None:
        _write_cache(path, now=current, latest=latest)
    return latest


def newer_release(
    installed: str,
    *,
    fetch: Callable[[], str | None] | None = None,
    cache_path: Path | None = None,
    now: float | None = None,
    environ: Mapping[str, str] = os.environ,
) -> NewerRelease | None:
    """Return the newer release to recommend, or None when ``installed`` is current."""

    if version_check_disabled(environ):
        return None
    latest = latest_release(fetch=fetch, cache_path=cache_path, now=now)
    if latest is None or not is_newer_release(latest, installed):
        return None
    return NewerRelease(installed=installed, latest=latest.strip())


def installed_release() -> str:
    from cayu._version import package_version

    return package_version()


def describe_newer_release(newer: NewerRelease) -> str:
    return f"cayu {newer.latest} is available; you're running {newer.installed}."


def _requirement_names(value: object) -> set[str]:
    entries = value if isinstance(value, list) else []
    return {name for entry in entries if isinstance(entry, str) and (name := _package_name(entry))}


def _project_depends_on_cayu(root: Path) -> bool:
    """Whether ``root`` is a uv project that lists ``cayu`` among its requirements."""

    if not (root / "uv.lock").is_file():
        return False
    try:
        document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    project = document.get("project")
    project = project if isinstance(project, dict) else {}
    groups: list[object] = [project.get("dependencies")]
    for table in (project.get("optional-dependencies"), document.get("dependency-groups")):
        if isinstance(table, dict):
            groups.extend(table.values())
    tool = document.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    if isinstance(uv, dict):
        groups.append(uv.get("dev-dependencies"))
    return any("cayu" in _requirement_names(group) for group in groups)


def _uv_tool_receipt_names_cayu(prefix: Path) -> bool:
    try:
        receipt = tomllib.loads((prefix / "uv-receipt.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    tool = receipt.get("tool")
    requirements = tool.get("requirements") if isinstance(tool, dict) else None
    if not isinstance(requirements, list):
        return False
    return any(
        isinstance(requirement, dict) and requirement.get("name") == "cayu"
        for requirement in requirements
    )


def install_method(*, prefix: Path | None = None, package_dir: Path | None = None) -> InstallMethod:
    """Say how the running Cayu was installed, from where its environment lives.

    - ``source``: the package is not under site-packages (an editable or
      ``PYTHONPATH`` checkout), so there is no release to update to.
    - ``uv-tool``: the environment holds a uv tool receipt that names ``cayu``.
    - ``uvx``: the environment is one of uv's cached ``archive-v<N>`` environments,
      which is where ``uvx`` runs tools.
    - ``project``: the environment sits inside a uv project (``pyproject.toml``
      and ``uv.lock``) that requires ``cayu``, such as its ``.venv``.
    - ``unknown``: anything else, such as pip or pipx.
    """

    try:
        package = (
            Path(__file__).resolve().parents[1] if package_dir is None else package_dir
        ).resolve()
        if not any(part in {"site-packages", "dist-packages"} for part in package.parts):
            return "source"
        environment = Path(sys.prefix if prefix is None else prefix).resolve()
        if _uv_tool_receipt_names_cayu(environment):
            return "uv-tool"
        if (
            _UVX_ENVIRONMENT.fullmatch(environment.parent.name)
            and (environment / "pyvenv.cfg").is_file()
        ):
            return "uvx"
        for root in environment.parents:
            if _project_depends_on_cayu(root):
                return "project"
    except (OSError, RuntimeError, ValueError):
        pass
    return "unknown"


def upgrade_hint(method: InstallMethod) -> str:
    if method == "project":
        return (
            "This project's cayu requirement decides its version. To move it to the new "
            "release, allow it in pyproject.toml if the requirement excludes it, then run "
            "`uv lock --upgrade-package cayu` and `uv sync`."
        )
    if method == "uv-tool":
        return "To update, run `uv tool upgrade cayu`."
    if method == "uvx":
        return "To run the latest release, use `uvx cayu@latest`."
    return "To update, upgrade the cayu package the way you installed it."


def requests_json_output(arguments: Sequence[str]) -> bool:
    """Whether a command line asks for JSON output (``--json``, ``--format json``...)."""

    previous = None
    for argument in arguments:
        if argument == "--":
            break
        name, has_value, value = argument.partition("=")
        if name in {"--json", "--jsonl"}:
            return True
        if name == "--format" and has_value and value in _JSON_FORMATS:
            return True
        if previous == "--format" and argument in _JSON_FORMATS:
            return True
        previous = argument
    return False


def in_ci(environ: Mapping[str, str] = os.environ) -> bool:
    return environ.get("CI", "").strip().lower() not in {"", "0", "false"}


def _is_terminal(stream: TextIO | None) -> bool:
    try:
        return stream is not None and stream.isatty()
    except (OSError, ValueError):
        return False


def notice_cache_path() -> Path | None:
    path = release_cache_path()
    return None if path is None else path.with_name(_NOTICE_FILENAME)


def _notified_since(path: Path, *, now: float) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    notified_at = document.get("notified_at") if isinstance(document, dict) else None
    if not isinstance(notified_at, int | float) or isinstance(notified_at, bool):
        return False
    return 0 <= now - notified_at < _CACHE_TTL_S


def claim_daily_notice(path: Path, *, latest: str, now: float | None = None) -> bool:
    """Record that a notice is shown now, unless one was shown in the last day.

    Only one process wins a claim: concurrent commands take an exclusive lock
    file, and a command that can't take it, read the record or write it shows
    nothing. The claim's time is read once the lock is held, so a command whose
    lookup finished later can't record an earlier time than the last winner.
    """

    lock = path.with_name(f"{path.name}.lock")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            if time.time() - lock.stat().st_mtime > _NOTICE_LOCK_STALE_S:
                lock.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    except OSError:
        return False
    try:
        os.close(descriptor)
        claimed_at = time.time() if now is None else now
        if _notified_since(path, now=claimed_at):
            return False
        return _write_json(path, {"notified_at": claimed_at, "latest": latest})
    finally:
        with contextlib.suppress(OSError):
            lock.unlink(missing_ok=True)


def print_newer_release_notice(
    arguments: Sequence[str],
    *,
    stream: TextIO | None = None,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Tell a person at a terminal, once a day, that a newer Cayu release exists.

    Prints on ``stream`` (stderr by default), never stdout. Silent when the check
    is turned off, in CI, when the stream is not a terminal, when the command
    asks for JSON output, from a source checkout, offline, or on the latest
    release. Never raises; returns whether it printed.
    """

    try:
        target = sys.stderr if stream is None else stream
        if version_check_disabled(environ) or in_ci(environ) or not _is_terminal(target):
            return False
        if requests_json_output(arguments):
            return False
        found = install_method()
        if found == "source":
            return False
        path = notice_cache_path()
        if path is None or _notified_since(path, now=time.time()):
            return False
        newer = newer_release(installed_release(), environ=environ)
        if newer is None or not claim_daily_notice(path, latest=newer.latest):
            return False
        if sys.stdout is not None:
            sys.stdout.flush()
        print(f"{describe_newer_release(newer)}\n{upgrade_hint(found)}", file=target)
        target.flush()
        return True
    except (Exception, KeyboardInterrupt):
        return False
