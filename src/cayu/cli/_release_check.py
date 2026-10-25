"""Best-effort check for a newer Cayu release on PyPI.

The check never blocks or fails the caller: a slow, offline, or malformed answer
reads as "no newer release". PyPI is asked at most once a day; the answer (or
the failure) is cached in the user's cache directory.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

NO_VERSION_CHECK_VARIABLE = "CAYU_NO_VERSION_CHECK"

_PYPI_PROJECT_URL = "https://pypi.org/pypi/cayu/json"
_FETCH_TIMEOUT_S = 2.0
_CACHE_TTL_S = 24 * 60 * 60
_CACHE_FILENAME = "release-check.json"
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


def _write_cache(path: Path, *, now: float, latest: str | None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"checked_at": now, "latest": latest}, handle)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError:
        pass


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
