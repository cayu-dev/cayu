"""Deterministic executable-specific selector boundary for benchmark evidence."""

from __future__ import annotations

import argparse
import errno
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from cayu.guides.command_selectors import pytest_selector

_CHECK_PROGRAM = Path(__file__).with_name("fixture_check_program.py")
_DECLARED_EFFECT = "none"


def _cannot_run_reason(error: OSError) -> str:
    if error.errno == errno.ENOENT:
        return "not_found"
    if error.errno in {errno.EACCES, errno.EPERM}:
        return "permission_denied"
    if error.errno == errno.ENOEXEC:
        return "invalid_executable_format"
    return "os_error"


_EntryState = tuple[str, int, bytes | str | None]


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _entry_state(path: Path) -> _EntryState | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    permissions = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        return ("symlink", permissions, os.readlink(path))
    if stat.S_ISREG(metadata.st_mode):
        return ("file", permissions, path.read_bytes())
    if stat.S_ISDIR(metadata.st_mode):
        return ("directory", permissions, None)
    return ("other", permissions, None)


def _entry_tree(root: Path):
    yield root
    state = _entry_state(root)
    if state is None or state[0] != "directory":
        return
    with os.scandir(root) as iterator:
        children = sorted(iterator, key=lambda child: child.name)
    for child in children:
        path = root / child.name
        yield from _entry_tree(path)


def _snapshot(
    workspace: Path,
    protected_paths: tuple[Path, ...],
) -> dict[Path, _EntryState | None]:
    paths: dict[Path, _EntryState | None] = {}
    for root in (workspace, *protected_paths):
        for path in _entry_tree(root):
            lexical_path = _lexical_absolute(path)
            paths[lexical_path] = _entry_state(lexical_path)
    return paths


def _observed_writes(
    before: dict[Path, _EntryState | None],
    workspace: Path,
    protected_paths: tuple[Path, ...],
) -> list[str]:
    after = _snapshot(workspace, protected_paths)
    paths = set(before) | set(after)
    return sorted(str(path) for path in paths if before.get(path) != after.get(path))


def run_check(
    *,
    workspace: Path,
    selectors: Iterable[str],
    executable: Path,
    timeout: float,
    protected_paths: tuple[Path, ...],
) -> dict[str, object]:
    workspace = workspace.resolve(strict=True)
    protected_paths = tuple(_lexical_absolute(path) for path in protected_paths)
    before = _snapshot(workspace, protected_paths)
    requested_selectors = tuple(selectors)
    selection_scope = "selected" if requested_selectors else "full"

    try:
        safe_selectors = tuple(
            pytest_selector(selector, workspace=workspace) for selector in requested_selectors
        )
    except ValueError:
        writes = _observed_writes(before, workspace, protected_paths)
        return {
            "cannot_run_errno": None,
            "cannot_run_reason": None,
            "declared_effect": _DECLARED_EFFECT,
            "effect_matches_observed_writes": not writes,
            "exit_code": None,
            "observed_writes": writes,
            "process_started": False,
            "selection_scope": selection_scope,
            "status": "rejected",
            "tests_executed": 0,
            "validated_selectors": [],
        }

    process_started = False
    cannot_run_errno: int | None = None
    cannot_run_reason: str | None = None
    exit_code: int | None = None
    tests_executed = 0
    try:
        process_started = True
        completed = subprocess.run(
            [str(executable), "-B", str(_CHECK_PROGRAM), "--", *safe_selectors],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except OSError as error:
        process_started = False
        status = "unavailable"
        cannot_run_errno = error.errno
        cannot_run_reason = _cannot_run_reason(error)
    except subprocess.TimeoutExpired:
        status = "timed_out"
    else:
        exit_code = completed.returncode
        try:
            payload = json.loads(completed.stdout)
            tests_executed = int(payload["tests_executed"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            tests_executed = 0
        if exit_code != 0:
            status = "failed"
        elif tests_executed == 0:
            status = "zero_tests_executed"
        else:
            status = "verified"

    writes = _observed_writes(before, workspace, protected_paths)
    effect_matches = not writes
    return {
        "cannot_run_errno": cannot_run_errno,
        "cannot_run_reason": cannot_run_reason,
        "declared_effect": _DECLARED_EFFECT,
        "effect_matches_observed_writes": effect_matches,
        "exit_code": exit_code,
        "observed_writes": writes,
        "process_started": process_started,
        "selection_scope": selection_scope,
        "status": status,
        "tests_executed": tests_executed,
        "validated_selectors": list(safe_selectors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--selector", action="append", default=[])
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--check-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--protected-path", action="append", default=[], type=Path)
    args = parser.parse_args()
    result = run_check(
        workspace=args.workspace,
        selectors=tuple(args.selector),
        executable=args.check_executable,
        timeout=args.timeout,
        protected_paths=tuple(args.protected_path),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
