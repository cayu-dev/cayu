from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path
from typing import Any


def write_baseline(workspace: Path, baseline: dict[str, str]) -> None:
    for relative, content in baseline.items():
        (workspace / relative).write_text(content, encoding="utf-8")


def apply_candidate(
    workspace: Path,
    candidate: dict[str, Any],
    baseline: dict[str, str],
) -> None:
    changes = candidate.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("Candidate did not produce workspace changes.")
    seen: set[str] = set()
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("Candidate file change was not an object.")
        relative = change.get("path")
        content = change.get("content")
        if (
            not isinstance(relative, str)
            or relative not in baseline
            or relative in seen
            or not isinstance(content, str)
        ):
            raise ValueError(f"Candidate produced an unsafe file change: {change!r}")
        seen.add(relative)
        (workspace / relative).write_text(content, encoding="utf-8")


def run_candidate_gates(workspace: Path, baseline: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    changed_files: list[str] = []
    diff_lines = 0
    for relative, original in baseline.items():
        actual = (workspace / relative).read_text(encoding="utf-8")
        if actual == original:
            continue
        changed_files.append(relative)
        diff_lines += sum(
            1
            for line in difflib.unified_diff(
                original.splitlines(),
                actual.splitlines(),
                lineterm="",
            )
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )
    return {
        "tests_passed": completed.returncode == 0,
        "test_output": (completed.stdout + completed.stderr)[-500:],
        "changed_files": changed_files,
        "test_files_changed": any(path.startswith("test_") for path in changed_files),
        "diff_lines": diff_lines,
    }
