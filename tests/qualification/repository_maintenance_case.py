"""Fixed behavioral corpus for the repository-maintenance application journey.

This module is qualification-owned, not copied into the agent-writable repository.
It neither executes candidate code nor authorizes delivery. The application must
obtain responses in its admitted Docker environment and bind them to the final
revision before using this behavioral result.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import cast

CASE_ID = "closed-integer-range-v1"
SEED_BASE_REVISION = "219ebbfb37f09b0a08e9ca98ad4474bf4640c06f"
ALLOWED_CHANGE_PATHS = ("range_ops.py", "tests/test_range_ops.py")

SEED_FILES = MappingProxyType(
    {
        "range_ops.py": '''"""Small integer range utilities."""


def in_closed_range(value: int, lower: int, upper: int) -> bool:
    """Return whether value belongs to the inclusive range [lower, upper]."""
    return lower <= value < upper
''',
        "tests/test_range_ops.py": """from range_ops import in_closed_range


def test_upper_endpoint_is_included():
    assert in_closed_range(2, -2, 2) is True


def test_outside_range_is_excluded():
    assert in_closed_range(3, -2, 2) is False
""",
        "pyproject.toml": """[project]
name = "closed-integer-range-fixture"
version = "0.0.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
""",
    }
)

# Exhaust the declared finite domain, including degenerate ranges and both sides
# of every endpoint. Ordering is part of the corpus identity and response contract.
PROBES = tuple(
    (value, lower, upper)
    for lower in range(-2, 3)
    for upper in range(lower, 3)
    for value in range(-3, 4)
)
EXPECTED_RESPONSES = tuple(lower <= value <= upper for value, lower, upper in PROBES)


class BehavioralOutcome(StrEnum):
    PASSED = "passed"
    INCORRECT = "incorrect"
    INVALID_RESPONSE = "invalid_response"


def check_behavioral_responses(responses: object) -> BehavioralOutcome:
    """Compare complete typed responses, never candidate-authored test status."""

    if type(responses) not in (list, tuple):
        return BehavioralOutcome.INVALID_RESPONSE
    values = cast("list[object] | tuple[object, ...]", responses)
    if len(values) != len(PROBES):
        return BehavioralOutcome.INVALID_RESPONSE
    snapshot = tuple(values)
    if len(snapshot) != len(PROBES) or any(type(response) is not bool for response in snapshot):
        return BehavioralOutcome.INVALID_RESPONSE
    if snapshot != EXPECTED_RESPONSES:
        return BehavioralOutcome.INCORRECT
    return BehavioralOutcome.PASSED


def corpus_fingerprint() -> str:
    """Pin seeded bytes, permitted edits, ordered inputs and independent answers."""

    material = {
        "case_id": CASE_ID,
        "seed_files": dict(SEED_FILES),
        "allowed_change_paths": ALLOWED_CHANGE_PATHS,
        "probes": PROBES,
        "expected_responses": EXPECTED_RESPONSES,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def materialize_seed_repository(destination: Path) -> str:
    """Create one new disposable fixture repository; return its exact base commit.

    Existing directories are never reused. This is a qualification fixture,
    not a product checkout, clone, publication, or candidate-code runner.
    """

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("The repository-maintenance fixture requires Git.")
    destination.mkdir()
    for relative, content in SEED_FILES.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
    environment = {
        "PATH": os.defpath,
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "Cayu Qualification",
        "GIT_AUTHOR_EMAIL": "qualification@example.invalid",
        "GIT_COMMITTER_NAME": "Cayu Qualification",
        "GIT_COMMITTER_EMAIL": "qualification@example.invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
    }

    def command(*arguments: str) -> bytes:
        return subprocess.run(
            [git, "-c", f"core.hooksPath={os.devnull}", "-C", str(destination), *arguments],
            env=environment,
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout

    command("init", "--object-format=sha1", "--initial-branch=main")
    command("add", "--", *sorted(SEED_FILES))
    command("-c", "commit.gpgsign=false", "commit", "-m", CASE_ID)
    return command("rev-parse", "HEAD").decode("ascii").strip()
