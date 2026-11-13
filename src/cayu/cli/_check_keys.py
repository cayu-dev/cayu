"""`cayu check` findings about keys a local run or a fresh checkout doesn't have.

A project made with automatic memory reads ``CAYU_MEMORY_EVIDENCE_KEY`` or its
git-ignored ``data/memory-evidence.key``; a CI checkout has neither, and the
check only describes the app, so it lends the factory a throwaway key. A
``ToolArgumentReview`` needs the public authority alias keyring, which a local
run usually lacks, so the check says what the approver sees without it.
"""

from __future__ import annotations

import ast
import os
import secrets
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
)

MEMORY_EVIDENCE_KEY_ENV = "CAYU_MEMORY_EVIDENCE_KEY"
_LOCAL_MEMORY_KEY = Path("data") / "memory-evidence.key"
_SETTINGS = Path("configuration") / "settings.py"
_SKIPPED_DIRECTORIES = frozenset(
    {"venv", "node_modules", "data", "tests", "web", "__pycache__", "build"}
)
_MAX_SCANNED_FILES = 2000


@contextmanager
def lent_memory_evidence_key(root: Path) -> Iterator[bool]:
    """Set a throwaway memory evidence key while the factory runs, when there is none.

    Only for a project whose generated settings read ``CAYU_MEMORY_EVIDENCE_KEY``.

    Yields whether it lent one. Describing the app derives no evidence and
    stores nothing under the key, so the throwaway value leaves no trace.
    """

    if (
        MEMORY_EVIDENCE_KEY_ENV in os.environ
        or (root / _LOCAL_MEMORY_KEY).is_file()
        or not _reads_memory_evidence_key(root)
    ):
        yield False
        return
    os.environ[MEMORY_EVIDENCE_KEY_ENV] = secrets.token_urlsafe(48)
    try:
        yield True
    finally:
        os.environ.pop(MEMORY_EVIDENCE_KEY_ENV, None)


def _reads_memory_evidence_key(root: Path) -> bool:
    """Whether the project has automatic memory and reads the key as `cayu new` generates."""

    try:
        source = (root / _SETTINGS).read_text(encoding="utf-8")
        document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    scaffold = document.get("tool", {}).get("cayu", {}).get("scaffold", {})
    capabilities = scaffold.get("capabilities") if isinstance(scaffold, dict) else None
    return (
        MEMORY_EVIDENCE_KEY_ENV in source
        and isinstance(capabilities, list)
        and "memory" in capabilities
    )


def lent_memory_key_diagnostic(*, deploy: bool) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        code="MEMORY_EVIDENCE_KEY_UNSET",
        severity=DiagnosticSeverity.INFO,
        subject="configuration",
        path=_LOCAL_MEMORY_KEY.as_posix(),
        message=(
            f"{MEMORY_EVIDENCE_KEY_ENV} is unset and {_LOCAL_MEMORY_KEY.as_posix()} doesn't "
            "exist, so the check built the application with a throwaway memory evidence "
            "key. Tests that build the app, and the app itself, still need one."
        ),
        hint=(
            (
                "A deployment needs its own stable key: Cayu Cloud sets "
                f"{MEMORY_EVIDENCE_KEY_ENV} for a deployed Agent; elsewhere, store one as a "
                "secret. "
                if deploy
                else ""
            )
            + f"In CI, set {MEMORY_EVIDENCE_KEY_ENV} to a throwaway value of at least 32 bytes "
            "before the tests, for example the output of `python -c 'import secrets; "
            "print(secrets.token_urlsafe(48))'`. Locally, `cayu new` created "
            f"{_LOCAL_MEMORY_KEY.as_posix()}; it is git-ignored, so a fresh checkout has none."
        ),
        tags=("configuration", "deploy") if deploy else ("configuration",),
        parameters={"deploy": deploy},
        documentation_anchor="cayu guide diagnostics#memory-evidence-key-unset",
        verification_command="cayu check --json",
    )


def uses_tool_argument_review(root: Path) -> str | None:
    """The first project module that imports ToolArgumentReview, as a relative path."""

    scanned = 0
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(
            part in _SKIPPED_DIRECTORIES or part.startswith(".") for part in relative.parts[:-1]
        ):
            continue
        scanned += 1
        if scanned > _MAX_SCANNED_FILES:
            return None
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if "ToolArgumentReview" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] == "cayu"
                and any(alias.name == "ToolArgumentReview" for alias in node.names)
            ):
                return relative.as_posix()
    return None


def review_without_keyring_diagnostic(module: str) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        code="HUMAN_REVIEW_KEYRING_MISSING",
        severity=DiagnosticSeverity.INFO,
        subject="human_review",
        path=module,
        message=(
            f"{module} uses ToolArgumentReview, but this run has no public authority alias "
            "keyring and the loaded app has no human review policy. Without one, an approver "
            "sees an approval's title, tool name and question but none of its arguments "
            "(the inbox reports `arguments_unavailable_reason: no_review_policy`) and decides "
            "without a review reference. A deployed Agent gets the keyring from Cayu Cloud and "
            "shows the review."
        ),
        hint=(
            "To see the review in a local run, run `cayu keys generate --env-file .env` once "
            "and start the app with `uv run --env-file .env ...`, so "
            f"{PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV} and {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} "
            "are set. A database opened with a keyring keeps requiring it, so use the same "
            "keys from then on."
        ),
        tags=("configuration",),
        parameters={"module": module},
        documentation_anchor="cayu guide diagnostics#human-review-keyring-missing",
        verification_command="cayu check --json",
    )


def review_without_purpose_diagnostic() -> ProjectDiagnostic:
    return ProjectDiagnostic(
        code="HUMAN_REVIEW_PURPOSE_MISSING",
        severity=DiagnosticSeverity.WARNING,
        subject="human_review",
        path="human_review_policy",
        message=(
            "The app's human review policy names no dashboard_purpose. Settling an approval "
            "or a question needs a protected review's reference, and an app requests the "
            "review with the purpose /api/app reports, which is null here. So an Agent's own "
            "pages can only show those items read-only, and people decide in the Developer "
            "dashboard, which needs `humanReviewPurpose` in its runtime config."
        ),
        hint=(
            "Give the policy a `dashboard_purpose` property that returns the purpose its "
            "authorize() accepts, as ToolArgumentReview does."
        ),
        tags=("configuration",),
        documentation_anchor="cayu guide diagnostics#human-review-purpose-missing",
        verification_command="cayu check --json",
    )
