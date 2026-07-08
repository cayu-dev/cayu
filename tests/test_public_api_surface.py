"""Regression guards for the public ``cayu`` import surface.

These names are load-bearing for developer-facing docs and examples: the README
imports them directly, and the deep contracts (custom session/task stores, fake
providers for tests, custom cloud runners/workspaces) all start from these base
ABCs. They were previously reachable only via submodule imports (e.g.
``from cayu.runtime import SessionStatus``) even though the README told readers
to ``from cayu import SessionStatus`` — which raised ``ImportError``. This test
pins them to the top level so that gap cannot silently reopen.
"""

from __future__ import annotations

import re
from pathlib import Path

import cayu

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROOT_IMPORT_PATTERN = re.compile(r"from cayu import (\(([^)]*)\)|([^\n(]+))", re.DOTALL)

# The session vocabulary the README's crash-recovery snippet depends on, plus
# the four runtime base ABCs a builder subclasses to extend the framework.
REQUIRED_TOP_LEVEL_EXPORTS = (
    "Session",
    "SessionStatus",
    "SessionStore",
    "InMemorySessionStore",
    "SessionStatusConflict",
    "ModelProvider",
    "ModelRequest",
    "ModelStreamEvent",
    "NativeStructuredOutputSchemaInvalid",
    "NativeStructuredOutputUnsupported",
    "Runner",
    "Workspace",
)


def test_required_names_are_importable_from_top_level() -> None:
    for name in REQUIRED_TOP_LEVEL_EXPORTS:
        assert hasattr(cayu, name), f"cayu.{name} is not exported from the top level"


def test_required_names_are_declared_in_dunder_all() -> None:
    # A name reachable via attribute access but absent from __all__ is invisible
    # to ``from cayu import *`` and to tooling that reads __all__ — pin both.
    for name in REQUIRED_TOP_LEVEL_EXPORTS:
        assert name in cayu.__all__, f"{name!r} missing from cayu.__all__"


def test_readme_recovery_snippet_imports_and_constructs() -> None:
    # The exact snippet printed in README.md (worker crash-recovery). It must run
    # verbatim as documented.
    from cayu import IncompleteSessionsRecoveryRequest, SessionStatus

    request = IncompleteSessionsRecoveryRequest(statuses={SessionStatus.INTERRUPTING})
    assert SessionStatus.INTERRUPTING in request.statuses


def test_every_documented_root_import_is_exported() -> None:
    # Audit the export surface as a set: every `from cayu import X` a reader can
    # copy out of the README, docs, or examples must resolve against
    # ``cayu.__all__`` — a doc that documents a missing name is the exact
    # papercut external consumers keep reporting one name at a time.
    documented: dict[str, Path] = {}
    paths = [_REPO_ROOT / "README.md"]
    for root in ("docs", "examples"):
        paths.extend(sorted((_REPO_ROOT / root).rglob("*")))
    for path in paths:
        if not path.is_file() or path.suffix not in {".md", ".py"}:
            continue
        for match in _ROOT_IMPORT_PATTERN.finditer(path.read_text(errors="ignore")):
            blob = match.group(2) or match.group(3) or ""
            for name in blob.split(","):
                name = name.split("#")[0].strip()
                if name.isidentifier():
                    documented.setdefault(name, path)

    assert documented, "doc scan found no root imports — the scanner is broken"
    exported = set(cayu.__all__)
    missing = {
        name: str(path.relative_to(_REPO_ROOT))
        for name, path in sorted(documented.items())
        if name not in exported
    }
    assert not missing, f"documented but not in cayu.__all__: {missing}"
