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

import cayu

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
