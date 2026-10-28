"""Public authoring imports stay small; specialized APIs have explicit owners."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

import cayu
from cayu._exports import EXPORTS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROOT_IMPORT_PATTERN = re.compile(r"from cayu import (\(([^)]*)\)|([^\n(]+))", re.DOTALL)
_COMMON_NAMES = {
    "Agent",
    "AgentSpec",
    "CayuApp",
    "CayuConfig",
    "Event",
    "EventType",
    "Message",
    "ResumeRequest",
    "RunOutcome",
    "RunRequest",
    "SessionStatus",
    "StructuredOutputResult",
    "Tool",
    "ToolContext",
    "ToolEffect",
    "ToolResult",
    "ToolSpec",
    "run_to_completion",
    "send_turn",
    "TurnRequest",
}


def test_required_names_are_importable_from_top_level() -> None:
    assert set(EXPORTS) == _COMMON_NAMES | {"__version__"}
    for name, (module, symbol) in EXPORTS.items():
        assert getattr(cayu, name) is getattr(importlib.import_module(module), symbol)


def test_required_names_are_declared_in_dunder_all() -> None:
    assert set(cayu.__all__) == _COMMON_NAMES | {"__version__"}


@pytest.mark.parametrize("package_name", ["cayu", "cayu.runtime"])
def test_removed_imports_have_public_replacements(package_name: str) -> None:
    replacements = json.loads((_REPO_ROOT / "docs/public-api-imports.json").read_text())
    package = importlib.import_module(package_name)
    manifest = importlib.import_module(package_name + "._exports")
    relevant = {
        old: new for old, new in replacements.items() if old.rsplit(".", 1)[0] == package_name
    }
    assert relevant
    for old, new in relevant.items():
        old_name = old.rsplit(".", 1)[1]
        module, symbol = new.rsplit(".", 1)
        assert all(not part.startswith("_") for part in module.split("."))
        assert old_name not in manifest.EXPORTS
        assert old_name not in package.__all__
        with pytest.raises(ImportError, match=re.escape(new.rsplit(".", 1)[0])):
            getattr(package, old_name)
        owner = importlib.import_module(module)
        assert getattr(owner, symbol) is not None, new


def test_runtime_exports_belong_to_runtime() -> None:
    from cayu.runtime._exports import EXPORTS as runtime_exports

    assert runtime_exports
    assert all(module.startswith("cayu.runtime.") for module, _ in runtime_exports.values())


def test_application_store_selection_is_public_from_storage() -> None:
    from cayu import storage
    from cayu.storage import application, targets

    assert storage.ApplicationStores is application.ApplicationStores
    assert storage.open_application_stores is application.open_application_stores
    assert storage.configured_database_url is targets.configured_database_url
    for name in ("ApplicationStores", "open_application_stores", "configured_database_url"):
        assert name in storage.__all__


def test_verified_task_worker_exports_share_the_canonical_owner() -> None:
    from cayu import verification
    from cayu.verification import verified_task_worker

    for name in (
        "VerifiedTaskHandler",
        "VerifiedTaskHandlerReport",
        "VerifiedTaskPreparationContext",
        "VerifiedTaskProposalContext",
        "VerifiedTaskWorker",
        "VerifiedTaskWorkerDraining",
    ):
        assert getattr(verification, name) is getattr(verified_task_worker, name)
        assert name in verification.__all__


def test_process_supervisor_and_event_loop_lag_are_public_parts() -> None:
    from cayu import observability, runtime
    from cayu.observability import event_loop_lag
    from cayu.runtime import _process_workers

    for name in (
        "ProcessWorkerCommand",
        "ProcessWorkerOutcome",
        "process_worker_environment",
        "supervise_process_workers",
        "supervisor_watchdog",
        "watch_supervisor",
    ):
        assert getattr(runtime, name) is getattr(_process_workers, name)
        assert name in runtime.__all__
    for name in ("EventLoopLagMonitor", "EventLoopLagSnapshot"):
        assert getattr(observability, name) is getattr(event_loop_lag, name)
        assert name in observability.__all__


def test_session_message_lifecycle_types_have_matching_public_exports() -> None:
    from cayu.sessions import messaging

    for name in (
        "SessionMessageAccessContext",
        "SessionMessageAccessDenied",
        "SessionMessageAccessPolicy",
        "SessionMessageActionRequest",
        "SessionMessageActionResult",
        "SessionMessageConditions",
        "SessionMessageConflict",
        "SessionMessageCursor",
        "SessionMessageInspection",
        "SessionMessageInspectionRecord",
        "SessionMessageQuery",
        "SessionMessageQueueStatus",
        "SessionMessageSource",
        "SessionMessageTarget",
    ):
        assert getattr(messaging, name).__module__ == "cayu.sessions.messaging"


def test_readme_recovery_snippet_imports_and_constructs() -> None:
    # The exact snippet printed in README.md (worker crash-recovery). It must run
    # verbatim as documented.
    from cayu.sessions.records import SessionStatus
    from cayu.sessions.recovery import IncompleteSessionsRecoveryRequest

    request = IncompleteSessionsRecoveryRequest(statuses={SessionStatus.INTERRUPTING})
    assert SessionStatus.INTERRUPTING in request.statuses


def test_every_documented_root_import_is_exported() -> None:
    # Explicit imports include optional-extra names that are intentionally absent
    # from __all__. Audit the complete declarations, then exercise lazy resolution
    # with the dev dependencies installed.
    documented: dict[str, Path] = {}
    paths = [_REPO_ROOT / "README.md"]
    for root in ("docs", "examples"):
        paths.extend(sorted((_REPO_ROOT / root).rglob("*")))
    paths.extend(sorted((_REPO_ROOT / "src" / "cayu" / "guides").glob("*.md")))
    for path in paths:
        if not path.is_file() or path.suffix not in {".md", ".py"}:
            continue
        source = path.read_text(errors="ignore")
        if path == _REPO_ROOT / "docs" / "release-notes.md":
            # Published notes describe their release's API and must stay intact.
            # Continue auditing any examples in the unreleased section.
            source = re.split(r"(?m)^## v\d+\.\d+\.\d+\b", source, maxsplit=1)[0]
        for match in _ROOT_IMPORT_PATTERN.finditer(source):
            blob = match.group(2) or match.group(3) or ""
            for name in blob.split(","):
                name = name.split("#")[0].strip()
                if name.isidentifier():
                    documented.setdefault(name, path)

    assert documented, "doc scan found no root imports — the scanner is broken"
    exported = set(EXPORTS)
    missing = {
        name: str(path.relative_to(_REPO_ROOT))
        for name, path in sorted(documented.items())
        if name not in exported
    }
    assert not missing, f"documented but not declared as a cayu export: {missing}"
    for name in documented:
        module_name, symbol = EXPORTS[name]
        assert name in dir(cayu), f"cayu.{name} is not discoverable"
        assert getattr(cayu, name) is getattr(importlib.import_module(module_name), symbol), name
