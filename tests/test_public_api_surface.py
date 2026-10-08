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

import importlib
import re
from pathlib import Path

import cayu
import cayu.runtime as cayu_runtime
from cayu._exports import EXPORTS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROOT_IMPORT_PATTERN = re.compile(r"from cayu import (\(([^)]*)\)|([^\n(]+))", re.DOTALL)

# The session vocabulary the README's crash-recovery snippet depends on, plus
# the runtime and workspace capability ABCs builders subclass to extend Cayu.
REQUIRED_TOP_LEVEL_EXPORTS = (
    "AgentAuthoringState",
    "BoundedTarReader",
    "Session",
    "SessionStatus",
    "SessionStore",
    "InvocationOriginClaim",
    "SessionInvocation",
    "session_invocation_for_run_request",
    "TarWriter",
    "InMemorySessionStore",
    "SessionStatusConflict",
    "ModelProvider",
    "ModelRequest",
    "ModelStreamEvent",
    "OpenAISubscriptionProvider",
    "NativeStructuredOutputSchemaInvalid",
    "NativeStructuredOutputUnsupported",
    "ProviderOperationStartRecoveryRequest",
    "Runner",
    "Workspace",
    "DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS",
    "MicrosandboxCleanupError",
    "WorkspaceCheckpointError",
    "WorkspaceCheckpointManifest",
    "WorkspaceCheckpointPolicy",
    "capture_workspace_checkpoint",
    "load_workspace_checkpoint",
    "pin_workspace_checkpoint",
    "release_workspace_checkpoint",
    "restore_workspace_checkpoint",
    "ExecutionDeadline",
    "ExecutionDeadlineExceeded",
    "current_execution_deadline",
    "execution_deadline_scope",
)

MANIFEST_TOP_LEVEL_EXPORTS = (
    "AppManifest",
    "DiagnosticSeverity",
    "ProjectCheckReport",
    "ProjectDiagnostic",
    "PublicServiceManifest",
    "RuntimeStoreDurability",
    "ServiceCheckEvidence",
    "check_manifest",
)

MANIFEST_RUNTIME_ONLY_EXPORTS = (
    "APP_MANIFEST_SCHEMA_VERSION",
    "AVAILABLE_CHECK_TAGS",
    "BUILTIN_DIAGNOSTIC_CODES",
    "CHECK_REPORT_SCHEMA_VERSION",
    "AgentManifest",
    "ApplicationDefaultsManifest",
    "CapabilityManifest",
    "EnvironmentManifest",
    "ProviderManifest",
    "RegistrationProvenance",
    "RequestFootprintConfigManifest",
    "RuntimeManifest",
    "StoreManifest",
    "ToolManifest",
    "ToolResultProjectionPolicyManifest",
)


def test_required_names_are_importable_from_top_level() -> None:
    for name in REQUIRED_TOP_LEVEL_EXPORTS:
        assert hasattr(cayu, name), f"cayu.{name} is not exported from the top level"


def test_application_store_selection_is_public_at_the_root() -> None:
    from cayu.storage import application, targets

    assert cayu.ApplicationStores is application.ApplicationStores
    assert cayu.open_application_stores is application.open_application_stores
    assert cayu.configured_database_url is targets.configured_database_url
    for name in ("ApplicationStores", "open_application_stores", "configured_database_url"):
        assert name in cayu.__all__


def test_execution_deadline_exports_remain_discoverable_from_root_and_runtime() -> None:
    from cayu import deadlines

    for name in (
        "ExecutionDeadline",
        "ExecutionDeadlineExceeded",
        "current_execution_deadline",
        "execution_deadline_scope",
    ):
        for package in (cayu, cayu_runtime):
            assert name in package.__all__
            assert getattr(package, name) is getattr(deadlines, name)


def test_verified_task_worker_exports_share_the_runtime_owner() -> None:
    from cayu.runtime import verified_task_worker

    for name in (
        "VerifiedTaskHandler",
        "VerifiedTaskHandlerReport",
        "VerifiedTaskPreparationContext",
        "VerifiedTaskProposalContext",
        "VerifiedTaskWorker",
        "VerifiedTaskWorkerDraining",
    ):
        assert getattr(cayu, name) is getattr(cayu_runtime, name)
        assert getattr(cayu, name) is getattr(verified_task_worker, name)
        assert name in cayu.__all__ and name in cayu_runtime.__all__


def test_required_names_are_declared_in_dunder_all() -> None:
    # A name reachable via attribute access but absent from __all__ is invisible
    # to ``from cayu import *`` and to tooling that reads __all__ — pin both.
    for name in REQUIRED_TOP_LEVEL_EXPORTS:
        assert name in cayu.__all__, f"{name!r} missing from cayu.__all__"


def test_session_message_lifecycle_types_have_matching_public_exports() -> None:
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
        assert name in cayu.__all__
        assert name in cayu_runtime.__all__
        assert getattr(cayu, name) is getattr(cayu_runtime, name)


def test_manifest_api_keeps_structural_types_out_of_the_root_namespace() -> None:
    for name in MANIFEST_TOP_LEVEL_EXPORTS:
        assert hasattr(cayu, name), f"cayu.{name} is a supported entry point"
        assert name in cayu.__all__
    for name in MANIFEST_RUNTIME_ONLY_EXPORTS:
        assert hasattr(cayu_runtime, name), f"cayu.runtime.{name} must remain public"
        assert not hasattr(cayu, name), f"cayu.{name} unnecessarily expands the root API"
        assert name not in cayu.__all__


def test_readme_recovery_snippet_imports_and_constructs() -> None:
    # The exact snippet printed in README.md (worker crash-recovery). It must run
    # verbatim as documented.
    from cayu.sessions.base import IncompleteSessionsRecoveryRequest
    from cayu.sessions.records import SessionStatus

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
        for match in _ROOT_IMPORT_PATTERN.finditer(path.read_text(errors="ignore")):
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
