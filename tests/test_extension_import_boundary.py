"""Readiness gate for extracting E2B and Microsandbox into optional packages (#1879).

An independently installed adapter package cannot import private Cayu modules
or names. This test parses the in-tree E2B and Microsandbox implementations and
fails if they import any private Cayu module or name outside ``_ALLOWED``.

``_ALLOWED`` is the remaining extraction debt. It must only shrink: when an
import moves to a public seam (for example ``cayu.extensions.runners`` or
``cayu.extensions.egress``), delete its entry; stale entries fail the test so
the list always reflects reality. Adding an entry needs a reason in review,
because it is new private coupling the extraction will have to undo.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

_ALLOWED: dict[str, frozenset[str]] = {
    "src/cayu/runners/e2b.py": frozenset(
        {
            "cayu._exception_groups.add_exception_note_safely",
            "cayu._exception_groups.exception_tree_contains",
            "cayu._task_wait.await_shielded_task_outcome",
            "cayu._task_wait.capture_awaitable_outcome",
            "cayu._task_wait.restore_task_cancellation_requests",
            "cayu._validation.require_clean_nonblank",
            "cayu._validation.require_durable_clean_nonblank",
            "cayu._validation.require_positive_finite_seconds",
            "cayu.runners._creation_cleanup.CreationCleanupProgress",
            "cayu.runners._creation_cleanup.drain_creation_cleanups",
            "cayu.runners._creation_cleanup.require_creation_cleanup_settled",
            "cayu.runners._creation_cleanup.retain_creation_cleanup",
            "cayu.runners._creation_cleanup.settle_creation_cleanup",
            "cayu.runners.base._clean_runner_preflight",
            "cayu.runners.base._clear_preflight_traceback_frames",
            "cayu.runners.base._contains_runner_fatal_signal",
        }
    ),
    "src/cayu/runners/microsandbox.py": frozenset(
        {
            "cayu._exception_groups.exception_group_children",
            "cayu._exception_state.exception_state",
            "cayu._exception_state.set_exception_state",
            "cayu._task_wait.await_shielded_task_outcome",
            "cayu._task_wait.capture_awaitable_outcome",
            "cayu._task_wait.restore_task_cancellation_requests",
            "cayu._task_wait.unexpected_child_cancellation_error",
            "cayu._validation.canonical_durable_json_bytes",
            "cayu._validation.copy_json_value",
            "cayu._validation.require_clean_nonblank",
            "cayu._validation.require_durable_clean_nonblank",
            "cayu.runners._admission_probes.EXECUTABLE_AVAILABILITY_SCRIPT",
            "cayu.runners._cleanup.RunnerCleanupProgress",
            "cayu.runners._cleanup.RunnerFailureProgress",
            "cayu.runners._cleanup._cleanup_artifact",
            "cayu.runners._cleanup.attach_runner_cancellation_failure",
            "cayu.runners._cleanup.runner_cancellation_failure",
            "cayu.runners._creation_cleanup.CreationCleanupProgress",
            "cayu.runners._creation_cleanup.CreationLease",
            "cayu.runners._creation_cleanup.acquire_creation_lease",
            "cayu.runners._creation_cleanup.drain_acquisition_restorations",
            "cayu.runners._creation_cleanup.drain_creation_cleanups",
            "cayu.runners._creation_cleanup.register_acquisition_restoration_retry",
            "cayu.runners._creation_cleanup.require_creation_cleanup_settled",
            "cayu.runners._creation_cleanup.retry_acquisition_settlement",
            "cayu.runners._creation_cleanup.settle_creation_cleanup",
            "cayu.runners.base._clean_runner_preflight",
            "cayu.runners.base._clear_preflight_traceback_frames",
            "cayu.runners.base._contains_runner_fatal_signal",
        }
    ),
    "src/cayu/egress/e2b_adapter.py": frozenset(
        {
            "cayu._exception_groups.iter_exception_tree",
            "cayu.egress.authority._build_adapter_verified_egress_authority_cutover_receipt",
        }
    ),
    "src/cayu/egress/microsandbox_adapter.py": frozenset(
        {
            "cayu._exception_groups.add_exception_note_safely",
            "cayu._exception_groups.exception_cause",
            "cayu.egress.adapter._await_bounded_cleanup_task",
            "cayu.egress.adapter._consume_accounted_task_cancellation",
            "cayu.egress.adapter._raise_primary_with_cleanup_cancellation",
            "cayu.runners._creation_cleanup.retry_acquisition_settlement",
        }
    ),
    "src/cayu/workspaces/e2b.py": frozenset(
        {
            "cayu._validation.require_clean_nonblank",
            "cayu._validation.require_positive_finite_seconds",
            "cayu.workspaces._guest_guard.guard_create",
            "cayu.workspaces._guest_guard.guard_delete",
            "cayu.workspaces._guest_guard.guard_delete_if_revision",
            "cayu.workspaces._guest_guard.guard_move_if_revision",
            "cayu.workspaces._guest_guard.guard_read",
            "cayu.workspaces._guest_guard.guard_replace",
            "cayu.workspaces._guest_guard.guard_require_absent",
            "cayu.workspaces._guest_guard.guard_write",
            "cayu.workspaces.base._WorkspaceListCollector",
            "cayu.workspaces.base._validate_absolute_guest_root",
            "cayu.workspaces.base._validate_workspace_offset",
            "cayu.workspaces.base._validate_workspace_positive_limit",
            "cayu.workspaces.base._validate_workspace_relative_path",
            "cayu.workspaces.base._validate_workspace_revision",
        }
    ),
    "src/cayu/workspaces/microsandbox.py": frozenset(
        {
            "cayu._validation.require_clean_nonblank",
            "cayu.workspaces._guest_guard.guard_create",
            "cayu.workspaces._guest_guard.guard_delete",
            "cayu.workspaces._guest_guard.guard_delete_if_revision",
            "cayu.workspaces._guest_guard.guard_move_if_revision",
            "cayu.workspaces._guest_guard.guard_read",
            "cayu.workspaces._guest_guard.guard_replace",
            "cayu.workspaces._guest_guard.guard_require_absent",
            "cayu.workspaces._guest_guard.guard_write",
            "cayu.workspaces.base._WorkspaceListCollector",
            "cayu.workspaces.base._validate_absolute_guest_root",
            "cayu.workspaces.base._validate_workspace_offset",
            "cayu.workspaces.base._validate_workspace_positive_limit",
            "cayu.workspaces.base._validate_workspace_relative_path",
            "cayu.workspaces.base._validate_workspace_revision",
        }
    ),
}


def _is_private(dotted: str) -> bool:
    return any(part.startswith("_") for part in dotted.split("."))


def _module_name(relative: str) -> str:
    return relative.removeprefix("src/").removesuffix(".py").replace("/", ".")


def _private_cayu_imports(source: str, *, module_name: str) -> set[str]:
    """Every private Cayu module or name the source imports, at any nesting level."""

    package = module_name.rsplit(".", 1)[0]
    tree = ast.parse(source, filename=module_name)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "cayu" and _is_private(alias.name):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if module.split(".")[0] != "cayu":
                continue
            for alias in node.names:
                target = f"{module}.{alias.name}"
                if _is_private(target):
                    found.add(target)
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.split(".")[0] == "cayu"
            and _is_private(node.args[0].value)
        ):
            # importlib.import_module("cayu._private") and similar dynamic loads.
            found.add(node.args[0].value)
    return found


@pytest.mark.parametrize("relative", sorted(_ALLOWED))
def test_provider_modules_import_only_public_or_allowlisted_cayu_names(relative: str) -> None:
    actual = _private_cayu_imports(
        (_ROOT / relative).read_text(encoding="utf-8"),
        module_name=_module_name(relative),
    )
    allowed = _ALLOWED[relative]
    new = sorted(actual - allowed)
    stale = sorted(allowed - actual)
    assert not new, (
        f"{relative} gained private Cayu imports {new}; use a public seam "
        "(cayu.extensions.*) or justify extending the extraction allowlist."
    )
    assert not stale, f"{relative} no longer imports {stale}; remove them from _ALLOWED."


def test_boundary_scanner_detects_private_imports() -> None:
    source = (
        "import cayu._private\n"
        "from cayu.runners import _cleanup\n"
        "from ._subprocess import copy_runner_env\n"
        "from cayu.extensions.runners import validate_timeout\n"
        "def later():\n"
        "    from cayu.runners.base import _clean_runner_preflight\n"
        "    importlib.import_module('cayu._task_wait')\n"
    )
    found = _private_cayu_imports(source, module_name="cayu.runners.fake")
    assert found == {
        "cayu._private",
        "cayu.runners._cleanup",
        "cayu.runners._subprocess.copy_runner_env",
        "cayu.runners.base._clean_runner_preflight",
        "cayu._task_wait",
    }
