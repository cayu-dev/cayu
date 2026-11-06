"""Keep new run, recovery and task code on the single settlement path.

Settlement is owned by ``runtime/_run_settlement.py`` (admitted invocations),
``runtime/_run_exit.py`` (exit classification) and ``tasks/_task_run_settlement.py``
(claimed tasks). A new ``except GeneratorExit`` or ``except BaseExceptionGroup``
(or a catch-all ``except BaseException``) in the engine, recovery or task
modules is a hand-written exit handler: it settles only the exits its author
anticipated. The allow-lists record the
handlers that remain today (see the settlement implementation notes for why
each stays) and may only shrink.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "src" / "cayu"

_GUARDED_HANDLER_TYPES = frozenset({"GeneratorExit", "BaseExceptionGroup", "BaseException"})

# (function qualname, handler type) -> count of remaining hand-written handlers.
_ALLOWED_HANDLERS: dict[str, dict[tuple[str, str], int]] = {
    "runtime/_recovery_coordinator.py": {
        ("RecoveryCoordinator._claim_manual_tool_round_recovery", "BaseException"): 1,
        (
            "RecoveryCoordinator._claim_manual_tool_round_recovery.reconstruct_claim_invocation_context",
            "BaseException",
        ): 1,
        ("RecoveryCoordinator._interrupt_deadline_settlement", "BaseException"): 1,
        ("RecoveryCoordinator._reconcile_tool_effect_claimed", "BaseException"): 1,
        ("RecoveryCoordinator._recover_tool_round_claimed", "BaseException"): 2,
        ("RecoveryCoordinator._recover_tool_round_claimed", "BaseExceptionGroup"): 1,
        ("RecoveryCoordinator._recover_tool_round_claimed", "GeneratorExit"): 1,
        ("RecoveryCoordinator._recovery_tool_hook_events", "BaseException"): 1,
        ("RecoveryCoordinator._resolve_tool_approval_owned", "BaseException"): 1,
        ("RecoveryCoordinator._resolve_user_input_owned", "BaseException"): 1,
        ("RecoveryCoordinator.continue_tool_approval_resolution", "GeneratorExit"): 2,
        ("RecoveryCoordinator.continue_user_input_resolution", "GeneratorExit"): 1,
        ("RecoveryCoordinator.recover_tool_approval", "BaseException"): 3,
        ("RecoveryCoordinator.recover_tool_approval", "BaseExceptionGroup"): 1,
        ("RecoveryCoordinator.recover_tool_approval", "GeneratorExit"): 1,
        ("RecoveryCoordinator.recover_tool_approval_request", "BaseException"): 1,
        ("RecoveryCoordinator.recover_tool_round", "BaseException"): 2,
        ("RecoveryCoordinator.recover_tool_round.supervise_recovery", "BaseException"): 3,
        ("RecoveryCoordinator.recover_tool_round_request", "BaseException"): 1,
        ("RecoveryCoordinator.recover_user_input", "BaseException"): 3,
        ("RecoveryCoordinator.recover_user_input", "BaseExceptionGroup"): 1,
        ("RecoveryCoordinator.recover_user_input", "GeneratorExit"): 1,
        ("RecoveryCoordinator.recover_user_input_request", "BaseException"): 2,
    },
    "runtime/_session_engine.py": {
        (
            "SessionEngine._handle_queued_messages_before_completion.commit_and_finish_completion",
            "BaseException",
        ): 1,
        ("SessionEngine._run_session_with_deadline", "BaseExceptionGroup"): 1,
        ("SessionEngine._run_session_with_deadline", "GeneratorExit"): 1,
        ("SessionEngine._run_session_with_group_observation", "BaseException"): 1,
        ("SessionEngine.compact_session", "BaseExceptionGroup"): 1,
        ("SessionEngine.compact_session", "GeneratorExit"): 1,
        ("SessionEngine.resume", "BaseException"): 3,
        ("SessionEngine.resume", "BaseExceptionGroup"): 1,
        ("SessionEngine.resume", "GeneratorExit"): 1,
        ("SessionEngine.run", "BaseException"): 2,
        ("SessionEngine.run", "BaseExceptionGroup"): 2,
        ("SessionEngine.run", "GeneratorExit"): 2,
    },
    "tasks/dispatch.py": {
        ("TaskStoreDispatcher._acknowledge_terminal_task", "BaseException"): 1,
        (
            "TaskStoreDispatcher._drain_dispatch_operation_under_cancellation_fence",
            "BaseException",
        ): 2,
        ("TaskStoreDispatcher._request_dispatch_cancellation_fence", "BaseException"): 2,
        ("TaskStoreDispatcher._request_dispatch_cancellation_fence", "GeneratorExit"): 2,
        ("TaskStoreDispatcher.process_next", "BaseException"): 1,
    },
    "tasks/worker.py": {
        ("_await_task_handler", "BaseExceptionGroup"): 1,
        ("_await_task_handler.owned_handler", "BaseException"): 1,
        ("_drain_ordinary_task_handler_under_cancellation_fence.cleanup", "BaseException"): 4,
        ("_drain_ordinary_task_handler_under_cancellation_fence.cleanup", "GeneratorExit"): 4,
        ("_drain_retry_task_handler_under_cancellation_fence.cleanup", "BaseException"): 4,
        ("_emit_interrupted_task_handoff_event", "GeneratorExit"): 2,
        ("_handle_with_heartbeat", "BaseException"): 1,
        ("_handle_with_heartbeat_owned", "BaseException"): 1,
        ("_quiesce_task_retry_handler_before_terminalization", "BaseException"): 1,
        ("_raise_task_retry_owner_cancellation", "BaseException"): 1,
        ("_raise_task_retry_owner_cancellation.cleanup", "BaseException"): 3,
        ("_raise_task_retry_owner_cancellation.cleanup", "GeneratorExit"): 1,
        ("_recover_expired_interrupted_task_handoffs", "GeneratorExit"): 1,
        ("_request_task_cancellation_fence", "BaseException"): 2,
        ("_request_task_cancellation_fence", "GeneratorExit"): 2,
        ("_restart_task_handler_after_shutdown.cleanup", "BaseException"): 2,
        ("_settle_interrupted_task_handoff_once", "BaseException"): 2,
        ("_settle_interrupted_task_handoff_once", "GeneratorExit"): 2,
        ("_validate_interrupted_task_handoff_receipt", "BaseException"): 1,
        ("_validate_interrupted_task_handoff_receipt", "GeneratorExit"): 1,
        ("run_task_worker", "BaseException"): 1,
        ("run_task_worker.run_slot.maintained_while_busy", "BaseException"): 1,
        ("run_task_worker.run_slot.run_step", "BaseException"): 2,
    },
}

# Entrances that admit an invocation. Between the admission call and their
# settlement scope only these yields may appear (an admission the store refused
# can replay an earlier profile decision; nothing was admitted).
_ADMISSION_ENTRANCES: dict[str, int] = {
    "SessionEngine.run": 0,
    "SessionEngine.resume": 1,
}


def _handler_type_names(node: ast.expr | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Tuple):
        return [name for element in node.elts for name in _handler_type_names(element)]
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return []


def _functions(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Every function with its qualname; same-name siblings are all kept."""

    found: list[tuple[str, ast.AST]] = []

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = ".".join([*scope, child.name])
                if not isinstance(child, ast.ClassDef):
                    found.append((name, child))
                visit(child, [*scope, child.name])
            else:
                visit(child, scope)

    visit(tree, [])
    return found


def _own_nodes(function: ast.AST) -> list[ast.AST]:
    """Nodes of ``function`` without those of nested functions or classes."""

    nodes: list[ast.AST] = []
    pending = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        nodes.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return nodes


@pytest.mark.parametrize("relative", sorted(_ALLOWED_HANDLERS))
def test_no_new_hand_written_exit_handlers(relative: str) -> None:
    tree = ast.parse((_ROOT / relative).read_text())
    found: collections.Counter[tuple[str, str]] = collections.Counter()
    for name, function in _functions(tree):
        for node in _own_nodes(function):
            if isinstance(node, ast.ExceptHandler):
                for type_name in _handler_type_names(node.type):
                    if type_name in _GUARDED_HANDLER_TYPES:
                        found[name, type_name] += 1
    allowed = _ALLOWED_HANDLERS[relative]
    grown = {key: count for key, count in found.items() if count > allowed.get(key, 0)}
    assert not grown, (
        f"New GeneratorExit/BaseExceptionGroup handlers in {relative}: {grown}. Settle "
        "exits through RunSettlement / classify_run_exit / tasks._task_run_settlement."
    )
    shrunk = {key: count for key, count in allowed.items() if found.get(key, 0) < count}
    assert not shrunk, f"Handlers were removed; shrink the allow-list in this test: {shrunk}"


def test_no_yield_between_admission_and_settlement_scope() -> None:
    tree = ast.parse((_ROOT / "runtime/_session_engine.py").read_text())
    functions = dict(_functions(tree))
    for name, allowed_yields in _ADMISSION_ENTRANCES.items():
        nodes = _own_nodes(functions[name])
        admissions = [
            node.lineno
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_invocation_lifecycle_command"
        ]
        scopes = [
            node.lineno
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "settling"
        ]
        assert admissions and len(scopes) == 1, name
        yields = [
            node.lineno
            for node in nodes
            if isinstance(node, ast.Yield | ast.YieldFrom)
            and min(admissions) < node.lineno < scopes[0]
        ]
        assert len(yields) == allowed_yields, (
            f"{name} yields at lines {yields} after admission, before its settlement scope."
        )
