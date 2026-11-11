"""Keep new run, recovery and task code on the single settlement path.

Settlement is owned by ``runtime/_run_settlement.py`` (admitted invocations,
recovered ones included), ``runtime/_run_exit.py`` (exit classification) and
``tasks/_task_run_settlement.py`` (claimed tasks). In the engine, recovery and
task modules, a new ``except GeneratorExit``, ``except BaseExceptionGroup``, a
catch-all ``except BaseException``, a bare ``except`` or ``except
CancelledError`` (also under ``except*``) is a hand-written exit handler: it
settles only the exits its author anticipated, and those modules count them per
function. In every module under ``runtime/``, ``tasks/`` and ``verification/``
(discovered by glob), a guarded handler that writes a session status, releases or
terminalizes a claim, or finalizes an abandonment, itself or through a function
of the same module that does, is listed by function. So is a recovery or engine
exit that finalizes an abandoned session itself instead of handing its ledger to
the settlement, a dispatcher branch that releases a claim outside the claim exit
table, and a dispatcher step that awaits work on a claim outside the lease owner.
A call reaches a function of the same module only by a bare name
(nested, enclosing or module scope) or through ``self``/``cls`` within its class.
Handler types are resolved through import aliases (also inside functions) and
module- or class-level constants. The allow-lists record the exits that remain
today and may only shrink.

Known limits, all syntactic: a handler that only records the exit in a flag that
a ``finally`` then settles, a settlement reached through ``getattr``, another
object or another module's helper, and a type resolved at run time are not seen.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "src" / "cayu"
_BARE = "<bare>"

_GUARDED_HANDLER_TYPES = frozenset(
    {"GeneratorExit", "BaseExceptionGroup", "BaseException", "CancelledError", _BARE}
)
# Discovered by glob; the settlement owners themselves are where exits belong.
_PACKAGES = ("runtime", "tasks", "verification")
_SETTLEMENT_OWNERS = frozenset(
    {"runtime/_run_exit.py", "runtime/_run_settlement.py", "tasks/_task_run_settlement.py"}
)

# (function qualname, handler type) -> count of remaining hand-written handlers.
_ALLOWED_HANDLERS: dict[str, dict[tuple[str, str], int]] = {
    "runtime/_provider_disposition_recovery.py": {
        (
            "ProviderDispositionRecovery.finish_pending_provider_operation_disposition",
            "BaseException",
        ): 1,
        (
            "ProviderDispositionRecovery.finish_pending_provider_operation_disposition",
            "BaseExceptionGroup",
        ): 1,
        ("ProviderDispositionRecovery.run_pending_provider_operation_fallback", "BaseException"): 1,
    },
    "runtime/_recovery_admission.py": {
        ("RecoveryAdmission.transition_recovery_session_to_running", "BaseException"): 3,
        ("RecoveryAdmission.transition_recovery_session_to_running", "CancelledError"): 1,
    },
    "runtime/_recovery_coordinator.py": {
        ("RecoveryCoordinator._claim_manual_tool_round_recovery", "BaseException"): 1,
        ("RecoveryCoordinator._claim_manual_tool_round_recovery", "CancelledError"): 1,
        (
            "RecoveryCoordinator._claim_manual_tool_round_recovery.reconstruct_claim_invocation_context",
            "BaseException",
        ): 1,
        ("RecoveryCoordinator._interrupt_deadline_settlement", "BaseException"): 1,
        ("RecoveryCoordinator._reconcile_tool_effect_claimed", "BaseException"): 1,
        ("RecoveryCoordinator._recover_tool_round_claimed", "BaseException"): 1,
        ("RecoveryCoordinator._recover_tool_round_claimed", "BaseExceptionGroup"): 1,
        ("RecoveryCoordinator._recovery_tool_hook_events", "BaseException"): 1,
        ("RecoveryCoordinator._resolve_tool_approval_owned", "BaseException"): 1,
        ("RecoveryCoordinator._resolve_user_input_owned", "BaseException"): 1,
        ("RecoveryCoordinator.continue_tool_approval_resolution", "GeneratorExit"): 1,
        ("RecoveryCoordinator.continue_user_input_resolution", "GeneratorExit"): 1,
        ("RecoveryCoordinator.recover_tool_approval", "BaseException"): 3,
        ("RecoveryCoordinator.recover_tool_approval", "BaseExceptionGroup"): 1,
        ("RecoveryCoordinator.recover_tool_approval", "CancelledError"): 1,
        ("RecoveryCoordinator.recover_tool_approval", "GeneratorExit"): 1,
        ("RecoveryCoordinator.recover_tool_approval_request", "BaseException"): 1,
        ("RecoveryCoordinator.recover_tool_round", "BaseException"): 2,
        ("RecoveryCoordinator.recover_tool_round.supervise_recovery", "BaseException"): 3,
        (
            "RecoveryCoordinator.recover_tool_round.supervise_recovery.stop_recovery_worker",
            "CancelledError",
        ): 1,
        ("RecoveryCoordinator.recover_tool_round_request", "BaseException"): 1,
        ("RecoveryCoordinator.recover_user_input", "BaseException"): 3,
        ("RecoveryCoordinator.recover_user_input", "BaseExceptionGroup"): 1,
        ("RecoveryCoordinator.recover_user_input", "CancelledError"): 1,
        ("RecoveryCoordinator.recover_user_input", "GeneratorExit"): 1,
        ("RecoveryCoordinator.recover_user_input_request", "BaseException"): 2,
    },
    "runtime/_session_engine.py": {
        (
            "SessionEngine._handle_queued_messages_before_completion.admit_successor",
            "CancelledError",
        ): 1,
        (
            "SessionEngine._handle_queued_messages_before_completion.commit_and_finish_completion",
            "BaseException",
        ): 1,
        ("SessionEngine._run_session_with_deadline", "BaseExceptionGroup"): 1,
        ("SessionEngine._run_session_with_deadline", "CancelledError"): 1,
        ("SessionEngine._run_session_with_deadline", "GeneratorExit"): 1,
        ("SessionEngine._run_session_with_group_observation", "BaseException"): 1,
        ("SessionEngine.compact_session", "BaseExceptionGroup"): 1,
        ("SessionEngine.compact_session", "CancelledError"): 1,
        ("SessionEngine.compact_session", "GeneratorExit"): 1,
        ("SessionEngine.resume", "BaseException"): 3,
        ("SessionEngine.resume", "BaseExceptionGroup"): 1,
        ("SessionEngine.resume", "GeneratorExit"): 1,
        ("SessionEngine.run", "BaseException"): 2,
        ("SessionEngine.run", "BaseExceptionGroup"): 2,
        ("SessionEngine.run", "CancelledError"): 2,
        ("SessionEngine.run", "GeneratorExit"): 2,
    },
    "tasks/dispatch.py": {
        ("TaskStoreDispatcher._acknowledge_terminal_task", "BaseException"): 1,
        ("TaskStoreDispatcher._await_with_task_lease", "CancelledError"): 1,
        (
            "TaskStoreDispatcher._drain_dispatch_operation_under_cancellation_fence",
            "BaseException",
        ): 2,
        ("TaskStoreDispatcher._heartbeat.heartbeat", "CancelledError"): 1,
        # Records a cancellation or process-control exit ending the claim for the
        # heartbeat stop; re-raises.
        ("TaskStoreDispatcher._process_next_owned", "BaseException"): 2,
        ("TaskStoreDispatcher._request_dispatch_cancellation_fence", "BaseException"): 2,
        ("TaskStoreDispatcher._request_dispatch_cancellation_fence", "GeneratorExit"): 2,
        ("TaskStoreDispatcher.process_next", "BaseException"): 1,
    },
    "tasks/worker.py": {
        ("_await_task_handler", "BaseExceptionGroup"): 1,
        ("_await_task_handler", "CancelledError"): 1,
        ("_await_task_handler.owned_handler", "BaseException"): 1,
        ("_await_task_retry_cleanup_resisting_cancellation", "CancelledError"): 1,
        ("_complete_task_retry_terminalization_resisting_cancellation", "CancelledError"): 1,
        ("_drain_ordinary_task_handler_under_cancellation_fence.cleanup", "BaseException"): 4,
        ("_drain_ordinary_task_handler_under_cancellation_fence.cleanup", "CancelledError"): 1,
        ("_drain_ordinary_task_handler_under_cancellation_fence.cleanup", "GeneratorExit"): 4,
        ("_drain_retry_task_handler_under_cancellation_fence.cleanup", "BaseException"): 4,
        ("_drain_retry_task_handler_under_cancellation_fence.cleanup", "CancelledError"): 1,
        ("_emit_interrupted_task_handoff_event", "CancelledError"): 2,
        ("_emit_interrupted_task_handoff_event", "GeneratorExit"): 2,
        ("_handle_with_heartbeat", "BaseException"): 1,
        ("_handle_with_heartbeat_owned", "BaseException"): 1,
        ("_heartbeat_until.inspect_heartbeat", "CancelledError"): 2,
        ("_heartbeat_until.reconcile_heartbeat_failure", "CancelledError"): 1,
        ("_quiesce_task_retry_handler_before_terminalization", "BaseException"): 1,
        ("_quiesce_task_retry_handler_before_terminalization", "CancelledError"): 2,
        ("_raise_task_retry_owner_cancellation", "BaseException"): 1,
        ("_raise_task_retry_owner_cancellation.cleanup", "BaseException"): 3,
        ("_raise_task_retry_owner_cancellation.cleanup", "CancelledError"): 1,
        ("_raise_task_retry_owner_cancellation.cleanup", "GeneratorExit"): 1,
        ("_recover_expired_interrupted_task_handoffs", "CancelledError"): 1,
        ("_recover_expired_interrupted_task_handoffs", "GeneratorExit"): 1,
        ("_renew_task_lease_before_deadline", "CancelledError"): 1,
        ("_request_task_cancellation_fence", "BaseException"): 2,
        ("_request_task_cancellation_fence", "GeneratorExit"): 2,
        ("_restart_task_handler_after_shutdown.cleanup", "BaseException"): 2,
        ("_restart_task_handler_after_shutdown.cleanup", "CancelledError"): 1,
        ("_settle_interrupted_task_handoff_once", "BaseException"): 2,
        ("_settle_interrupted_task_handoff_once", "CancelledError"): 2,
        ("_settle_interrupted_task_handoff_once", "GeneratorExit"): 2,
        ("_settle_interrupted_task_handoff_with_retry", "CancelledError"): 1,
        ("_stop_busy_slot_maintenance", "CancelledError"): 1,
        ("_validate_interrupted_task_handoff_receipt", "BaseException"): 1,
        ("_validate_interrupted_task_handoff_receipt", "GeneratorExit"): 1,
        ("run_task_worker", "BaseException"): 1,
        ("run_task_worker.drain_slot", "CancelledError"): 1,
        ("run_task_worker.run_slot.maintained_while_busy", "BaseException"): 1,
        ("run_task_worker.run_slot.run_step", "BaseException"): 2,
    },
}

# Recovered invocations settle through ``RunSettlement``: the exit owner hands
# its ledger to the recovery handoff cleanup. Finalizing an abandoned session
# directly is left only where an admission or a recovery claim fails before its
# ledger exists (the claim's acknowledgement was lost, or its context could not
# be rebuilt).
_RECOVERY_MODULES = (
    "runtime/_provider_disposition_recovery.py",
    "runtime/_recovery_admission.py",
    "runtime/_recovery_coordinator.py",
    "runtime/_session_engine.py",
)
_ALLOWED_DIRECT_ABANDONMENT: dict[str, dict[str, int]] = {
    "runtime/_session_engine.py": {
        # A cancellation during setup, before the run's settlement scope opens.
        "SessionEngine.run.abandon_cancelled_setup": 1,
        # An expired execution deadline fails the task and abandons the session
        # in one supervised step.
        "SessionEngine.run.settle_expired_task": 1,
        # The predecessor already settled; a cancelled successor admission
        # abandons only the session.
        "SessionEngine._handle_queued_messages_before_completion.admit_successor"
        ".finalize_admission": 1,
        # A linked task's authority check failing right after admission.
        "SessionEngine.resume": 1,
        # The run's own abandonment step, which its settlement calls.
        "SessionEngine._run_session_with_deadline.finalize_abandoned_run": 1,
    },
    "runtime/_recovery_admission.py": {
        "RecoveryAdmission.transition_recovery_session_to_running": 2,
    },
    "runtime/_recovery_coordinator.py": {
        "RecoveryCoordinator._claim_manual_tool_round_recovery": 4,
        "RecoveryCoordinator._claim_manual_tool_round_recovery.reconstruct_claim_invocation_context": 1,
    },
}
_ABANDONMENT_FINALIZERS = frozenset(
    {
        "finalize_abandoned_session_by_id",
        "finalize_abandoned_session_run",
        "finalize_abandoned_without_registered_runtime",
    }
)
_HANDOFF_CLEANUPS = frozenset({"cleanup_recovery_handoff", "cleanup_entrypoint_handoff"})
# Handoff cleanups that pass ``ledger=None``: a nested continuation whose
# enclosing owner (the manual-recovery supervisor or the claimed recovery)
# settles the exit, or the provider entrance, where the pending disposition owns
# the invocation until the fallback stream starts and settles its own exits.
_NESTED_HANDOFFS: dict[str, dict[str, int]] = {
    "runtime/_provider_disposition_recovery.py": {
        "ProviderDispositionRecovery.finish_pending_provider_operation_disposition": 1,
    },
    "runtime/_recovery_coordinator.py": {
        "RecoveryCoordinator._reconcile_tool_effect_claimed": 1,
        "RecoveryCoordinator._recover_tool_round_claimed": 1,
        "RecoveryCoordinator.recover_tool_round_request": 1,
    },
}
# Handoff cleanups whose ledger is ``owner if condition else None``: ownership
# passes between two owners at a known point. The entrance owns a manual
# recovery until its claimed recovery stream starts and settles its own exits
# (``inner_started``); the pending provider disposition owns a fallback until
# its replacement dispatch is durable.
_CONDITIONAL_HANDOFFS: dict[str, dict[str, int]] = {
    "runtime/_provider_disposition_recovery.py": {
        "ProviderDispositionRecovery.run_pending_provider_operation_fallback": 1,
    },
    "runtime/_recovery_coordinator.py": {
        "RecoveryCoordinator.recover_tool_approval_request": 1,
        "RecoveryCoordinator.recover_user_input_request": 1,
    },
}
# Dispatcher claim releases outside ``release_drained_claim``: the release
# primitive itself, and an unstarted claim whose lease ran out before dispatch.
_ALLOWED_CLAIM_RELEASES: dict[str, int] = {
    "TaskStoreDispatcher._process_next_owned.claim_next_task": 1,
    "TaskStoreDispatcher._release_claimed_task": 1,
}
_CLAIM_RELEASES = frozenset({"release_task", "_release_claimed_task"})

# Calls that settle a session as an operator interruption.
_INTERRUPTION_SETTLEMENTS = frozenset(
    {
        "handle_session_interrupted",
        "handle_session_interrupted_preserving_failure",
        "interrupt_recovery",
    }
)
# The rules that decide whether a grouped exit is the caller's cancellation, which
# abandons the run unless an interrupt was requested. Session exits ask
# ``runtime._run_exit.grouped_caller_cancellation``; recovery keeps its own wider
# abandonment signal.
_GROUPED_CANCELLATION_RULES = frozenset(
    {"grouped_caller_cancellation", "_recovery_abandonment_signal"}
)

# Entrances that admit an invocation. Between the admission call and their
# settlement scope only these yields may appear (an admission the store refused
# can replay an earlier profile decision; nothing was admitted).
_ADMISSION_ENTRANCES: dict[str, int] = {
    "SessionEngine.run": 0,
    "SessionEngine.resume": 1,
}

# Calls that settle an exit: a session status write, an interaction transition or
# interruption, an abandonment, the invocation fence, and a claim's release or
# terminal write.
_SETTLEMENT_CALLS = frozenset(
    {
        "update_status",
        "transition_status",
        "transition_status_and_checkpoint",
        "transition_status_if_no_queued_messages",
        "publish_interaction_transition",
        "publish_sibling_interaction_transition",
        *_INTERRUPTION_SETTLEMENTS,
        *_ABANDONMENT_FINALIZERS,
        "release_session_invocation",
        "settle_session_invocation",
        "release_run_fence",
        *_CLAIM_RELEASES,
        "release_attached_task_worker",
        "restart_task",
        "complete_task",
        "fail_task",
        "terminalize_task",
    }
)
# Handlers that settle an exit themselves, per function, with why each stays.
_ALLOWED_SETTLING_HANDLERS: dict[str, dict[str, int]] = {
    # An operator interruption settled after the one grouped-cancellation rule
    # (the grouped-exit test below checks the order).
    "runtime/_recovery_coordinator.py": {
        "RecoveryCoordinator._recover_tool_round_claimed": 1,
        "RecoveryCoordinator.recover_tool_approval": 1,
        "RecoveryCoordinator.recover_user_input": 1,
    },
    "runtime/_session_engine.py": {
        "SessionEngine._run_session_with_deadline": 3,
        "SessionEngine.run": 2,
    },
    # The task handler's cancellation exits: they settle a cancelled claim through
    # the worker's cancellation owners, ``settle_ordinary_cancellation`` and
    # ``_raise_task_retry_owner_cancellation``.
    "tasks/worker.py": {"_await_task_handler": 2},
    # The abandonment finalizer's own path when its interrupting transition fails.
    "runtime/_session_finalization.py": {
        "SessionFinalization.finalize_abandoned_session_run": 1,
    },
}

# The dispatcher's claimed region: everything after it unpacks ``claimed``.
_CLAIM_OWNER = "TaskStoreDispatcher._process_next_owned"
_CLAIM_BINDING = "claimed"
# Steps awaited directly on a claimed task, by callee, with why each may be.
_CLAIMED_REGION_STEPS: dict[str, int] = {
    # The lease owner itself.
    "_await_with_task_lease": 6,
    # The claim exit table: rejection, release of work no handler drains, a lost
    # claim's authoritative status, and an acknowledged cancellation fence.
    "_reject_claimed_dispatch": 7,
    "_release_drained_dispatch_claim": 4,
    "_reclaimed_dispatch_handle": 8,
    "_request_dispatch_cancellation_fence": 1,
    "_settle_dispatch_cancellation_after_quiescence": 1,
    # The execution settlement's start mark, acknowledgement and return proof.
    "enter": 1,
    "attempt": 3,
    "verify_returned_execution": 3,
    # Authority reads before the start mark; a failure releases the claim.
    "_load_dispatch_session_invocation": 1,
    "_queued_dispatch_settlement_state": 1,
    # Stopping the heartbeat in the lease owner's finally.
    "_stop_heartbeat": 2,
}


def _type_aliases(tree: ast.AST) -> dict[str, list[ast.expr]]:
    """Names that may stand for exception types: import aliases anywhere in the
    module, and module- or class-level constants (annotated or not)."""

    aliases: dict[str, list[ast.expr]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname is not None:
                    aliases[alias.asname] = [ast.Name(id=alias.name)]
    scopes = [tree, *(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))]
    for scope in scopes:
        for node in getattr(scope, "body", []):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            if not isinstance(target, ast.Name):
                continue
            if isinstance(value, ast.Tuple):
                aliases[target.id] = list(value.elts)
            elif isinstance(value, ast.Name | ast.Attribute):
                aliases[target.id] = [value]
    return aliases


def _handler_type_names(
    node: ast.expr | None,
    aliases: dict[str, list[ast.expr]] | None = None,
    _seen: frozenset[str] = frozenset(),
) -> list[str]:
    if node is None:
        return [_BARE]
    if isinstance(node, ast.Tuple):
        return [
            name for element in node.elts for name in _handler_type_names(element, aliases, _seen)
        ]
    if isinstance(node, ast.Starred):
        return _handler_type_names(node.value, aliases, _seen)
    if isinstance(node, ast.Name):
        if aliases and node.id in aliases and node.id not in _seen:
            return [
                name
                for element in aliases[node.id]
                for name in _handler_type_names(element, aliases, _seen | {node.id})
            ]
        return [node.id]
    if isinstance(node, ast.Attribute):
        # ``self._STOP`` or ``Owner._STOP``: a class-level constant.
        if (
            aliases
            and node.attr in aliases
            and node.attr not in _seen
            and isinstance(node.value, ast.Name)
        ):
            return [
                name
                for element in aliases[node.attr]
                for name in _handler_type_names(element, aliases, _seen | {node.attr})
            ]
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


def _guarded_handlers(tree: ast.AST) -> collections.Counter[tuple[str, str]]:
    """Guarded handlers per (function, caught type); a tuple counts each guarded type."""

    found: collections.Counter[tuple[str, str]] = collections.Counter()
    aliases = _type_aliases(tree)
    for name, function in _functions(tree):
        for node in _own_nodes(function):
            if isinstance(node, ast.ExceptHandler):
                for type_name in _handler_type_names(node.type, aliases):
                    if type_name in _GUARDED_HANDLER_TYPES:
                        found[name, type_name] += 1
    return found


def _discovered_modules() -> list[str]:
    return [
        relative
        for package in _PACKAGES
        for path in sorted((_ROOT / package).rglob("*.py"))
        if (relative := path.relative_to(_ROOT).as_posix()) not in _SETTLEMENT_OWNERS
    ]


@pytest.mark.parametrize("relative", sorted(_ALLOWED_HANDLERS))
def test_no_new_hand_written_exit_handlers(relative: str) -> None:
    found = _guarded_handlers(ast.parse((_ROOT / relative).read_text()))
    allowed = _ALLOWED_HANDLERS[relative]
    grown = {key: count for key, count in found.items() if count > allowed.get(key, 0)}
    assert not grown, (
        f"New hand-written exit handlers in {relative}: {grown}. Settle "
        "exits through RunSettlement / classify_run_exit / tasks._task_run_settlement."
    )
    shrunk = {key: count for key, count in allowed.items() if found.get(key, 0) < count}
    assert not shrunk, f"Handlers were removed; shrink the allow-list in this test: {shrunk}"


def _class_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                names.add(".".join([*scope, child.name]))
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                visit(child, [*scope, child.name])
            else:
                visit(child, scope)

    visit(tree, [])
    return names


def _resolved_call(
    call: ast.Call, caller: str, functions: set[str], classes: set[str]
) -> str | None:
    """The same-module function a call reaches, or ``None``.

    A bare name resolves to a function nested in the caller or an enclosing
    scope, else to a module function; ``self.name`` and ``cls.name`` to a method
    of the caller's class. A call on any other receiver resolves to nothing.
    """

    scope = caller.split(".")
    if isinstance(call.func, ast.Name):
        for depth in range(len(scope), -1, -1):
            candidate = ".".join([*scope[:depth], call.func.id])
            if candidate in functions:
                return candidate
        return None
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"self", "cls"}
    ):
        for depth in range(len(scope) - 1, 0, -1):
            owner = ".".join(scope[:depth])
            if owner in classes:
                candidate = f"{owner}.{call.func.attr}"
                return candidate if candidate in functions else None
    return None


def _settles(
    nodes: list[ast.AST],
    caller: str,
    settling: set[str],
    functions: set[str],
    classes: set[str],
) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            _called_name(node) in _SETTLEMENT_CALLS
            or _resolved_call(node, caller, functions, classes) in settling
        )
        for node in nodes
    )


def _settling_functions(tree: ast.AST) -> set[str]:
    """Qualnames of this module's functions that settle, directly or through a
    function of the module they call (see ``_resolved_call``)."""

    owned = {name: _own_nodes(function) for name, function in _functions(tree)}
    functions, classes = set(owned), _class_names(tree)
    settling: set[str] = set()
    while True:
        grown = {
            name
            for name, nodes in owned.items()
            if name not in settling and _settles(nodes, name, settling, functions, classes)
        }
        if not grown:
            return settling
        settling |= grown


def _settling_handlers(tree: ast.AST) -> collections.Counter[str]:
    """Guarded handlers, per function, that settle in their own body or by calling
    a settling function of the same module."""

    aliases = _type_aliases(tree)
    settling = _settling_functions(tree)
    functions = {name for name, _ in _functions(tree)}
    classes = _class_names(tree)
    found: collections.Counter[str] = collections.Counter()
    for name, function in _functions(tree):
        for node in _own_nodes(function):
            if (
                isinstance(node, ast.ExceptHandler)
                and set(_handler_type_names(node.type, aliases)) & _GUARDED_HANDLER_TYPES
                and _settles(_own_nodes(node), name, settling, functions, classes)
            ):
                found[name] += 1
    return found


def test_no_new_handler_settles_an_exit_by_hand() -> None:
    found = {
        relative: sites
        for relative in _discovered_modules()
        if (sites := _settling_handlers(ast.parse((_ROOT / relative).read_text())))
    }
    grown = {
        (relative, name): count
        for relative, sites in found.items()
        for name, count in sites.items()
        if count > _ALLOWED_SETTLING_HANDLERS.get(relative, {}).get(name, 0)
    }
    assert not grown, (
        f"Exit handlers that settle a session or a claim themselves: {grown}. Hand the "
        "exit to RunSettlement or the claim exit table instead."
    )
    shrunk = {
        (relative, name): count
        for relative, allowed in _ALLOWED_SETTLING_HANDLERS.items()
        for name, count in allowed.items()
        if found.get(relative, collections.Counter())[name] < count
    }
    assert not shrunk, f"Handlers no longer settle by hand; shrink the allow-list: {shrunk}"


def _claimed_region_awaits(function: ast.AST) -> collections.Counter[str]:
    """Direct awaits after the claim is unpacked, by callee; nested functions excluded."""

    nodes = _own_nodes(function)
    claimed_at = min(
        node.lineno
        for node in nodes
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name)
        and node.value.id == _CLAIM_BINDING
    )
    found: collections.Counter[str] = collections.Counter()
    for node in nodes:
        if isinstance(node, ast.Await) and node.lineno > claimed_at:
            value = node.value
            found[(_called_name(value) if isinstance(value, ast.Call) else None) or "?"] += 1
    return found


def test_dispatcher_awaits_claimed_work_under_its_lease_owner() -> None:
    """Work on a claim runs inside ``_await_with_task_lease``; only claim-exit and
    settlement steps are awaited directly."""

    functions = dict(_functions(ast.parse((_ROOT / "tasks/dispatch.py").read_text())))
    _assert_allow_list(
        _claimed_region_awaits(functions[_CLAIM_OWNER]),
        _CLAIMED_REGION_STEPS,
        "Dispatcher awaits on a claimed task outside _await_with_task_lease (run the work "
        "through it, or exit through the claim exit table)",
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "async def f(self):\n    claimed = await self.claim()\n"
            "    task, lease = claimed\n    await self._recover_stalled_session(task)\n"
            "    await self._release_drained_dispatch_claim(task=task)\n",
            {"_recover_stalled_session": 1, "_release_drained_dispatch_claim": 1},
        ),
        (
            "async def f(self):\n    claimed = await self.claim()\n"
            "    task, lease = claimed\n\n    async def recover():\n"
            "        await self._recover_stalled_session(task)\n\n"
            "    return await self._await_with_task_lease(recover(), task=task)\n",
            {"_await_with_task_lease": 1},
        ),
    ],
    ids=["recovery-outside-the-lease-owner", "recovery-under-the-lease-owner"],
)
def test_the_claimed_region_ratchet_sees_unowned_awaits(
    source: str, expected: dict[str, int]
) -> None:
    [function] = ast.parse(source).body
    assert _claimed_region_awaits(function) == collections.Counter(expected)


@pytest.mark.parametrize(
    ("source", "handlers", "settling"),
    [
        (
            "async def f():\n    try:\n        await g()\n"
            "    except asyncio.CancelledError:\n"
            "        await store.update_status(sid, SessionStatus.INTERRUPTED)\n        raise\n",
            {("f", "CancelledError"): 1},
            {"f": 1},
        ),
        (
            "async def f():\n    try:\n        await g()\n"
            "    except (asyncio.CancelledError, Exception):\n        raise\n"
            "    except:\n        raise\n    try:\n        await g()\n"
            "    except* BaseException:\n        pass\n",
            {("f", "CancelledError"): 1, ("f", "BaseException"): 1, ("f", _BARE): 1},
            {},
        ),
        (
            "async def f():\n    try:\n        await g()\n    except ValueError:\n"
            "        await tasks.release_task(1, 2)\n    except BaseException:\n"
            "        async def later():\n            await tasks.release_task(1, 2)\n"
            "        raise\n",
            {("f", "BaseException"): 1},
            {},
        ),
        (
            "async def _fail(sid):\n    await store.update_status(sid, SessionStatus.FAILED)\n"
            "async def _settle(sid):\n    await _fail(sid)\n"
            "async def f():\n    try:\n        await g()\n"
            "    except BaseException:\n        await _settle(sid)\n        raise\n",
            {("f", "BaseException"): 1},
            {"f": 1},
        ),
        (
            "from asyncio import CancelledError as Stopped\n"
            "_STOP = (Stopped, KeyboardInterrupt)\n"
            "async def f():\n    try:\n        await g()\n"
            "    except _STOP:\n        await tasks.release_task(1, 2)\n        raise\n",
            {("f", "CancelledError"): 1},
            {"f": 1},
        ),
        (
            "async def close(sid):\n    await store.update_status(sid, SessionStatus.FAILED)\n"
            "async def f(stream):\n    try:\n        await g()\n"
            "    except BaseException:\n        await stream.close()\n        raise\n",
            {("f", "BaseException"): 1},
            {},
        ),
        (
            "class Owner:\n    async def _fail(self, sid):\n"
            "        await self.store.transition_status(sid, to_status=SessionStatus.FAILED)\n\n"
            "    async def run(self):\n        try:\n            await g()\n"
            "        except asyncio.CancelledError:\n            await self._fail(sid)\n"
            "            raise\n",
            {("Owner.run", "CancelledError"): 1},
            {"Owner.run": 1},
        ),
        (
            "_STOP: tuple = (asyncio.CancelledError,)\n"
            "class Owner:\n    _ENDS = (GeneratorExit,)\n\n"
            "    async def run(self):\n        from asyncio import CancelledError as Gone\n"
            "        try:\n            await g()\n        except _STOP:\n            raise\n"
            "        try:\n            await g()\n        except self._ENDS:\n            raise\n"
            "        try:\n            await g()\n        except Gone:\n            raise\n",
            {("Owner.run", "CancelledError"): 2, ("Owner.run", "GeneratorExit"): 1},
            {},
        ),
    ],
    ids=[
        "cancelled-handler-settles",
        "grouped-star-and-bare",
        "unguarded-or-nested-release",
        "settles-through-a-module-helper",
        "aliased-and-constant-types",
        "unrelated-receiver-with-a-settling-name",
        "settles-through-a-method-of-its-class",
        "annotated-class-and-local-alias-types",
    ],
)
def test_the_handler_ratchets_see_cancellation_and_settling_handlers(
    source: str,
    handlers: dict[tuple[str, str], int],
    settling: dict[str, int],
) -> None:
    tree = ast.parse(source)
    assert _guarded_handlers(tree) == collections.Counter(handlers)
    assert _settling_handlers(tree) == collections.Counter(settling)


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


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _unclassified_group_handlers(tree: ast.AST) -> list[str]:
    """Handlers that can catch a group and interrupt before asking the rule.

    That is ``except BaseExceptionGroup``, ``except BaseException``, a bare
    ``except`` or ``except*``. The rule must be called in the handler's own body,
    not a nested function, before its first interruption call. The check is
    syntactic: it sees order, not the branch the rule's answer takes.
    """

    unclassified: list[str] = []
    aliases = _type_aliases(tree)
    for name, function in _functions(tree):
        for node in _own_nodes(function):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is not None and not set(_handler_type_names(node.type, aliases)) & {
                "BaseExceptionGroup",
                "BaseException",
            }:
                continue
            calls = [call for call in _own_nodes(node) if isinstance(call, ast.Call)]
            interruptions = [
                call.lineno for call in calls if _called_name(call) in _INTERRUPTION_SETTLEMENTS
            ]
            rules = [
                call.lineno for call in calls if _called_name(call) in _GROUPED_CANCELLATION_RULES
            ]
            if interruptions and (not rules or min(rules) >= min(interruptions)):
                unclassified.append(f"{name} (line {node.lineno})")
    return unclassified


@pytest.mark.parametrize("relative", sorted(_ALLOWED_HANDLERS))
def test_grouped_exits_classify_cancellation_before_interrupting(relative: str) -> None:
    """A grouped exit settles as an operator interruption only after the one rule.

    A grouped cancellation without an interrupt request settles like the plain
    cancellation it carries; a handler that interrupts first would route it
    differently.
    """

    unclassified = _unclassified_group_handlers(ast.parse((_ROOT / relative).read_text()))
    assert not unclassified, (
        f"Handlers in {relative} that can catch a group interrupt the session before "
        f"asking grouped_caller_cancellation: {unclassified}"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def f():\n    try:\n        pass\n    except* BaseException as e:\n"
            "        interrupt_recovery()\n        grouped_caller_cancellation(e)\n",
            ["f (line 4)"],
        ),
        (
            "def g():\n    try:\n        pass\n    except BaseExceptionGroup as e:\n"
            "        def h():\n            grouped_caller_cancellation(e)\n"
            "        interrupt_recovery()\n",
            ["g (line 4)"],
        ),
        (
            "def k():\n    try:\n        pass\n    except:\n"
            "        grouped_caller_cancellation(None)\n        interrupt_recovery()\n",
            [],
        ),
    ],
    ids=["rule-after-interruption", "rule-in-nested-function", "rule-first"],
)
def test_grouped_exit_ratchet_sees_order_handlers_and_nesting(
    source: str, expected: list[str]
) -> None:
    assert _unclassified_group_handlers(ast.parse(source)) == expected


def _calls_by_function(tree: ast.AST, names: frozenset[str]) -> collections.Counter[str]:
    """Calls to ``names`` per function, including those inside its lambdas."""

    found: collections.Counter[str] = collections.Counter()
    for name, function in _functions(tree):
        pending = list(ast.iter_child_nodes(function))
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if isinstance(node, ast.Call) and _called_name(node) in names:
                found[name] += 1
            pending.extend(ast.iter_child_nodes(node))
    return found


def _calls_handoff_cleanup(call: ast.Call) -> bool:
    """A handoff cleanup call, also one choosing between the two cleanups inline."""

    if isinstance(call.func, ast.IfExp):
        branches = (call.func.body, call.func.orelse)
        return any(
            isinstance(branch, ast.Attribute) and branch.attr in _HANDOFF_CLEANUPS
            for branch in branches
        )
    return _called_name(call) in _HANDOFF_CLEANUPS


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _handoff_ownership(tree: ast.AST) -> dict[str, collections.Counter[str] | list[str]]:
    """Classify each handoff cleanup's ``ledger=`` by who settles the exit.

    ``nested``: ``None``. ``conditional``: a named ledger in one arm of a
    conditional and ``None`` in the other. Anything that is neither a plain
    named ledger nor one of these is ``unrecognised``; a call naming no ledger
    at all is ``unnamed``.
    """

    nested: collections.Counter[str] = collections.Counter()
    conditional: collections.Counter[str] = collections.Counter()
    unnamed: list[str] = []
    unrecognised: list[str] = []
    for name, function in _functions(tree):
        for node in _own_nodes(function):
            if not isinstance(node, ast.Call) or not _calls_handoff_cleanup(node):
                continue
            ledger = next((k.value for k in node.keywords if k.arg == "ledger"), None)
            where = f"{name} (line {node.lineno})"
            if ledger is None:
                unnamed.append(where)
            elif _is_none(ledger):
                nested[name] += 1
            elif isinstance(ledger, ast.IfExp) and (
                _is_none(ledger.body) != _is_none(ledger.orelse)
            ):
                owner = ledger.orelse if _is_none(ledger.body) else ledger.body
                if isinstance(owner, ast.Name):
                    conditional[name] += 1
                else:
                    unrecognised.append(where)
            elif not isinstance(ledger, ast.Name):
                unrecognised.append(where)
    return {
        "nested": nested,
        "conditional": conditional,
        "unnamed": unnamed,
        "unrecognised": unrecognised,
    }


def _beyond(found: collections.Counter[str], allowed: dict[str, int]) -> dict[str, int]:
    return {name: count for name, count in found.items() if count > allowed.get(name, 0)}


def _below(found: collections.Counter[str], allowed: dict[str, int]) -> dict[str, int]:
    return {name: count for name, count in allowed.items() if found.get(name, 0) < count}


def _assert_allow_list(found: collections.Counter[str], allowed: dict[str, int], what: str) -> None:
    assert not _beyond(found, allowed), f"{what}: {_beyond(found, allowed)}"
    assert not _below(found, allowed), (
        f"{what} were removed; shrink the allow-list: {_below(found, allowed)}"
    )


@pytest.mark.parametrize("relative", _RECOVERY_MODULES)
def test_recovery_exits_settle_through_run_settlement(relative: str) -> None:
    tree = ast.parse((_ROOT / relative).read_text())
    _assert_allow_list(
        _calls_by_function(tree, _ABANDONMENT_FINALIZERS),
        _ALLOWED_DIRECT_ABANDONMENT.get(relative, {}),
        f"Recovery exits in {relative} that finalize an abandoned session themselves "
        "(hand the exit owner's ledger to the handoff cleanup)",
    )
    ownership = _handoff_ownership(tree)
    assert not ownership["unnamed"], (
        f"Handoff cleanups in {relative} that name no exit owner: {ownership['unnamed']}"
    )
    assert not ownership["unrecognised"], (
        f"Handoff cleanups in {relative} whose ledger is neither a named owner, None, nor "
        f"'owner if condition else None': {ownership['unrecognised']}"
    )
    _assert_allow_list(
        ownership["nested"],
        _NESTED_HANDOFFS.get(relative, {}),
        f"Handoff cleanups in {relative} that settle nothing (pass the exit owner's ledger "
        "unless an enclosing owner settles this exit)",
    )
    _assert_allow_list(
        ownership["conditional"],
        _CONDITIONAL_HANDOFFS.get(relative, {}),
        f"Handoff cleanups in {relative} whose ownership changes hands conditionally",
    )


def test_dispatcher_releases_claims_through_the_claim_exit_table() -> None:
    tree = ast.parse((_ROOT / "tasks/dispatch.py").read_text())
    _assert_allow_list(
        _calls_by_function(tree, _CLAIM_RELEASES),
        _ALLOWED_CLAIM_RELEASES,
        "Dispatcher claim releases outside the claim exit table (release a claim no "
        "handler is left to drain with tasks._task_run_settlement.release_drained_claim)",
    )


_NEW_EXITS: list[tuple[str, dict[str, object]]] = [
    (
        "def f():\n    run(lambda: s.finalize_abandoned_session_by_id(1))\n"
        "    s.finalize_abandoned_session_run(2)\n"
        "    s.finalize_abandoned_without_registered_runtime(3)\n",
        {"direct": {"f": 3}},
    ),
    (
        "async def g():\n    await a.cleanup_recovery_handoff(ledger=None)\n"
        "    await (a.cleanup_entrypoint_handoff if c else b)(stream=None)\n",
        {"nested": {"g": 1}, "unnamed": ["g (line 3)"]},
    ),
    (
        "async def h(ledger):\n    await a.cleanup_recovery_handoff(ledger=ledger)\n"
        "    await self._tasks.release_task(1, 2)\n",
        {"releases": {"h": 1}},
    ),
    (
        "async def k(ledger):\n"
        "    await a.cleanup_recovery_handoff(ledger=None if started else ledger)\n"
        "    await a.cleanup_recovery_handoff(ledger=None if started else None)\n"
        "    await a.cleanup_recovery_handoff(ledger=make())\n",
        {"conditional": {"k": 1}, "unrecognised": ["k (line 3)", "k (line 4)"]},
    ),
]


@pytest.mark.parametrize(
    ("source", "expected"),
    _NEW_EXITS,
    ids=["direct-abandonment", "handoff-without-owner", "raw-release", "conditional-ledgers"],
)
def test_recovery_and_dispatch_ratchets_see_new_exits(
    source: str, expected: dict[str, object]
) -> None:
    tree = ast.parse(source)
    ownership = _handoff_ownership(tree)
    assert _calls_by_function(tree, _ABANDONMENT_FINALIZERS) == expected.get("direct", {})
    assert _calls_by_function(tree, _CLAIM_RELEASES) == expected.get("releases", {})
    for kind in ("nested", "conditional"):
        assert ownership[kind] == collections.Counter(expected.get(kind, {})), kind
    for kind in ("unnamed", "unrecognised"):
        assert sorted(ownership[kind]) == expected.get(kind, []), kind
