"""Keep new run, recovery and task code on the single settlement path.

Settlement is owned by ``runtime/_run_settlement.py`` (admitted invocations,
recovered ones included), ``runtime/_run_exit.py`` (exit classification) and
``tasks/_task_run_settlement.py`` (claimed tasks). A new ``except GeneratorExit``
or ``except BaseExceptionGroup`` (or a catch-all ``except BaseException``) in the
engine, recovery or task modules is a hand-written exit handler: it settles only
the exits its author anticipated. So is a recovery exit that finalizes an
abandoned session itself instead of handing its ledger to the settlement, and a
dispatcher branch that releases a claim outside the claim exit table. The
allow-lists record the exits that remain today (see the settlement
implementation notes for why each stays) and may only shrink.
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
        ("RecoveryCoordinator._recover_tool_round_claimed", "BaseException"): 1,
        ("RecoveryCoordinator._recover_tool_round_claimed", "BaseExceptionGroup"): 1,
        ("RecoveryCoordinator._recovery_tool_hook_events", "BaseException"): 1,
        ("RecoveryCoordinator._resolve_tool_approval_owned", "BaseException"): 1,
        ("RecoveryCoordinator._resolve_user_input_owned", "BaseException"): 1,
        ("RecoveryCoordinator.continue_tool_approval_resolution", "GeneratorExit"): 1,
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

# Recovered invocations settle through ``RunSettlement``: the exit owner hands
# its ledger to the recovery handoff cleanup. Finalizing an abandoned session
# directly is left only where an admission or a recovery claim fails before its
# ledger exists (the claim's acknowledgement was lost, or its context could not
# be rebuilt).
_RECOVERY_MODULES = (
    "runtime/_provider_disposition_recovery.py",
    "runtime/_recovery_admission.py",
    "runtime/_recovery_coordinator.py",
)
_ALLOWED_DIRECT_ABANDONMENT: dict[str, dict[str, int]] = {
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
    for name, function in _functions(tree):
        for node in _own_nodes(function):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is not None and not set(_handler_type_names(node.type)) & {
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
