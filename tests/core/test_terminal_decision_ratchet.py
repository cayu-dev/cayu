"""Keep every new terminal interaction transition on its durable decision.

A transition that ends an interaction COMPLETED, FAILED or INTERRUPTED commits
before the events and the fence release it owes. Passing ``owed_outcome`` makes
the transition install, in the same store transaction, the decision that the
run, settlement and recovery all finish those writes from. A terminal
transition published without one leaves a crash or close window between its
commit and its owed writes that nothing repairs.

The allow-list records the transitions that still publish without a decision:
recovery and abandonment paths that finish their own evidence under a recovery
claim, transitions whose interaction is already terminal, and outcomes that keep
an earlier election of their own. It may only shrink.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "src" / "cayu"
_TRANSITIONS = frozenset(
    {"publish_interaction_transition", "publish_sibling_interaction_transition"}
)
_NON_TERMINAL_STATUSES = frozenset({"RUNNING", "PENDING", "INTERRUPTING"})

# (function qualname) -> count of terminal transitions published without a decision.
_ALLOWED_UNDECIDED: dict[str, dict[str, int]] = {
    "runtime/_session_engine.py": {
        # Admission and setup failures before the turn's body runs.
        "SessionEngine.resume": 2,
        "SessionEngine.run": 1,
        # A completion that must finalize its workspace first commits RUNNING.
        "SessionEngine._complete_session_if_no_queued_messages": 1,
    },
    "runtime/_session_finalization.py": {
        # Finishes an interaction a sibling already settled.
        "SessionFinalization._emit_turn_completed": 1,
        # Abandonment and provider-operation recovery publish under their own claim.
        "SessionFinalization.complete_abandoned_turn": 1,
        "SessionFinalization.fail_provider_operation_resolution": 1,
        # The recovery-claim replay inside the transition publisher itself.
        "SessionFinalization.publish_interaction_transition": 1,
        # Forwards its caller's outcome; each caller is counted at its own call.
        "SessionFinalization.publish_sibling_interaction_transition": 1,
        # A limit stop publishes no TURN_COMPLETED when no turn is being tracked.
        "SessionFinalization.stop_session_for_limit_reached": 2,
    },
    "runtime/_session_outcome_publication.py": {
        # A workspace finalization failure after the completion committed.
        "SessionOutcomePublication.complete_session": 1,
        # A linked task's failure keeps its earlier election or the event
        # identities its task carries; an ambiguous provider start is interrupted
        # without a turn of its own.
        "SessionOutcomePublication.fail": 3,
    },
    "runtime/_session_recovery.py": {
        "SessionRecovery._publish_durable_recovery_interaction_transition": 1,
        "SessionRecovery._publish_durable_recovery_interaction_transition"
        ".reconcile_terminal_interaction": 1,
    },
    "runtime/_session_task_lifecycle.py": {
        # Replays a linked task failure from its earlier election.
        "SessionTaskLifecycle.replay_runtime_task_failure_if_needed": 2,
    },
}


def _functions(tree: ast.AST) -> list[tuple[str, ast.AST]]:
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
    nodes: list[ast.AST] = []
    pending = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        nodes.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return nodes


def _is_terminal_transition(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _TRANSITIONS
    ):
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    # Only an outcome built at the call is a decision; a name or a conditional
    # can be None, and None publishes without one.
    if isinstance(keywords.get("owed_outcome"), ast.Call | ast.Await):
        return False
    status = keywords.get("to_status")
    # A status chosen at run time may be terminal; only a literal live status is not.
    return not (
        isinstance(status, ast.Attribute)
        and isinstance(status.value, ast.Name)
        and status.value.id == "SessionStatus"
        and status.attr in _NON_TERMINAL_STATUSES
    )


def _undecided_transitions() -> dict[str, collections.Counter[str]]:
    found: dict[str, collections.Counter[str]] = {}
    for path in sorted((_ROOT / "runtime").rglob("*.py")):
        tree = ast.parse(path.read_text())
        counts: collections.Counter[str] = collections.Counter()
        for name, function in _functions(tree):
            counts[name] += sum(_is_terminal_transition(node) for node in _own_nodes(function))
        counts = +counts
        if counts:
            found[str(path.relative_to(_ROOT))] = counts
    return found


def test_no_new_terminal_transition_without_its_decision() -> None:
    found = _undecided_transitions()
    grown = {
        (relative, name): count
        for relative, counts in found.items()
        for name, count in counts.items()
        if count > _ALLOWED_UNDECIDED.get(relative, {}).get(name, 0)
    }
    assert not grown, (
        f"Terminal interaction transitions without a durable decision: {grown}. Pass "
        "owed_outcome so the transition installs what its outcome still owes."
    )
    shrunk = {
        (relative, name): count
        for relative, allowed in _ALLOWED_UNDECIDED.items()
        for name, count in allowed.items()
        if found.get(relative, collections.Counter())[name] < count
    }
    assert not shrunk, f"Transitions now carry their decision; shrink the allow-list: {shrunk}"


@pytest.mark.parametrize(
    ("source", "terminal"),
    [
        ("store.publish_sibling_interaction_transition(to_status=SessionStatus.FAILED)", True),
        ("store.publish_interaction_transition(to_status=status)", True),
        (
            "store.publish_sibling_interaction_transition("
            "to_status=SessionStatus.INTERRUPTED, owed_outcome=OwedOutcome(context, event))",
            False,
        ),
        ("store.publish_interaction_transition(to_status=status, owed_outcome=None)", True),
        ("store.publish_interaction_transition(to_status=status, owed_outcome=owed)", True),
        (
            "store.publish_interaction_transition("
            "to_status=status, owed_outcome=None if done else outcome(context))",
            True,
        ),
        ("store.publish_interaction_transition(to_status=SessionStatus.RUNNING)", False),
    ],
)
def test_the_ratchet_recognises_terminal_transitions(source: str, terminal: bool) -> None:
    assert _is_terminal_transition(ast.parse(source, mode="eval").body) is terminal
