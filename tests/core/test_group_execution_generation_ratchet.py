"""Task-group executions inside ``src/cayu`` are matched by claim generation.

A worker name and a start time do not identify a group execution: a later
claim may reuse both, for example a continuation under the original worker
name. This ratchet fails on

- a settlement identity (``settled_execution=`` or the argument of
  ``_settle_task_group_execution``) that is not the whole claim: a claim helper's
  result, a copy of a claim, or a parameter passed through unchanged;
- a comparison of an execution obligation, or of its ``worker_id`` or
  ``started_at`` (alone or in a tuple), outside ``claim_holds_execution``, the
  one place that also compares the generation. A presence check against
  ``None`` is not a match.

The allow-list holds the remaining start-time comparisons, each about a row or
an invocation that holds no claim. It may only shrink: a listed site that no
longer offends fails too, until it is removed from the list.
"""

from __future__ import annotations

import ast
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tests.core.task_invocation_fixtures import unattributed_task_invocation

from cayu.tasks._cancellation import _expired_dispatched_task_cancellation
from cayu.tasks._group_quiescence import _keeps_claim, claim_holds_execution
from cayu.tasks._lifecycle import work_attempt_claim_execution
from cayu.tasks.groups import TaskGroupExecutionObligation
from cayu.tasks.records import Task, TaskStatus

_SOURCE = Path(__file__).resolve().parents[2] / "src" / "cayu"
_EVIDENCE_FIELDS = frozenset({"worker_id", "started_at"})
_AUTHORITY = "cayu/tasks/_group_quiescence.py::claim_holds_execution"
# Helpers that return the claim (a task row) whose execution a settlement names.
_CLAIM_HELPERS = frozenset(
    {"claimed_execution", "recovered_completion_execution", "work_attempt_claim_execution"}
)
_CLAIM_COPY = "copy_task"
_SETTLEMENT_METHOD = "_settle_task_group_execution"
# "path::function" -> number of remaining evidence comparisons, with why each stays.
_ALLOWED: dict[str, int] = {
    # A row without a worker holds no claim: releasing a holder advances the
    # generation, so the row keeps its last execution by dispatch marker.
    "cayu/tasks/_group_quiescence.py::_observes_execution": 1,
    # An invocation observation must name the attached row's dispatch marker. It
    # grants nothing to a worker-owned execution; an ownerless one is authenticated
    # by its invocation obligation.
    "cayu/tasks/_group_quiescence.py::plan_group_graph_transition": 1,
    # Resuming a held, ownerless task requires its last dispatch's settlement.
    "cayu/tasks/_group_quiescence.py::_fresh_resumed_execution": 1,
}


def _is_execution_name(name: str) -> bool:
    # ``settled_execution`` is the presented claim (a task row), not an obligation.
    return "execution" in name and not name.endswith("settled_execution")


class _Scanner(ast.NodeVisitor):
    def __init__(self, relative: str) -> None:
        self.relative = relative
        self.scope: list[str] = []
        self.execution_names: list[set[str]] = [set()]
        self.claim_names: list[set[str]] = [set()]
        self.comparisons: Counter[str] = Counter()
        self.inline_settlements: list[str] = []

    def _site(self) -> str:
        return f"{self.relative}::{'.'.join(self.scope) or '<module>'}"

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.execution_names.append(set())
        # A parameter passed through unchanged is the caller's claim.
        self.claim_names.append(
            set()
            if isinstance(node, ast.ClassDef)
            else {argument.arg for argument in ast.walk(node.args) if isinstance(argument, ast.arg)}
        )
        self.generic_visit(node)
        self.claim_names.pop()
        self.execution_names.pop()
        self.scope.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # ``for item in snapshot.quiescence.executions`` binds obligations.
        if isinstance(node.iter, ast.Attribute) and node.iter.attr == "executions":
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    self.execution_names[-1].add(target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.visit_comprehension(
            ast.comprehension(target=node.target, iter=node.iter, ifs=[], is_async=0)
        )
        for child in (*node.body, *node.orelse):
            self.visit(child)

    def visit_Assign(self, node: ast.Assign) -> None:
        # ``prior = executions.get(task_id)`` binds an obligation under any name.
        if self._from_executions(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.execution_names[-1].add(target.id)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if self._is_claim(node.value):
                    self.claim_names[-1].add(target.id)
                else:
                    self.claim_names[-1].discard(target.id)
        self.generic_visit(node)

    def _is_claim(self, node: ast.expr) -> bool:
        """The whole claim: a helper's result, a copy of a claim, or a passed-through name."""
        if isinstance(node, ast.Name):
            return node.id in self.claim_names[-1]
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return False
        if node.func.id in _CLAIM_HELPERS:
            return True
        return (
            node.func.id == _CLAIM_COPY
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name | ast.Attribute)
        )

    def _is_execution(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return _is_execution_name(node.id) or any(
                node.id in names for names in self.execution_names
            )
        return self._from_executions(node)

    @staticmethod
    def _from_executions(node: ast.expr) -> bool:
        """``snapshot.quiescence.executions[0]`` or ``executions.get(task_id)``."""
        if isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        return (isinstance(node, ast.Attribute) and node.attr == "executions") or (
            isinstance(node, ast.Name) and node.id == "executions"
        )

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        presence = any(
            isinstance(operand, ast.Constant) and operand.value is None for operand in operands
        )
        if not presence and any(
            self._is_evidence(element)
            for operand in operands
            for element in (operand.elts if isinstance(operand, ast.Tuple) else (operand,))
        ):
            self.comparisons[self._site()] += 1
        self.generic_visit(node)

    def _is_evidence(self, node: ast.expr) -> bool:
        """An obligation itself, or its worker name or start time."""
        if isinstance(node, ast.Attribute) and node.attr in _EVIDENCE_FIELDS:
            return self._is_execution(node.value)
        # A whole obligation, only where it is known to be bound from ``executions``.
        if isinstance(node, ast.Name):
            return any(node.id in names for names in self.execution_names)
        return self._is_execution(node)

    def visit_Call(self, node: ast.Call) -> None:
        method = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if method == _SETTLEMENT_METHOD and not (node.args and self._is_claim(node.args[0])):
            self.inline_settlements.append(f"{self._site()}:{node.lineno}")
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg == "settled_execution" and not self._is_claim(node.value):
            self.inline_settlements.append(f"{self._site()}:{node.value.lineno}")
        self.generic_visit(node)


def _scan() -> _Scanner:
    combined = _Scanner("")
    for path in sorted(_SOURCE.rglob("*.py")):
        relative = path.relative_to(_SOURCE.parent).as_posix()
        scanner = _Scanner(relative)
        scanner.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        combined.comparisons.update(scanner.comparisons)
        combined.inline_settlements.extend(scanner.inline_settlements)
    return combined


def test_group_execution_settlements_present_the_whole_claim():
    assert _scan().inline_settlements == [], (
        "Present a group execution settlement as the whole claim (a task row, with "
        "its claim generation) from a claim helper, a copy of a claim or a passed-"
        "through parameter, never one rebuilt from its worker name and start time."
    )


def test_group_executions_are_matched_by_generation_only_in_one_place():
    comparisons = dict(_scan().comparisons)
    assert comparisons.pop(_AUTHORITY, 0) >= 1, "The generation authority lost its evidence check."
    assert comparisons == _ALLOWED, (
        "Match a group execution with claim_holds_execution, which compares the "
        "claim generation; a worker name or start time is evidence only. Shrink the "
        "allow-list when a listed site no longer offends."
    )


_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _claim(*, generation: int, worker_id: str = "worker") -> Task:
    now = _NOW
    return Task(
        id="member",
        type="member",
        status=TaskStatus.CLAIMED,
        worker_id=worker_id,
        lease_expires_at=now,
        started_at=now,
        claim_generation=generation,
        created_at=now,
        updated_at=now,
        invocation=unattributed_task_invocation(),
    )


def test_claim_generation_is_the_authority_for_a_group_execution():
    started = _claim(generation=3)
    assert started.started_at is not None
    execution = TaskGroupExecutionObligation(
        task_id="member", claim_generation=3, worker_id="worker", started_at=started.started_at
    )
    assert claim_holds_execution(execution, started)
    # Reclaim's cancellation fence keeps the holder and advances the generation by one.
    fenced = _expired_dispatched_task_cancellation(started, updated_at=started.started_at)
    assert claim_holds_execution(execution, fenced.model_copy(update={"claim_generation": 4}))
    assert not claim_holds_execution(execution, started.model_copy(update={"claim_generation": 4}))
    # The same worker name and start time under a later claim is another execution.
    assert not claim_holds_execution(execution, _claim(generation=5))
    assert not claim_holds_execution(execution, _claim(generation=2))
    assert not claim_holds_execution(execution, _claim(generation=3, worker_id="other"))


def test_a_fenced_loser_keeps_its_claim_only_under_the_same_generation():
    held = _claim(generation=3)
    assert _keeps_claim(held, held.model_copy(update={"lease_expires_at": None}))
    fenced = _expired_dispatched_task_cancellation(held, updated_at=_NOW)
    assert _keeps_claim(held, fenced.model_copy(update={"claim_generation": 4}))
    # A later claim under the same worker name cannot resume a fenced loser.
    assert not _keeps_claim(held, _claim(generation=5))
    released = held.model_copy(update={"worker_id": None, "lease_expires_at": None})
    assert not _keeps_claim(released, held)
    assert _keeps_claim(released, released)


# Verified work releases a task's worker in these planners. A work-attempt
# settlement presents the row's generation as the claim that started the execution
# (``work_attempt_claim_execution``), which holds only while they never advance it.
_VERIFIED_RELEASE_PLANNERS = (
    "cayu/tasks/_verified_work_policy.py",
    "cayu/runtime/_work_attempt_lifecycle_policy.py",
)


def test_verified_work_releases_keep_the_claim_generation():
    for relative in _VERIFIED_RELEASE_PLANNERS:
        tree = ast.parse((_SOURCE.parent / relative).read_text(encoding="utf-8"))
        assert not [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "claim_generation"
        ], f"{relative} changes the claim generation; revisit work_attempt_claim_execution."
    started = _claim(generation=3)
    released = started.model_copy(update={"worker_id": None, "lease_expires_at": None})
    claim = work_attempt_claim_execution(released, "worker")
    assert claim is not None
    assert (claim.worker_id, claim.claim_generation) == ("worker", 3)
    execution = TaskGroupExecutionObligation(
        task_id="member", claim_generation=3, worker_id="worker", started_at=_NOW
    )
    assert claim_holds_execution(execution, claim)
    assert (
        work_attempt_claim_execution(released.model_copy(update={"started_at": None}), "w") is None
    )
