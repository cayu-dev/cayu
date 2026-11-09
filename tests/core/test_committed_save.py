"""A save cancelled after it may have committed is re-read before it fails."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest
from tests.core.test_settlement_ratchet import _called_name, _own_nodes

from cayu.runtime._committed_save import save_ended_cancelled, save_reconciling_cancellation

_ROOT = Path(__file__).resolve().parents[2] / "src" / "cayu"

# Durable saves whose child task must reconcile a cancelled outcome.
_COMMITTING_SAVES = frozenset(
    {
        "commit_and_fan_out",
        "complete_model_completion_stage",
        "promote_model_completion_stage",
        "apply_invocation_lifecycle_command",
        "fence_or_rebind_active_invocation",
        "transition_status_and_checkpoint",
    }
)


def _cancelled_with(*others: BaseException) -> BaseException:
    cancellation = asyncio.CancelledError()
    return BaseExceptionGroup("save", [cancellation, *others]) if others else cancellation


def _run(
    failure: BaseException | None,
    reconciled: object = "committed",
    reconcile_failure: BaseException | None = None,
):
    reconciles: list[str] = []

    async def save() -> object:
        if failure is not None:
            raise failure
        return "saved"

    async def reconcile() -> object:
        reconciles.append("reconcile")
        if reconcile_failure is not None:
            raise reconcile_failure
        return reconciled

    async def scenario() -> object:
        return await save_reconciling_cancellation(save, reconcile=reconcile)

    try:
        return asyncio.run(scenario()), reconciles
    except BaseException as error:
        return error, reconciles


def test_a_save_that_returns_is_not_reconciled() -> None:
    assert _run(None) == ("saved", [])


@pytest.mark.parametrize(
    "failure",
    [_cancelled_with(), _cancelled_with(OSError("cleanup failed"))],
    ids=["cancelled", "grouped"],
)
def test_a_cancelled_save_returns_what_committed(failure: BaseException) -> None:
    assert save_ended_cancelled(failure)
    assert _run(failure) == ("committed", ["reconcile"])


def test_a_cancelled_save_that_did_not_commit_keeps_its_failure() -> None:
    failure = _cancelled_with(OSError("cleanup failed"))
    assert _run(failure, reconciled=None) == (failure, ["reconcile"])


def test_a_failed_reconciliation_keeps_the_original_failure() -> None:
    failure = _cancelled_with(OSError("cleanup failed"))
    error, reconciles = _run(failure, reconcile_failure=RuntimeError("read failed"))
    assert error is failure and reconciles == ["reconcile"]
    assert "Reconciling the cancelled save also failed: RuntimeError." in failure.__notes__


def test_a_cancelled_reconciliation_keeps_the_original_failure() -> None:
    failure = _cancelled_with(OSError("cleanup failed"))
    error, reconciles = _run(failure, reconcile_failure=asyncio.CancelledError())
    assert error is failure and reconciles == ["reconcile"]
    assert "Reconciling the cancelled save also failed: CancelledError." in failure.__notes__


def test_a_grouped_cancellation_of_the_reconciliation_keeps_the_original_failure() -> None:
    failure = _cancelled_with(OSError("cleanup failed"))
    reconcile_failure = BaseExceptionGroup(
        "reconciliation", [asyncio.CancelledError(), OSError("read cleanup failed")]
    )
    error, reconciles = _run(failure, reconcile_failure=reconcile_failure)
    assert error is failure and reconciles == ["reconcile"]
    assert "Reconciling the cancelled save also failed: BaseExceptionGroup." in failure.__notes__


@pytest.mark.parametrize(
    "failure",
    [
        OSError("write failed"),
        ExceptionGroup("write", [OSError("write failed")]),
        _cancelled_with(KeyboardInterrupt()),
        _cancelled_with(GeneratorExit()),
    ],
    ids=["failure", "failure-group", "process-control", "generator-exit"],
)
def test_other_outcomes_are_not_reconciled(failure: BaseException) -> None:
    assert not save_ended_cancelled(failure)
    assert _run(failure) == (failure, [])


def _passes_own_task(call: ast.Call) -> bool:
    return any(keyword.arg == "own_task" for keyword in call.keywords)


# Child tasks that are spawned with a task-creating call.
_SPAWNS = frozenset({"create_task", "ensure_future", "shield"})
# Committing children whose caller reconciles any failure itself, by reading the
# durable outcome back. This list may only shrink.
_RECONCILED_BY_CALLER = {
    # The caller reloads its owned recovery claim after any failure.
    ("runtime/_recovery_ownership.py", "reserve_and_fence_incomplete_recovery"),
    # The interaction-transition publication replays its transition after any failure.
    ("runtime/_session_finalization.py", "apply_invocation_lifecycle_command"),
}


def _committing_children() -> tuple[set[tuple[str, str]], list[str]]:
    """Every spawned child whose body runs a durable save without reconciling it.

    A body is a call, or a local variable assigned one in the same function; it
    commits when it is a listed save, or a function of the same module that calls
    one directly. Calls passing ``own_task`` to a function without that
    parameter are returned too.
    """

    trees = {path: ast.parse(path.read_text()) for path in sorted(_ROOT.rglob("*.py"))}
    own_task_functions = {
        node.name
        for tree in trees.values()
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(arg.arg == "own_task" for arg in node.args.kwonlyargs + node.args.args)
    }
    unreconciled: set[tuple[str, str]] = set()
    invalid_own_task: list[str] = []
    for path, tree in trees.items():
        relative = str(path.relative_to(_ROOT))
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        saving_functions = {
            function.name
            for function in functions
            if any(
                isinstance(node, ast.Call)
                and _called_name(node) in _COMMITTING_SAVES
                and not _passes_own_task(node)
                for node in _own_nodes(function)
            )
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _passes_own_task(node)
                and _called_name(node) not in own_task_functions
            ):
                invalid_own_task.append(f"{relative}:{node.lineno}")
        for function in functions:
            nodes = _own_nodes(function)
            assigned = {
                node.targets[0].id: node.value
                for node in nodes
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
            }
            for node in nodes:
                if not (isinstance(node, ast.Call) and _called_name(node) in _SPAWNS and node.args):
                    continue
                body = node.args[0]
                if isinstance(body, ast.Name):
                    body = assigned.get(body.id, body)
                if not isinstance(body, ast.Call) or _passes_own_task(body):
                    continue
                name = _called_name(body)
                if name is not None and (name in _COMMITTING_SAVES or name in saving_functions):
                    unreconciled.add((relative, name))
    return unreconciled, invalid_own_task


def test_committing_child_tasks_reconcile_a_cancelled_save() -> None:
    """A child task that runs a durable save re-reads it when it ends cancelled.

    Its body is ``save_reconciling_cancellation``, or passes ``own_task=True`` to
    a save that reconciles through it, unless its caller reconciles every failure
    itself. The check is syntactic: it follows calls, local variables and
    same-module wrappers, not saves reached through other modules.
    """

    unreconciled, invalid_own_task = _committing_children()
    assert unreconciled <= _RECONCILED_BY_CALLER, (
        "Child tasks run durable saves without reconciling a cancelled save: "
        f"{sorted(unreconciled - _RECONCILED_BY_CALLER)}. Use save_reconciling_cancellation."
    )
    assert unreconciled >= _RECONCILED_BY_CALLER, (
        f"Shrink the caller-reconciled list: {sorted(_RECONCILED_BY_CALLER - unreconciled)}"
    )
    assert not invalid_own_task, f"own_task passed to a save without it: {invalid_own_task}"
