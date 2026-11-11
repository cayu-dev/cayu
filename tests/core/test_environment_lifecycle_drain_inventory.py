"""Every registry of work that can outlive its caller is covered by the drain.

The inventory test finds every retained-work registry the lifecycle creates and
requires the drain to reference it, or to name what covers it instead, so a
drain cannot report settled while retained work still runs.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterable

from cayu.runtime._environment_lifecycle import EnvironmentLifecycle

# Constructors of registries whose entries are work that outlives its caller.
_RETAINED_WORK_TYPES = frozenset(
    {"BoundedInvocationOperationRegistry", "KeptTeardowns", "RetainedEnvironmentEndWatches"}
)
# Annotations of containers that hold such work, or the owners that retain it.
_RETAINED_WORK_ANNOTATIONS = ("asyncio.Task", "_ActiveEnvironmentSetup")
# The methods that make up the drain.
_DRAIN = ("drain_retained_cleanups", "_settling_retained_work")
# Registries the drain covers through something else, and what.
_COVERED_ELSEWHERE = {
    # An observation that outlives its caller is held behind the environment's
    # workspace mutation fence (``fail_closed`` with its settlement probe). The
    # drain waits for that fence through the retained owner, and the
    # environment-end watch releases it once the environment is gone.
    "_final_workspace_observation_operations": "workspace mutation fence",
}


def _retained_work_attributes(init: ast.FunctionDef) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(init):
        if isinstance(node, ast.AnnAssign):
            targets, annotation, value = [node.target], ast.unparse(node.annotation), node.value
        elif isinstance(node, ast.Assign):
            targets, annotation, value = node.targets, "", node.value
        else:
            continue
        constructed = (
            value.func.id
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            else None
        )
        if constructed not in _RETAINED_WORK_TYPES and not any(
            marker in annotation for marker in _RETAINED_WORK_ANNOTATIONS
        ):
            continue
        found.update(
            target.attr
            for target in targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        )
    return found


def _referenced_attributes(functions: Iterable[ast.AST]) -> set[str]:
    return {
        node.attr
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def _undrained(source: str) -> set[str]:
    [class_node] = ast.parse(textwrap.dedent(source)).body
    assert isinstance(class_node, ast.ClassDef)
    methods = {
        node.name: node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    init = methods["__init__"]
    assert isinstance(init, ast.FunctionDef)
    drain = [methods[name] for name in _DRAIN if name in methods]
    return _retained_work_attributes(init) - _referenced_attributes(drain)


def test_the_drain_covers_every_retained_work_registry() -> None:
    undrained = _undrained(inspect.getsource(EnvironmentLifecycle))
    assert undrained == set(_COVERED_ELSEWHERE), (
        "EnvironmentLifecycle registries of retained work that drain_retained_cleanups "
        "never waits for; add them to _settling_retained_work or the drain's owner checks: "
        f"{sorted(undrained - set(_COVERED_ELSEWHERE))}; no longer outside the drain: "
        f"{sorted(set(_COVERED_ELSEWHERE) - undrained)}"
    )


def test_an_outlived_final_observation_is_held_behind_the_workspace_fence() -> None:
    """What ``_COVERED_ELSEWHERE`` relies on: every path that leaves the
    observation running closes the fence on that exact task."""

    from cayu.runtime import _environment_lifecycle as lifecycle

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(lifecycle._observe_final_workspace_revision))
    )
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    tracked = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "track"
        and [ast.unparse(arg) for arg in call.args] == ["observation_task"]
    ]
    held = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "fail_closed"
        and ast.unparse(call.args[0])
        == "workspace_mutation_task_settlement_probe(observation_task)"
    ]
    # The cancellation, unproven-settlement, pending-cancellation and timeout paths.
    assert len(tracked) == 1
    assert len(held) == 4


def test_the_inventory_sees_a_registry_left_out_of_the_drain() -> None:
    source = """
        class Lifecycle:
            def __init__(self):
                self._kept = KeptTeardowns()
                self._tasks: dict[str, asyncio.Task[None]] = {}
                self._names: set[str] = set()
                self._new = BoundedInvocationOperationRegistry(max_operations=1)

            def _settling_retained_work(self):
                return (*self._kept.settling(),)

            async def drain_retained_cleanups(self):
                return not self._tasks and not self._settling_retained_work()
    """
    assert _undrained(source) == {"_new"}
