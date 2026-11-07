"""Durable transcript indexes and model-facing positions meet only in the transcript view.

After a model switch the model-facing transcript omits opaque-only assistant
rows, so its positions are not durable indexes. These checks keep new code from
converting between the two by hand: projections are built in
``cayu.runtime._transcript_view`` (or at a listed call that never indexes
durable data with the result), recorded input indexes in the Runtime come from the view, and recovery
requests carry the durable transcript rather than a list.

They are syntactic: they keep the known shapes out, not every possible
conversion.
"""

from __future__ import annotations

import ast
from collections import Counter
from functools import cache
from pathlib import Path

import cayu

SOURCE = Path(cayu.__file__).parent
PROJECTIONS = frozenset(
    {
        "project_portable_transcript",
        "project_portable_transcript_prefix",
        "_project_model_target_snapshot",
    }
)
PROJECTION_OWNERS = frozenset({"runtime/_model_target.py", "runtime/_transcript_view.py"})

# (module, enclosing function, projection) -> (number of calls, why they may project directly).
PROJECTION_ALLOWLIST = {
    (
        "runtime/_transcript.py",
        "_project_model_target_snapshot",
        "project_portable_transcript_prefix",
    ): (
        1,
        "Snapshot projection used by the listed preflight checks.",
    ),
    ("runtime/_session_fork.py", "fork_session", "project_portable_transcript"): (
        1,
        "Fork preflight of the copied transcript; positions are not kept.",
    ),
    ("runtime/_model_step_executor.py", "execute", "project_portable_transcript"): (
        1,
        "In-run fallback projects the in-memory list; positions map through "
        "projection_position_map.",
    ),
    (
        "runtime/_model_step_executor.py",
        "preflight_portable_model_material",
        "project_portable_transcript",
    ): (
        1,
        "Portability preflight; the result is discarded.",
    ),
    ("runtime/_session_engine.py", "resume", "project_portable_transcript"): (
        1,
        "Preflight of the new request messages, which carry no durable index.",
    ),
    ("runtime/_session_engine.py", "resume", "_project_model_target_snapshot"): (
        1,
        "Preflight that the stored projection cursor still fits; the result is discarded.",
    ),
    ("runtime/_session_engine.py", "_run_session_with_deadline", "project_portable_transcript"): (
        1,
        "In-run fallback projects the loop's model-facing list as the executor did.",
    ),
    ("runtime/_compaction/explicit.py", "run", "_project_model_target_snapshot"): (
        1,
        "Explicit compaction summarises the model-facing list and records a durable cursor.",
    ),
    (
        "sessions/_execution_profile_rules.py",
        "_validate_session_model_transition",
        "project_portable_transcript",
    ): (
        1,
        "Switch admission preflight; the result is discarded.",
    ),
}

# (module, enclosing function) -> (number of keywords, why the index is not from the view).
INPUT_INDEX_ALLOWLIST = {
    ("runtime/_session_engine.py", "run"): (
        2,
        "A new session has neither a projection nor retained-away rows, so its "
        "list length is its durable cursor.",
    ),
}


@cache
def _modules() -> tuple[tuple[str, ast.Module], ...]:
    return tuple(
        (path.relative_to(SOURCE).as_posix(), ast.parse(path.read_text(), str(path)))
        for path in sorted(SOURCE.rglob("*.py"))
    )


def _functions(tree: ast.AST):
    """Yield (enclosing function name, node) for every node in a module."""

    def visit(node: ast.AST, function: str):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        yield function, node
        for child in ast.iter_child_nodes(node):
            yield from visit(child, function)

    yield from visit(tree, "<module>")


def _aliases(tree: ast.Module) -> dict[str, str]:
    return {
        alias.asname: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname is not None
    }


def _callee(call: ast.Call, aliases: dict[str, str]) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return aliases.get(call.func.id, call.func.id)
    return None


def test_projections_are_built_by_the_view_or_a_listed_call() -> None:
    used: Counter[tuple[str, str, str]] = Counter()
    unlisted = []
    for module, tree in _modules():
        if module in PROJECTION_OWNERS:
            continue
        aliases = _aliases(tree)
        for function, node in _functions(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _callee(node, aliases)
            if name not in PROJECTIONS:
                continue
            site = (module, function, name)
            used[site] += 1
            if site not in PROJECTION_ALLOWLIST:
                unlisted.append(f"{module}:{node.lineno} {function}() calls {name}")
    assert not unlisted, (
        "Build model-facing transcripts with _transcript_view.model_transcript_view so each "
        f"row keeps its durable index: {unlisted}"
    )
    changed = {
        site: (used[site], count)
        for site, (count, _reason) in PROJECTION_ALLOWLIST.items()
        if used[site] != count
    }
    assert not changed, (
        "Listed projection calls changed (found, listed); list each new call with its "
        f"reason or remove stale entries: {changed}"
    )


def _from_tail_start(value: ast.AST) -> bool:
    return not any(isinstance(node, ast.BinOp) for node in ast.walk(value)) and any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tail_start"
        for node in ast.walk(value)
    )


def test_recorded_input_indexes_come_from_the_view() -> None:
    used: Counter[tuple[str, str]] = Counter()
    computed = []
    for module, tree in _modules():
        if not module.startswith("runtime/"):
            continue
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for function, node in _functions(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "message_start_index":
                    continue
                value = keyword.value
                if isinstance(value, ast.Name) and function in functions:
                    assigned = [
                        statement.value
                        for statement in ast.walk(functions[function])
                        if isinstance(statement, ast.Assign)
                        and any(
                            isinstance(target, ast.Name) and target.id == value.id
                            for target in statement.targets
                        )
                    ]
                    from_view = bool(assigned) and all(map(_from_tail_start, assigned))
                else:
                    from_view = _from_tail_start(value)
                if from_view:
                    continue
                used[(module, function)] += 1
                if (module, function) not in INPUT_INDEX_ALLOWLIST:
                    computed.append(f"{module}:{node.lineno} {function}()")
    assert not computed, (
        "A recorded input index is durable; take it from ModelTranscriptView.tail_start: "
        f"{computed}"
    )
    changed = {
        site: (used[site], count)
        for site, (count, _reason) in INPUT_INDEX_ALLOWLIST.items()
        if used[site] != count
    }
    assert not changed, f"Listed input indexes changed (found, listed): {changed}"


def test_recovery_requests_carry_the_durable_transcript() -> None:
    lists = [
        f"{module}:{node.lineno}"
        for module, tree in _modules()
        for _function, node in _functions(tree)
        if isinstance(node, ast.Call)
        and _callee(node, {}) == "RecoverySessionRunRequest"
        and any(keyword.arg == "messages" for keyword in node.keywords)
    ]
    assert not lists, (
        "Pass transcript=DurableTranscript.from_snapshot(...); the engine projects it for "
        f"the model once: {lists}"
    )
