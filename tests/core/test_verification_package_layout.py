"""Verification composition preserves public parts and lower-layer boundaries."""

from __future__ import annotations

import ast
import importlib
import os
import pickle
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

import cayu
from cayu.verification._exports import EXPORTS

_PUBLIC_MODULES = (
    "completion_verifiers",
    "completion_result_resolvers",
    "verified_task_worker",
)
_COMPATIBILITY = {f"cayu.runtime.{name}": f"cayu.verification.{name}" for name in _PUBLIC_MODULES}


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_public_verification_imports_and_old_pickle_globals_share_canonical_types(module_name):
    canonical = importlib.import_module(f"cayu.verification.{module_name}")
    legacy = importlib.import_module(f"cayu.runtime.{module_name}")
    packages = [
        importlib.import_module(name) for name in ("cayu", "cayu.runtime", "cayu.verification")
    ]
    for name, (module, symbol) in EXPORTS.items():
        if module != canonical.__name__:
            continue
        value = getattr(canonical, symbol)
        assert getattr(legacy, name) is value
        assert all(
            getattr(package, name) is value and name in package.__all__ for package in packages
        )
        if isinstance(value, type):
            assert value.__module__ == canonical.__name__
            assert pickle.loads(f"c{legacy.__name__}\n{name}\n.".encode()) is value
    for name, value in vars(canonical).items():
        if (
            name.startswith("copy_completion_")
            and getattr(value, "__module__", None) == canonical.__name__
        ):
            assert getattr(legacy, name) is value


def test_old_pickled_requests_preserve_their_serialized_authority():
    from cayu.verification import (
        CompletionResultResolutionRequest,
        CompletionVerifierExecutionRequest,
    )

    requests = (
        CompletionVerifierExecutionRequest(
            proposal_id="proposal", claim_id="claim", decision_id="decision", worker_id="worker"
        ),
        CompletionResultResolutionRequest(
            task_id="task", decision_id="decision", idempotency_key="apply"
        ),
    )
    for request in requests:
        legacy_pickle = pickle.dumps(request, protocol=0).replace(
            b"cayu.verification.", b"cayu.runtime."
        )
        restored = pickle.loads(legacy_pickle)
        assert type(restored) is type(request)
        assert restored.model_dump(mode="json") == request.model_dump(mode="json")
        assert pickle.loads(pickle.dumps(request)) == request


@pytest.mark.parametrize(
    "first_module",
    (
        "cayu.verification",
        "cayu.verification.completion_verifiers",
        "cayu.verification.completion_result_resolvers",
        "cayu.verification.verified_task_worker",
        "cayu.runtime.verified_task_worker",
        "cayu.applications",
    ),
)
def test_verification_import_order_and_optional_construction(first_module):
    script = """
import importlib
import sys
from typing import get_type_hints

first = importlib.import_module(sys.argv[1])
if sys.argv[1] == "cayu.verification":
    assert "cayu.applications" not in sys.modules
    assert "cayu.verification.verified_task_worker" not in sys.modules
    assert "cayu.verification._verified_completion" not in sys.modules
from cayu import CayuApp
from cayu.verification import VerifiedTaskWorker, CompletionVerifierExecutionRequest
from cayu.runtime.verified_task_worker import VerifiedTaskWorker as LegacyWorker
from cayu.verification._verified_completion import VerifiedCompletionCoordinator, VerifiedTaskDecisionExecution
assert LegacyWorker is VerifiedTaskWorker
assert get_type_hints(VerifiedTaskDecisionExecution)["_coordinator"] is VerifiedCompletionCoordinator
assert get_type_hints(CompletionVerifierExecutionRequest)["proposal_id"] is str
app = CayuApp(enable_logging=False)
assert app.task_store is None
assert app._verified_completion.resolver._application_coordinator is app._verified_completion.application
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _verification_target(module: str) -> bool:
    return module == "cayu.verification" or module.startswith("cayu.verification.")


_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSION_SCOPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _parameters(args: ast.arguments):
    yield from (*args.posonlyargs, *args.args, *args.kwonlyargs)
    yield from (arg for arg in (args.vararg, args.kwarg) if arg is not None)


def _scope_headers(node: ast.AST):
    """Expressions evaluated in the scope containing a definition or comprehension."""
    if isinstance(node, _FUNCTION_SCOPES):
        yield from getattr(node, "decorator_list", ())
        yield from node.args.defaults
        yield from (value for value in node.args.kw_defaults if value is not None)
        yield from (arg.annotation for arg in _parameters(node.args) if arg.annotation is not None)
        if getattr(node, "returns", None) is not None:
            yield node.returns
    elif isinstance(node, ast.ClassDef):
        yield from node.decorator_list
        yield from node.bases
        yield from (keyword.value for keyword in node.keywords)
    elif isinstance(node, _COMPREHENSION_SCOPES):
        yield node.generators[0].iter


def _scope_body(tree: ast.AST):
    if isinstance(tree, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        yield from tree.body
    elif isinstance(tree, ast.Lambda):
        yield tree.body
    elif isinstance(tree, _COMPREHENSION_SCOPES):
        if isinstance(tree, ast.DictComp):
            yield tree.key
            yield tree.value
        else:
            yield tree.elt
        for index, generator in enumerate(tree.generators):
            yield generator.target
            yield from generator.ifs
            if index:
                yield generator.iter
    else:
        yield from ast.iter_child_nodes(tree)


def _attribute_targets(tree: ast.AST, package: str, inherited: dict[str, str] | None = None):
    """Resolve direct import bindings without leaking aliases between lexical scopes."""
    scopes = (*_FUNCTION_SCOPES, ast.ClassDef, *_COMPREHENSION_SCOPES)
    nodes = []
    pending = list(_scope_body(tree))
    while pending:
        node = pending.pop()
        nodes.append(node)
        if isinstance(node, scopes):
            pending.extend(_scope_headers(node))
        else:
            pending.extend(ast.iter_child_nodes(node))

    bindings = dict(inherited or {})
    if isinstance(tree, _FUNCTION_SCOPES):
        for arg in _parameters(tree.args):
            bindings.pop(arg.arg, None)
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            bindings.pop(node.id, None)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bindings.pop(node.name, None)
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                bindings[alias.asname or name] = alias.name if alias.asname else name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = resolve_name("." * node.level + base, package)
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{base}.{alias.name}"

    for node in nodes:
        if isinstance(node, ast.Attribute):
            parts = []
            value = node
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name) and value.id in bindings:
                yield ".".join((bindings[value.id], *reversed(parts)))
        elif isinstance(node, scopes):
            # Nested bodies do not close over a class namespace; their headers do.
            outer = inherited if isinstance(tree, ast.ClassDef) else bindings
            yield from _attribute_targets(node, package, outer)


def _targets(tree: ast.Module, package: str):
    yield from _attribute_targets(tree, package)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = resolve_name("." * node.level + base, package)
            yield base
            for alias in node.names:
                yield f"{base}.{alias.name}"
                if base in {"cayu", "cayu.runtime"} and alias.name in EXPORTS:
                    yield EXPORTS[alias.name][0]
                if base in {"cayu", "cayu.runtime"} and alias.name == "*":
                    yield "cayu.verification"
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            assigned = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "EXPORTS" for target in assigned):
                for module, _symbol in ast.literal_eval(node.value).values():
                    yield module


def _canonical_target(target: str) -> str:
    for package in ("cayu.runtime", "cayu"):
        if target.startswith(package + "."):
            name, _, suffix = target[len(package) + 1 :].partition(".")
            if name in EXPORTS:
                module, symbol = EXPORTS[name]
                return ".".join(part for part in (module, symbol, suffix) if part)
    for legacy, canonical in _COMPATIBILITY.items():
        if target == legacy or target.startswith(legacy + "."):
            return canonical + target[len(legacy) :]
    return target


@pytest.mark.parametrize(
    "source, package",
    (
        ("import cayu.verification as v", "cayu.runtime"),
        ("from ..verification import VerifiedTaskWorker", "cayu.runtime"),
        ("from cayu import VerifiedTaskWorker as Worker", "cayu.tasks"),
        ("from . import DeterministicCompletionVerifier", "cayu.runtime"),
        ("from cayu.runtime import completion_verifiers", "cayu.storage"),
        ("import cayu as c\nWorker = c.VerifiedTaskWorker", "cayu.tasks"),
        (
            "import cayu.runtime as r\nVerifier = r.DeterministicCompletionVerifier",
            "cayu.storage",
        ),
        ("from cayu import runtime as r\nResolver = r.CompletionResultResolver", "cayu.tasks"),
        ("import cayu.runtime\nWorker = cayu.runtime.VerifiedTaskWorker", "cayu.tasks"),
        ("from .. import runtime as r\nWorker = r.VerifiedTaskWorker", "cayu.tasks"),
        ("import cayu\nWorker = cayu.VerifiedTaskWorker", "cayu.tasks"),
        ("import cayu as c\ndef worker():\n    return c.VerifiedTaskWorker", "cayu.tasks"),
        ("def worker():\n    import cayu as c\n    return c.VerifiedTaskWorker", "cayu.tasks"),
        (
            "from cayu.runtime.completion_result_resolvers import CompletionResultResolver",
            "cayu.tasks",
        ),
        (
            "EXPORTS = {'Worker': ('cayu.verification.verified_task_worker', 'VerifiedTaskWorker')}",
            "cayu.runtime",
        ),
        (
            "EXPORTS: dict[str, tuple[str, str]] = {'Owner': ('cayu.verification._verified_completion', 'VerifiedCompletionCoordinator')}",
            "cayu.runtime",
        ),
    ),
)
def test_boundary_recognizes_relative_aliased_and_lazy_imports(source, package):
    assert any(
        _verification_target(_canonical_target(target))
        for target in _targets(ast.parse(source), package)
    )


@pytest.mark.parametrize(
    "source",
    (
        "import unrelated as c\nWorker = c.VerifiedTaskWorker",
        "import cayu as c\nApp = c.CayuApp",
        "import cayu as c\ndef worker(c):\n    return c.VerifiedTaskWorker",
        "import cayu as c\ndef worker():\n    c = unrelated\n    return c.VerifiedTaskWorker",
        "def first():\n    import cayu as c\ndef second(c):\n    return c.VerifiedTaskWorker",
        "class First:\n    import cayu as c\n    def worker(self, c):\n        return c.VerifiedTaskWorker",
    ),
)
def test_boundary_does_not_confuse_unrelated_or_local_names_with_imports(source):
    assert not any(
        _verification_target(_canonical_target(target))
        for target in _targets(ast.parse(source), "cayu.tasks")
    )


@pytest.mark.parametrize(
    "source, expected",
    (
        ("def factory(c=c.VerifiedTaskWorker):\n    return c", True),
        ("async def factory(c=c.VerifiedTaskWorker):\n    return c", True),
        ("factory = lambda c=c.VerifiedTaskWorker: c", True),
        ("def factory(value=lambda c: c):\n    return c.VerifiedTaskWorker", True),
        ("def factory(*, c=c.VerifiedTaskWorker):\n    return c", True),
        ("def use(c: c.VerifiedTaskWorker):\n    return c", True),
        ("def use(c) -> c.VerifiedTaskWorker:\n    return c", True),
        ("@c.VerifiedTaskWorker\ndef use(c):\n    return c", True),
        ("class Custom(c.VerifiedTaskWorker):\n    c = None", True),
        ("class Custom(metaclass=c.VerifiedTaskWorker):\n    c = None", True),
        ("class Custom:\n    import cayu as v\n    def use(v=v.VerifiedTaskWorker): pass", True),
        (
            "class Custom:\n    import cayu as v\n    def use(self): return v.VerifiedTaskWorker",
            False,
        ),
        (
            "def factory(modules):\n    local = [c for c in modules]\n    return c.VerifiedTaskWorker",
            True,
        ),
        ("workers = [c for c in c.VerifiedTaskWorker]", True),
        ("workers = [c.VerifiedTaskWorker for c in modules]", False),
        ("workers = {c.VerifiedTaskWorker for c in modules}", False),
        ("workers = {c: c.VerifiedTaskWorker for c in modules}", False),
        ("workers = (c.VerifiedTaskWorker for c in modules)", False),
        ("workers = [item for c in modules for item in c.VerifiedTaskWorker]", False),
        ("workers = [item for item in modules if c.VerifiedTaskWorker]", True),
        ("workers = [[c.VerifiedTaskWorker for item in values] for c in modules]", False),
        (
            "workers = [[c.VerifiedTaskWorker for c in values] for item in c.VerifiedTaskWorker]",
            True,
        ),
        (
            "class Custom:\n    import cayu as v\n    workers = [v for v in v.VerifiedTaskWorker]",
            True,
        ),
        (
            "class Custom:\n    import cayu as v\n    workers = [v.VerifiedTaskWorker for v in modules]",
            False,
        ),
    ),
)
def test_boundary_resolves_headers_and_comprehensions_in_their_evaluation_scope(source, expected):
    targets = _targets(ast.parse("import cayu as c\n" + source), "cayu.tasks")
    assert any(_verification_target(_canonical_target(target)) for target in targets) is expected


def test_verification_dependencies_are_limited_to_composition_and_compatibility():
    root = Path(cayu.__file__).resolve().parent
    compatibility_files = {
        "cayu.__init__",
        "cayu._exports",
        "cayu.runtime.__init__",
        "cayu.runtime._exports",
    }
    composition = ("cayu.applications", "cayu.server", "cayu.cli", "cayu.coding_products")
    for path in sorted((*root.rglob("*.py"), *root.rglob("*.pyi"))):
        module = "cayu." + ".".join(path.relative_to(root).with_suffix("").parts)
        package = module.rpartition(".")[0]
        targets = set(_targets(ast.parse(path.read_text()), package))
        verification = {
            target for target in targets if _verification_target(_canonical_target(target))
        }
        if module.startswith("cayu.verification."):
            assert all(target == _canonical_target(target) for target in verification), (
                f"{module}: use canonical imports"
            )
        elif module in _COMPATIBILITY:
            # Only the matching public facade is permitted, never private orchestration.
            assert all(
                target == _COMPATIBILITY[module] or target.startswith(_COMPATIBILITY[module] + ".")
                for target in verification
            ), module
        elif module in compatibility_files:
            allowed = {owner for owner, _symbol in EXPORTS.values()}
            assert all(
                any(target == owner or target.startswith(owner + ".") for owner in allowed)
                for target in verification
            ), module
        elif not any(module == owner or module.startswith(owner + ".") for owner in composition):
            assert not verification, (
                f"{module}: verification belongs to composition: {sorted(verification)}"
            )
