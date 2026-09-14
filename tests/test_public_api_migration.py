"""Canonical API and dependency boundaries for the public concept migration."""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MOVES = json.loads((_ROOT / "docs/public-api-migration.json").read_text())
_PACKAGES = json.loads((_ROOT / "docs/public-api-packages.json").read_text())
_PACKAGE_CONVERSIONS = {"cayu.memory", "cayu.testing"}


@pytest.mark.parametrize("module_name", sorted(set(_MOVES.values())))
def test_canonical_implementations_are_importable(module_name):
    script = f"""
import sys
import importlib
sys.path.insert(0, {str(_ROOT / "src")!r})
module = importlib.import_module({module_name!r})
assert module.__name__ == {module_name!r}
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=60)


@pytest.mark.parametrize("package_name", _PACKAGES)
def test_public_manifests_resolve_and_match_static_declarations(package_name):
    package = importlib.import_module(package_name)
    manifest = importlib.import_module(package_name + "._exports")
    declarations = ast.parse(Path(package.__file__).with_suffix(".pyi").read_text())
    declared = {
        alias.asname or alias.name
        for node in declarations.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert declared == set(manifest.EXPORTS)
    assert set(package.__all__) <= declared
    assert len(package.__all__) == len(set(package.__all__))
    for name, (module_name, symbol) in manifest.EXPORTS.items():
        assert module_name != package_name, "lazy exports must not resolve through themselves"
        assert getattr(package, name) is getattr(importlib.import_module(module_name), symbol)


def test_importing_agent_contracts_does_not_load_execution_or_optional_adapters():
    script = f"""
import sys
import textwrap
sys.path.insert(0, {str(_ROOT / "src")!r})
from cayu.agents import AgentSpec
from cayu.messages import Message
assert AgentSpec(name='test', model='test').name == 'test'
assert Message.text('user', 'hello').role == 'user'
for module in ('cayu.applications', 'cayu.evals', 'cayu.storage.postgres',
               'cayu.providers.openai', 'cayu.server', 'cayu.runners.docker'):
    assert module not in sys.modules, module
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=30)


def test_unknown_public_attributes_raise_attribute_error():
    import cayu

    with pytest.raises(AttributeError, match="no attribute"):
        _ = cayu.not_a_cayu_public_symbol


def test_runtime_code_and_export_manifests_use_canonical_implementation_paths():
    legacy = (set(_MOVES) - _PACKAGE_CONVERSIONS) | {"cayu.core"}
    removed_import = re.compile(
        r"\b(?:from|import)\s+(?:" + "|".join(re.escape(name) for name in sorted(legacy)) + r")\b"
    )
    paths = (
        path
        for directory in ("src", "tests", "examples", "scripts", "maintenance")
        for path in (_ROOT / directory).rglob("*.py")
    )
    for path in paths:
        source = path.read_text()
        assert removed_import.search(source) is None, f"{path} contains a removed import"
        trees = [ast.parse(source)]
        # Subprocess programs and scaffold templates must use the same paths.
        for node in ast.walk(trees[0]):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "cayu" not in node.value or "import" not in node.value:
                    continue
                # Fragments still receive the textual removed-module check.
                with contextlib.suppress(SyntaxError):
                    trees.append(ast.parse(textwrap.dedent(node.value)))
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.module not in legacy, f"{path}:{node.lineno} imports {node.module}"
                    for alias in node.names:
                        target = f"{node.module}.{alias.name}"
                        assert target not in legacy, f"{path}:{node.lineno} imports {target}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name not in legacy, (
                            f"{path}:{node.lineno} imports {alias.name}"
                        )
    for package_name in _PACKAGES:
        manifest = importlib.import_module(package_name + "._exports")
        assert not {module for module, _ in manifest.EXPORTS.values()} & legacy


@pytest.mark.parametrize(
    ("module", "symbol"),
    [
        ("cayu.applications", "CayuApp"),
        ("cayu.sessions", "RunRequest"),
        ("cayu.tasks", "Task"),
        ("cayu.context", "DefaultContextPolicy"),
        ("cayu.memory", "AutomaticRecallPolicy"),
        ("cayu.approvals", "ToolApprovalDecision"),
        ("cayu.budgets", "BudgetPolicy"),
        ("cayu.snapshots", "AgentSnapshot"),
        ("cayu.delivery.github", "GitHubDeliveryError"),
        ("cayu.tools", "Tool"),
        ("cayu.tools", "AuxiliaryInferencePolicy"),
        ("cayu.tools.inference", "InferenceInvoker"),
        ("cayu.tools.inference", "InferenceLimits"),
        ("cayu.providers.response", "ModelResponse"),
        ("cayu.workflows", "Workflow"),
        ("cayu.storage", "postgres"),
    ],
)
def test_concept_can_be_the_first_cayu_import_in_a_fresh_process(module, symbol):
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(_ROOT / 'src')!r}); "
        f"from {module} import {symbol}; "
        f"assert {symbol} is not None"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=30)


@pytest.mark.parametrize("package_name", _PACKAGES)
def test_package_exports_resolve_before_implementation_modules_are_preloaded(package_name):
    script = f"""
import sys
import importlib
sys.path.insert(0, {str(_ROOT / "src")!r})
package = importlib.import_module({package_name!r})
exports = importlib.import_module({package_name!r} + '._exports').EXPORTS
for name, (module_name, symbol) in exports.items():
    assert module_name != package.__name__, name
    actual = getattr(package, name)
    assert actual is getattr(importlib.import_module(module_name), symbol), name
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=60)


@pytest.mark.parametrize(
    ("module_name", "symbol", "record_type"),
    [
        ("cayu.memory.interventions", "MemoryInterventionSpec", "cayu.memory-intervention-spec"),
        (
            "cayu.memory.interventions",
            "MemoryInterventionOperation",
            "cayu.memory-intervention-operation",
        ),
        (
            "cayu.memory.interventions",
            "MemoryInterventionReceipt",
            "cayu.memory-intervention-receipt",
        ),
        (
            "cayu.memory.interventions",
            "MemoryInterventionTrialBinding",
            "cayu.memory-intervention-trial",
        ),
        (
            "cayu.memory.interventions",
            "MemoryInterventionComparability",
            "cayu.memory-intervention-comparability",
        ),
        (
            "cayu.memory.execution",
            "MemoryInterventionExecutionRecord",
            "cayu.memory-intervention-execution",
        ),
        ("cayu.evals.memory_reporting", "MemoryExperimentReport", "cayu.memory-experiment-report"),
    ],
)
def test_module_moves_do_not_rename_serialized_record_discriminators(
    module_name, symbol, record_type
):
    model = getattr(importlib.import_module(module_name), symbol)
    field = model.model_json_schema()["properties"]["record_type"]
    assert field["const"] == record_type
    assert field["default"] == record_type


def test_removed_modules_and_stubs_are_absent_from_source_tree():
    for original in _MOVES:
        for suffix in (".py", ".pyi"):
            path = _ROOT / "src" / (original.replace(".", "/") + suffix)
            assert not path.exists(), path
    assert not (_ROOT / "src/cayu/core").exists()


def test_documentation_and_packaged_guides_use_current_module_paths():
    legacy = (set(_MOVES) - _PACKAGE_CONVERSIONS) | {"cayu.core"}
    pattern = re.compile(
        r"`(?:"
        + "|".join(re.escape(name) for name in sorted(legacy))
        + r")`"
        + r"|\b(?:from|import)\s+(?:"
        + "|".join(re.escape(name) for name in sorted(legacy))
        + r")\b"
        + r"|(?:"
        + "|".join(re.escape(name.replace(".", "/") + ".py") for name in sorted(legacy))
        + r")"
    )
    for directory in ("docs", "examples", "src/cayu/guides"):
        for path in (_ROOT / directory).rglob("*.md"):
            # The historical-to-current lookup table is the explicit migration boundary.
            if path == _ROOT / "docs/public-concepts.md":
                continue
            assert pattern.search(path.read_text()) is None, path


def test_root_does_not_advertise_unsupported_postgres_module_export():
    script = f"""
import sys
sys.path.insert(0, {str(_ROOT / "src")!r})
import cayu
assert 'postgres' not in dir(cayu)
assert 'postgres' not in cayu.__all__
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=30)
