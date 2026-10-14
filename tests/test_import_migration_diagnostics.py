"""Removed routes fail with repairs in source and installed-package use."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.generate_import_migrations import render

from cayu._import_migration import import_replacement

_ROOT = Path(__file__).resolve().parents[1]


def test_installed_migration_inventory_matches_documented_routes():
    documented = json.loads((_ROOT / "docs/public-api-imports.json").read_text())
    shipped = _ROOT / "src/cayu/data/public_api_imports.json"
    assert shipped.read_text() == render(documented)
    expanded = {
        f"{package}.{name}": f"{owner}.{name}"
        for package, owners in json.loads(shipped.read_text()).items()
        for owner, names in owners.items()
        for name in names
    }
    assert expanded == documented
    for old, replacement in documented.items():
        assert import_replacement(*old.rsplit(".", 1)) == replacement


@pytest.mark.parametrize("package", ["cayu", "cayu.runtime"])
@pytest.mark.parametrize("access", ["from_import", "attribute"])
def test_removed_import_errors_preserve_the_repair_without_loading_the_owner(package, access):
    statement = (
        f"from {package} import RetryPolicy"
        if access == "from_import"
        else f"import {package}; {package}.RetryPolicy"
    )
    program = f"""
import sys
sys.path.insert(0, {str(_ROOT / "src")!r})
try:
    exec({statement!r})
except ImportError as error:
    assert 'from cayu.providers import RetryPolicy' in str(error), str(error)
else:
    raise AssertionError('Removed import resolved')
assert 'cayu.providers' not in sys.modules
assert 'cayu.applications' not in sys.modules
assert 'cayu.evals' not in sys.modules
for name in ('psycopg', 'fastapi', 'openai', 'anthropic', 'mcp'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-I", "-c", program], check=True, timeout=30)


def test_supported_imports_do_not_load_migration_data_and_unknown_names_stay_unknown():
    program = f"""
import sys
sys.path.insert(0, {str(_ROOT / "src")!r})
import cayu
from cayu import AgentSpec
from cayu.runtime import ExecutionProfileBehaviorIdentity
assert 'cayu._import_migration' not in sys.modules
assert not hasattr(cayu, 'not_a_cayu_public_symbol')
assert not hasattr(cayu, '__not_a_protocol__')
assert 'RetryPolicy' not in dir(cayu)
assert 'RetryPolicy' not in cayu.__all__
"""
    subprocess.run([sys.executable, "-I", "-c", program], check=True, timeout=30)


@pytest.mark.parametrize(
    ("package", "module", "names"),
    [
        ("cayu.providers", "cayu.providers.retry_policy", ["RetryPolicy", "copy_retry_policy"]),
        ("cayu.budgets", "cayu.budgets.run_limits", ["copy_run_limits", "has_run_limits"]),
        (
            "cayu.tools",
            "cayu.tools.base",
            [
                "ToolExecutionRequirement",
                "ToolExecutableRequirement",
                "ToolRunnerCapabilityRequirement",
            ],
        ),
        ("cayu.coding_products", "cayu.coding_products", ["CodingProductExecutionInspection"]),
    ],
)
def test_authoring_exports_keep_their_owner_and_wildcard_visibility(package, module, names):
    public = importlib.import_module(package)
    owner = importlib.import_module(module)
    for name in names:
        assert getattr(public, name) is getattr(owner, name)
        assert name in public.__all__
