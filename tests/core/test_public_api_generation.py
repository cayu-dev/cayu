"""The checked-in lazy API stubs derive from runtime declarations and policy."""

from __future__ import annotations

import ast
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts/generate_public_api_stubs.py"
_GENERATOR = runpy.run_path(str(_SCRIPT))


@pytest.fixture
def api_checkout(tmp_path):
    """Declaration-only checkout whose implementation imports would fail."""

    package = tmp_path / "src/cayu"
    package.mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/public-api-packages.json").write_text('["cayu"]')
    shutil.copyfile(_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copyfile(_ROOT / "src/cayu/_api.py", package / "_api.py")
    (package / "__init__.py").write_text('raise AssertionError("imported Cayu")\n')
    (package / "implementation.py").write_text('raise AssertionError("imported implementation")\n')
    (package / "_exports.py").write_text(
        "EXPORTS = {\n"
        "    'PublicAlias': ('cayu.implementation', 'Canonical'),\n"
        "    'Ordinary': ('cayu.implementation', 'Ordinary'),\n"
        "    'Optional': ('cayu.storage.postgres', 'PostgresSessionStore'),\n"
        "    'ExplicitOnly': ('cayu.implementation', 'ExplicitOnly'),\n"
        "    '__version__': ('cayu._version', '__version__'),\n"
        "}\n"
        "PUBLIC_NAMES = ['PublicAlias', 'Ordinary', 'Optional']\n"
    )
    return tmp_path


def test_generation_preserves_aliases_and_runtime_wildcards_without_imports(api_checkout):
    # Use an isolated interpreter with no installed Cayu or optional dependencies.
    # Blocking imports also catches accidental use of a globally installed copy.
    program = """
import importlib.abc
import runpy
import sys
from pathlib import Path
class RejectCayu(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cayu' or fullname.startswith('cayu.'):
            raise AssertionError(f'Imported {fullname} during declaration generation')
sys.meta_path.insert(0, RejectCayu())
generate = runpy.run_path(sys.argv[1])['generate']
raise SystemExit(generate(Path(sys.argv[2]), check=False))
"""
    subprocess.run(
        [sys.executable, "-I", "-c", program, str(_SCRIPT), str(api_checkout)],
        check=True,
        timeout=60,
    )
    source = (api_checkout / "src/cayu/__init__.pyi").read_text()
    assert "from cayu.implementation import Canonical as PublicAlias" in source
    assert "from cayu.storage.postgres import PostgresSessionStore as Optional" in source
    assert "from cayu.implementation import ExplicitOnly as ExplicitOnly" in source
    assert "from cayu._version import __version__ as __version__" in source
    tree = ast.parse(source)
    wildcard = next(node for node in tree.body if isinstance(node, ast.Assign))
    assert ast.literal_eval(wildcard.value) == ["Ordinary", "PublicAlias"]


@pytest.mark.parametrize("damage", ["wrong_target", "missing", "extra_name"])
def test_check_detects_drift_without_writing(api_checkout, damage, capsys):
    generate = _GENERATOR["generate"]
    assert generate(api_checkout, check=False) == 0
    stub = api_checkout / "src/cayu/__init__.pyi"
    if damage == "missing":
        stub.unlink()
    elif damage == "wrong_target":
        stub.write_text(
            stub.read_text().replace(
                "import Canonical as PublicAlias", "import Wrong as PublicAlias"
            )
        )
    else:
        stub.write_text(stub.read_text() + "from cayu.implementation import Extra as Extra\n")
    before = stub.read_bytes() if stub.exists() else None
    assert generate(api_checkout, check=True) == 1
    assert (stub.read_bytes() if stub.exists() else None) == before
    output = capsys.readouterr().out
    assert "src/cayu/__init__.pyi" in output
    assert "uv run python scripts/generate_public_api_stubs.py" in output
    assert generate(api_checkout, check=False) == 0
    assert generate(api_checkout, check=True) == 0


def test_generation_is_deterministic_and_does_not_rewrite_current_files(api_checkout):
    generate = _GENERATOR["generate"]
    assert generate(api_checkout, check=False) == 0
    stub = api_checkout / "src/cayu/__init__.pyi"
    original = stub.read_bytes()
    modified = stub.stat().st_mtime_ns
    assert generate(api_checkout, check=False) == 0
    assert generate(api_checkout, check=True) == 0
    assert stub.read_bytes() == original
    assert stub.stat().st_mtime_ns == modified


def test_generation_uses_current_shared_optional_policy(api_checkout):
    helper = api_checkout / "src/cayu/_api.py"
    helper.write_text(
        helper.read_text().replace('"cayu.storage.postgres",', '"cayu.implementation",')
    )
    assert _GENERATOR["generate"](api_checkout, check=False) == 0
    tree = ast.parse((api_checkout / "src/cayu/__init__.pyi").read_text())
    wildcard = next(node for node in tree.body if isinstance(node, ast.Assign))
    assert ast.literal_eval(wildcard.value) == ["Optional"]


@pytest.mark.parametrize("damage", ["missing", "unlisted", "duplicate", "empty"])
def test_generation_rejects_incomplete_package_inventory(api_checkout, damage):
    inventory = api_checkout / "docs/public-api-packages.json"
    if damage == "missing":
        inventory.write_text('["cayu", "cayu.missing"]')
    elif damage == "duplicate":
        inventory.write_text('["cayu", "cayu"]')
    elif damage == "empty":
        inventory.write_text("[]")
    else:
        package = api_checkout / "src/cayu/unlisted"
        package.mkdir()
        (package / "_exports.py").write_text("EXPORTS = {}; PUBLIC_NAMES = []\n")
    with pytest.raises(ValueError, match="inventory differs|duplicate packages|nonempty list"):
        _GENERATOR["generate"](api_checkout, check=False)
    assert not (api_checkout / "src/cayu/__init__.pyi").exists()


def test_formatting_failure_leaves_checked_in_stubs_untouched(api_checkout):
    generate = _GENERATOR["generate"]
    assert generate(api_checkout, check=False) == 0
    stub = api_checkout / "src/cayu/__init__.pyi"
    original = stub.read_bytes()
    manifest = api_checkout / "src/cayu/_exports.py"
    manifest.write_text(manifest.read_text().replace("'Canonical'", "'invalid-symbol'"))
    with pytest.raises(subprocess.CalledProcessError):
        generate(api_checkout, check=False)
    assert stub.read_bytes() == original


def test_generated_public_api_stubs_are_current():
    result = subprocess.run(
        [sys.executable, "-I", str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    packages = json.loads((_ROOT / "docs/public-api-packages.json").read_text())
    assert f"All {len(packages)} public API stubs are current." in result.stdout
