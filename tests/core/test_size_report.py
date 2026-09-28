from __future__ import annotations

import ast
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/size_report.py"
_REPORT = runpy.run_path(str(_SCRIPT))
build_report = _REPORT["build_report"]
SourceInventory = _REPORT["SourceInventory"]


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    files = {
        "src/cayu/_exports.py": "PUBLIC_NAMES = ['Agent']\nraise RuntimeError('never import')\n",
        "src/cayu/sessions/base.py": "class SessionStore:\n    def load(self): pass\n",
        "src/cayu/runtime/_recovery_coordinator.py": (
            "class RecoveryCoordinator:\n    def __init__(self, store, *, clock=None): pass\n"
        ),
        "tests/test_example.py": "def test_example(): pass\n",
        ".test_durations": '{"tests/test_example.py::test_example": 1.25}',
        ".gitattributes": "tests/test_example.py export-ignore\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgSign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "Fixture",
    )
    return tmp_path, git("rev-parse", "HEAD")


def test_report_reads_commit_blobs_despite_dirty_untracked_and_export_ignored_files(repository):
    root, revision = repository
    before = build_report(root, revision)
    (root / "src/cayu/_exports.py").write_text("this is not valid Python")
    (root / "tests/test_untracked.py").write_text("also invalid Python")
    after = build_report(root, revision)
    assert before == after
    assert after["revision"] == revision
    assert after["metrics"]["source_python_files"] == 3
    assert after["metrics"]["test_python_files"] == 1
    assert after["metrics"]["root_public_names"] == 1
    assert after["metrics"]["recovery_constructor_named_parameters"] == 2
    assert after["committed_duration_snapshot"] == {"entries": 1, "seconds": 1.25}


def test_import_aliases_relative_modules_and_unique_dependency_counts():
    inventory = SourceInventory({"cayu.runtime._round", "cayu.runtime.execution_units"})
    inventory.observe(
        "src/cayu/storage/adapter.py",
        b"from ..runtime import _round as r\n"
        b"from cayu.runtime._round import pending_tool_round_from_checkpoint as load\n"
        b"if False:\n    from cayu.runtime.execution_units import Identity\n"
        b"load({})\nr.prepare_tool_round_publication()\n",
    )
    inventory.observe(
        "tests/test_provider.py",
        b"from cayu.providers.base import ModelProvider as Provider\n"
        b"from cayu.runtime import _round\n"
        b"class Fake(Provider): pass\n",
    )
    inventory.observe(
        "tests/support/providers.py",
        b"from cayu import ModelProvider\nclass Shared(ModelProvider): pass\n",
    )
    report = inventory.report()
    assert report["runtime_dependencies"]["src/cayu/storage/adapter.py"] == [
        "cayu.runtime._round",
        "cayu.runtime.execution_units",
    ]
    assert report["metrics"]["storage_runtime_modules"] == 2
    assert report["metrics"]["test_private_module_import_statements"] == 1
    assert report["metrics"]["test_direct_provider_subclasses_outside_support"] == 1
    assert report["ownership_call_sites"]["pending_tool_round_from_checkpoint"] == [
        "src/cayu/storage/adapter.py:5"
    ]


def test_inventory_keeps_all_source_replacements_and_large_test_boundary():
    inventory = SourceInventory(set())
    for index in range(25):
        inventory.observe(f"src/cayu/owner_{index}.py", b"# owner\n" * (index + 1))
    inventory.observe("tests/test_below.py", b"# test\n" * 9_999)
    inventory.observe("tests/test_boundary.py", b"# test\n" * 10_000)
    inventory.observe("tests/test_above.py", b"# test\n" * 10_001)

    report = inventory.report()
    assert report["source_module_lines"] == {
        f"src/cayu/owner_{index}.py": index + 1 for index in range(25)
    }
    assert report["test_modules_at_least_10000_lines"] == {
        "tests/test_above.py": 10_001,
        "tests/test_boundary.py": 10_000,
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(b"", 0, id="empty"),
        pytest.param(b"# first", 1, id="unterminated"),
        pytest.param(b"# first\n# second\n", 2, id="lf"),
        pytest.param(b"# first\r\n# second\r\n", 2, id="crlf"),
        pytest.param(b"# first\r# second\r", 2, id="cr"),
        pytest.param(b"# first\r\n# second\r# third\n# fourth", 4, id="mixed"),
    ],
)
def test_inventory_counts_physical_lines_in_source_tests_and_stubs(source, expected):
    inventory = SourceInventory(set())
    inventory.observe("src/cayu/example.py", source)
    inventory.observe("tests/test_example.py", source)
    inventory.observe("src/cayu/example/__init__.pyi", source)

    report = inventory.report()
    assert report["source_module_lines"] == {"src/cayu/example.py": expected}
    assert report["metrics"]["source_python_lines"] == expected
    assert report["metrics"]["test_python_lines"] == expected
    assert report["metrics"]["export_declaration_lines"] == expected


def test_string_separators_do_not_make_a_test_module_oversized():
    source = 'description = "\v\f\x1c\x1d\x1e\x85\u2028\u2029"\n'
    inventory = SourceInventory(set())
    inventory.observe("src/cayu/example.py", source.encode())
    inventory.observe("tests/test_example.py", (source + "# test\n" * 9_998).encode())

    report = inventory.report()
    assert report["source_module_lines"] == {"src/cayu/example.py": 1}
    assert report["metrics"]["test_python_lines"] == 9_999
    assert report["test_modules_at_least_10000_lines"] == {}


@pytest.mark.parametrize(
    "value",
    [
        '{"case": -1}',
        '{"case": true}',
        '{"case": NaN}',
        '{"case": Infinity}',
        '{"case": 1, "case": 2}',
        "[]",
    ],
)
def test_invalid_durations_fail_instead_of_producing_plausible_totals(value):
    with pytest.raises(ValueError):
        _REPORT["_duration_snapshot"](value.encode())


@pytest.mark.parametrize("source", ["PUBLIC_NAMES = dynamic()", "PUBLIC_NAMES = ['A', 'A']"])
def test_unmeasurable_public_surface_fails_closed(source):
    with pytest.raises(ValueError):
        _REPORT["_literal_names"](ast.parse(source), "PUBLIC_NAMES")


@pytest.mark.parametrize(
    "change",
    [
        pytest.param("PUBLIC_NAMES = ['A', 'B']", id="reassignment"),
        pytest.param("PUBLIC_NAMES = dynamic()", id="dynamic-reassignment"),
        pytest.param("PUBLIC_NAMES += ['B']", id="augmented-assignment"),
        pytest.param("PUBLIC_NAMES.append('B')", id="method-mutation"),
        pytest.param("PUBLIC_NAMES[:] = ['A', 'B']", id="subscript-mutation"),
        pytest.param("alias = PUBLIC_NAMES\nalias.append('B')", id="alias-mutation"),
        pytest.param("if enabled:\n    PUBLIC_NAMES = ['B']", id="conditional-assignment"),
        pytest.param("del PUBLIC_NAMES", id="deletion"),
        pytest.param("from other import names as PUBLIC_NAMES", id="import-rebinding"),
        pytest.param("def PUBLIC_NAMES(): pass", id="function-rebinding"),
    ],
)
def test_public_names_changes_fail_instead_of_undercounting(change):
    source = "PUBLIC_NAMES = ['A']\n" + change
    with pytest.raises(ValueError, match="single standalone literal assignment"):
        _REPORT["_literal_names"](ast.parse(source), "PUBLIC_NAMES")


def test_public_names_chained_alias_fails_instead_of_undercounting():
    source = "PUBLIC_NAMES = alias = ['A']\nalias.append('B')"
    with pytest.raises(ValueError, match="single standalone literal assignment"):
        _REPORT["_literal_names"](ast.parse(source), "PUBLIC_NAMES")


@pytest.mark.parametrize("literal", ["['A', 'B']", "('A', 'B')"])
def test_standalone_public_names_allow_unrelated_declarations(literal):
    source = f"EXPORTS = {{}}\nPUBLIC_NAMES = {literal}\nEXPORTS['C'] = ('module', 'C')"
    assert _REPORT["_literal_names"](ast.parse(source), "PUBLIC_NAMES") == ["A", "B"]


def test_check_reproduces_recorded_revision_and_rejects_changed_report(repository, tmp_path):
    root, revision = repository
    expected = build_report(root, revision)
    saved = tmp_path / "baseline.json"
    saved.write_text(json.dumps(expected))
    command = [sys.executable, str(_SCRIPT), "--repo", str(root), "--check", str(saved)]
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0
    expected["metrics"]["source_python_lines"] += 1
    saved.write_text(json.dumps(expected))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 1
    assert "differs" in failed.stderr


def test_invalid_revision_fails_without_falling_back_to_worktree(repository):
    root, _ = repository
    with pytest.raises(subprocess.CalledProcessError):
        build_report(root, "--help")
