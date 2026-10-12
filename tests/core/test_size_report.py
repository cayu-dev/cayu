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
check_source_size = _REPORT["check_source_size"]


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


def test_report_follows_session_store_owner_without_losing_historical_revisions(repository):
    root, before_revision = repository
    old_owner = root / "src/cayu/sessions/base.py"
    old_owner.rename(root / "src/cayu/sessions/store.py")
    old_owner.write_text("from .store import SessionStore as SessionStore\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
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
            "Separate store owner",
        ],
        check=True,
    )

    after = build_report(root, "HEAD")
    before = build_report(root, before_revision)
    assert after["metrics"]["session_store_public_methods"] == 1
    assert before["metrics"]["session_store_public_methods"] == 1
    assert "src/cayu/sessions/store.py" in after["source_module_lines"]
    assert "src/cayu/sessions/store.py" not in before["source_module_lines"]


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


@pytest.mark.parametrize("mode", ["inventory", "source-size"])
@pytest.mark.parametrize("revision", [None, "", "HEAD"], ids=["omitted", "empty", "explicit-head"])
def test_cli_revision_selection_preserves_explicit_empty_values(repository, mode, revision):
    root, commit = repository
    command = [sys.executable, str(_SCRIPT), "--repo", str(root)]
    if mode == "source-size":
        command.extend(["--check-source-size", str(_size_policy(root))])
    if revision is not None:
        command.extend(["--revision", revision])
    result = subprocess.run(command, capture_output=True, text=True)

    if revision == "":
        assert result.returncode == 1
        assert "size_report:" in result.stderr
        assert not result.stdout
    else:
        assert result.returncode == 0, result.stderr
        if mode == "inventory":
            assert json.loads(result.stdout)["revision"] == commit
        else:
            assert f"passed at {commit} (3 files, 0 capped exceptions)" in result.stdout


@pytest.mark.parametrize("revision", ["", "HEAD"], ids=["empty", "explicit-head"])
def test_cli_baseline_check_rejects_every_explicit_revision(repository, revision):
    root, commit = repository
    saved = root / "baseline.json"
    saved.write_text(json.dumps(build_report(root, commit)))
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--repo",
            str(root),
            "--check",
            str(saved),
            "--revision",
            revision,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--revision cannot be combined with --check" in result.stderr
    assert not result.stdout


def _size_policy(root: Path, exceptions=None) -> Path:
    path = root / "size-policy.json"
    path.write_text(
        json.dumps({"schema_version": 1, "default_max_lines": 9999, "exceptions": exceptions or {}})
    )
    return path


def _commit_source_fixture(root: Path, *, stage: bool = True) -> None:
    if stage:
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
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
            "Source size fixture",
        ],
        check=True,
    )


@pytest.mark.parametrize("suffix", ["py", "pyi"])
@pytest.mark.parametrize("lines", [9999, 10000])
def test_source_size_guard_applies_to_new_python_and_stub_paths(repository, suffix, lines):
    root, _ = repository
    path = root / f"src/cayu/replacement.{suffix}"
    path.write_text("# maintained source\n" * lines)
    policy = _size_policy(root)
    _commit_source_fixture(root)

    if lines == 9999:
        _, files, exceptions = check_source_size(root, "HEAD", policy)
        assert (files, exceptions) == (4, 0)
    else:
        with pytest.raises(ValueError, match=rf"replacement\.{suffix}: 10,000.*9,999"):
            check_source_size(root, "HEAD", policy)


@pytest.mark.parametrize("lines", [10000, 10001, 10002])
def test_source_size_exception_allows_only_its_documented_cap(repository, lines):
    root, _ = repository
    name = "src/cayu/worker.py"
    (root / name).write_text("# worker\n" * lines)
    policy = _size_policy(root, {name: {"max_lines": 10001, "reason": "Shipped worker boundary"}})
    _commit_source_fixture(root)

    if lines <= 10001:
        assert check_source_size(root, "HEAD", policy)[1:] == (4, 1)
    else:
        with pytest.raises(ValueError, match="worker.py: 10,002.*10,001"):
            check_source_size(root, "HEAD", policy)


@pytest.mark.parametrize("remaining_lines", [None, 9999])
def test_source_size_guard_rejects_obsolete_exceptions(repository, remaining_lines):
    root, _ = repository
    name = "src/cayu/worker.py"
    if remaining_lines is not None:
        (root / name).write_text("# worker\n" * remaining_lines)
    policy = _size_policy(root, {name: {"max_lines": 10000, "reason": "Existing worker"}})
    _commit_source_fixture(root)
    with pytest.raises(ValueError, match="worker.py: remove the exception"):
        check_source_size(root, "HEAD", policy)


def test_source_size_exception_does_not_follow_an_oversized_replacement(repository):
    root, _ = repository
    (root / "src/cayu/replacement.py").write_text("# worker\n" * 10000)
    policy = _size_policy(
        root, {"src/cayu/old_worker.py": {"max_lines": 10000, "reason": "Existing worker"}}
    )
    _commit_source_fixture(root)
    with pytest.raises(ValueError) as raised:
        check_source_size(root, "HEAD", policy)
    assert "replacement.py: 10,000 lines exceeds" in str(raised.value)
    assert "old_worker.py: remove the exception" in str(raised.value)


@pytest.mark.parametrize(
    "invalid_policy",
    [
        [],
        {"schema_version": True, "default_max_lines": 9999, "exceptions": {}},
        {"schema_version": 2, "default_max_lines": 9999, "exceptions": {}},
        {"schema_version": 1, "default_max_lines": True, "exceptions": {}},
        {"schema_version": 1, "default_max_lines": 0, "exceptions": {}},
        {"schema_version": 1, "default_max_lines": 9999, "exceptions": []},
        {"schema_version": 1, "default_max_lines": 9999, "exceptions": {}, "typo": 1},
    ],
)
def test_source_size_guard_rejects_invalid_policy(repository, invalid_policy):
    root, _ = repository
    policy = root / "invalid-policy.json"
    policy.write_text(json.dumps(invalid_policy))
    with pytest.raises(ValueError, match="Invalid source-size policy"):
        check_source_size(root, "HEAD", policy)


@pytest.mark.parametrize(
    ("name", "exception"),
    [
        ("tests/test_example.py", {"max_lines": 10000, "reason": "Wrong population"}),
        ("src/cayu/../other.py", {"max_lines": 10000, "reason": "Noncanonical"}),
        ("src/cayu//other.py", {"max_lines": 10000, "reason": "Noncanonical"}),
        ("src/cayu/other.txt", {"max_lines": 10000, "reason": "Wrong suffix"}),
        ("src/cayu/other.py", {"max_lines": True, "reason": "Invalid cap"}),
        ("src/cayu/other.py", {"max_lines": 9999, "reason": "Not an exception"}),
        ("src/cayu/other.py", {"max_lines": 10000, "reason": " "}),
        ("src/cayu/other.py", {"max_lines": 10000, "reason": "Reason", "typo": 1}),
    ],
)
def test_source_size_guard_rejects_invalid_exception(repository, name, exception):
    root, _ = repository
    with pytest.raises(ValueError, match="Invalid source-size exception"):
        check_source_size(root, "HEAD", _size_policy(root, {name: exception}))


def test_source_size_guard_rejects_duplicate_policy_keys(repository):
    root, _ = repository
    policy = root / "invalid-policy.json"
    policy.write_text(
        '{"schema_version":1,"default_max_lines":9999,"exceptions":{},"exceptions":{}}'
    )
    with pytest.raises(ValueError, match="Duplicate source-size policy key"):
        check_source_size(root, "HEAD", policy)


def test_source_size_guard_uses_committed_blobs_in_shallow_repositories(repository, tmp_path):
    root, original = repository
    policy = _size_policy(root)
    oversized = root / "src/cayu/oversized.py"
    oversized.write_text("raise RuntimeError('never import this source')\n" * 10000)
    _commit_source_fixture(root)
    clone = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", root.as_uri(), str(clone)], check=True)
    assert (
        subprocess.check_output(
            ["git", "-C", str(clone), "rev-parse", "--is-shallow-repository"], text=True
        ).strip()
        == "true"
    )
    (clone / "src/cayu/oversized.py").write_text("# dirty worktree is smaller\n")
    (clone / "src/cayu/untracked.py").write_text("# untracked\n" * 10001)
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--repo", str(clone), "--check-source-size", str(policy)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "oversized.py: 10,000 lines exceeds" in result.stderr
    assert "untracked.py" not in result.stderr
    assert "never import this source" not in result.stderr

    before = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--repo",
            str(root),
            "--revision",
            original,
            "--check-source-size",
            str(policy),
        ],
        capture_output=True,
        text=True,
    )
    assert before.returncode == 0, before.stderr
    assert f"passed at {original} (3 files, 0 capped exceptions)" in before.stdout


def test_source_size_guard_rejects_source_symlinks(repository):
    root, _ = repository
    policy = _size_policy(root)
    target = subprocess.check_output(
        ["git", "-C", str(root), "hash-object", "-w", "--stdin"],
        input="_exports.py",
        text=True,
    ).strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "update-index",
            "--add",
            "--cacheinfo",
            f"120000,{target},src/cayu/alias.py",
        ],
        check=True,
    )
    _commit_source_fixture(root, stage=False)
    with pytest.raises(ValueError, match="regular file: src/cayu/alias.py"):
        check_source_size(root, "HEAD", policy)


def test_source_size_guard_rejects_an_empty_source_population(repository):
    root, _ = repository
    subprocess.run(["git", "-C", str(root), "rm", "-r", "-q", "src"], check=True)
    _commit_source_fixture(root)
    with pytest.raises(ValueError, match="no production Python or stub files"):
        check_source_size(root, "HEAD", _size_policy(root))
