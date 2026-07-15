from __future__ import annotations

import errno
import inspect
import json
import runpy
import shutil
import subprocess
import sys
from importlib import import_module
from importlib.resources import files
from pathlib import Path

import pytest

from cayu.cli import main

_ROOT = Path(__file__).parents[2]
_FIXTURE_ROOT = _ROOT / "tests" / "fixtures" / "command_selector_boundary"
_RUNNER = _FIXTURE_ROOT / "safe_selector_check.py"
_CHECK_PROGRAM = _FIXTURE_ROOT / "fixture_check_program.py"
_REPOSITORY = _FIXTURE_ROOT / "repository"
_INVALID_SELECTORS = (
    "--help",
    "--junitxml=outside.xml",
    "/tests/pass.py",
    "../tests/pass.py",
    "",
    r"tests\pass.py",
    "tests//pass.py",
    "tests/./pass.py",
    "tests/pass.py/",
    "src/pass.py",
    "tests/pass.txt",
    "tests/pass.py::test ok",
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    shutil.copytree(_REPOSITORY, workspace)
    return workspace


def _run_fixture(
    workspace: Path,
    *selectors: str,
    timeout: float = 1.0,
    executable: Path | None = None,
    protected_paths: tuple[Path, ...] = (),
) -> dict:
    command = [
        sys.executable,
        str(_RUNNER),
        "--workspace",
        str(workspace),
        "--timeout",
        str(timeout),
    ]
    if executable is not None:
        command.extend(("--check-executable", str(executable)))
    for selector in selectors:
        command.append(f"--selector={selector}")
    for path in protected_paths:
        command.extend(("--protected-path", str(path)))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _regular_file_state(path: Path) -> tuple[bytes, int] | None:
    if not path.exists():
        return None
    metadata = path.stat()
    return (path.read_bytes(), metadata.st_mode)


@pytest.mark.parametrize(
    "selector",
    _INVALID_SELECTORS,
)
def test_model_controlled_selectors_cannot_bypass_the_check_or_write_outside(
    tmp_path: Path,
    selector: str,
) -> None:
    workspace = _workspace(tmp_path)
    marker = tmp_path / "outside-marker.txt"
    marker.write_text("keep\n", encoding="utf-8")
    report = tmp_path / "outside.xml"

    if selector.startswith("--junitxml="):
        selector = f"--junitxml={report}"
    result = _run_fixture(
        workspace,
        selector,
        protected_paths=(marker, report),
    )

    assert result == {
        "cannot_run_errno": None,
        "cannot_run_reason": None,
        "declared_effect": "none",
        "effect_matches_observed_writes": True,
        "exit_code": None,
        "observed_writes": [],
        "process_started": False,
        "selection_scope": "selected",
        "status": "rejected",
        "tests_executed": 0,
        "validated_selectors": [],
    }
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not report.exists()


@pytest.mark.parametrize(
    ("attack", "unsafe_expected"),
    (
        ("create", b"created\n"),
        ("overwrite", b"overwritten\n"),
        ("remove", None),
    ),
)
def test_create_overwrite_and_remove_options_are_real_attacks_rejected_before_launch(
    tmp_path: Path,
    attack: str,
    unsafe_expected: bytes | None,
) -> None:
    unsafe_target = tmp_path / f"unsafe-{attack}.txt"
    safe_target = tmp_path / f"safe-{attack}.txt"
    if attack != "create":
        unsafe_target.write_bytes(b"unsafe original\n")
        safe_target.write_bytes(b"safe original\n")
    safe_before = _regular_file_state(safe_target)

    subprocess.run(
        [sys.executable, str(_CHECK_PROGRAM), f"--{attack}={unsafe_target}"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (unsafe_target.read_bytes() if unsafe_target.exists() else None) == unsafe_expected

    result = _run_fixture(
        _workspace(tmp_path / "safe"),
        f"--{attack}={safe_target}",
        protected_paths=(safe_target,),
    )

    assert result["status"] == "rejected"
    assert result["process_started"] is False
    assert result["effect_matches_observed_writes"] is True
    assert result["observed_writes"] == []
    assert _regular_file_state(safe_target) == safe_before


def test_rejected_selector_reports_protected_state_changed_during_validation(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    marker = tmp_path / "outside-marker.txt"
    marker.write_text("before\n", encoding="utf-8")
    runner = runpy.run_path(str(_RUNNER))

    def selectors():
        marker.write_text("changed during validation\n", encoding="utf-8")
        yield "--help"

    result = runner["run_check"](
        workspace=workspace,
        selectors=selectors(),
        executable=Path(sys.executable),
        timeout=1,
        protected_paths=(marker,),
    )

    assert result["status"] == "rejected"
    assert result["process_started"] is False
    assert result["effect_matches_observed_writes"] is False
    assert result["observed_writes"] == [str(marker)]


def test_zero_exit_without_executed_tests_is_not_verified(tmp_path: Path) -> None:
    result = _run_fixture(_workspace(tmp_path), "tests/zero.py")

    assert result["exit_code"] == 0
    assert result["tests_executed"] == 0
    assert result["status"] == "zero_tests_executed"


def test_selected_subset_is_distinguishable_from_a_full_check(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    full = _run_fixture(workspace)
    selected = _run_fixture(workspace, "tests/pass.py::test_ok")

    assert full["selection_scope"] == "full"
    assert full["validated_selectors"] == []
    assert selected["selection_scope"] == "selected"
    assert selected["validated_selectors"] == ["tests/pass.py::test_ok"]
    assert selected["tests_executed"] == 1
    assert selected["status"] == "verified"


def test_check_outcomes_remain_distinct(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    verified = _run_fixture(workspace, "tests/pass.py")
    failed = _run_fixture(workspace, "tests/fail.py")
    timed_out = _run_fixture(workspace, "tests/slow.py", timeout=0.05)
    unavailable = _run_fixture(
        workspace,
        "tests/pass.py",
        executable=tmp_path / "missing-python",
    )

    assert verified["status"] == "verified"
    assert verified["exit_code"] == 0
    assert verified["tests_executed"] == 1
    assert failed["status"] == "failed"
    assert failed["exit_code"] == 1
    assert timed_out["status"] == "timed_out"
    assert timed_out["exit_code"] is None
    assert unavailable["status"] == "unavailable"
    assert unavailable["exit_code"] is None
    assert unavailable["cannot_run_reason"] == "not_found"


def test_non_executable_checker_is_reported_as_unavailable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    non_executable = tmp_path / "non-executable-checker"
    non_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    non_executable.chmod(0o644)

    result = _run_fixture(
        workspace,
        "tests/pass.py",
        executable=non_executable,
    )

    assert result["status"] == "unavailable"
    assert result["cannot_run_reason"] == "permission_denied"
    assert result["exit_code"] is None
    assert result["process_started"] is False


def test_invalid_executable_format_is_reported_as_unavailable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    invalid_executable = tmp_path / "invalid-executable"
    invalid_executable.write_text("not an executable format\n", encoding="utf-8")
    invalid_executable.chmod(0o755)

    result = _run_fixture(
        workspace,
        "tests/pass.py",
        executable=invalid_executable,
    )

    assert result["status"] == "unavailable"
    assert result["cannot_run_reason"] == "invalid_executable_format"
    assert result["cannot_run_errno"] == errno.ENOEXEC
    assert result["exit_code"] is None
    assert result["process_started"] is False


def test_declared_effect_is_checked_against_observed_writes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    outside_output = tmp_path / "outside-output.txt"

    result = _run_fixture(
        workspace,
        "tests/fail.py",
        "tests/writes.py",
        protected_paths=(outside_output,),
    )

    assert result["exit_code"] == 1
    assert result["tests_executed"] == 2
    assert result["declared_effect"] == "none"
    assert result["effect_matches_observed_writes"] is False
    assert result["status"] == "failed"
    assert result["observed_writes"] == [str(outside_output)]
    assert outside_output.read_text(encoding="utf-8") == "unexpected write\n"


def test_effect_inventory_detects_empty_directory_creation_and_removal(
    tmp_path: Path,
) -> None:
    create_workspace = _workspace(tmp_path / "create")
    created_directory = create_workspace / "internal-empty-dir"

    created = _run_fixture(create_workspace, "tests/mkdir.py")

    assert created["status"] == "verified"
    assert created["effect_matches_observed_writes"] is False
    assert created["observed_writes"] == [str(created_directory)]
    assert created_directory.is_dir()

    remove_workspace = _workspace(tmp_path / "remove")
    removed_directory = remove_workspace / "internal-empty-dir"
    removed_directory.mkdir()

    removed = _run_fixture(remove_workspace, "tests/rmdir.py")

    assert removed["status"] == "verified"
    assert removed["effect_matches_observed_writes"] is False
    assert removed["observed_writes"] == [str(removed_directory)]
    assert not removed_directory.exists()


def test_effect_inventory_preserves_internal_symlink_identity_and_target_changes(
    tmp_path: Path,
) -> None:
    create_workspace = _workspace(tmp_path / "create")
    created_link = create_workspace / "internal-link"

    created = _run_fixture(create_workspace, "tests/symlink.py")

    assert created["status"] == "verified"
    assert created["effect_matches_observed_writes"] is False
    assert created["observed_writes"] == [str(created_link)]
    assert created_link.is_symlink()
    assert created_link.readlink() == Path("tests/pass.py")

    retarget_workspace = _workspace(tmp_path / "retarget")
    retargeted_link = retarget_workspace / "internal-link"
    retargeted_link.symlink_to("tests/pass.py")

    retargeted = _run_fixture(retarget_workspace, "tests/retarget_symlink.py")

    assert retargeted["status"] == "verified"
    assert retargeted["effect_matches_observed_writes"] is False
    assert retargeted["observed_writes"] == [str(retargeted_link)]
    assert retargeted_link.readlink() == Path("tests/fail.py")


def test_effect_inventory_snapshots_the_effective_symlinked_workspace(
    tmp_path: Path,
) -> None:
    real_workspace = _workspace(tmp_path / "real")
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(real_workspace, target_is_directory=True)
    created_directory = real_workspace / "internal-empty-dir"

    result = _run_fixture(workspace_link, "tests/mkdir.py")

    assert result["status"] == "verified"
    assert result["effect_matches_observed_writes"] is False
    assert result["observed_writes"] == [str(created_directory)]
    assert created_directory.is_dir()


def test_package_guide_selector_recipe_matches_the_regression_fixture(
    tmp_path: Path,
    capsys,
) -> None:
    workspace = _workspace(tmp_path)
    selector_module = import_module("cayu.guides.command_selectors")
    canonical_selector = selector_module.pytest_selector
    guide_source = files("cayu.guides").joinpath("authoring.md").read_text(encoding="utf-8")
    assert guide_source.count("<!-- cayu-guide-include:pytest-selector -->") == 1
    assert "def pytest_selector" not in guide_source

    assert main(["guide", "authoring"]) == 0
    authoring = capsys.readouterr().out
    assert "<!-- cayu-guide-include:pytest-selector -->" not in authoring
    assert "command-selector-recipe" not in authoring
    assert "```python\nimport re\nfrom pathlib import Path, PurePosixPath" in authoring
    assert '_NODE_ID = re.compile(r"[A-Za-z0-9_.\\[\\]-]+")' in authoring
    assert inspect.getsource(canonical_selector).strip() in authoring
    assert canonical_selector("tests/pass.py::test_ok", workspace=workspace) == (
        "tests/pass.py::test_ok"
    )
    for invalid in _INVALID_SELECTORS:
        with pytest.raises(ValueError):
            canonical_selector(invalid, workspace=workspace)
