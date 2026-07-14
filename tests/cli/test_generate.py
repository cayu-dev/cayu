from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.generate import GeneratorApplyError, apply_slice_plan, plan_slice


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generate_slice_dry_run_is_deterministic_and_write_free(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    before = _files(project)
    monkeypatch.chdir(project)

    command = [
        "generate",
        "slice",
        "analyst",
        "--tool",
        "analyze_document",
        "--effect",
        "none",
        "--dry-run",
        "--json",
    ]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert first["schema_version"] == "1"
    assert first["status"] == "ready"
    assert [edit["path"] for edit in first["edits"]] == [
        "agents/analyst.py",
        "app.py",
        "evals/analyst.py",
        "tests/test_analyst.py",
        "tools/analyze_document.py",
    ]
    assert first["verification_commands"] == [
        "cayu inspect --json",
        "cayu check --json",
        "pytest tests/test_analyst.py",
        "cayu eval run evals.analyst:build_eval",
    ]
    assert _files(project) == before


def test_generate_slice_json_is_a_write_free_plan_without_dry_run(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert _files(project) == before


def test_generate_slice_applies_once_and_passes_public_verification(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    command = [
        "generate",
        "slice",
        "analyst",
        "--tool",
        "analyze_document",
        "--effect",
        "external",
    ]

    assert main(command) == 0
    assert capsys.readouterr().out.startswith("Applied analyst: ready")
    after_apply = _files(project)
    assert "AlwaysRequireApprovalToolPolicy" in after_apply["app.py"].decode()

    assert main([*command, "--json"]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["status"] == "already_present"
    assert repeated["edits"] == []
    assert _files(project) == after_apply

    for module_name in ("app", "agents.analyst", "tools.analyze_document"):
        sys.modules.pop(module_name, None)
    assert main(["inspect", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert [agent["name"] for agent in manifest["agents"]] == ["analyst", "assistant"]
    assert main(["check", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []
    eval_output = project / "analyst-eval.json"
    assert (
        main(
            [
                "eval",
                "run",
                "evals.analyst:build_eval",
                "--output",
                str(eval_output),
            ]
        )
        == 0
    )
    assert json.loads(eval_output.read_text(encoding="utf-8"))["status"] == "passed"

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_analyst.py", "-q"],
        cwd=project,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_generate_slice_missing_registration_seam_requires_manual_action_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    app_path = project / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8").replace(
            "    # <cayu:generated-registrations>\n    # </cayu:generated-registrations>\n",
            "",
        ),
        encoding="utf-8",
    )
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 1
    )
    plan = json.loads(capsys.readouterr().out)

    assert plan["status"] == "manual_action_required"
    assert plan["conflicts"][0]["path"] == "app.py"
    assert _files(project) == before


def test_generate_slice_conflict_preserves_user_files_and_registration(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    user_file = project / "agents" / "analyst.py"
    user_file.write_text("# user-owned\n", encoding="utf-8")
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 1
    )
    plan = json.loads(capsys.readouterr().out)

    assert plan["status"] == "conflict"
    assert plan["conflicts"] == [
        {
            "path": "agents/analyst.py",
            "operation": "create",
            "reason": "path exists with user-authored or different content",
        }
    ]
    assert _files(project) == before


def test_generate_slice_rejects_python_keywords_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "class",
                "--tool",
                "await",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)

    assert result["error"]["code"] == "GENERATOR_PLAN_FAILED"
    assert "Python keyword" in result["error"]["message"]
    assert _files(project) == before


def test_generate_slice_rejects_symlinked_generated_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    agents = project / "agents"
    agents.rename(project / "real-agents")
    agents.symlink_to(outside, target_is_directory=True)
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 1
    )
    plan = json.loads(capsys.readouterr().out)

    assert plan["status"] == "conflict"
    assert plan["conflicts"][0] == {
        "path": "agents/analyst.py",
        "operation": "create",
        "reason": "generated path contains a symbolic link: agents",
    }
    assert not (outside / "analyst.py").exists()
    assert _files(project) == before


def test_apply_slice_plan_rejects_stale_preimages_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    app_path = project / "app.py"
    app_path.write_text(app_path.read_text(encoding="utf-8") + "\n# concurrent edit\n")
    before = _files(project)

    with pytest.raises(GeneratorApplyError, match="changed after the plan was created"):
        apply_slice_plan(plan)

    assert _files(project) == before


def test_apply_slice_plan_rejects_a_symlink_introduced_after_planning(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(project)
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    agents = project / "agents"
    agents.rename(project / "real-agents")
    agents.symlink_to(outside, target_is_directory=True)
    before = _files(project)

    with pytest.raises(GeneratorApplyError, match="symbolic link: agents"):
        apply_slice_plan(plan)

    assert not (outside / "analyst.py").exists()
    assert _files(project) == before


def test_apply_slice_plan_rolls_back_a_mid_commit_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    before = _files(project)
    real_replace = Path.replace
    commits = 0

    def fail_second_commit(source: Path, target: Path) -> Path:
        nonlocal commits
        if ".cayu-generate-" in source.as_posix() and ".cayu-generate-" not in target.as_posix():
            commits += 1
            if commits == 2:
                raise OSError("simulated commit failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_commit)

    with pytest.raises(GeneratorApplyError, match="simulated commit failure"):
        apply_slice_plan(plan)

    assert _files(project) == before
    assert not list(project.glob(".cayu-generate-*"))
