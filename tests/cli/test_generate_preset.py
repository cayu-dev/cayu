from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_plan import normalize_application_plan


def _bypass_coding_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    from cayu.cli import scaffold

    git = shutil.which("git")
    assert git is not None

    def bypass(*, parent: Path) -> tuple[str, str]:
        del parent
        return git, "test-owned-rg"

    monkeypatch.setattr(scaffold, "_preflight_coding_commands", bypass)


def _source_files(root: Path) -> dict[str, bytes]:
    skipped = {".git", ".cayu", "data", "__pycache__"}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and skipped.isdisjoint(path.relative_to(root).parts)
    }


def _agent_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    assert main(["new", "convert-me", "--provider", "openai", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    return tmp_path / "convert-me"


def _convert(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, dict]:
    code = main(["generate", "preset", "coding", "--execution", "docker", *arguments, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_agent_project_converts_to_a_fresh_docker_coding_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bypass_coding_dependency_preflight(monkeypatch)
    project = _agent_project(tmp_path, capsys)
    (project / "domain" / "rules.py").write_text("LIMIT = 3\n", encoding="utf-8")
    before = _source_files(project)
    monkeypatch.chdir(project)

    code, planned = _convert(capsys, "--dry-run")

    assert code == 0
    assert planned["status"] == "ready"
    assert planned["source"]["preset"] == "agent"
    assert planned["target"] == {
        "preset": "coding",
        "provider": "openai",
        "execution": "docker",
        "capabilities": ["artifacts", "delegation", "evals", "human-input", "knowledge", "tasks"],
    }
    operations = {edit["path"]: edit["operation"] for edit in planned["edits"]}
    assert operations["Dockerfile.coding"] == "create"
    assert operations["app.py"] == "update_region"
    assert [item["path"] for item in planned["removals"]] == [
        "observability/CAPABILITY.md",
        "tests/test_memory.py",
        "tests/test_standard_capabilities.py",
    ]
    assert _source_files(project) == before

    code, applied = _convert(capsys)

    assert code == 0
    assert applied["status"] == "ready"
    expected = project_files(
        "convert-me",
        application_plan=normalize_application_plan(
            name="convert-me",
            agent_name="convert-me",
            preset="coding",
            provider="openai",
            execution="docker",
        ),
    )
    converted = _source_files(project)
    assert converted.pop("domain/rules.py") == b"LIMIT = 3\n"
    assert converted == {path: content.encode() for path, content in expected.items()}
    assert (project / "data" / "artifacts").is_dir()

    code, again = _convert(capsys)
    assert code == 0
    assert again["status"] == "already_present"


def test_changed_files_are_conflicts_and_nothing_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bypass_coding_dependency_preflight(monkeypatch)
    project = _agent_project(tmp_path, capsys)
    with (project / "agents" / "registration.py").open("a", encoding="utf-8") as handle:
        handle.write("# my change\n")
    with (project / "tests" / "test_memory.py").open("a", encoding="utf-8") as handle:
        handle.write("# my test\n")
    (project / "Dockerfile.coding").write_text("FROM scratch\n", encoding="utf-8")
    before = _source_files(project)
    monkeypatch.chdir(project)

    code, plan = _convert(capsys)

    assert code == 1
    assert plan["status"] == "conflict"
    assert {item["path"] for item in plan["conflicts"]} == {
        "Dockerfile.coding",
        "agents/registration.py",
        "tests/test_memory.py",
    }
    assert any("--agent-name convert-me" in note for note in plan["notes"])
    assert _source_files(project) == before

    assert main(["generate", "preset", "coding", "--execution", "docker", "--table"]) == 1
    text = capsys.readouterr().out
    assert text.startswith("Planned preset coding (execution docker): conflict\n")
    assert "  conflict: agents/registration.py — changed since it was generated;" in text


def test_agent_name_override_renames_only_the_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _agent_project(tmp_path, capsys)
    monkeypatch.chdir(project)

    command = ["generate", "preset", "agent", "--provider", "openai", "--agent-name", "worker"]
    assert main([*command, "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["status"] == "ready", plan["conflicts"]
    assert 'name="worker"' in (project / "agents" / "agent.py").read_text(encoding="utf-8")


def test_execution_defaults_to_the_declared_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "sandboxed", "--execution", "docker", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    monkeypatch.chdir(tmp_path / "sandboxed")

    assert main(["generate", "preset", "agent", "--without", "memory", "--dry-run", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["target"]["execution"] == "docker"
    removed = {item["path"] for item in plan["removals"]}
    assert "tests/test_memory.py" in removed
    assert "environments/sandbox.py" not in removed


def test_interrupted_conversion_still_removes_dropped_files_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cayu.cli import generate_preset

    project = _agent_project(tmp_path, capsys)
    monkeypatch.chdir(project)
    committed = generate_preset.apply_slice_plan

    def commit_then_crash(plan):
        committed(plan)
        raise RuntimeError("process died after the edit transaction committed")

    monkeypatch.setattr(generate_preset, "apply_slice_plan", commit_then_crash)
    command = ["generate", "preset", "agent", "--provider", "openai", "--without", "memory"]
    with pytest.raises(RuntimeError, match="process died"):
        main([*command, "--json"])
    capsys.readouterr()
    monkeypatch.setattr(generate_preset, "apply_slice_plan", committed)

    assert main([*command, "--json"]) == 0
    capsys.readouterr()
    assert not (project / "tests" / "test_memory.py").exists()
    assert not (project / "memory").exists()


def test_moving_the_local_database_is_called_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _agent_project(tmp_path, capsys)
    monkeypatch.chdir(project)

    code, plan = _convert(capsys, "--dry-run")

    assert code == 0
    assert any("data/cayu.db" in note and ".cayu/runtime/cayu.db" in note for note in plan["notes"])


def test_conversion_needs_a_scaffold_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "custom"\n\n[tool.cayu]\nfactory = "app:build_app"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    code, plan = _convert(capsys, "--dry-run")

    assert code == 1
    assert plan["status"] == "manual_action_required"
    assert "[tool.cayu.scaffold]" in plan["conflicts"][0]["reason"]


def test_invalid_target_plan_fails_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _agent_project(tmp_path, capsys)
    monkeypatch.chdir(project)

    assert main(["generate", "preset", "service", "--execution", "docker", "--json"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "GENERATOR_PLAN_FAILED"
    assert "not supported by preset 'service'" in error["message"]
