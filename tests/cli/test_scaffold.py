"""Tests for ``cayu new`` (the project scaffold)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from cayu.cli import main


def test_cayu_new_creates_a_valid_importable_project(tmp_path: Path, capsys) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    proj = tmp_path / "myproj"
    for filename in ("app.py", "pyproject.toml", "README.md", ".gitignore"):
        assert (proj / filename).exists()
    for dirname in ("agents", "tools", "evals", "tests"):
        assert (proj / dirname).is_dir()

    # The generated app.py must import cleanly: every cayu export in the template
    # exists and the syntax is valid. build_app() is not called at import, so no
    # API key is needed here.
    spec = importlib.util.spec_from_file_location("scaffolded_app", proj / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(proj))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(proj))
    assert hasattr(module, "build_app")

    pyproject = (proj / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dependencies = ["cayu"]' in pyproject
    assert '[project.optional-dependencies]\nconsole = ["cayu[console]"]' in pyproject
    assert 'dev = ["pytest"]' in pyproject
    assert '[tool.cayu]\nfactory = "app:build_app"' in pyproject
    readme = (proj / "README.md").read_text(encoding="utf-8")
    assert "cayu inspect --json" in readme
    assert "pip install -e '.[console,dev]'" in readme
    assert "uv sync --extra console --extra dev" in readme
    assert "cayu generate slice NAME --tool TOOL --effect EFFECT" in readme
    assert "cayu check --json" in capsys.readouterr().out


def test_cayu_new_emits_safe_agent_instructions_and_credential_free_proof(
    tmp_path: Path,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myproj"

    assert (project / "AGENTS.md").is_file()
    assert (project / "agents" / "assistant.py").is_file()
    assert (project / "tools" / "greet.py").is_file()
    assert (project / "tests" / "test_assistant.py").is_file()
    assert (project / "evals" / "assistant.py").is_file()
    assert not (project / "workflows").exists()
    assert not (project / "memory").exists()

    app_source = (project / "app.py").read_text(encoding="utf-8")
    assert "ExecCommandTool" not in app_source
    assert "# <cayu:generated-imports>" in app_source
    assert "# <cayu:generated-registrations>" in app_source
    tool_source = (project / "tools" / "greet.py").read_text(encoding="utf-8")
    agent_source = (project / "agents" / "assistant.py").read_text(encoding="utf-8")
    eval_source = (project / "evals" / "assistant.py").read_text(encoding="utf-8")
    assert 'GREET_TOOL_NAME = "greet"' in tool_source
    assert "from tools.greet import GREET_TOOL_NAME" in agent_source
    assert "workflow_tool_names=(GREET_TOOL_NAME,)" in agent_source
    assert "name=GREET_TOOL_NAME" in tool_source
    assert "ToolCalled(GREET_TOOL_NAME)" in eval_source
    assert "ToolEffect.NONE" in tool_source
    assert "ToolEffect.EXTERNAL" not in tool_source

    instructions = (project / "AGENTS.md").read_text(encoding="utf-8")
    assert "cayu inspect --json" in instructions
    assert "cayu check --json" in instructions
    assert "cayu generate slice" in instructions
    assert "pytest" in instructions
    assert "cayu eval run evals.assistant:build_eval" in instructions
    assert "prompt_tool_alignment" in instructions
    assert "registered_tool_names" in instructions
    assert "manifest_fingerprint" in instructions
    assert "Do not claim live verification" in instructions


def test_cayu_new_refuses_a_nonempty_directory(tmp_path: Path) -> None:
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "existing.txt").write_text("keep me")
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 1
    assert (proj / "existing.txt").read_text() == "keep me"


def test_cayu_new_refuses_an_existing_file(tmp_path: Path) -> None:
    proj = tmp_path / "myproj"
    proj.write_text("keep me")

    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 1
    assert proj.read_text() == "keep me"


def test_cayu_new_rejects_invalid_names(tmp_path: Path) -> None:
    assert main(["new", "../escape", "--dir", str(tmp_path)]) == 1
    assert main(["new", "has space", "--dir", str(tmp_path)]) == 1
