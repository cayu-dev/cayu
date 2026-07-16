"""Tests for ``cayu new`` (the project scaffold)."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from cayu import CayuApp, InMemorySessionStore, InMemoryTaskStore, ScriptedModelProvider
from cayu.cli import main
from cayu.cli.project import project_context


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
    with project_context(proj):
        spec.loader.exec_module(module)
    assert hasattr(module, "build_app")
    assert not any(isinstance(value, CayuApp) for value in vars(module).values())

    first_app = module.build_app(
        provider=ScriptedModelProvider([]),
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )
    second_app = module.build_app(
        provider=ScriptedModelProvider([]),
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )
    assert first_app is not second_app

    pyproject = (proj / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dependencies = ["cayu"]' in pyproject
    assert '[project.optional-dependencies]\nconsole = ["cayu[console]"]' in pyproject
    assert 'dev = ["pytest"]' in pyproject
    assert '[tool.cayu]\nfactory = "app:build_app"' in pyproject
    readme = (proj / "README.md").read_text(encoding="utf-8")
    assert "cayu inspect --json" in readme
    assert "cayu guide anatomy" in readme
    assert readme.index("## Application structure") < readme.index("## Setup and prove the project")
    assert "pip install -e '.[console,dev]'" in readme
    assert "uv sync --extra console --extra dev" in readme
    assert "cayu generate slice NAME --tool TOOL --effect EFFECT" in readme
    assert "cayu check --json" in capsys.readouterr().out


def test_project_context_isolates_and_restores_project_packages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_tools = {
        name: module for name, module in sys.modules.items() if name.partition(".")[0] == "tools"
    }
    for name in previous_tools:
        sys.modules.pop(name, None)
    try:
        host_root = tmp_path / "host"
        host_tools_path = host_root / "tools"
        host_tools_path.mkdir(parents=True)
        (host_tools_path / "__init__.py").write_text("MARKER = 'host'\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(host_root))
        importlib.import_module("tools")
        host_tools = sys.modules["tools"]
        project_root = tmp_path / "project"
        project_root.mkdir()
        project_tools = project_root / "tools"
        project_tools.mkdir()
        (project_tools / "__init__.py").write_text("", encoding="utf-8")
        (project_tools / "greet.py").write_text("MARKER = 'project'\n", encoding="utf-8")

        with project_context(project_root):
            greet = importlib.import_module("tools.greet")
            assert greet.MARKER == "project"

        assert sys.modules["tools"] is host_tools
        assert host_tools.MARKER == "host"
        assert "tools.greet" not in sys.modules
    finally:
        for name in tuple(sys.modules):
            if name.partition(".")[0] == "tools":
                sys.modules.pop(name, None)
        sys.modules.update(previous_tools)


def test_project_context_does_not_leak_modules_between_projects(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "project_marker.py").write_text("VALUE = 'first'\n", encoding="utf-8")
    (second / "project_marker.py").write_text("VALUE = 'second'\n", encoding="utf-8")

    with project_context(first):
        assert importlib.import_module("project_marker").VALUE == "first"
    assert "project_marker" not in sys.modules

    with project_context(second):
        assert importlib.import_module("project_marker").VALUE == "second"
    assert "project_marker" not in sys.modules


def test_project_context_preserves_loaded_standard_library_modules(tmp_path: Path) -> None:
    import secrets

    stdlib_secrets = secrets
    (tmp_path / "secrets.py").write_text("TOKEN = 'project'\n", encoding="utf-8")

    with project_context(tmp_path):
        loaded = importlib.import_module("secrets")

    assert loaded is stdlib_secrets
    assert not hasattr(loaded, "TOKEN")


def test_project_context_does_not_shadow_unloaded_standard_library_modules(
    tmp_path: Path,
) -> None:
    previous_fractions = {
        name: module
        for name, module in sys.modules.items()
        if name.partition(".")[0] == "fractions"
    }
    for name in previous_fractions:
        sys.modules.pop(name, None)
    try:
        (tmp_path / "fractions.py").write_text("TOKEN = 'project'\n", encoding="utf-8")

        with project_context(tmp_path):
            loaded = importlib.import_module("fractions")
            assert loaded.Fraction(1, 2).numerator == 1
            assert not hasattr(loaded, "TOKEN")

        assert "fractions" not in sys.modules
    finally:
        for name in tuple(sys.modules):
            if name.partition(".")[0] == "fractions":
                sys.modules.pop(name, None)
        sys.modules.update(previous_fractions)


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
    assert "cayu guide anatomy" in instructions
    assert "cayu guide tool-effects" in instructions
    assert "cayu inspect --json" in instructions
    assert "cayu check --json" in instructions
    assert "cayu check --fail-on warning --json" in instructions
    assert "cayu generate slice" in instructions
    assert "pytest" in instructions
    assert "cayu eval run evals.assistant:build_eval" in instructions
    assert "registered tool manifest" in instructions
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
