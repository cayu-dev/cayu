"""Tests for ``cayu new`` (the project scaffold)."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu import (
    CayuApp,
    EvalStatus,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    load_eval_run,
    run_to_completion,
)
from cayu.cli import main
from cayu.cli.project import project_context


def test_cayu_new_creates_a_valid_importable_project(tmp_path: Path, capsys) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    proj = tmp_path / "myproj"
    for filename in (
        "app.py",
        "configuration.py",
        "pyproject.toml",
        "README.md",
        ".gitignore",
    ):
        assert (proj / filename).exists()
    for dirname in ("agents", "evals", "tests"):
        assert (proj / dirname).is_dir()
    assert not (proj / "tools").exists()

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

    app_source = (proj / "app.py").read_text(encoding="utf-8")
    configuration_source = (proj / "configuration.py").read_text(encoding="utf-8")
    assert "AnthropicProvider" in app_source
    assert "OpenAISubscriptionProvider" in app_source
    assert "CAYU_OPENAI_SUBSCRIPTION" not in app_source
    assert "_SCAFFOLDED_PROVIDER = None" in configuration_source
    assert 'os.environ.get("CAYU_PROVIDER", _SCAFFOLDED_PROVIDER)' in configuration_source
    pyproject = (proj / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dependencies = ["cayu>=0.1.0rc3"]' in pyproject
    assert 'console = ["cayu[console]"]' not in pyproject
    assert 'dev = ["pytest"]' in pyproject
    assert '[tool.cayu]\nfactory = "app:build_app"' in pyproject
    assert 'eval_target = "evals.agent:build_eval"' in pyproject
    assert '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"' in pyproject
    assert 'SQLiteSessionStore("data/cayu.db")' in app_source
    assert 'SQLiteTaskStore("data/cayu.db")' in app_source
    assert "sessions.sqlite" not in app_source
    readme = (proj / "README.md").read_text(encoding="utf-8")
    assert "uv run cayu inspect --json" in readme
    assert "uv run cayu guide anatomy" in readme
    assert readme.index("## Application structure") < readme.index("## Setup and prove the project")
    assert "pip install -e" not in readme
    assert "uv sync --extra dev" in readme
    assert "uv run cayu eval run" in readme
    assert "uv run cayu session list" in readme
    assert "uv run cayu auth openai login" in readme
    assert "CAYU_PROVIDER=openai-subscription" in readme
    assert "CAYU_PROVIDER=anthropic" in readme
    assert "ANTHROPIC_API_KEY" in readme
    assert "subscription holder's own local" in readme
    assert "development and evaluation" in readme
    assert "not intended for production" in readme
    assert "bypassing plan limits" in readme
    assert "cayu eval run evals.agent:build_eval" not in readme
    assert "uv run cayu guide authoring#cayu-map" in readme
    assert "github.com" not in readme
    assert 'uv run python run.py --message "YOUR REQUEST"' in readme
    assert "model-only" in readme
    assert "cayu generate slice" not in readme
    output = capsys.readouterr().out
    assert "uv sync --extra dev" in output
    assert "uv run cayu check --json" in output
    assert "none selected" in output


def test_scaffold_subscription_mode_selects_a_compatible_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert (
        main(
            [
                "new",
                "myproj",
                "--dir",
                str(tmp_path),
                "--provider",
                "openai-subscription",
            ]
        )
        == 0
    )
    project = tmp_path / "myproj"
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("Subscription result."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="openai_subscription",
    )
    monkeypatch.setattr("cayu.OpenAISubscriptionProvider", lambda: provider)

    spec = importlib.util.spec_from_file_location("subscription_scaffold_app", project / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with project_context(project):
        spec.loader.exec_module(module)
        app = module.build_app(
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )
        outcome = asyncio.run(
            run_to_completion(
                app,
                RunRequest(
                    agent_name="myproj",
                    messages=[Message.text("user", "Test with my subscription")],
                ),
            )
        )

    assert outcome.ok
    assert provider.requests[0].model == "gpt-5.4"


def test_scaffold_does_not_infer_provider_from_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myproj"
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    spec = importlib.util.spec_from_file_location("neutral_scaffold_app", project / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with project_context(project):
        spec.loader.exec_module(module)
        provider = module.configured_provider()
        app = module.build_app(
            provider=provider,
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )

    assert isinstance(provider, ScriptedModelProvider)
    assert provider.name == "unconfigured"
    assert app.describe().agents[0].model == "provider-model-unconfigured"
    with pytest.raises(RuntimeError, match="no provider is selected"):
        module.validate_run_configuration(app, "myproj")


def test_scaffold_provider_env_explicitly_overrides_scaffold_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path), "--provider", "openai"]) == 0
    project = tmp_path / "myproj"
    provider = ScriptedModelProvider([], name="anthropic")
    monkeypatch.setenv("CAYU_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr("cayu.AnthropicProvider", lambda *, api_key: provider)
    monkeypatch.setattr(
        "cayu.OpenAIProvider",
        lambda **kwargs: pytest.fail("credential presence must not override CAYU_PROVIDER"),
    )

    spec = importlib.util.spec_from_file_location("anthropic_scaffold_app", project / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with project_context(project):
        spec.loader.exec_module(module)
        selected = module.configured_provider()
        app = module.build_app(
            provider=selected,
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )

    assert selected is provider
    agent = app.describe().agents[0]
    assert agent.configured_provider == "anthropic"
    assert agent.model == "claude-sonnet-4-6"


@pytest.mark.parametrize(
    ("scaffold_provider", "environment_provider"),
    [
        pytest.param(None, None, id="neutral"),
        pytest.param("openai", None, id="openai"),
        pytest.param("anthropic", None, id="anthropic"),
        pytest.param("openai-subscription", None, id="openai-subscription"),
        pytest.param(None, "openai", id="neutral-env-openai"),
        pytest.param(None, "anthropic", id="neutral-env-anthropic"),
        pytest.param(None, "openai-subscription", id="neutral-env-openai-subscription"),
    ],
)
def test_scaffolded_credential_free_proof_ignores_live_provider_selection(
    tmp_path: Path,
    scaffold_provider: str | None,
    environment_provider: str | None,
) -> None:
    command = ["new", "myproj", "--dir", str(tmp_path)]
    if scaffold_provider is not None:
        command.extend(("--provider", scaffold_provider))
    assert main(command) == 0
    project = tmp_path / "myproj"
    environment = os.environ.copy()
    for name in ("CAYU_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        environment.pop(name, None)
    if environment_provider is not None:
        environment["CAYU_PROVIDER"] = environment_provider
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")

    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert test_result.returncode == 0, test_result.stdout + test_result.stderr

    eval_result = subprocess.run(
        [sys.executable, "-m", "cayu", "eval", "run"],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert eval_result.returncode == 0, eval_result.stdout + eval_result.stderr
    assert json.loads(eval_result.stdout)["status"] == "passed"


def test_python_m_cayu_routes_to_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cayu", "version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("cayu ")
    assert result.stderr == ""


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
    assert (project / "agents" / "agent.py").is_file()
    assert not (project / "tools" / "greet.py").exists()
    assert (project / "tests" / "test_agent.py").is_file()
    assert (project / "evals" / "agent.py").is_file()
    assert not (project / "workflows").exists()
    assert not (project / "memory").exists()

    app_source = (project / "app.py").read_text(encoding="utf-8")
    assert "ExecCommandTool" not in app_source
    assert "# <cayu:generated-imports>" in app_source
    assert "# <cayu:generated-registrations>" in app_source
    agent_source = (project / "agents" / "agent.py").read_text(encoding="utf-8")
    eval_source = (project / "evals" / "agent.py").read_text(encoding="utf-8")
    assert 'name="myproj"' in agent_source
    assert "_SYSTEM_PROMPT_PARTS: list[str] = []" in agent_source
    assert "system_prompt=" in agent_source
    assert "workflow_tool_names=" in agent_source
    assert "ToolCalled" not in eval_source

    instructions = (project / "AGENTS.md").read_text(encoding="utf-8")
    assert "uv run cayu guide anatomy" in instructions
    assert "uv run cayu inspect --json" in instructions
    assert "uv run cayu check --json" in instructions
    assert "uv run pytest" in instructions
    assert "uv run cayu eval run" in instructions
    assert "cayu eval run evals.agent:build_eval" not in instructions
    assert "Edit the existing agent, test, and eval" in instructions
    assert "Tools are optional" in instructions
    assert "uv run cayu guide authoring#cayu-map" in instructions
    assert "uv run cayu guide references" in instructions
    assert "github.com" not in instructions
    assert "Deployment is a separate task" in instructions
    assert "Clarify users, jobs, triggers" not in instructions
    assert "cayu generate slice" not in instructions
    assert "uv run cayu generate tool TOOL_NAME --agent myproj --effect EFFECT" in instructions


def test_cayu_new_uses_supported_hyphenated_project_name_for_the_agent(
    tmp_path: Path,
) -> None:
    assert main(["new", "code-review", "--dir", str(tmp_path)]) == 0

    agent_source = (tmp_path / "code-review" / "agents" / "agent.py").read_text(encoding="utf-8")
    assert 'name="code-review"' in agent_source


def test_scaffolded_default_eval_runs_from_nested_directory_without_api_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myproj"
    nested = project / "agents" / "reviewer"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert main(["eval", "run", "--output", "eval-run.json"]) == 0

    report_path = project / "eval-run.json"
    report = load_eval_run(report_path)
    assert report.status == EvalStatus.PASSED
    assert report.suite_id == "agent-output"
    assert not (nested / "eval-run.json").exists()


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
