"""Tests for ``cayu new`` (the project scaffold)."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cayu import (
    CayuApp,
    ChatCompletionsProvider,
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EvalStatus,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    StructuredCommandToolPolicy,
    load_eval_run,
    run_to_completion,
)
from cayu import __version__ as cayu_version
from cayu.cli import main
from cayu.cli._bounded_command import BoundedCommandResult
from cayu.cli.project import project_context

_RESERVED_TEMPLATE_TOKENS = (
    "__PROJECT_NAME__",
    "__AGENT_NAME__",
    "__REVIEWER_NAME__",
    "__CAYU_VERSION__",
    "__PROVIDER_DISPLAY__",
    "__PROVIDER_LITERAL__",
    "__PROVIDER_GUIDE_POINTER__",
)


def _bypass_coding_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep non-probe scaffold tests independent of host ripgrep availability."""

    from cayu.cli import scaffold

    git = shutil.which("git")
    assert git is not None

    def bypass(*, parent: Path) -> tuple[str, str]:
        del parent
        return git, "test-owned-rg"

    monkeypatch.setattr(scaffold, "_preflight_coding_commands", bypass)


def _compatible_coding_probe_result() -> BoundedCommandResult:
    """Return all positive evidence consumed by one dependency-preflight test."""

    evidence = (
        b"probe.txt\0staged.txt\0search.txt\0"
        b" M probe.txt\0A  staged.txt\0AM probe.txt\0"
        b"-cayu scaffold probe\n+cayu scaffold semantic probe\n"
        b"-cayu dependency baseline\n+cayu dependency changed\n"
        b"staged.txt\n+cayu staged semantic probe\n"
        b"1\t1\tprobe.txt\0"
        b"1\t0\tprobe.txt\0"
        b"1\t0\tstaged.txt\0"
        b"probe.txt|1|cayu scaffold semantic probe\n"
        b"search.txt|1|cayu dependency needle\n"
        b"probe.txt\01\nsearch.txt\01\n"
    )
    return BoundedCommandResult(
        returncode=0,
        output=evidence,
        output_truncated=False,
    )


def _install_hermetic_coding_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide exact test-owned Git/rg probe seams without invoking host tools."""

    from cayu.cli import scaffold

    monkeypatch.setattr(
        scaffold.shutil,
        "which",
        lambda command: f"/test-bin/{command}" if command in {"git", "rg"} else None,
    )

    def compatible(argv, **kwargs) -> BoundedCommandResult:
        del argv, kwargs
        return _compatible_coding_probe_result()

    monkeypatch.setattr(scaffold, "run_bounded_command", compatible)


def test_cayu_new_creates_a_valid_importable_project(tmp_path: Path, capsys) -> None:
    class FalsySessionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __bool__(self) -> bool:
            return False

    class FalsyTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __bool__(self) -> bool:
            return False

    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    proj = tmp_path / "myproj"
    for filename in (
        "app.py",
        "configuration/settings.py",
        "agents/registration.py",
        "CLAUDE.md",
        "pyproject.toml",
        "README.md",
        ".gitignore",
    ):
        assert (proj / filename).exists()
    for dirname in (
        "agents",
        "configuration",
        "domain",
        "environments",
        "evals",
        "integrations",
        "knowledge",
        "memory",
        "observability",
        "operations",
        "policies",
        "prompts",
        "tests",
        "tools",
        "workflows",
        "data",
    ):
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

    session_store = FalsySessionStore()
    task_store = FalsyTaskStore()
    first_app = module.build_app(
        provider=ScriptedModelProvider([]),
        session_store=session_store,
        task_store=task_store,
        knowledge_store=InMemoryKnowledgeStore(),
    )
    second_app = module.build_app(
        provider=ScriptedModelProvider([]),
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        knowledge_store=InMemoryKnowledgeStore(),
    )
    assert first_app is not second_app
    assert first_app.session_store is session_store
    assert first_app.task_store is task_store
    manifest = first_app.describe()
    assert manifest.stores.knowledge == "InMemoryKnowledgeStore"
    assert manifest.runtime.request_footprint.fingerprinting_enabled is True
    assert manifest.runtime.request_footprint.fingerprint_key_id == "standard-local-v1"
    agent = manifest.agents[0]
    assert agent.context_policy == "AutomaticRecallContextPolicy"
    assert agent.tool_policy == "ParameterConstrainedToolPolicy"
    assert {tool.name for tool in agent.tools} == {
        "ask_user",
        "list_artifacts",
        "list_knowledge",
        "read_knowledge",
        "remember_knowledge",
        "search_knowledge",
    }
    assert next(
        tool for tool in agent.tools if tool.name == "remember_knowledge"
    ).policy_coverage == ("conditional")
    environment = manifest.environments[0]
    assert environment.artifact_store == "LocalArtifactStore"
    assert environment.knowledge_store == "InMemoryKnowledgeStore"
    assert environment.runner is None
    assert environment.vault is None
    assert environment.mcp_servers == ()
    memory_key = proj / "data" / "memory-evidence.key"
    assert len(memory_key.read_text(encoding="utf-8").strip().encode("utf-8")) >= 32
    if os.name != "nt":
        assert memory_key.stat().st_mode & 0o777 == 0o600

    app_source = (proj / "app.py").read_text(encoding="utf-8")
    configuration_source = (proj / "configuration/settings.py").read_text(encoding="utf-8")
    provider_source = (proj / "configuration/providers.py").read_text(encoding="utf-8")
    storage_source = (proj / "configuration/storage.py").read_text(encoding="utf-8")
    assert "AnthropicProvider" in provider_source
    assert "ChatCompletionsProvider" in provider_source
    assert "OpenAISubscriptionProvider" in provider_source
    assert "CAYU_OPENAI_SUBSCRIPTION" not in provider_source
    assert "_SCAFFOLDED_PROVIDER = None" in configuration_source
    assert 'os.environ.get("CAYU_PROVIDER", _SCAFFOLDED_PROVIDER)' in configuration_source
    pyproject = (proj / "pyproject.toml").read_text(encoding="utf-8")
    assert f'dependencies = ["cayu=={cayu_version}"]' in pyproject
    assert 'console = ["cayu[console]"]' not in pyproject
    assert f'dev = ["cayu[server]=={cayu_version}", "pytest"]' in pyproject
    assert '[tool.uv]\ncache-dir = ".cayu/uv-cache"' in pyproject
    assert (proj / ".gitignore").read_text(encoding="utf-8").startswith(".cayu/\n")
    assert '[tool.cayu]\nfactory = "app:build_app"' in pyproject
    assert 'eval_target = "evals.agent:build_eval"' in pyproject
    assert '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"' in pyproject
    assert 'else SQLiteSessionStore(\n                "data/cayu.db",' in storage_source
    assert "public_authority_alias_codec_from_environment()" in storage_source
    assert 'SQLiteTaskStore("data/cayu.db")' in storage_source
    assert "sessions.sqlite" not in storage_source
    assert "def build_app(" in app_source
    assert "class " not in app_source
    assert (proj / "CLAUDE.md").read_text(encoding="utf-8") == "@AGENTS.md\n"
    readme = (proj / "README.md").read_text(encoding="utf-8")
    assert "uv run --no-sync cayu inspect --json" in readme
    assert "uv run --no-sync cayu guide anatomy" in readme
    assert readme.index("## Application structure") < readme.index("## Setup and prove the project")
    assert "pip install -e" not in readme
    assert "uv sync --extra dev" in readme
    assert "uv run --no-sync cayu eval run" in readme
    assert "Run setup and proof commands in the listed order" in readme
    assert "Do not parallelize application-" in (proj / "AGENTS.md").read_text(encoding="utf-8")
    assert "uv run --no-sync cayu session list" in readme
    assert "uv run --no-sync cayu serve --dev" in readme
    assert "http://127.0.0.1:8000/cayu/" in readme
    assert "developer/operator control plane" in readme
    assert "no Evals-specific Python configuration" in readme
    assert "uv run --no-sync cayu guide evals-first" in readme
    assert "cayu guide evals-ai-quality" in readme
    assert "Fresh Evals execution remains gated" not in readme
    assert "Never mount it with unauthenticated open access on a public listener" in readme
    assert "client-IP checks are not authentication" in readme
    assert "uv run --no-sync cayu auth openai login" in readme
    assert "CAYU_PROVIDER=openai-subscription" in readme
    assert "CAYU_PROVIDER=anthropic" in readme
    assert "ANTHROPIC_API_KEY" in readme
    assert "CAYU_PROVIDER=openrouter" in readme
    assert "OPENROUTER_API_KEY" in readme
    assert "CAYU_MODEL=vendor/model" in readme
    assert 'provider_options["openrouter"]' in readme
    assert "subscription holder's own local" in readme
    assert "development and evaluation" in readme
    assert "not intended for production" in readme
    assert "bypassing plan limits" in readme
    assert "cayu eval run evals.agent:build_eval" not in readme
    assert "uv run --no-sync cayu guide authoring#cayu-map" in readme
    assert "github.com" not in readme
    assert 'uv run --no-sync python run.py --message "YOUR REQUEST"' in readme
    assert "cayu generate slice" not in readme
    output = capsys.readouterr().out
    assert "uv sync --extra dev" in output
    assert "uv run --no-sync cayu check --fail-on warning --json" in output
    assert "uv run --no-sync cayu serve --dev" in output
    assert "http://127.0.0.1:8000/cayu/" in output
    assert "none selected" in output


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX umask semantics")
def test_cayu_new_preserves_restrictive_creation_umask(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import os
import stat
from pathlib import Path
from cayu.cli import main

parent = Path({str(tmp_path)!r})
os.umask(0o077)
if main(["new", "private-project", "--dir", str(parent), "--json"]) != 0:
    raise SystemExit(71)
project = parent / "private-project"
expected = {{
    project: 0o700,
    project / "data": 0o700,
    project / "data" / "artifacts": 0o700,
    project / "app.py": 0o600,
    project / "data" / "memory-evidence.key": 0o600,
}}
for path, mode in expected.items():
    if stat.S_IMODE(path.stat().st_mode) != mode:
        raise SystemExit(72)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX mode semantics")
def test_cayu_new_preserves_preexisting_empty_target_mode(tmp_path: Path) -> None:
    target = tmp_path / "existing-project"
    target.mkdir()
    target.chmod(0o711)

    assert main(["new", target.name, "--dir", str(tmp_path), "--json"]) == 0

    assert stat.S_IMODE(target.stat(follow_symlinks=False).st_mode) == 0o711


def test_cayu_new_coding_emits_explicit_composition_and_clean_git_baseline(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_coding_dependency_preflight(monkeypatch)
    assert main(["new", "coder", "--preset", "coding", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "coder"

    for filename in (
        "environments/command_probe.py",
        "environments/coding.py",
        "knowledge/coding.py",
        "operations/coding.py",
        "policies/coding.py",
        "prompts/coding.py",
        "tools/coding.py",
        "agents/reviewer.py",
        "tests/test_coding_composition.py",
    ):
        assert (project / filename).is_file()
    assert (
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    assert (
        subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == "Initial Cayu coding composition"
    )

    composition = (project / "operations/coding.py").read_text(encoding="utf-8")
    command_probe = (project / "environments/command_probe.py").read_text(encoding="utf-8")
    assert (project / ".gitignore").read_text(encoding="utf-8").startswith(".cayu/\n")
    assert "from environments.command_probe import" in composition
    assert "cayu.cli._bounded_command" not in composition
    assert "run_bounded_command" in command_probe
    for public_surface in (
        "LocalWorkspace",
        "LocalRunner",
        "LocalArtifactStore",
        "SearchTextTool",
        "GitChangesTool",
        "SubagentTool",
        "SubagentResultTool",
        "UserInputTool",
        "ParameterConstrainedToolPolicy",
    ):
        assert public_surface in composition
    assert "AllRegisteredToolsExposurePolicy" in (project / "agents/registration.py").read_text(
        encoding="utf-8"
    )
    assert "SQLiteKnowledgeStore" in (project / "configuration/coding_storage.py").read_text(
        encoding="utf-8"
    )
    assert "mode=SubagentExecutionMode.BACKGROUND" in composition
    assert 'os.environ.get("CAYU_WORKSPACE_ROOT", ".")' in (
        project / "environments/coding.py"
    ).read_text(encoding="utf-8")
    assert "inherit_env=False" in composition
    assert '_STATE_ROOT = _PROJECT_ROOT / ".cayu" / "runtime"' in composition
    assert "excluded_directory_names=_PROTECTED_WORKSPACE_DIRECTORY_NAMES" in composition
    assert "LocalWorkspace.require_path_operations_supported()" in composition
    assert "exclude_directories=_SEARCH_EXCLUDED_DIRECTORIES" in composition
    assert "protected_entry_names=_PROTECTED_WORKSPACE_DIRECTORY_NAMES" in composition
    readme = (project / "README.md").read_text(encoding="utf-8")
    instructions = (project / "AGENTS.md").read_text(encoding="utf-8")
    for identity in (
        "_PRIMARY_TOOL_POLICY_IDENTITY",
        "_SUBAGENT_RESULT_TOOL_IDENTITY",
        "_coding_environment_identity()",
        "REVIEWER_EXECUTION_PROFILE_IDENTITY",
        "_subagent_tool_identity()",
    ):
        assert identity in readme
        assert identity in instructions
    assert "older continuations fail closed" in " ".join(readme.split())
    assert "stale durable continuations fail closed" in " ".join(instructions.split())
    assert "project-owned standard-library support" in " ".join(readme.split())
    assert "state is stored below that protected `.cayu` boundary" in " ".join(readme.split())
    assert "runtime-private `.cayu` directories excluded" in " ".join(instructions.split())
    assert "do not replace it with an import from Cayu's private modules" in " ".join(
        instructions.split()
    )

    spec = importlib.util.spec_from_file_location("coding_scaffolded_app", project / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with project_context(project):
        spec.loader.exec_module(module)
        monkeypatch.setitem(
            module.build_coding_app.__globals__,
            "_verify_coding_dependencies",
            lambda root: None,
        )
        app = module.build_app(
            provider=ScriptedModelProvider([]),
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )
    manifest = app.describe()
    assert {agent.name for agent in manifest.agents} == {"coder", "coder-reviewer"}
    registered_primary = app._agents["coder"]
    assert registered_primary.tools["subagent"].tool.spec.execution_profile_identity is None
    assert registered_primary.tools["subagent_result"].tool.spec.execution_profile_identity is None
    primary = next(agent for agent in manifest.agents if agent.name == "coder")
    assert {tool.name for tool in primary.tools} >= {
        "list_files",
        "search_text",
        "read_file",
        "apply_patch",
        "write_file",
        "edit_file",
        "delete_file",
        "git_changes",
        "list_artifacts",
        "list_knowledge",
        "search_knowledge",
        "read_knowledge",
        "remember_knowledge",
        "subagent",
        "subagent_result",
        "ask_user",
    }
    assert manifest.defaults.environment == "coding"
    assert manifest.stores.knowledge == "SQLiteKnowledgeStore"

    output = capsys.readouterr().out
    assert "pytest -q tests/test_coding_composition.py" in output
    assert '--agent coder --message "YOUR REQUEST"' in output


def test_cayu_new_docker_coding_emits_explicit_checks_and_immutable_image_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_coding_dependency_preflight(monkeypatch)
    assert (
        main(
            [
                "new",
                "docker-coder",
                "--preset",
                "coding",
                "--execution",
                "docker",
                "--coding-toolchain",
                "python",
                "--dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    project = tmp_path / "docker-coder"
    expected_support = {
        ".dockerignore",
        "Dockerfile.coding",
        "build_coding_image.py",
        "docker-coding-build.json",
        "docker-coding-image.json",
        "domain/coding_product.py",
        "tests/test_project.py",
        "workflows/coding_product.py",
    }
    assert all((project / path).is_file() for path in expected_support)
    private_cayu_imports = {
        path.relative_to(project).as_posix()
        for path in project.rglob("*.py")
        if "from cayu._" in path.read_text(encoding="utf-8")
        or "import cayu._" in path.read_text(encoding="utf-8")
    }
    assert private_cayu_imports == set()
    assert (
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )

    composition_source = (project / "operations/coding.py").read_text(encoding="utf-8")
    primary_source = (project / "agents" / "agent.py").read_text(encoding="utf-8")
    dockerfile = (project / "Dockerfile.coding").read_text(encoding="utf-8")
    builder = (project / "build_coding_image.py").read_text(encoding="utf-8")
    readme = (project / "README.md").read_text(encoding="utf-8")
    domain_source = (project / "domain" / "coding_product.py").read_text(encoding="utf-8")
    workflow_source = (project / "workflows" / "coding_product.py").read_text(encoding="utf-8")
    assert "DockerCodingEnvironmentFactory" in composition_source
    assert "DockerWorkspaceTransferLimits" in composition_source
    assert "RunCheckTool" in composition_source
    assert "RunCommandTool" in composition_source
    assert "StructuredCommandToolPolicy" in composition_source
    assert "DockerCodingToolchainProfile" in composition_source
    assert "NamedCheck" in composition_source
    assert "_ExactCheckCommandPolicy" in composition_source
    assert '"docker-coding-image.json",' in composition_source
    assert "does not match the current schema version 3" in composition_source
    assert "_PYTHON_TOOLCHAIN_DEPENDENCY_PATHS" in composition_source
    assert "values=list(check_names)" in composition_source
    assert "values=list(command_selectors)" in composition_source
    assert 'code_trust="untrusted"' not in composition_source
    assert "ExecutionRequirements.untrusted" not in composition_source
    assert "ExecCommandTool" not in composition_source
    assert "LocalRunner" not in composition_source
    assert r"(?:\.cayu|\.git|\.runtime)" in composition_source
    assert "PRIMARY_EXECUTION_PROFILE_IDENTITY" in primary_source
    assert "ARG CAYU_BASE_IMAGE" in dockerfile
    assert "ARG CAYU_DEBIAN_SNAPSHOT" in dockerfile
    assert "snapshot.debian.org/archive/debian" in dockerfile
    assert "uv sync --frozen --extra dev --no-install-project" in dockerfile
    assert "from=cayu-wheel" in dockerfile
    assert "pip install" in dockerfile
    assert "cayu_wheel_sha256" in builder
    assert '"--build-context"' in builder
    assert "Docker is the P1 bounded path" in readme
    assert "#1191" in readme
    assert "does not claim exact" in readme
    assert "patch_ready_for_delivery" in readme
    assert "class CodingProductTask" in domain_source
    assert "class CodingProductApplication" in workflow_source
    assert "await asyncio.to_thread(" in workflow_source
    assert "source_git_authority_validator=self._validate_source_git_authority" in workflow_source
    assert "admit_or_recover_coding_product_request" in workflow_source
    assert "register_coding_product_contract" in workflow_source
    assert json.loads((project / "docker-coding-image.json").read_text())["content_digest"] is None
    build_configuration = json.loads((project / "docker-coding-build.json").read_text())
    assert build_configuration["cayu_wheel"] is None
    assert build_configuration["cayu_wheel_sha256"] is None
    assert build_configuration["debian_snapshot"] is None
    assert build_configuration["debian_suite"] is None
    scaffold_output = capsys.readouterr().out
    assert "Build and record image" in scaffold_output
    assert "admitted trusted-repository Docker checks and commands" in scaffold_output

    check_cli = importlib.import_module("cayu.cli.check")

    def build_app_without_host_dependency_probe(
        target: str,
        *,
        command: str = "Project",
    ) -> CayuApp:
        del command
        assert target == "app:build_app"
        composition = importlib.import_module("operations.coding")
        monkeypatch.setattr(composition, "_verify_coding_dependencies", lambda root: None)
        app_module = importlib.import_module("app")
        return app_module.build_app()

    # This assertion owns the immutable-image diagnostic, not the host runner's
    # optional ripgrep installation. The ordinary dependency preflight has
    # dedicated semantic tests below.
    monkeypatch.setattr(
        check_cli,
        "build_project_app",
        build_app_without_host_dependency_probe,
    )
    monkeypatch.chdir(project)
    assert main(["check", "--json"]) == 2
    diagnostic = json.loads(capsys.readouterr().out)
    assert diagnostic["error"]["code"] == "PROJECT_CHECK_FAILED"
    assert "no immutable image ID" in diagnostic["error"]["message"]
    assert "environment" not in diagnostic["error"]

    spec = importlib.util.spec_from_file_location(
        "docker_coding_scaffolded_app", project / "app.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    image_id = "sha256:" + ("a" * 64)
    with project_context(project):
        spec.loader.exec_module(module)
        globals_ = module.build_coding_app.__globals__
        monkeypatch.setitem(globals_, "_verify_coding_dependencies", lambda root: None)
        monkeypatch.setitem(
            globals_,
            "_configured_docker_authority",
            lambda root: (
                DockerImageIdentity(
                    reference="docker-coder:test",
                    content_digest=image_id,
                ),
                "/usr/bin/docker",
            ),
        )
        app = module.build_app(
            provider=ScriptedModelProvider([]),
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )

    environment = app._environments["coding"]
    assert environment.factory_backed is True
    assert isinstance(environment.factory, DockerCodingEnvironmentFactory)
    primary = app._agents["docker-coder"]
    reviewer = app._agents["docker-coder-reviewer"]
    assert set(reviewer.tools) == set()
    assert "run_check" in primary.tools
    assert "apply_patch" in primary.tools
    assert "run_command" in primary.tools
    assert "exec_command" not in primary.tools
    assert primary.execution_requirements.code_trust == "trusted"
    assert primary.execution_requirements.network_access == "deny_by_default"
    assert primary.tools["run_check"].tool.schema["properties"]["check"]["enum"] == [
        "format",
        "lint",
        "test",
    ]
    assert primary.tools["run_command"].publish_arguments is False
    assert isinstance(primary.tool_policy, StructuredCommandToolPolicy)
    assert primary.tools["run_command"].tool.schema["properties"]["selector"]["enum"] == [
        "focused-test",
        "lint-file",
        "python-version",
    ]
    test_check = primary.tools["run_check"].tool._checks_by_name["test"]
    assert tuple(test_check.command.argv or ()) == (
        "/opt/cayu-project/.venv/bin/pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
    )

    generated_environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    inherited_pythonpath = generated_environment.get("PYTHONPATH")
    generated_environment["PYTHONPATH"] = (
        source_root
        if inherited_pythonpath is None
        else os.pathsep.join((source_root, inherited_pythonpath))
    )
    generated_suite = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_architecture.py",
            "tests/test_coding_composition.py",
        ],
        cwd=project,
        env=generated_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert generated_suite.returncode == 0, generated_suite.stdout + generated_suite.stderr


def test_generated_docker_builder_uses_one_immutable_input_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_coding_dependency_preflight(monkeypatch)
    assert (
        main(
            [
                "new",
                "snapshot-builder",
                "--preset",
                "coding",
                "--execution",
                "docker",
                "--coding-toolchain",
                "python",
                "--dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    project = tmp_path / "snapshot-builder"
    build_configuration_path = project / "docker-coding-build.json"
    build_configuration = json.loads(build_configuration_path.read_text(encoding="utf-8"))
    build_configuration.update(
        {
            "base_image": "python@sha256:" + ("b" * 64),
            "uv_version": "0.9.0",
            "debian_snapshot": "20260101T000000Z",
            "debian_suite": "bookworm",
            "git_package": "1:2.39.5-0+deb12u2",
            "ripgrep_package": "13.0.0-4+b2",
        }
    )
    build_configuration_path.write_text(
        json.dumps(build_configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    image_configuration_path = project / "docker-coding-image.json"
    empty_image_configuration = image_configuration_path.read_bytes()
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    original_lock = (project / "uv.lock").read_bytes()
    spec = importlib.util.spec_from_file_location(
        "snapshot_builder_build_coding_image",
        project / "build_coding_image.py",
    )
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/usr/bin/docker")
    real_subprocess_run = subprocess.run
    built_image_id = "sha256:" + ("a" * 64)
    mutate_during_build = False
    substitute_tag = False
    captured_build_contexts: list[dict[str, bytes]] = []

    def fake_run(command, **kwargs):
        del kwargs
        nonlocal mutate_during_build, substitute_tag
        argv = tuple(command)
        if argv[1] == "build":
            context = Path(argv[-1])
            assert context != project
            captured_build_contexts.append(
                {
                    path.relative_to(context).as_posix(): path.read_bytes()
                    for path in context.rglob("*")
                    if path.is_file()
                }
            )
            image_id_receipt = Path(argv[argv.index("--iidfile") + 1])
            image_id_receipt.write_text(built_image_id + "\n", encoding="ascii")
            if mutate_during_build:
                (project / "uv.lock").write_bytes(original_lock + b"# changed during build\n")
            return subprocess.CompletedProcess(command, 0)
        if argv[1:3] == ("image", "inspect"):
            if argv[4] == "{{.Architecture}}":
                assert argv[-1] == built_image_id
                return subprocess.CompletedProcess(command, 0, stdout=b"amd64\n")
            assert argv[-1] == build_configuration["image_reference"]
            tagged_image_id = "sha256:" + (("c" if substitute_tag else "a") * 64)
            return subprocess.CompletedProcess(
                command, 0, stdout=(tagged_image_id + "\n").encode("ascii")
            )
        if argv[1] == "run":
            image_argument = argv.index("--entrypoint") + 2
            assert argv[image_argument] == built_image_id
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(f"unexpected Docker command: {argv}")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    assert builder.main() == 0
    recorded = json.loads(image_configuration_path.read_text(encoding="utf-8"))
    assert recorded["content_digest"] == built_image_id
    recorded_dependencies = {
        item["path"]: item["content_sha256"] for item in recorded["dependency_inputs"]
    }
    assert recorded_dependencies["uv.lock"] == (
        "sha256:" + hashlib.sha256(original_lock).hexdigest()
    )
    assert captured_build_contexts == [
        {
            path: builder._read_project_input(path, max_bytes=limit)
            for path, limit in builder._BUILD_CONTEXT_INPUT_LIMITS.items()
        }
    ]
    generated_environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    inherited_pythonpath = generated_environment.get("PYTHONPATH")
    generated_environment["PYTHONPATH"] = (
        source_root
        if inherited_pythonpath is None
        else os.pathsep.join((source_root, inherited_pythonpath))
    )
    configured_self_test = real_subprocess_run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_coding_composition.py::test_image_build_requires_reviewed_pinned_inputs",
        ],
        cwd=project,
        env=generated_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert configured_self_test.returncode == 0, (
        configured_self_test.stdout + configured_self_test.stderr
    )

    image_configuration_path.write_bytes(empty_image_configuration)
    substitute_tag = True
    with pytest.raises(RuntimeError, match="tag changed"):
        builder.main()
    assert image_configuration_path.read_bytes() == empty_image_configuration

    image_configuration_path.write_bytes(empty_image_configuration)
    substitute_tag = False
    mutate_during_build = True
    with pytest.raises(RuntimeError, match="changed during the image build"):
        builder.main()
    assert image_configuration_path.read_bytes() == empty_image_configuration
    assert captured_build_contexts[-1]["uv.lock"] == original_lock


def test_coding_execution_requires_the_coding_composition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "new",
                "invalid-docker",
                "--execution",
                "docker",
                "--dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert not (tmp_path / "invalid-docker").exists()
    assert "execution" in capsys.readouterr().err


def test_coding_toolchain_requires_docker_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "new",
                "invalid-toolchain",
                "--preset",
                "coding",
                "--coding-toolchain",
                "python",
                "--dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert not (tmp_path / "invalid-toolchain").exists()
    assert "requires --execution docker" in capsys.readouterr().err


def test_cayu_new_coding_rejects_unsupported_local_workspace_before_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unsupported() -> None:
        raise RuntimeError(
            "LocalWorkspace requires POSIX descriptor-relative filesystem primitives."
        )

    monkeypatch.setattr(
        LocalWorkspace,
        "require_path_operations_supported",
        staticmethod(unsupported),
    )

    assert main(["new", "coder", "--preset", "coding", "--dir", str(tmp_path)]) == 1

    assert not (tmp_path / "coder").exists()
    assert "requires POSIX descriptor-relative filesystem primitives" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows capability evidence")
def test_cayu_new_coding_rejects_native_windows_workspace_before_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "coder", "--preset", "coding", "--dir", str(tmp_path)]) == 1

    assert not (tmp_path / "coder").exists()
    assert "requires POSIX descriptor-relative filesystem primitives" in capsys.readouterr().err


def test_cayu_new_coding_rejects_missing_dependencies(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    from cayu.cli import scaffold

    real_which = scaffold.shutil.which
    monkeypatch.setattr(
        scaffold.shutil,
        "which",
        lambda command: None if command == "rg" else real_which(command),
    )
    assert main(["new", "missing-rg", "--preset", "coding", "--dir", str(tmp_path)]) == 1
    assert not (tmp_path / "missing-rg").exists()
    assert "requires these commands on PATH: rg" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "unsupported_flag"),
    [
        ("git", "--force"),
        ("git", "ls-files"),
        ("git", "--cached"),
        ("git", "--numstat"),
        ("rg", "--files-with-matches"),
        ("rg", "--count-matches"),
        ("rg", "--glob"),
        ("rg", "--iglob"),
        ("rg", "--ignore-case"),
    ],
)
def test_cayu_new_coding_rejects_incompatible_runtime_command_dialects(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    unsupported_flag: str,
) -> None:
    from cayu.cli import scaffold
    from cayu.cli._bounded_command import BoundedCommandResult

    _install_hermetic_coding_preflight(monkeypatch)

    def incompatible(argv, **kwargs):
        if Path(argv[0]).name == command and unsupported_flag in argv:
            return BoundedCommandResult(
                returncode=7,
                output=b"",
                output_truncated=False,
            )
        return _compatible_coding_probe_result()

    monkeypatch.setattr(scaffold, "run_bounded_command", incompatible)
    project_name = f"incompatible-{command}-{unsupported_flag.removeprefix('--')}"
    assert (
        main(
            [
                "new",
                project_name,
                "--preset",
                "coding",
                "--dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert not (tmp_path / project_name).exists()
    assert f"{command} failed with exit code 7" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "semantic_argument"),
    [
        ("git", "status"),
        ("rg", "--files"),
    ],
)
def test_cayu_new_coding_rejects_false_success_dependency_shims(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    semantic_argument: str,
) -> None:
    from cayu.cli import scaffold
    from cayu.cli._bounded_command import BoundedCommandResult

    _install_hermetic_coding_preflight(monkeypatch)

    def false_success(argv, **kwargs):
        if Path(argv[0]).name == command and semantic_argument in argv:
            return BoundedCommandResult(
                returncode=0,
                output=b"",
                output_truncated=False,
            )
        return _compatible_coding_probe_result()

    monkeypatch.setattr(scaffold, "run_bounded_command", false_success)
    project_name = f"false-success-{command}"
    assert (
        main(
            [
                "new",
                project_name,
                "--preset",
                "coding",
                "--dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert not (tmp_path / project_name).exists()
    assert f"{command} semantic probe failed" in capsys.readouterr().err


def test_cayu_new_coding_confines_git_authority_and_ignores_global_hooks(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold

    _bypass_coding_dependency_preflight(monkeypatch)

    canary = tmp_path / "canary"
    canary.mkdir()
    setup_hooks = tmp_path / "setup-hooks"
    setup_hooks.mkdir()
    git = scaffold.shutil.which("git")
    assert git is not None
    setup_environment = scaffold._sanitized_scaffold_git_environment(cwd=canary)
    scaffold._run_scaffold_command(
        scaffold._safe_git_argv(
            git,
            "init",
            "-b",
            "main",
            f"--template={setup_hooks}",
            hooks_dir=setup_hooks,
        ),
        cwd=canary,
        env=setup_environment,
    )
    (canary / "owned.txt").write_text("unchanged\n", encoding="utf-8")
    scaffold._run_scaffold_command(
        scaffold._safe_git_argv(git, "add", "--", ".", hooks_dir=setup_hooks),
        cwd=canary,
        env=setup_environment,
    )
    scaffold._run_scaffold_command(
        scaffold._safe_git_argv(
            git,
            "-c",
            "user.name=Cayu Test",
            "-c",
            "user.email=test@cayu.local",
            "commit",
            "-m",
            "canary",
            hooks_dir=setup_hooks,
        ),
        cwd=canary,
        env=setup_environment,
    )
    ambient_index = tmp_path / "ambient-index"
    ambient_index.write_bytes((canary / ".git" / "index").read_bytes())
    before = ambient_index.read_bytes()

    fake_home = tmp_path / "home"
    hooks = fake_home / "hooks"
    hooks.mkdir(parents=True)
    global_git = fake_home / ".config" / "git"
    global_git.mkdir(parents=True)
    (global_git / "ignore").write_text("*.py\n", encoding="utf-8")
    (global_git / "attributes").write_text("*.md binary\n", encoding="utf-8")
    hook_marker = tmp_path / "ambient-hook-ran"
    hook = hooks / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch {hook_marker}\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    (fake_home / ".gitconfig").write_text(
        f"[core]\n\thooksPath = {hooks}\n[commit]\n\tgpgSign = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("GIT_DIR", str(canary / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(canary))
    monkeypatch.setenv("GIT_INDEX_FILE", str(ambient_index))

    assert main(["new", "safe-coder", "--preset", "coding", "--dir", str(tmp_path)]) == 0
    assert ambient_index.read_bytes() == before
    assert not hook_marker.exists()
    project = tmp_path / "safe-coder"
    assert (project / ".git").is_dir()
    tracked_output = scaffold._run_scaffold_command(
        scaffold._safe_git_argv(
            git,
            "ls-files",
            "--cached",
            "-z",
            "--",
            hooks_dir=setup_hooks,
        ),
        cwd=project,
        env=scaffold._sanitized_scaffold_git_environment(cwd=project),
    )
    assert {path for path in tracked_output.split("\0") if path} == set(
        scaffold.project_files("safe-coder", preset="coding")
    )
    assert "Scaffolded" in capsys.readouterr().out


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX sticky-directory modes")
def test_cayu_new_coding_rejects_nonsticky_shared_parent(
    tmp_path: Path,
    capsys,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)

    assert main(["new", "unsafe-coder", "--preset", "coding", "--dir", str(shared)]) == 1
    assert not (shared / "unsafe-coder").exists()
    assert "group/world-writable unless it is sticky" in capsys.readouterr().err


def test_scaffold_parent_mode_does_not_apply_posix_bits_on_windows() -> None:
    from cayu.cli import scaffold

    assert scaffold._unsafe_shared_scaffold_parent_mode(0o777, platform="posix") is True
    assert scaffold._unsafe_shared_scaffold_parent_mode(0o1777, platform="posix") is False
    assert scaffold._unsafe_shared_scaffold_parent_mode(0o777, platform="nt") is False


def test_cayu_new_coding_removes_generated_files_after_git_failure(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold

    _bypass_coding_dependency_preflight(monkeypatch)

    original = scaffold._run_scaffold_command

    def fail_target_commit(argv, *, cwd, env=None, allowed_exit_codes=frozenset({0})):
        if cwd.name.startswith(".cayu-tree-stage-") and "commit" in argv:
            raise scaffold._ScaffoldCommandError("forced commit failure")
        return original(
            argv,
            cwd=cwd,
            env=env,
            allowed_exit_codes=allowed_exit_codes,
        )

    monkeypatch.setattr(scaffold, "_run_scaffold_command", fail_target_commit)
    assert main(["new", "broken-coder", "--preset", "coding", "--dir", str(tmp_path)]) == 1
    assert not (tmp_path / "broken-coder").exists()
    assert "forced commit failure" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX process-signal delivery")
def test_scaffold_git_signal_is_not_masked_by_post_command_identity_failure(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "stage"
    working_directory.mkdir()
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import os
import signal
from pathlib import Path
from cayu.cli import scaffold

working_directory = Path({str(working_directory)!r})
checks = 0

def assert_unchanged():
    global checks
    checks += 1
    if checks > 1:
        raise OSError("secondary identity failure")

def interrupt_command(*args, **kwargs):
    del args, kwargs
    os.kill(os.getpid(), signal.SIGINT)
    raise SystemExit(91)

scaffold._run_scaffold_command = interrupt_command
identity = scaffold._scaffold_directory_identity(working_directory)
try:
    scaffold._run_scaffold_git_command(
        ["git", "status"],
        cwd=working_directory,
        env={{}},
        expected_directory_identity=identity,
        assert_directory_unchanged=assert_unchanged,
    )
except KeyboardInterrupt:
    if checks != 1:
        raise SystemExit(77)
    raise SystemExit(75)
raise SystemExit(76)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 75


def test_cayu_new_coding_preserves_preexisting_empty_target_after_git_failure(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold

    _bypass_coding_dependency_preflight(monkeypatch)

    target = tmp_path / "existing-coder"
    target.mkdir(mode=0o700)
    before = target.stat()
    original = scaffold._run_scaffold_command

    def fail_target_commit(argv, *, cwd, env=None, allowed_exit_codes=frozenset({0})):
        if cwd.name.startswith(".cayu-tree-stage-") and "commit" in argv:
            raise scaffold._ScaffoldCommandError("forced commit failure")
        return original(
            argv,
            cwd=cwd,
            env=env,
            allowed_exit_codes=allowed_exit_codes,
        )

    monkeypatch.setattr(scaffold, "_run_scaffold_command", fail_target_commit)
    assert main(["new", target.name, "--preset", "coding", "--dir", str(tmp_path)]) == 1
    after = target.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert after.st_mode == before.st_mode
    assert list(target.iterdir()) == []
    assert "forced commit failure" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlink semantics")
def test_cayu_new_coding_cleanup_does_not_follow_replaced_generated_directory(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold

    _bypass_coding_dependency_preflight(monkeypatch)

    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "reviewer.py"
    canary.write_text("outside-owned\n", encoding="utf-8")
    original = scaffold._run_scaffold_command

    def replace_agents_then_fail(argv, *, cwd, env=None, allowed_exit_codes=frozenset({0})):
        if cwd.name.startswith(".cayu-tree-stage-") and "commit" in argv:
            shutil.rmtree(cwd / "agents")
            (cwd / "agents").symlink_to(outside, target_is_directory=True)
            raise scaffold._ScaffoldCommandError("forced commit failure")
        return original(
            argv,
            cwd=cwd,
            env=env,
            allowed_exit_codes=allowed_exit_codes,
        )

    monkeypatch.setattr(scaffold, "_run_scaffold_command", replace_agents_then_fail)
    assert main(["new", "symlink-coder", "--preset", "coding", "--dir", str(tmp_path)]) == 1
    assert canary.read_text(encoding="utf-8") == "outside-owned\n"
    assert not (tmp_path / "symlink-coder").exists()
    assert "forced commit failure" in capsys.readouterr().err


def test_cayu_new_coding_does_not_clean_replacement_target(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_coding_dependency_preflight(monkeypatch)

    target = tmp_path / "replaced-coder"
    replacement_canary = "replacement-owned\n"
    from cayu.cli import _guarded_tree_publication as publication

    original = publication._publication_fault

    def replace_before_publish(phase: str) -> None:
        if phase != "stage_created":
            original(phase)
            return
        target.mkdir()
        (target / ".git").mkdir()
        (target / ".git" / "replacement-owned").write_text(
            replacement_canary,
            encoding="utf-8",
        )
        (target / "pyproject.toml").write_text(replacement_canary, encoding="utf-8")
        (target / "owned.txt").write_text(replacement_canary, encoding="utf-8")

    monkeypatch.setattr(publication, "_publication_fault", replace_before_publish)
    assert main(["new", target.name, "--preset", "coding", "--dir", str(tmp_path)]) == 1
    assert (target / "pyproject.toml").read_text(encoding="utf-8") == replacement_canary
    assert (target / "owned.txt").read_text(encoding="utf-8") == replacement_canary
    assert (target / ".git" / "replacement-owned").read_text(encoding="utf-8") == (
        replacement_canary
    )
    assert "destination appeared" in capsys.readouterr().err


def test_cayu_new_does_not_adopt_concurrent_empty_target(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "concurrent-empty"
    from cayu.cli import _guarded_tree_publication as publication

    original = publication._publication_fault
    replacement_identity: tuple[int, int] | None = None

    def create_empty_target_after_admission(phase: str) -> None:
        nonlocal replacement_identity
        if phase != "stage_created":
            original(phase)
            return
        target.mkdir()
        value = target.stat(follow_symlinks=False)
        replacement_identity = (value.st_dev, value.st_ino)

    monkeypatch.setattr(
        publication,
        "_publication_fault",
        create_empty_target_after_admission,
    )

    assert main(["new", target.name, "--dir", str(tmp_path), "--json"]) == 1

    current = target.stat(follow_symlinks=False)
    assert replacement_identity is not None
    assert (current.st_dev, current.st_ino) == replacement_identity
    assert list(target.iterdir()) == []
    assert "destination appeared" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("failure", "detail"),
    [
        ("start", "could not start"),
        ("timeout", "timed out"),
        ("read", "output could not be read"),
    ],
)
def test_scaffold_command_probe_bounds_start_and_timeout_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    detail: str,
) -> None:
    from cayu.cli import scaffold

    def fail(*args, **kwargs):
        del args, kwargs
        if failure == "start":
            raise scaffold.BoundedCommandStartError
        if failure == "timeout":
            raise scaffold.BoundedCommandTimeoutError
        raise scaffold.BoundedCommandReadError

    monkeypatch.setattr(scaffold, "run_bounded_command", fail)
    with pytest.raises(scaffold._ScaffoldCommandError, match=detail):
        scaffold._run_scaffold_command(["rg", "--version"], cwd=tmp_path)


def test_scaffold_command_probe_reports_incompatible_exit_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold
    from cayu.cli._bounded_command import BoundedCommandResult

    canary = "PRIVATE_COMMAND_OUTPUT"

    def incompatible(argv, **kwargs):
        del argv, kwargs
        return BoundedCommandResult(
            returncode=9,
            output=canary.encode(),
            output_truncated=False,
        )

    monkeypatch.setattr(scaffold, "run_bounded_command", incompatible)
    with pytest.raises(scaffold._ScaffoldCommandError) as raised:
        scaffold._run_scaffold_command(["rg", "--version"], cwd=tmp_path)
    assert "exit code 9" in str(raised.value)
    assert canary not in str(raised.value)


def test_cayu_new_service_emits_the_supported_secure_product_shell(
    tmp_path: Path,
    capsys,
) -> None:
    assert main(["new", "myservice", "--preset", "service", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myservice"

    for filename in (
        "service.py",
        "product_store.py",
        "tests/test_public_service_security.py",
    ):
        assert (project / filename).is_file()
    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert f'dependencies = ["cayu[server]=={cayu_version}"]' in pyproject
    assert 'dev = ["pytest", "ruff>=0.15.15,<0.16"]' in pyproject
    assert 'service_factory = "service:build_service"' in pyproject
    assert 'factory = "app:build_app"' in pyproject

    service_source = (project / "service.py").read_text(encoding="utf-8")
    product_store_source = (project / "product_store.py").read_text(encoding="utf-8")
    assert "create_agent_service(" in service_source
    assert "await asyncio.to_thread" in product_store_source
    assert "claim_execution" in product_store_source
    assert "heartbeat_execution" in product_store_source
    assert "release_execution" in product_store_source
    assert "record_result_receipt" in product_store_source
    assert "result_receipt" in product_store_source
    assert "find_by_session_id" in product_store_source
    assert "execution_claim_id" in product_store_source
    assert "ProductExecutionClaimLost" in product_store_source
    assert "AuthenticatedProductAccess" in service_source
    assert "AuthenticatedAccess" in service_source
    assert "PlaceholderOperatorAccess" in service_source
    assert "ProductPrincipal" in service_source
    assert "ProjectControlPlaneContext" in service_source
    assert "project_context: ProjectControlPlaneContext | None = None" in service_source
    assert "project_context=project_context" in service_source
    assert "PRODUCT_AUTH_TOKENS_JSON" in service_source
    assert "CAYU_OPERATOR_BEARER_TOKEN" in service_source
    assert "@app." not in service_source
    assert "create_agent_service(" in service_source

    security_test = (project / "tests/test_public_service_security.py").read_text(encoding="utf-8")
    assert "resolve_execution_profile_identity" in security_test
    for phrase in (
        "anonymous",
        "cross_tenant",
        "idempotency",
        "control_plane",
        "redact",
        "background",
        "oversized",
        "profiled_session_identity",
        "AdmitInvocationCommand",
        "apply_invocation_lifecycle_command",
    ):
        assert phrase in security_test

    readme = (project / "README.md").read_text(encoding="utf-8")
    agents = (project / "AGENTS.md").read_text(encoding="utf-8")
    for guidance in (readme, agents):
        assert "cayu check --deploy --fail-on warning --json" in guidance
        assert "pytest -q tests/test_public_service_security.py" in guidance
        assert "customer" in guidance
        assert "operator" in guidance
        assert "tenant-qualified" in guidance
        assert "arbitrary ASGI" in guidance
    assert "TLS-terminating" in readme
    assert "Never send either bearer token" in readme
    assert "at most 1 MiB of encoded JSON" in readme
    assert "private, no-store" in readme

    output = capsys.readouterr().out
    assert "cayu check --deploy --fail-on warning --json" in output
    assert "pytest -q tests/test_public_service_security.py" in output
    assert "Product API: http://127.0.0.1:8000/api/operations" in output
    assert "Operator control plane: http://127.0.0.1:8000/cayu/" in output


def test_scaffolded_service_replacement_recovery_uses_profiled_admission(
    tmp_path: Path,
) -> None:
    assert main(["new", "myservice", "--preset", "service", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myservice"
    environment = os.environ.copy()
    for name in ("CAYU_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, tempfile; from pathlib import Path; "
                "tests = runpy.run_path('tests/test_public_service_security.py'); "
                "context = tempfile.TemporaryDirectory(prefix='cayu-scaffold-profile-'); "
                "tests['test_replacement_worker_continues_same_durable_session']"
                "(Path(context.name)); context.cleanup()"
            ),
        ],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scaffolded_service_deploy_check_fails_closed_then_accepts_configured_auth(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "service", "--preset", "service", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "service"
    capsys.readouterr()
    monkeypatch.chdir(project)
    monkeypatch.delenv("PRODUCT_AUTH_TOKENS_JSON", raising=False)
    monkeypatch.delenv("CAYU_OPERATOR_BEARER_TOKEN", raising=False)

    assert main(["check", "--deploy", "--json"]) == 1
    unsafe = json.loads(capsys.readouterr().out)
    assert [item["code"] for item in unsafe["diagnostics"]] == [
        "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE",
        "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
    ]

    monkeypatch.setenv("PRODUCT_AUTH_TOKENS_JSON", '{"customer-token":null}')
    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "operator-token")
    assert main(["check", "--deploy", "--json"]) == 1
    invalid_principal = json.loads(capsys.readouterr().out)
    assert [item["code"] for item in invalid_principal["diagnostics"]] == [
        "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE"
    ]

    monkeypatch.setenv(
        "PRODUCT_AUTH_TOKENS_JSON",
        json.dumps({"tøk": {"tenant_id": "tenant-a", "subject_id": "alice"}}),
    )
    assert main(["check", "--deploy", "--json"]) == 1
    invalid_product_token = json.loads(capsys.readouterr().out)
    assert [item["code"] for item in invalid_product_token["diagnostics"]] == [
        "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE"
    ]

    monkeypatch.setenv(
        "PRODUCT_AUTH_TOKENS_JSON",
        '{"customer-token":{"tenant_id":"tenant-a","subject_id":"alice"}}',
    )
    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "øperator-token")
    assert main(["check", "--deploy", "--json"]) == 1
    invalid_operator_token = json.loads(capsys.readouterr().out)
    assert [item["code"] for item in invalid_operator_token["diagnostics"]] == [
        "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE"
    ]

    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "customer-token")
    assert main(["check", "--deploy", "--json"]) == 1
    overlapping = json.loads(capsys.readouterr().out)
    assert [item["code"] for item in overlapping["diagnostics"]] == [
        "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE"
    ]

    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "operator-token")
    assert main(["check", "--deploy", "--fail-on", "warning", "--json"]) == 0
    supported = json.loads(capsys.readouterr().out)
    assert supported["diagnostics"] == []
    assert supported["service_evidence"]["configuration"] == "supported"


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


def test_scaffold_openrouter_builds_first_class_provider_with_explicit_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path), "--provider", "openrouter"]) == 0
    project = tmp_path / "myproj"
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("CAYU_MODEL", "stealth/ox-alpha")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.test/cayu")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Cayu")
    monkeypatch.setenv("OPENROUTER_ROUTER_METADATA", "enabled")

    spec = importlib.util.spec_from_file_location("openrouter_scaffold_app", project / "app.py")
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
        module.validate_run_configuration(app, "myproj")

    assert app.describe().agents[0].model == "stealth/ox-alpha"
    assert isinstance(provider, ChatCompletionsProvider)
    assert provider.name == "openrouter"
    assert provider.api_key_env == "OPENROUTER_API_KEY"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    headers = provider._headers()
    assert headers["HTTP-Referer"] == "https://example.test/cayu"
    assert headers["X-OpenRouter-Title"] == "Cayu"
    assert headers["X-OpenRouter-Metadata"] == "enabled"


def test_scaffold_openrouter_requires_model_before_live_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path), "--provider", "openrouter"]) == 0
    project = tmp_path / "myproj"
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.delenv("CAYU_MODEL", raising=False)

    spec = importlib.util.spec_from_file_location(
        "openrouter_missing_model_scaffold_app",
        project / "app.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with project_context(project):
        spec.loader.exec_module(module)
        session_store = InMemorySessionStore()
        app = module.build_app(
            session_store=session_store,
            task_store=InMemoryTaskStore(),
        )
        with pytest.raises(RuntimeError, match="requires an explicit CAYU_MODEL"):
            module.validate_run_configuration(app, "myproj")

        outcome = asyncio.run(
            run_to_completion(
                app,
                RunRequest(
                    agent_name="myproj",
                    session_id="openrouter-missing-model-direct-sdk",
                    messages=[Message.text("user", "must not dispatch")],
                ),
            )
        )

    assert app.describe().agents[0].model == "openrouter-model-unconfigured"
    assert outcome.events == ()
    assert outcome.error == (
        "RuntimeError: provider 'openrouter' requires an explicit CAYU_MODEL model slug"
    )
    assert asyncio.run(session_store.load("openrouter-missing-model-direct-sdk")) is None


def test_scaffold_openrouter_requires_selected_provider_key_before_live_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path), "--provider", "openrouter"]) == 0
    project = tmp_path / "myproj"
    monkeypatch.setenv("CAYU_MODEL", "vendor/model")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    spec = importlib.util.spec_from_file_location(
        "openrouter_missing_key_scaffold_app",
        project / "app.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with project_context(project):
        spec.loader.exec_module(module)
        app = module.build_app(
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
            module.validate_run_configuration(app, "myproj")


def test_scaffold_openrouter_rejects_invalid_router_metadata_setting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path), "--provider", "openrouter"]) == 0
    project = tmp_path / "myproj"
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENROUTER_ROUTER_METADATA", "verbose")

    spec = importlib.util.spec_from_file_location(
        "openrouter_invalid_metadata_scaffold_app",
        project / "app.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with project_context(project):
        spec.loader.exec_module(module)
        with pytest.raises(RuntimeError, match="must be 'enabled' or 'disabled'"):
            module.configured_provider()


@pytest.mark.parametrize(
    ("scaffold_provider", "environment_provider"),
    [
        pytest.param(None, None, id="neutral"),
        pytest.param("openai", None, id="openai"),
        pytest.param("anthropic", None, id="anthropic"),
        pytest.param("openai-subscription", None, id="openai-subscription"),
        pytest.param("openrouter", None, id="openrouter"),
        pytest.param(None, "openai", id="neutral-env-openai"),
        pytest.param(None, "anthropic", id="neutral-env-anthropic"),
        pytest.param(None, "openai-subscription", id="neutral-env-openai-subscription"),
        pytest.param(None, "openrouter", id="neutral-env-openrouter"),
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
    for name in (
        "CAYU_PROVIDER",
        "CAYU_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ):
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


def test_standard_scaffold_inspect_and_check_are_credential_free(
    tmp_path: Path,
) -> None:
    assert main(["new", "standard", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "standard"
    environment = os.environ.copy()
    for name in (
        "CAYU_PROVIDER",
        "CAYU_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "CAYU_MEMORY_EVIDENCE_KEY",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")

    inspected = subprocess.run(
        [sys.executable, "-m", "cayu", "inspect", "--json"],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert inspected.returncode == 0, inspected.stdout + inspected.stderr
    manifest = json.loads(inspected.stdout)
    assert manifest["stores"]["knowledge"] == "SQLiteKnowledgeStore"
    assert manifest["runtime"]["request_footprint"]["fingerprinting_enabled"] is True
    assert manifest["agents"][0]["context_policy"] == "AutomaticRecallContextPolicy"

    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "cayu",
            "check",
            "--fail-on",
            "warning",
            "--json",
        ],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["diagnostics"] == []


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
    assert (project / "workflows" / "__init__.py").is_file()
    assert (project / "memory" / "context.py").is_file()
    capability_cards = (
        "knowledge/CAPABILITY.md",
        "memory/CAPABILITY.md",
        "environments/ARTIFACTS.md",
        "operations/TASKS.md",
        "operations/HUMAN_INPUT.md",
        "operations/APPROVALS.md",
        "operations/RECOVERY.md",
        "evals/CAPABILITY.md",
        "observability/CAPABILITY.md",
    )
    for relative in capability_cards:
        card = (project / relative).read_text(encoding="utf-8")
        for heading in (
            "**Behavior:**",
            "**Use it when:**",
            "**Project state:**",
            "**Restricted/unavailable:**",
            "**Explicit seam:**",
            "**Verify:**",
        ):
            assert heading in card

    app_source = (project / "app.py").read_text(encoding="utf-8")
    assert "ExecCommandTool" not in app_source
    registration_source = (project / "agents/registration.py").read_text(encoding="utf-8")
    assert "# <cayu:generated-imports>" in registration_source
    assert "# <cayu:generated-registrations>" in registration_source
    agent_source = (project / "agents" / "agent.py").read_text(encoding="utf-8")
    eval_source = (project / "evals" / "agent.py").read_text(encoding="utf-8")
    assert 'name="myproj"' in agent_source
    assert "_SYSTEM_PROMPT_PARTS: list[str] = list(SYSTEM_PROMPT_PARTS)" in agent_source
    assert "system_prompt=" in agent_source
    assert "workflow_tool_names=" in agent_source
    assert "ToolCalled" not in eval_source

    instructions = (project / "AGENTS.md").read_text(encoding="utf-8")
    readme = (project / "README.md").read_text(encoding="utf-8")
    for capability in (
        "approvals",
        "artifacts",
        "evals",
        "human-input",
        "knowledge",
        "memory",
        "observability",
        "recovery",
        "tasks",
    ):
        assert f"| `{capability}` | configured |" in readme
        assert f"| `{capability}` | configured |" in instructions
    assert "uv run --no-sync cayu guide anatomy" in instructions
    assert "uv run --no-sync cayu inspect --json" in instructions
    assert "uv run --no-sync cayu check --fail-on warning --json" in instructions
    assert "uv run --no-sync pytest" in instructions
    assert "uv run --no-sync cayu eval run" in instructions
    assert "uv run --no-sync cayu serve --dev" in instructions
    assert "http://127.0.0.1:8000/cayu/" in instructions
    assert "developer/operator control plane" in instructions
    assert "uv run --no-sync cayu guide evals-first" in instructions
    assert "end-user UI" in instructions
    assert "Never mount it with `OpenAccess()` on a public listener" in instructions
    assert "Client-IP and forwarded-header checks are not authentication" in instructions
    assert "cayu eval run evals.agent:build_eval" not in instructions
    assert "Edit the existing agent, test, and eval" in instructions
    assert "uv run --no-sync cayu guide authoring#cayu-map" in instructions
    assert "uv run --no-sync cayu guide references" in instructions
    assert "github.com" not in instructions
    assert "Deployment is a separate task" in instructions
    assert "Clarify users, jobs, triggers" not in instructions
    assert "cayu generate slice" not in instructions
    assert (
        "uv run --no-sync cayu generate tool TOOL_NAME --agent myproj --effect EFFECT"
        in instructions
    )


def test_cayu_new_routes_provider_questions_to_the_package_compatibility_guide(
    tmp_path: Path,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myproj"

    for relative_path in ("README.md", "AGENTS.md"):
        text = " ".join((project / relative_path).read_text(encoding="utf-8").split())
        assert "OpenRouter is a first-class scaffold choice" in text
        assert "other compatible endpoints work through Cayu's generic adapter" in text
        assert "uv run --no-sync cayu guide providers#compatible-chat-completions" in text
        for service in ("OpenRouter", "Fireworks", "Baseten", "OpenCode Go"):
            assert service in text


def test_cayu_new_routes_durable_operations_to_the_package_quickstart(tmp_path: Path) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myproj"

    readme = " ".join((project / "README.md").read_text(encoding="utf-8").split())
    agents = " ".join((project / "AGENTS.md").read_text(encoding="utf-8").split())
    assert "For durable operational changes" in readme
    assert "propose, authorize, act once, verify, inspect, and recover" in readme
    assert "uv run --no-sync cayu guide durable-operations" in readme
    assert "If the job observes, proposes, authorizes, executes, verifies, or recovers" in agents
    assert "uv run --no-sync cayu guide durable-operations" in agents


def test_cayu_new_uses_supported_hyphenated_project_name_for_the_agent(
    tmp_path: Path,
) -> None:
    assert main(["new", "code-review", "--dir", str(tmp_path)]) == 0

    agent_source = (tmp_path / "code-review" / "agents" / "agent.py").read_text(encoding="utf-8")
    assert 'name="code-review"' in agent_source


def test_cayu_new_uses_an_explicit_agent_name_across_the_generated_contract(
    tmp_path: Path,
) -> None:
    assert (
        main(
            [
                "new",
                "kimi-test-agent",
                "--agent-name",
                "code-review-agent",
                "--dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    project = tmp_path / "kimi-test-agent"

    assert 'name = "kimi-test-agent"' in (project / "pyproject.toml").read_text()
    assert 'name="code-review-agent"' in (project / "agents" / "agent.py").read_text()
    assert 'agent_name="code-review-agent"' in (project / "tests" / "test_agent.py").read_text()
    assert 'agent_name="code-review-agent"' in (project / "evals" / "agent.py").read_text()
    readme = " ".join((project / "README.md").read_text().split())
    assert "registered agent identity is `code-review-agent`" in readme
    instructions = " ".join((project / "AGENTS.md").read_text().split())
    assert "registered agent identity is `code-review-agent`" in instructions
    assert "--agent code-review-agent --effect EFFECT" in instructions

    environment = os.environ.copy()
    for variable in ("CAYU_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        environment.pop(variable, None)
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


@pytest.mark.parametrize("reserved_token", _RESERVED_TEMPLATE_TOKENS)
def test_cayu_new_preserves_reserved_tokens_in_the_default_agent_identity(
    tmp_path: Path,
    reserved_token: str,
) -> None:
    project_name = f"project{reserved_token}"
    assert main(["new", project_name, "--dir", str(tmp_path)]) == 0
    project = tmp_path / project_name

    assert f'name = "{project_name}"' in (project / "pyproject.toml").read_text()
    assert f'name="{project_name}"' in (project / "agents" / "agent.py").read_text()
    assert f'agent_name="{project_name}"' in (project / "tests" / "test_agent.py").read_text()
    assert f'agent_name="{project_name}"' in (project / "evals" / "agent.py").read_text()
    readme = " ".join((project / "README.md").read_text().split())
    assert f"registered agent identity is `{project_name}`" in readme
    instructions = " ".join((project / "AGENTS.md").read_text().split())
    assert f"registered agent identity is `{project_name}`" in instructions


@pytest.mark.parametrize("reserved_token", _RESERVED_TEMPLATE_TOKENS)
def test_cayu_new_preserves_reserved_tokens_in_an_explicit_agent_identity(
    tmp_path: Path,
    reserved_token: str,
) -> None:
    agent_name = f"agent{reserved_token}"
    assert (
        main(
            [
                "new",
                "project",
                "--agent-name",
                agent_name,
                "--dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    project = tmp_path / "project"

    assert 'name = "project"' in (project / "pyproject.toml").read_text()
    assert f'name="{agent_name}"' in (project / "agents" / "agent.py").read_text()
    assert f'agent_name="{agent_name}"' in (project / "tests" / "test_agent.py").read_text()
    assert f'agent_name="{agent_name}"' in (project / "evals" / "agent.py").read_text()
    readme = " ".join((project / "README.md").read_text().split())
    assert f"registered agent identity is `{agent_name}`" in readme
    instructions = " ".join((project / "AGENTS.md").read_text().split())
    assert f"registered agent identity is `{agent_name}`" in instructions


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


def test_cayu_new_rejects_an_invalid_explicit_agent_name_before_creating_files(
    tmp_path: Path,
    capsys,
) -> None:
    assert (
        main(
            [
                "new",
                "valid-project",
                "--agent-name",
                "code review agent",
                "--dir",
                str(tmp_path),
            ]
        )
        == 1
    )

    assert "invalid agent name 'code review agent'" in capsys.readouterr().err
    assert not (tmp_path / "valid-project").exists()
