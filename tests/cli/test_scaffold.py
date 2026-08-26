"""Tests for ``cayu new`` (the project scaffold)."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cayu import (
    CayuApp,
    ChatCompletionsProvider,
    EvalStatus,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
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

    session_store = FalsySessionStore()
    task_store = FalsyTaskStore()
    first_app = module.build_app(
        provider=ScriptedModelProvider([]),
        session_store=session_store,
        task_store=task_store,
    )
    second_app = module.build_app(
        provider=ScriptedModelProvider([]),
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )
    assert first_app is not second_app
    assert first_app.session_store is session_store
    assert first_app.task_store is task_store

    app_source = (proj / "app.py").read_text(encoding="utf-8")
    configuration_source = (proj / "configuration.py").read_text(encoding="utf-8")
    assert "AnthropicProvider" in app_source
    assert "ChatCompletionsProvider" in app_source
    assert "OpenAISubscriptionProvider" in app_source
    assert "CAYU_OPENAI_SUBSCRIPTION" not in app_source
    assert "_SCAFFOLDED_PROVIDER = None" in configuration_source
    assert 'os.environ.get("CAYU_PROVIDER", _SCAFFOLDED_PROVIDER)' in configuration_source
    pyproject = (proj / "pyproject.toml").read_text(encoding="utf-8")
    assert f'dependencies = ["cayu>={cayu_version}"]' in pyproject
    assert 'console = ["cayu[console]"]' not in pyproject
    assert f'dev = ["cayu[server]>={cayu_version}", "pytest"]' in pyproject
    assert '[tool.cayu]\nfactory = "app:build_app"' in pyproject
    assert 'eval_target = "evals.agent:build_eval"' in pyproject
    assert '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"' in pyproject
    assert 'else SQLiteSessionStore(\n                "data/cayu.db",' in app_source
    assert "public_authority_alias_codec_from_environment()" in app_source
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
    assert "uv run cayu serve --dev" in readme
    assert "http://127.0.0.1:8000/cayu/" in readme
    assert "developer/operator control plane" in readme
    assert "no Evals-specific Python configuration" in readme
    assert "Fresh Evals execution remains gated" not in readme
    assert "Never mount it with unauthenticated open access on a public listener" in readme
    assert "client-IP checks are not authentication" in readme
    assert "uv run cayu auth openai login" in readme
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
    assert "uv run cayu guide authoring#cayu-map" in readme
    assert "github.com" not in readme
    assert 'uv run python run.py --message "YOUR REQUEST"' in readme
    assert "model-only" in readme
    assert "cayu generate slice" not in readme
    output = capsys.readouterr().out
    assert "uv sync --extra dev" in output
    assert "uv run cayu check --json" in output
    assert "uv run cayu serve --dev" in output
    assert "http://127.0.0.1:8000/cayu/" in output
    assert "none selected" in output


def test_cayu_new_coding_emits_explicit_composition_and_clean_git_baseline(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_coding_dependency_preflight(monkeypatch)
    assert main(["new", "coder", "--composition", "coding", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "coder"

    for filename in (
        "command_probe.py",
        "composition.py",
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

    composition = (project / "composition.py").read_text(encoding="utf-8")
    command_probe = (project / "command_probe.py").read_text(encoding="utf-8")
    assert (project / ".gitignore").read_text(encoding="utf-8").startswith(".cayu/\n")
    assert "from command_probe import" in composition
    assert "cayu.cli._bounded_command" not in composition
    assert "run_bounded_command" in command_probe
    for public_surface in (
        "LocalWorkspace",
        "LocalRunner",
        "LocalArtifactStore",
        "SQLiteKnowledgeStore",
        "SearchTextTool",
        "GitChangesTool",
        "SubagentTool",
        "SubagentResultTool",
        "UserInputTool",
        "AllRegisteredToolsExposurePolicy",
        "ParameterConstrainedToolPolicy",
    ):
        assert public_surface in composition
    assert "mode=SubagentExecutionMode.BACKGROUND" in composition
    assert 'os.environ.get("CAYU_WORKSPACE_ROOT", ".")' in composition
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

    assert main(["new", "coder", "--composition", "coding", "--dir", str(tmp_path)]) == 1

    assert not (tmp_path / "coder").exists()
    assert "requires POSIX descriptor-relative filesystem primitives" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows capability evidence")
def test_cayu_new_coding_rejects_native_windows_workspace_before_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "coder", "--composition", "coding", "--dir", str(tmp_path)]) == 1

    assert not (tmp_path / "coder").exists()
    assert "requires POSIX descriptor-relative filesystem primitives" in capsys.readouterr().err


def test_cayu_new_coding_rejects_service_and_missing_dependencies(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    assert (
        main(
            [
                "new",
                "service-coder",
                "--composition",
                "coding",
                "--template",
                "service",
                "--dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert not (tmp_path / "service-coder").exists()
    assert "cannot be combined" in capsys.readouterr().err

    from cayu.cli import scaffold

    real_which = scaffold.shutil.which
    monkeypatch.setattr(
        scaffold.shutil,
        "which",
        lambda command: None if command == "rg" else real_which(command),
    )
    assert main(["new", "missing-rg", "--composition", "coding", "--dir", str(tmp_path)]) == 1
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
                "--composition",
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
                "--composition",
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

    assert main(["new", "safe-coder", "--composition", "coding", "--dir", str(tmp_path)]) == 0
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
        scaffold.project_files("safe-coder", composition="coding")
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

    assert main(["new", "unsafe-coder", "--composition", "coding", "--dir", str(shared)]) == 1
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
        if cwd.name.startswith(".broken-coder.cayu-scaffold-") and "commit" in argv:
            raise scaffold._ScaffoldCommandError("forced commit failure")
        return original(
            argv,
            cwd=cwd,
            env=env,
            allowed_exit_codes=allowed_exit_codes,
        )

    monkeypatch.setattr(scaffold, "_run_scaffold_command", fail_target_commit)
    assert main(["new", "broken-coder", "--composition", "coding", "--dir", str(tmp_path)]) == 1
    assert (tmp_path / "broken-coder").is_dir()
    assert list((tmp_path / "broken-coder").iterdir()) == []
    assert "forced commit failure" in capsys.readouterr().err


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
        if cwd.name.startswith(".existing-coder.cayu-scaffold-") and "commit" in argv:
            raise scaffold._ScaffoldCommandError("forced commit failure")
        return original(
            argv,
            cwd=cwd,
            env=env,
            allowed_exit_codes=allowed_exit_codes,
        )

    monkeypatch.setattr(scaffold, "_run_scaffold_command", fail_target_commit)
    assert main(["new", target.name, "--composition", "coding", "--dir", str(tmp_path)]) == 1
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
        if cwd.name.startswith(".symlink-coder.cayu-scaffold-") and "commit" in argv:
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
    assert main(["new", "symlink-coder", "--composition", "coding", "--dir", str(tmp_path)]) == 1
    assert canary.read_text(encoding="utf-8") == "outside-owned\n"
    assert list((tmp_path / "symlink-coder").iterdir()) == []
    assert "forced commit failure" in capsys.readouterr().err


def test_cayu_new_coding_does_not_clean_replacement_target(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold

    _bypass_coding_dependency_preflight(monkeypatch)

    target = tmp_path / "replaced-coder"
    moved = tmp_path / "original-coder"
    replacement_canary = "replacement-owned\n"
    original = scaffold._publish_staged_scaffold

    def replace_before_publish(*, staging, target, expected_target_identity, target_mode):
        target.rename(moved)
        target.mkdir()
        (target / ".git").mkdir()
        (target / ".git" / "replacement-owned").write_text(
            replacement_canary,
            encoding="utf-8",
        )
        (target / "pyproject.toml").write_text(replacement_canary, encoding="utf-8")
        (target / "owned.txt").write_text(replacement_canary, encoding="utf-8")
        return original(
            staging=staging,
            target=target,
            expected_target_identity=expected_target_identity,
            target_mode=target_mode,
        )

    monkeypatch.setattr(scaffold, "_publish_staged_scaffold", replace_before_publish)
    assert main(["new", target.name, "--composition", "coding", "--dir", str(tmp_path)]) == 1
    assert (target / "pyproject.toml").read_text(encoding="utf-8") == replacement_canary
    assert (target / "owned.txt").read_text(encoding="utf-8") == replacement_canary
    assert (target / ".git" / "replacement-owned").read_text(encoding="utf-8") == (
        replacement_canary
    )
    assert moved.is_dir()
    assert "target identity changed" in capsys.readouterr().err


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
    assert main(["new", "myservice", "--template", "service", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myservice"

    for filename in (
        "service.py",
        "product_store.py",
        "tests/test_public_service_security.py",
    ):
        assert (project / filename).is_file()
    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert f'dependencies = ["cayu[server]>={cayu_version}"]' in pyproject
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
        "SessionInvocationAdmission",
        "admit_session_invocation",
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
    assert main(["new", "myservice", "--template", "service", "--dir", str(tmp_path)]) == 0
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
    assert main(["new", "service", "--template", "service", "--dir", str(tmp_path)]) == 0
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
    assert "uv run cayu serve --dev" in instructions
    assert "http://127.0.0.1:8000/cayu/" in instructions
    assert "developer/operator control plane" in instructions
    assert "end-user UI" in instructions
    assert "Never mount it with `OpenAccess()` on a public listener" in instructions
    assert "Client-IP and forwarded-header checks are not authentication" in instructions
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


def test_cayu_new_routes_provider_questions_to_the_package_compatibility_guide(
    tmp_path: Path,
) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myproj"

    for relative_path in ("README.md", "AGENTS.md"):
        text = " ".join((project / relative_path).read_text(encoding="utf-8").split())
        assert "OpenRouter is a first-class scaffold choice" in text
        assert "other compatible endpoints work through Cayu's generic adapter" in text
        assert "uv run cayu guide providers#compatible-chat-completions" in text
        for service in ("OpenRouter", "Fireworks", "Baseten", "OpenCode Go"):
            assert service in text


def test_cayu_new_routes_durable_operations_to_the_package_quickstart(tmp_path: Path) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "myproj"

    readme = " ".join((project / "README.md").read_text(encoding="utf-8").split())
    agents = " ".join((project / "AGENTS.md").read_text(encoding="utf-8").split())
    assert "For durable operational changes" in readme
    assert "propose, authorize, act once, verify, inspect, and recover" in readme
    assert "uv run cayu guide durable-operations" in readme
    assert "If the job observes, proposes, authorizes, executes, verifies, or recovers" in agents
    assert "uv run cayu guide durable-operations" in agents


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
