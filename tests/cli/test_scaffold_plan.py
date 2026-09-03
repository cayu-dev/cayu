from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cayu import CayuApp
from cayu.cli import main
from cayu.cli.project import project_context
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_plan import normalize_application_plan


def _json_output(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_dry_run_and_apply_use_the_same_normalized_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = [
        "new",
        "observed",
        "--dir",
        str(tmp_path),
        "--with",
        "observability",
        "--dry-run",
        "--json",
    ]
    assert main(command) == 0
    dry_run = _json_output(capsys)
    assert dry_run["status"] == "planned"
    assert not (tmp_path / "observed").exists()

    command.remove("--dry-run")
    assert main(command) == 0
    applied = _json_output(capsys)
    assert applied["status"] == "created"
    assert applied["plan"] == dry_run["plan"]
    assert applied["agent_context"] == dry_run["agent_context"]
    assert applied["plan"]["capabilities"] == [
        "approvals",
        "artifacts",
        "evals",
        "human-input",
        "knowledge",
        "memory",
        "observability",
        "recovery",
        "tasks",
    ]
    assert applied["plan"]["private_files"] == ["data/memory-evidence.key"]
    runtime = (tmp_path / "observed/configuration/runtime.py").read_text(encoding="utf-8")
    assert "enable_logging=True" in runtime
    instructions = (tmp_path / "observed/AGENTS.md").read_text(encoding="utf-8")
    expected_reference = (
        "uv run --no-sync cayu new observed_reference --preset agent --database sqlite "
        "--provider neutral --execution none --json "
        '--agent-name "observed" --dir "$reference_parent"'
    )
    assert f"{expected_reference} --dry-run" in instructions
    assert expected_reference in instructions
    assert "installed, exactly pinned Cayu version" in instructions
    assert "project-local uv cache" in instructions


def test_dry_run_rejects_the_same_nonempty_target_as_apply(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "occupied"
    target.mkdir()
    existing = target / "existing.txt"
    existing.write_text("user-owned\n", encoding="utf-8")

    assert (
        main(
            [
                "new",
                target.name,
                "--dir",
                str(tmp_path),
                "--dry-run",
                "--json",
            ]
        )
        == 1
    )

    payload = _json_output(capsys)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "TARGET_NOT_EMPTY"
    assert existing.read_text(encoding="utf-8") == "user-owned\n"


def test_default_agent_plan_is_the_complete_safe_local_profile() -> None:
    plan = normalize_application_plan(name="standard", agent_name="standard")

    assert plan.capabilities == (
        "approvals",
        "artifacts",
        "evals",
        "human-input",
        "knowledge",
        "memory",
        "observability",
        "recovery",
        "tasks",
    )
    files = project_files("standard", application_plan=plan)
    assert "AutomaticRecallContextPolicy" in files["memory/context.py"]
    assert "SQLiteKnowledgeStore" in files["configuration/storage.py"]
    assert "LocalArtifactStore" in files["environments/local.py"]
    assert "StaticToolExposurePolicy" in files["policies/tools.py"]
    assert "tests/test_memory.py" in files


def test_without_memory_disables_recall_but_retains_its_truthful_home() -> None:
    plan = normalize_application_plan(
        name="no-memory",
        agent_name="no-memory",
        without_capabilities=("memory",),
    )
    files = project_files("no-memory", application_plan=plan)

    assert "memory" not in plan.capabilities
    assert "memory/context.py" in files
    assert "if not False:" in files["memory/context.py"]
    assert "tests/test_memory.py" not in files
    assert "available but not configured" in files["memory/CAPABILITY.md"]


def test_standard_capability_acceptance_is_emitted_only_for_its_complete_profile() -> None:
    default_files = project_files("standard")
    without_memory = project_files("standard", without_capabilities=("memory",))
    without_observability = project_files(
        "standard",
        without_capabilities=("observability",),
    )

    assert "tests/test_standard_capabilities.py" in default_files
    assert "tests/test_standard_capabilities.py" not in without_memory
    assert "tests/test_standard_capabilities.py" not in without_observability


def test_joint_dependency_exclusions_are_order_independent() -> None:
    knowledge_first = normalize_application_plan(
        name="reduced",
        agent_name="reduced",
        without_capabilities=("knowledge", "memory"),
    )
    memory_first = normalize_application_plan(
        name="reduced",
        agent_name="reduced",
        without_capabilities=("memory", "knowledge"),
    )

    assert knowledge_first == memory_first
    assert "knowledge" not in knowledge_first.capabilities
    assert "memory" not in knowledge_first.capabilities


def test_recovery_opt_out_changes_the_generated_runtime_profile() -> None:
    selected = project_files("standard")
    excluded = project_files("standard", without_capabilities=("recovery",))
    changed = {
        path for path in set(selected) | set(excluded) if selected.get(path) != excluded.get(path)
    }

    assert {
        "environments/local.py",
        "policies/tools.py",
        "tools/registration.py",
    } <= changed
    for path in (
        "environments/local.py",
        "policies/tools.py",
        "tools/registration.py",
    ):
        assert "if True" in selected[path]
        assert "if False" in excluded[path]


def test_user_names_cannot_collide_with_internal_capability_tokens() -> None:
    name = "project__TASKS_ENABLED__"
    files = project_files(name)

    assert f'name="{name}"' in files["agents/agent.py"]
    assert f'agent_name="{name}"' in files["tests/test_agent.py"]
    assert "projectTrue" not in "\n".join(files.values())


@pytest.mark.parametrize("execution", ("none", "docker"))
def test_coding_knowledge_opt_out_reaches_the_overlay(execution: str) -> None:
    files = project_files(
        "coder",
        preset="coding",
        execution=execution,
        without_capabilities=("knowledge",),
    )

    assert "_KNOWLEDGE_ENABLED = False" in files["operations/coding.py"]
    assert '"list_knowledge"' not in files["tools/coding.py"]
    assert "No knowledge store or knowledge tools are configured" in files["README.md"]


def test_without_flags_disable_each_safe_default_deterministically(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    excluded = (
        "approvals",
        "artifacts",
        "evals",
        "human-input",
        "memory",
        "observability",
        "recovery",
        "tasks",
    )
    command = ["new", "reduced", "--dir", str(tmp_path), "--json"]
    for capability in excluded:
        command.extend(("--without", capability))

    assert main(command) == 0
    payload = _json_output(capsys)
    project = tmp_path / "reduced"
    assert payload["plan"]["capabilities"] == ["knowledge"]
    assert not (project / "data" / "memory-evidence.key").exists()
    assert not (project / "data" / "artifacts").exists()
    assert not (project / "tests" / "test_memory.py").exists()
    assert not (project / "evals" / "agent.py").exists()
    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert "eval_target" not in pyproject
    assert "enable_logging=False" in (project / "configuration" / "runtime.py").read_text(
        encoding="utf-8"
    )
    assert 'SQLiteTaskStore("data/cayu.db") if False else None' in (
        project / "configuration" / "storage.py"
    ).read_text(encoding="utf-8")
    assert "if not False:" in (project / "memory" / "context.py").read_text(encoding="utf-8")
    for capability in excluded:
        card_path = {
            "approvals": "operations/APPROVALS.md",
            "artifacts": "environments/ARTIFACTS.md",
            "evals": "evals/CAPABILITY.md",
            "human-input": "operations/HUMAN_INPUT.md",
            "memory": "memory/CAPABILITY.md",
            "observability": "observability/CAPABILITY.md",
            "recovery": "operations/RECOVERY.md",
            "tasks": "operations/TASKS.md",
        }[capability]
        assert "available but not configured" in (project / card_path).read_text(encoding="utf-8")


def test_excluding_a_required_default_fails_before_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "new",
                "broken-scope",
                "--without",
                "knowledge",
                "--dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 1
    )
    payload = _json_output(capsys)
    assert payload["error"]["code"] == "CAPABILITY_DEPENDENCY_CONFLICT"
    assert "memory" in payload["error"]["message"]
    assert not (tmp_path / "broken-scope").exists()


def test_service_can_disable_selectable_observability_without_erasing_its_home() -> None:
    plan = normalize_application_plan(
        name="service",
        agent_name="service",
        preset="service",
        without_capabilities=("observability",),
    )
    files = project_files("service", application_plan=plan)

    assert "observability" not in plan.capabilities
    assert "observability/__init__.py" in files
    assert "enable_logging=False" in files["configuration/runtime.py"]


def test_legacy_coding_flags_normalize_to_the_canonical_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical = [
        "new",
        "coder",
        "--dir",
        str(tmp_path),
        "--preset",
        "coding",
        "--execution",
        "docker",
        "--dry-run",
        "--json",
    ]
    legacy = [
        "new",
        "coder",
        "--dir",
        str(tmp_path),
        "--composition",
        "coding",
        "--coding-execution",
        "docker",
        "--dry-run",
        "--json",
    ]

    assert main(canonical) == 0
    canonical_payload = _json_output(capsys)
    assert main(legacy) == 0
    assert _json_output(capsys) == canonical_payload


def test_postgres_coding_plan_selects_every_active_database_store() -> None:
    plan = normalize_application_plan(
        name="coder",
        agent_name="coder",
        preset="coding",
        database="postgres",
    )
    files = project_files("coder", application_plan=plan)
    pyproject = files["pyproject.toml"]
    storage = files["configuration/coding_storage.py"]
    readme = files["README.md"]
    instructions = files["AGENTS.md"]

    assert 'dependencies = ["cayu[postgres]==' in pyproject
    assert 'dev = ["cayu[postgres,server]==' in pyproject
    assert 'backend = "postgres"\nenv = "CAYU_DATABASE_URL"' in pyproject
    for store in ("PostgresSessionStore", "PostgresTaskStore", "PostgresKnowledgeStore"):
        assert store in storage
    assert "SQLite" not in storage
    assert "SQLite" not in readme
    assert "SQLite" not in instructions
    assert "data/cayu.db" not in readme
    assert "configured durable Postgres Evals store" in readme
    assert "durable Postgres knowledge" in readme
    assert "session, task, and knowledge state lives in the configured Postgres stores" in readme
    assert "same configured Postgres database" in readme
    assert "same configured Postgres database" in instructions


def test_postgres_service_plan_is_rejected_before_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "service"

    assert (
        main(
            [
                "new",
                target.name,
                "--preset",
                "service",
                "--database",
                "postgres",
                "--dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 1
    )

    payload = _json_output(capsys)
    assert payload["error"] == {
        "code": "UNSUPPORTED_ADAPTER",
        "message": "database 'postgres' is not supported by preset 'service'",
    }
    assert not target.exists()


def test_minimal_is_recorded_and_omits_the_complete_convention(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "tiny", "--minimal", "--dir", str(tmp_path), "--json"]) == 0
    payload = _json_output(capsys)
    project = tmp_path / "tiny"

    assert payload["plan"]["minimal"] is True
    assert payload["plan"]["capabilities"] == []
    assert payload["plan"]["private_files"] == []
    assert (project / "configuration.py").is_file()
    assert not (project / "configuration").exists()
    assert not (project / "workflows").exists()
    assert (project / "data").is_dir()
    assert not (project / "data" / "memory-evidence.key").exists()
    assert not (project / "data" / "artifacts").exists()
    assert "minimal = true" in (project / "pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        (("--database", "postgres"), "MINIMAL_DATABASE_UNSUPPORTED"),
        (("--with", "observability"), "MINIMAL_CAPABILITY_UNSUPPORTED"),
    ],
)
def test_minimal_rejects_unimplemented_combinations_before_writes(
    arguments: tuple[str, ...],
    code: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "tiny", "--minimal", *arguments, "--dir", str(tmp_path), "--json"]) == 1

    assert _json_output(capsys)["error"]["code"] == code
    assert not (tmp_path / "tiny").exists()


def test_service_plan_declares_and_supplies_non_secret_auth_proof_environment() -> None:
    plan = normalize_application_plan(
        name="service",
        agent_name="service",
        preset="service",
    )

    payload = plan.as_dict()
    assert payload["environment"] == [
        "CAYU_OPERATOR_BEARER_TOKEN",
        "PRODUCT_AUTH_TOKENS_JSON",
    ]
    check = next(
        command for command in plan.verification_commands() if "cayu check --deploy" in command
    )
    assert check.startswith("PRODUCT_AUTH_TOKENS_JSON=")
    assert "CAYU_OPERATOR_BEARER_TOKEN=local-operator-token" in check


def test_docker_plan_puts_image_construction_before_runtime_verification() -> None:
    plan = normalize_application_plan(
        name="coder",
        agent_name="coder",
        preset="coding",
        execution="docker",
    )

    commands = plan.verification_commands()
    assert commands[:3] == (
        "uv lock",
        "uv sync --extra dev",
        "uv run --no-sync python build_coding_image.py",
    )
    assert commands.index("uv run --no-sync python build_coding_image.py") < commands.index(
        "uv run --no-sync cayu inspect --json"
    )


def test_docker_coding_plan_records_explicit_toolchain_and_command_authority() -> None:
    plan = normalize_application_plan(
        name="coder",
        agent_name="coder",
        preset="coding",
        execution="docker",
        coding_toolchain="python",
        coding_command_authority="structured",
    )

    assert plan.as_dict()["coding"] == {
        "toolchain": "python",
        "command_authority": "structured",
    }
    files = project_files("coder", application_plan=plan)
    instructions = files["AGENTS.md"]
    assert "--coding-toolchain python" in instructions
    assert "--coding-command-authority structured" in instructions


def test_unconfigured_extension_only_capability_fails_before_target_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "new",
                "invalid",
                "--with",
                "mcp",
                "--dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 1
    )
    payload = _json_output(capsys)
    assert payload["error"]["code"] == "CAPABILITY_NOT_SELECTABLE"
    assert not (tmp_path / "invalid").exists()


def test_discovery_reports_truthful_capability_status_and_complete_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "--list-capabilities", "--json"]) == 0
    payload = _json_output(capsys)
    by_name = {item["name"]: item for item in payload["capabilities"]}

    assert by_name["observability"]["status"] == "selectable"
    assert by_name["memory"]["status"] == "preset-owned"
    assert by_name["memory"]["requires"] == ["knowledge"]
    for field in (
        "files",
        "dependencies",
        "environment",
        "requires",
        "conflicts",
        "supported_presets",
        "supported_databases",
        "supported_executions",
        "verification",
    ):
        assert field in by_name["observability"]


def test_discovery_reports_only_coherent_preset_databases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "--list-presets", "--json"]) == 0
    payload = _json_output(capsys)
    by_name = {item["name"]: item for item in payload["presets"]}

    assert by_name["agent"]["supported_databases"] == ["sqlite", "postgres"]
    assert by_name["service"]["supported_databases"] == ["sqlite"]
    assert by_name["coding"]["supported_databases"] == ["sqlite", "postgres"]


def test_importing_every_default_project_module_is_inert(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "inert", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "inert"
    modules = []
    for path in sorted(project.rglob("*.py")):
        relative = path.relative_to(project)
        if relative.parts[0] == "tests" or relative.name == "__init__.py":
            continue
        modules.append(".".join(relative.with_suffix("").parts))

    with project_context(project):
        for name in modules:
            module = importlib.import_module(name)
            assert not any(isinstance(value, CayuApp) for value in vars(module).values())

    for name in modules:
        sys.modules.pop(name, None)


def test_non_coding_publication_failure_leaves_no_new_target(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold

    def fail_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("forced publication failure")

    monkeypatch.setattr(scaffold, "publish_guarded_tree", fail_publish)
    assert main(["new", "atomic", "--dir", str(tmp_path), "--json"]) == 1
    assert _json_output(capsys)["error"]["code"] == "SCAFFOLD_PUBLICATION_FAILED"
    assert not (tmp_path / "atomic").exists()


def test_non_coding_publication_preserves_bounded_recovery_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold
    from cayu.cli._guarded_tree_publication import GuardedTreePublicationError

    cleanup_failure = OSError("retained cleanup owner")
    publication_failure = GuardedTreePublicationError(
        "publication_conflict",
        "publication ownership changed",
        paths=(".cayu-tree-stage-owned",),
    )
    publication_failure.add_note("guarded publication remains recoverable by exact retry")

    def fail_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise publication_failure from cleanup_failure

    monkeypatch.setattr(scaffold, "publish_guarded_tree", fail_publish)
    assert main(["new", "diagnostic", "--dir", str(tmp_path), "--json"]) == 1

    payload = _json_output(capsys)
    message = payload["error"]["message"]
    assert payload["error"]["code"] == "SCAFFOLD_PUBLICATION_FAILED"
    assert "affected paths: '.cayu-tree-stage-owned'" in message
    assert "guarded publication remains recoverable by exact retry" in message
    assert not (tmp_path / "diagnostic").exists()


def test_scaffold_publication_translation_retains_ordered_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold
    from cayu.cli._guarded_tree_publication import GuardedTreePublicationError

    cleanup_failure = OSError("retained cleanup owner")
    publication_failure = GuardedTreePublicationError(
        "publication_conflict",
        "publication ownership changed",
        paths=(".cayu-tree-stage-owned",),
    )
    publication_failure.add_note("guarded publication remains recoverable by exact retry")

    def fail_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise publication_failure from cleanup_failure

    monkeypatch.setattr(scaffold, "publish_guarded_tree", fail_publish)
    plan = normalize_application_plan(name="diagnostic", agent_name="diagnostic")

    with pytest.raises(scaffold._ScaffoldCommandError) as raised:
        scaffold._publish_new_project(
            target=tmp_path / "diagnostic",
            files={"app.py": "# generated\n"},
            plan=plan,
        )

    assert raised.value.__cause__ is publication_failure
    assert publication_failure.__cause__ is cleanup_failure
    assert raised.value.__notes__ == publication_failure.__notes__
    assert "affected paths: '.cayu-tree-stage-owned'" in str(raised.value)


@pytest.mark.parametrize(
    "destination_state",
    ("unchanged", "modified", "empty", "absent", "recreated_empty"),
)
def test_non_coding_scaffold_retry_authenticates_published_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    destination_state: str,
) -> None:
    command = ["new", "retryable", "--dir", str(tmp_path), "--json"]
    target = tmp_path / "retryable"

    assert main(command) == 0
    _json_output(capsys)
    original_private_key = (target / "data" / "memory-evidence.key").read_bytes()

    if destination_state == "modified":
        (target / "operator-owned.txt").write_text("keep\n", encoding="utf-8")
    elif destination_state == "empty":
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    elif destination_state in {"absent", "recreated_empty"}:
        shutil.rmtree(target)
        if destination_state == "recreated_empty":
            target.mkdir()

    result = main(command)
    payload = _json_output(capsys)

    if destination_state == "modified":
        assert result == 1
        assert payload["error"]["code"] == "TARGET_NOT_EMPTY"
        assert (target / "operator-owned.txt").read_text(encoding="utf-8") == "keep\n"
        return

    assert result == 0
    assert payload["status"] == "created"
    assert (target / "app.py").is_file()
    if destination_state == "unchanged":
        assert (target / "data" / "memory-evidence.key").read_bytes() == original_private_key


def test_coding_preflight_failure_retains_recoverable_settlement_diagnostic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import _guarded_tree_publication as publication
    from cayu.cli import scaffold

    git = shutil.which("git")
    assert git is not None
    monkeypatch.setattr(
        scaffold.LocalWorkspace,
        "require_path_operations_supported",
        staticmethod(lambda: None),
    )
    preflight_calls = 0

    def fail_once_preflight(*, parent: Path) -> tuple[str, str]:
        nonlocal preflight_calls
        del parent
        preflight_calls += 1
        if preflight_calls == 1:
            raise scaffold._ScaffoldCommandError("dependency probe failed")
        return git, "test-owned-rg"

    monkeypatch.setattr(scaffold, "_preflight_coding_commands", fail_once_preflight)
    original_rollback = publication._rollback_before_publication
    rollback_calls = 0

    def fail_once_rollback(journal: Any, *, parent: Any) -> str:
        nonlocal rollback_calls
        rollback_calls += 1
        if rollback_calls == 1:
            raise OSError("retained cleanup owner")
        return original_rollback(journal, parent=parent)

    monkeypatch.setattr(publication, "_rollback_before_publication", fail_once_rollback)
    command = [
        "new",
        "recoverable-coder",
        "--composition",
        "coding",
        "--dir",
        str(tmp_path),
        "--json",
    ]

    assert main(command) == 1
    failure = _json_output(capsys)
    assert failure["error"]["code"] == "CODING_PREFLIGHT_FAILED"
    assert "dependency probe failed" in failure["error"]["message"]
    assert "guarded publication remains recoverable" in failure["error"]["message"]
    target = tmp_path / "recoverable-coder"
    assert not target.exists()
    assert list(tmp_path.glob(".cayu-tree-publication-*.jsonl"))

    assert main(command) == 0
    _json_output(capsys)
    assert (target / "app.py").is_file()
    assert not [
        path
        for path in tmp_path.glob(".cayu-tree-publication-*.jsonl")
        if not path.name.endswith("-receipt.jsonl")
    ]


@pytest.mark.parametrize("crash_phase", ("original_backed_up", "tree_renamed"))
def test_non_coding_scaffold_recovers_publication_after_process_death(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    crash_phase: str,
) -> None:
    project = tmp_path / "recoverable"
    if crash_phase == "original_backed_up":
        project.mkdir()
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import os
from cayu.cli import main
from cayu.cli import _guarded_tree_publication as publication

def fault(phase):
    if phase == {crash_phase!r}:
        os._exit(83)

publication._publication_fault = fault
main(["new", "recoverable", "--dir", {str(tmp_path)!r}])
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )
    assert completed.returncode == 83
    private_key = (
        (project / "data" / "memory-evidence.key").read_bytes()
        if crash_phase == "tree_renamed"
        else None
    )

    assert main(["new", "recoverable", "--dir", str(tmp_path), "--json"]) == 0

    assert _json_output(capsys)["status"] == "created"
    recovered_private_key = (project / "data" / "memory-evidence.key").read_bytes()
    if private_key is not None:
        assert recovered_private_key == private_key
    assert not list(tmp_path.glob(".cayu-tree-stage-*"))
    assert not list(tmp_path.glob(".cayu-tree-backup-*"))
    active_journals = [
        path
        for path in tmp_path.glob(".cayu-tree-publication-*.jsonl")
        if not path.name.endswith("-receipt.jsonl")
    ]
    assert active_journals == []


def test_non_coding_scaffold_settles_different_active_request_before_publication(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "recoverable"
    project.mkdir()
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import os
from cayu.cli import main
from cayu.cli import _guarded_tree_publication as publication

def fault(phase):
    if phase == "original_backed_up":
        os._exit(84)

publication._publication_fault = fault
main(["new", "recoverable", "--minimal", "--dir", {str(tmp_path)!r}])
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )
    assert completed.returncode == 84
    assert list(tmp_path.glob(".cayu-tree-publication-*.jsonl"))

    assert main(["new", "recoverable", "--dir", str(tmp_path), "--json"]) == 0

    assert _json_output(capsys)["status"] == "created"
    assert (project / "data" / "memory-evidence.key").is_file()
    assert not [
        path
        for path in tmp_path.glob(".cayu-tree-publication-*.jsonl")
        if not path.name.endswith("-receipt.jsonl")
    ]
