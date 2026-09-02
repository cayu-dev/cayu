from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

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
    assert "observability" in applied["plan"]["capabilities"]
    runtime = (tmp_path / "observed/configuration/runtime.py").read_text(encoding="utf-8")
    assert "enable_logging=True" in runtime
    instructions = (tmp_path / "observed/AGENTS.md").read_text(encoding="utf-8")
    expected_reference = (
        "uv run --no-sync cayu new observed_reference --preset agent --database sqlite "
        "--provider neutral --execution none --with observability --json "
        '--agent-name "observed" --dir "$reference_parent"'
    )
    assert f"{expected_reference} --dry-run" in instructions
    assert expected_reference in instructions
    assert "installed, exactly pinned Cayu version" in instructions
    assert "project-local uv cache" in instructions


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
    assert (project / "configuration.py").is_file()
    assert not (project / "configuration").exists()
    assert not (project / "workflows").exists()
    assert (project / "data").is_dir()
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


def test_extension_only_capability_fails_before_target_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "new",
                "invalid",
                "--with",
                "memory",
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
    assert by_name["memory"]["status"] == "extension-only"
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

    def fail_publish(**kwargs: object) -> None:
        del kwargs
        raise OSError("forced publication failure")

    monkeypatch.setattr(scaffold, "_publish_staged_scaffold", fail_publish)
    assert main(["new", "atomic", "--dir", str(tmp_path), "--json"]) == 1
    assert _json_output(capsys)["error"]["code"] == "SCAFFOLD_PUBLICATION_FAILED"
    assert not (tmp_path / "atomic").exists()
