"""The long-running preset generates a working, tested example of each pattern."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_plan import ScaffoldPlanError, normalize_application_plan

_PATTERN_FILES = (
    "integrations/payments.py",
    "tools/payments.py",
    "policies/tools.py",
    "policies/budgets.py",
    "workflows/invoices.py",
    "operations/timers.py",
    "tests/test_restart.py",
    "tests/test_spend_cap.py",
    "evals/agent.py",
)


def _credential_free_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "CAYU_PROVIDER",
        "CAYU_MODEL",
        "CAYU_DATABASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    return environment


def _run(project: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=project,
        env=_credential_free_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_long_running_project_passes_its_own_tests_eval_and_check(tmp_path: Path) -> None:
    assert main(["new", "invoices", "--preset", "long-running", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "invoices"
    for relative in _PATTERN_FILES:
        assert (project / relative).is_file(), relative

    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["cayu"]["workers"] == {"timers": "operations.timers:run_timer_worker"}
    assert pyproject["tool"]["cayu"]["scaffold"]["preset"] == "long-running"
    agents = (project / "AGENTS.md").read_text(encoding="utf-8")
    assert "## Jobs that wait, approve, or spend" in agents
    assert "## Long-running patterns" in agents
    assert "cayu guide long-running" in agents
    # The example composes library parts rather than reimplementing them.
    assert "AlwaysRequireApprovalToolPolicy" in (project / "policies/tools.py").read_text()
    assert "TaskSchedulePolicy" in (project / "operations/timers.py").read_text()
    assert "execution_timeout_s=None" in (project / "operations/timers.py").read_text()
    assert "default_price_book" in (project / "policies/budgets.py").read_text()
    assert "RestartHarness" in (project / "tests/test_restart.py").read_text()

    tests = _run(project, "-m", "pytest", "-q", "-p", "no:cacheprovider")
    assert tests.returncode == 0, tests.stdout + tests.stderr
    assert "12 passed" in tests.stdout

    evaluation = _run(project, "-m", "cayu", "eval", "run")
    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    assert json.loads(evaluation.stdout)["status"] == "passed"

    checked = _run(project, "-m", "cayu", "check", "--fail-on", "warning", "--json")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["diagnostics"] == []


def test_long_running_preset_is_listed_and_requires_its_patterns(capsys) -> None:
    assert main(["new", "--list-presets", "--json"]) == 0
    presets = {entry["name"]: entry for entry in json.loads(capsys.readouterr().out)["presets"]}
    assert presets["long-running"]["supported_executions"] == ["none"]

    plan = normalize_application_plan(name="jobs", agent_name="jobs", preset="long-running")
    assert {"approvals", "human-input", "tasks", "evals"} <= set(plan.capabilities)
    assert "knowledge" not in plan.capabilities
    for capability in ("approvals", "human-input", "tasks"):
        with pytest.raises(ScaffoldPlanError) as excinfo:
            normalize_application_plan(
                name="jobs",
                agent_name="jobs",
                preset="long-running",
                without_capabilities=(capability,),
            )
        assert excinfo.value.code == "capability_required"
    with pytest.raises(ScaffoldPlanError) as excinfo:
        normalize_application_plan(
            name="jobs", agent_name="jobs", preset="long-running", execution="docker"
        )
    assert excinfo.value.code == "unsupported_adapter"


def test_optional_evals_can_be_left_out() -> None:
    files = project_files("jobs", preset="long-running", without_capabilities=("evals",))
    assert "evals/agent.py" not in files
    assert "tests/test_restart.py" in files


@pytest.mark.parametrize("preset", ["agent", "service", "coding"])
def test_every_preset_points_waiting_and_spending_jobs_to_the_guide(
    preset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if preset == "coding":
        from cayu.cli import scaffold

        monkeypatch.setattr(scaffold, "_preflight_coding_commands", lambda *, parent: ("", ""))
    agents = project_files("probe", preset=preset)["AGENTS.md"]
    assert "uv run --no-sync cayu guide long-running" in agents
    assert "`cayu guide long-running#timers`" in agents


def test_payment_tool_rejects_oversized_invoice_before_paying(tmp_path: Path) -> None:
    assert main(["new", "invoices", "--preset", "long-running", "--dir", str(tmp_path)]) == 0
    checked = _run(
        tmp_path / "invoices",
        "-c",
        """
import asyncio
from jsonschema import Draft202012Validator
from cayu import ToolContext
from cayu.tools._terminal_bounds import tool_result_payload_bytes
from integrations.payments import LocalPaymentService
from tools.payments import PayInvoiceTool

async def check():
    payments = LocalPaymentService()
    tool = PayInvoiceTool(payments)
    validator = Draft202012Validator(tool.spec.input_schema)
    oversized = {"invoice_id": "x" * 70000, "amount_cents": 1}
    assert list(validator.iter_errors(oversized))
    result = await tool.run(
        ToolContext(session_id="payment", idempotency_key="oversized"), oversized
    )
    assert result.is_error
    assert payments.payments == {}
    assert tool_result_payload_bytes(result) <= tool.spec.max_terminal_payload_bytes
    for index, invoice_id in enumerate(("x" * 256, "\\x01" * 256, "🌍" * 256)):
        arguments = {"invoice_id": invoice_id, "amount_cents": 1}
        validator.validate(arguments)
        key = f"payment-{index}"
        result = await tool.run(
            ToolContext(session_id="payment", idempotency_key=key), arguments
        )
        assert not result.is_error
        assert result.structured == {"payment_id": payments.payments[key].payment_id}
        assert tool_result_payload_bytes(result) <= tool.spec.max_terminal_payload_bytes
    assert len(payments.payments) == 3

asyncio.run(check())
""",
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
