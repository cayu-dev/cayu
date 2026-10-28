"""Cayu's own configuration models built from literals are not import side effects."""

from __future__ import annotations

from pathlib import Path

import pytest

from cayu.budgets import RunLimits
from cayu.cli.scaffold_check import (
    _CAYU_CONFIGURATION_MODELS,
    _check_import_inertness,
)
from cayu.providers import RetryPolicy
from cayu.tasks import TaskRetryBudget, TaskRetryPolicy


def _findings(root: Path, source: str) -> list[tuple[str, str, str]]:
    (root / "operations").mkdir()
    (root / "operations/__init__.py").write_text("")
    (root / "operations/task_health.py").write_text(source)
    return [
        (item.code, item.path, item.parameters["reason"]) for item in _check_import_inertness(root)
    ]


def test_module_level_task_retry_budget_is_declarative(tmp_path):
    source = (
        "from cayu.tasks import TaskRetryBudget\n\n"
        "TASK_BUDGET = TaskRetryBudget(max_attempts=1, max_elapsed_seconds=None)\n"
    )
    assert _findings(tmp_path, source) == []


def test_literal_configuration_models_are_declarative(tmp_path):
    source = """import cayu.tasks as tasks
from decimal import Decimal

from cayu.budgets import RunLimits
from cayu.budgets.run_limits import RunLimits as Limits
from cayu.providers import RetryPolicy
from cayu.tasks import TaskRetryBudget, TaskRetryPolicy
from cayu.tasks.retry_budgets import TaskRetryBudget as Budget

DEFAULT = TaskRetryBudget()
UNLIMITED = TaskRetryBudget.unlimited()
QUALIFIED = tasks.TaskRetryBudget(max_attempts=2, max_elapsed_seconds=600)
QUALIFIED_UNLIMITED = tasks.TaskRetryBudget.unlimited()
ALIASED = Budget(max_attempts=5)
POLICY = TaskRetryPolicy(
    max_attempts=3,
    max_elapsed_seconds=900.0,
    max_estimated_cost=Decimal("2.50"),
    cost_currency="USD",
)
LIMITS = RunLimits(max_total_tokens=100_000, scope="session", on_limit="finish")
ALIASED_LIMITS = Limits(max_tool_calls=20)
PROVIDER_RETRY = RetryPolicy(max_attempts=3, retry_on_status_codes=(429, 503))
"""
    assert _findings(tmp_path, source) == []


@pytest.mark.parametrize(
    "statement",
    [
        "BUDGET = TaskRetryBudget(max_attempts=load())",
        "BUDGET = TaskRetryBudget(**load())",
        "BUDGET = TaskRetryBudget(**{'max_attempts': 1})",
        "BUDGET = TaskRetryBudget(max_attempt=1)",
        "BUDGET = TaskRetryBudget.unlimited(load())",
        "BUDGET = TaskRetryBudget.model_validate(load())",
        "BUDGET = TaskRetryBudget.model_construct()",
        "POLICY = TaskRetryPolicy(max_attempts=3, max_estimated_cost=Decimal(0.1))",
        "POLICY = TaskRetryPolicy(max_attempts=3, max_estimated_cost=Decimal(load()))",
        "POLICY = TaskRetryPolicy(max_attempts=3, max_estimated_cost=load())",
        "LIMITS = RunLimits(5)",
        "LIMITS = RunLimits(max_total_tokens=load())",
    ],
)
def test_computed_configuration_values_stay_rejected(tmp_path, statement):
    source = (
        "from decimal import Decimal\n"
        "from cayu.budgets import RunLimits\n"
        "from cayu.tasks import TaskRetryBudget, TaskRetryPolicy\n"
        "from vendor import load\n" + statement + "\n"
    )
    line = 4 + len(statement.splitlines())
    assert _findings(tmp_path, source) == [
        (
            "SCAFFOLD_IMPORT_SIDE_EFFECT",
            f"operations/task_health.py:{line}",
            "unsupported_expression",
        )
    ]


def test_rebound_model_name_is_not_trusted(tmp_path):
    source = (
        "from cayu.tasks import TaskRetryBudget\n"
        "from vendor import load\n"
        "TaskRetryBudget = load()\n"
        "BUDGET = TaskRetryBudget(max_attempts=1)\n"
    )
    assert ("SCAFFOLD_IMPORT_SIDE_EFFECT", "operations/task_health.py:4") in [
        (code, path) for code, path, _reason in _findings(tmp_path, source)
    ]


def test_project_local_cayu_module_is_not_trusted(tmp_path):
    (tmp_path / "cayu").mkdir()
    (tmp_path / "cayu/__init__.py").write_text("")
    (tmp_path / "cayu/tasks.py").write_text("TaskRetryBudget = dict\n")
    source = "from cayu.tasks import TaskRetryBudget\nBUDGET = TaskRetryBudget(max_attempts=1)\n"
    assert ("SCAFFOLD_IMPORT_SIDE_EFFECT", "operations/task_health.py:2") in [
        (code, path) for code, path, _reason in _findings(tmp_path, source)
    ]


@pytest.mark.parametrize("model", [RetryPolicy, TaskRetryBudget, TaskRetryPolicy, RunLimits])
def test_reviewed_fields_match_each_configuration_model(model):
    # A new field must be reviewed in the check before literal construction admits it.
    reviewed = {
        fields
        for (_module, name), fields in _CAYU_CONFIGURATION_MODELS.items()
        if name == model.__name__
    }
    assert reviewed == {frozenset(model.model_fields)}
