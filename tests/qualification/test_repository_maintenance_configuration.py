"""Configured production factory; no paid calls or live database proof."""

import importlib
import json
import tomllib
import warnings

import pytest

from cayu import PostgresBudgetLedger
from cayu.cli.project import build_project_app
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_deployment import deployment as deployment


def test_declared_factory_constructs_fresh_budgeted_graphs(deployment, project, monkeypatch):
    from psycopg_pool import AsyncConnectionPool

    _module, source, provider = deployment
    monkeypatch.setenv("CAYU_WORKSPACE_ROOT", str(source))
    monkeypatch.setenv("CAYU_MAINTENANCE_BUDGET_JSON", denial_policy().model_dump_json())

    async def forbidden(*args, **kwargs):
        pytest.fail("Factory connected to PostgreSQL")

    monkeypatch.setattr(AsyncConnectionPool, "open", forbidden)
    config = tomllib.loads((project / "pyproject.toml").read_text())
    target = config["tool"]["cayu"]["factory"]
    assert target == "app:build_maintenance_app"
    first, second = build_project_app(target), build_project_app(target)
    assert first is not second
    assert first.budget_policy is not None and second.budget_policy is not None
    assert first.budget_policy == second.budget_policy == denial_policy()
    assert first.budget_policy is not second.budget_policy
    assert type(first.budget_ledger) is PostgresBudgetLedger
    assert first.budget_ledger is not second.budget_ledger
    assert first.session_store is not second.session_store
    assert first.get_provider() is provider and not provider.requests
    first.budget_policy.limits[0].currency = "EUR"
    assert second.budget_policy.limits[0].currency == "USD"


@pytest.mark.parametrize(
    "invalid",
    ["missing", "empty", "json", "duplicate", "oversize", "constant", "shape", "cap", "unpriced"],
)
def test_bad_budget_rejects_before_construction_without_diagnostics(
    deployment, monkeypatch, caplog, capsys, invalid
):
    module, _source, _provider = deployment
    root = importlib.import_module("app")
    canary = "private-budget-canary"
    value = denial_policy().model_dump(mode="json")
    raw = json.dumps(value)
    if invalid == "missing":
        monkeypatch.delenv("CAYU_MAINTENANCE_BUDGET_JSON", raising=False)
    else:
        if invalid == "empty":
            raw = ""
        elif invalid == "json":
            raw = canary
        elif invalid == "duplicate":
            raw = '{"limits":[],"limits":[],"secret":"' + canary + '"}'
        elif invalid == "oversize":
            raw = canary * 6000
        elif invalid == "constant":
            raw = '{"limits":NaN,"secret":"' + canary + '"}'
        elif invalid == "shape":
            raw = json.dumps({"limits": [canary]})
        elif invalid == "cap":
            value["limits"][0]["max_estimated_cost"] = "1.01"
            raw = json.dumps(value)
        else:
            value["limits"][0]["allow_unpriced"] = True
            raw = json.dumps(value)
        monkeypatch.setenv("CAYU_MAINTENANCE_BUDGET_JSON", raw)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid policy reached application construction")

    monkeypatch.setattr(module, "build_maintenance_deployment", forbidden)
    with (
        warnings.catch_warnings(record=True) as recorded,
        pytest.raises(ValueError, match="maintenance budget configuration") as caught,
    ):
        root.build_maintenance_app()
    assert canary not in str(caught.value) + repr(caught.value)
    output = capsys.readouterr()
    assert not recorded
    assert canary not in output.out + output.err + caplog.text
