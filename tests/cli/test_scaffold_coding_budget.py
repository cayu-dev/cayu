"""Generated coding constructors delegate budget authority to the existing Runtime."""

import importlib
from decimal import Decimal

import pytest
from tests.core.test_queued_session_messages import RecordingOneShotProvider
from tests.qualification.repository_maintenance_case import materialize_seed_repository

from cayu import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    DockerImageIdentity,
    InMemoryBudgetLedger,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalArtifactStore,
    ModelPrice,
    PriceBook,
)
from cayu.cli.project import project_context
from cayu.cli.scaffold import project_files


def denial_policy():
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("0.5"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name=RecordingOneShotProvider.name,
                            model="fake-model",
                            input_per_million=Decimal("1000000"),
                            output_per_million=Decimal("1000000"),
                        ),
                    )
                ),
                reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=0),
            ),
        )
    )


@pytest.mark.parametrize(
    "execution,entrance",
    [(None, "build_app"), ("docker", "build_app"), ("docker", "build_coding_product_application")],
)
def test_coding_constructors_forward_budget_owners(tmp_path, monkeypatch, execution, entrance):
    root = tmp_path / "consumer"
    for relative, content in project_files(
        "budget-coder", preset="coding", database="sqlite", execution=execution
    ).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = tmp_path / "source"
    materialize_seed_repository(source)
    with project_context(root):
        operations = importlib.import_module("operations.coding")
        module = importlib.import_module("app")
        if execution == "docker":
            profile = operations._python_toolchain_profile(
                DockerImageIdentity(reference="fixture@sha256:" + "a" * 64)
            )
            monkeypatch.setattr(
                operations,
                "_configured_docker_authority",
                lambda _root: (profile, "/usr/bin/docker"),
            )
        policy, ledger = denial_policy(), InMemoryBudgetLedger()
        constructed = getattr(module, entrance)(
            provider=RecordingOneShotProvider(),
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
            knowledge_store=InMemoryKnowledgeStore(access_scope=operations._knowledge_scope()),
            workspace_root=source,
            artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="budget-artifacts"),
            budget_policy=policy,
            budget_ledger=ledger,
        )
        app = constructed.app if entrance == "build_coding_product_application" else constructed
        assert app.budget_ledger is ledger
        assert app.budget_policy == policy and app.budget_policy is not policy
        policy.limits[0].max_estimated_cost = Decimal("100")
        assert app.budget_policy.limits[0].max_estimated_cost == Decimal("0.5")
