"""Explicit lineage is part of exact coding-product admission, not ambient context."""

import asyncio
from typing import Any

import pytest
from tests.core.test_coding_product_recovery import product as product

from cayu import CodingProductAdmissionError, admit_or_recover_coding_product_request


@pytest.mark.parametrize("field", ["parent_session_id", "causal_budget_id"])
def test_runner_rejects_unbound_lineage_before_any_artifact_or_dispatch(product, field):
    runner, request, run, _ = product

    async def scenario():
        with pytest.raises(ValueError, match="identities conflict"):
            await runner.run(request, run.model_copy(update={field: "different-root"}))
        listed = await runner.repository.store.list(session_id=request.session_id)
        assert listed.artifacts == () and listed.total_count == 0 and not listed.truncated

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["parent_session_id", "causal_budget_id"])
def test_admission_replay_requires_each_expected_lineage_field(product, field):
    runner, request, run, _ = product

    async def scenario():
        arguments: dict[str, Any] = dict(
            repository=runner.repository,
            product_run_id=request.product_run_id,
            session_id=request.session_id,
            agent_name=request.agent_name,
            task_id=request.task.task_id,
            messages=run.messages,
            source_workspace=runner.source_workspace,
            source_origin_id=request.source.origin_id,
            source_destination_id=request.source.destination_id,
            source_git_baseline=request.source.git_baseline,
            runtime=request.runtime,
            settlement=request.settlement,
            observation_limits=request.source.observation_limits,
            parent_session_id="workflow-root",
            causal_budget_id="workflow-budget",
        )
        admitted = await admit_or_recover_coding_product_request(**arguments)
        assert admitted.parent_session_id == "workflow-root"
        assert admitted.causal_budget_id == "workflow-budget"
        assert await admit_or_recover_coding_product_request(**arguments) == admitted
        arguments[field] = "different-root"
        with pytest.raises(CodingProductAdmissionError, match="different caller authority"):
            await admit_or_recover_coding_product_request(**arguments)
        assert (
            await runner.repository.load_request(
                request.product_run_id, session_id=request.session_id
            )
            == admitted
        )
        assert admitted.fingerprint != request.fingerprint

    asyncio.run(scenario())
