"""Installed example extends the real emitted application preparation owner."""

import asyncio
from dataclasses import replace

import pytest

from cayu import CodingProductRunner
from cayu.guides.coding_host import (
    BusinessStore,
    Reservation,
    SettlementConflict,
    extend_generated_application,
)
from tests.qualification.test_repository_maintenance_request import consumer as consumer
from tests.qualification.test_repository_maintenance_request import project as project


def test_generated_run_reserves_before_dispatch_and_never_synthesizes_recovery(
    consumer, tmp_path, monkeypatch
):
    application, task, provider, *_ = consumer
    from tests.cli.test_coding_host_settlement_example import reservation

    pricing = reservation().pricing
    business = BusinessStore(tmp_path / "business.sqlite")
    wrapped = extend_generated_application(
        application, business=business, tenant="example", pricing=pricing, cost_basis="synthetic"
    )
    dispatched = []

    async def dispatch(self, request, run_request, **kwargs):
        expected = Reservation(
            tenant="example",
            public_id=task.product_run_id,
            request=request,
            pricing=pricing,
            cost_basis="synthetic",
        )
        assert business.read(expected) is None
        dispatched.append(request)
        # Stop exactly at the public runner boundary: this is admission proof,
        # not simulated successful execution or worker-loss qualification.
        raise LookupError("public dispatch boundary")

    async def scenario():
        monkeypatch.setattr(CodingProductRunner, "run", dispatch)
        with pytest.raises(LookupError, match="public dispatch boundary"):
            await wrapped.run(task)
        assert len(dispatched) == 1 and not provider.requests
        expected = Reservation(
            tenant="example",
            public_id=task.product_run_id,
            request=dispatched[0],
            pricing=pricing,
            cost_basis="synthetic",
        )
        other = replace(
            task,
            product_run_id="competing-product",
            session_id="competing-session",
            task_id="competing-task",
        )
        with pytest.raises(SettlementConflict, match="unsettled owner"):
            await wrapped.run(other)
        assert len(dispatched) == 1
        business.settle(expected, {"classification": "cancelled"})

        async def read_only(self, request, **kwargs):
            assert request == expected.request
            return "historical reconstruction"

        monkeypatch.setattr(CodingProductRunner, "recover_settled_execution", read_only)
        assert await wrapped.run(task) == "historical reconstruction"
        assert len(dispatched) == 1
        missing = extend_generated_application(
            application,
            business=BusinessStore(tmp_path / "missing.sqlite"),
            tenant="example",
            pricing=pricing,
            cost_basis="synthetic",
        )
        with pytest.raises(SettlementConflict, match="absent or conflicting"):
            await missing.recover_settled(task)
        assert not provider.requests

    asyncio.run(scenario())
