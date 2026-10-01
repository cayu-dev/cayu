from __future__ import annotations

import asyncio

import pytest
from tests.evals.test_workflow_eval_target import _register_app

from cayu import Environment, EnvironmentSpec, RunRequest
from cayu.evals.runner import (
    _inspect_settled_execution_profile,
    _ProfileRevalidationSettlementUnproven,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity


def _profile_app():
    app = _register_app()
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="settlement",
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:workflow-profile-settlement",
                    behavior_version="1",
                    implementation_version="1",
                ),
            )
        ),
        default=True,
    )
    return app, app._environments["settlement"].workspace_mutation_fence


@pytest.mark.parametrize("remaining_seconds", [0.0, 0.05])
def test_first_profile_inspection_obeys_remaining_close_budget(remaining_seconds):
    async def exercise():
        app, fence = _profile_app()
        release = asyncio.Event()
        entered = asyncio.Event()

        async def settle():
            entered.set()
            await release.wait()
            return True

        fence.fail_closed(settle)
        try:
            with pytest.raises(_ProfileRevalidationSettlementUnproven):
                await asyncio.wait_for(
                    _inspect_settled_execution_profile(
                        app,
                        RunRequest(agent_name="first", messages=[]),
                        deadline=asyncio.get_running_loop().time() + remaining_seconds,
                    ),
                    timeout=2,
                )
            assert entered.is_set() is (remaining_seconds > 0)
        finally:
            release.set()
            await fence.wait_until_available()

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_first_profile_inspection_waits_for_settlement_and_preserves_cancellation(cancel):
    async def exercise():
        app, fence = _profile_app()
        request = RunRequest(agent_name="first", messages=[])
        expected = await app.inspect_run_execution_profile(request)
        release = asyncio.Event()
        entered = asyncio.Event()

        async def settle():
            entered.set()
            await release.wait()
            return True

        fence.fail_closed(settle)
        inspection = asyncio.create_task(
            _inspect_settled_execution_profile(
                app, request, deadline=asyncio.get_running_loop().time() + 10
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            if cancel:
                inspection.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await inspection
                assert inspection.cancelling() == 1
            else:
                release.set()
                assert await inspection == expected
        finally:
            release.set()
            await fence.wait_until_available()
            if not inspection.done():
                inspection.cancel()
            await asyncio.gather(inspection, return_exceptions=True)

    asyncio.run(exercise())


def test_inspection_owned_timeout_is_not_a_settlement_deadline(monkeypatch):
    async def exercise():
        app, _ = _profile_app()

        async def fails(request):
            raise TimeoutError("inspection-owned timeout")

        monkeypatch.setattr(app, "inspect_run_execution_profile", fails)
        with pytest.raises(TimeoutError, match="inspection-owned timeout"):
            await _inspect_settled_execution_profile(
                app,
                RunRequest(agent_name="first", messages=[]),
                deadline=asyncio.get_running_loop().time() + 10,
            )

    asyncio.run(exercise())
